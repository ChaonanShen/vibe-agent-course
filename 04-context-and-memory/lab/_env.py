"""Demo 共用：LLM 配置 + 流式客户端封装。

- 优先使用显式的 `BASE_URL` / `API_KEY`；
- 若在 CNB 平台运行（未显式配置），则自动从 `CNB_API_ENDPOINT` + `CNB_REPO_SLUG`
  拼接 Base URL，并使用 `CNB_TOKEN` 作为 API Key。
- `MODEL_ID` 未设置时默认 `deepseek-v4.1-flash`。
- `count_tokens()` 默认用 DeepSeek-V3 词表估算，`TOKENIZER_SPEC` 可换词表（含离线文件），
  `TOKEN_ESTIMATE_RATIO` 可做比例校准。
- 接口仅支持流式输出，chat() 内部强制 stream=True，
  拼接完整响应后返回 message 对象，模拟非流式行为。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ============================================================
# 环境变量读取
# ============================================================

def get_base_url() -> str:
    """获取 LLM API 的 Base URL。"""
    base_url = os.environ.get("BASE_URL")
    if base_url:
        return base_url

    endpoint = os.environ.get("CNB_API_ENDPOINT", "")
    repo_slug = os.environ.get("CNB_REPO_SLUG", "")
    if endpoint and repo_slug:
        return f"{endpoint}/{repo_slug}/-/ai-ide/v2"

    raise EnvironmentError(
        "请设置 BASE_URL，或同时设置 CNB_API_ENDPOINT 和 CNB_REPO_SLUG 环境变量"
    )


def get_api_key() -> str:
    """获取 API Key。优先读取 `API_KEY`，否则读取 `CNB_TOKEN`。"""
    api_key = os.environ.get("API_KEY") or os.environ.get("CNB_TOKEN")
    if not api_key:
        raise EnvironmentError("请设置 API_KEY 或 CNB_TOKEN 环境变量")
    return api_key


def get_model_id() -> str:
    """获取模型 ID，默认 `deepseek-v4.1-flash`。"""
    return os.environ.get("MODEL_ID", "deepseek-v4.1-flash")


def make_client():
    """创建一个 OpenAI 兼容客户端。"""
    from openai import OpenAI

    return OpenAI(
        base_url=get_base_url(),
        api_key=get_api_key(),
        timeout=int(os.environ.get("LLM_TIMEOUT", "60")),
    )


# ============================================================
# Token 计数：本地估算（发请求之前就能拿到）
# ============================================================
# 为什么需要它：
#   上下文窗口上限、计费、Prompt Caching 全都按 token 算，字符数和消息条数
#   都不是模型实际读到的量（中文 1 字 ≈ 1 token，英文 4 字符 ≈ 1 token；
#   一条消息可能 5 个字也可能 5000 字）。而「要不要压缩」必须在发请求之前
#   决定，那时还拿不到 API 返回的 usage.prompt_tokens（那是事后的真实值），
#   只能本地估。demo_13a 读 usage 属于事后观测；demo_13b 的规模对比
#   和 demo_13d / 15 的压缩阈值判断则统一走这里。
#
# 词表：默认用 DeepSeek 开源的 deepseek-ai/DeepSeek-V3 词表近似课程内部模型
#   deepseek-v4.1-flash —— 后者没有公开词表，V3 的中文压缩率与它最接近
#   （约 1.6~1.7 汉字/token），比 OpenAI 的 cl100k_base 准得多。
#
# 配置方式（环境变量 TOKENIZER_SPEC）：
#   hf:deepseek-ai/DeepSeek-V3    默认；首次联网下载 tokenizer.json（约 7MB），
#                                 之后走本地缓存，不再联网
#   file:/path/to/tokenizer.json  离线：直接用本地文件，完全不联网
#   cl100k_base / o200k_base      兜底：没装 tokenizers 时用 tiktoken 的 BPE
#
# 固定校准系数 TOKEN_ESTIMATE_RATIO（默认 1.0）：词表仍有残差时，用它一次性
#   压掉系统性偏差。测法：`count_tokens(messages)` 与 API 返回的
#   `usage.prompt_tokens` 求比值，跑几轮取平均；估算偏高设 <1，偏低设 >1。
#   注意它只修"比例"，不修"结构性差异"，所以优先把词表配对；
#   要精确数字仍然只有 API 返回的 usage.prompt_tokens。
#
#   本课程实测（13a 前 5 轮，V3 词表）：ratio = real/est ≈ 0.97，即估算仅
#   偏高 3%，默认 1.0 已经够用；想更贴可设 TOKEN_ESTIMATE_RATIO=0.97。

_TOKENIZER_SPEC = os.environ.get("TOKENIZER_SPEC", "hf:deepseek-ai/DeepSeek-V3")
_TOKEN_ESTIMATE_RATIO = float(os.environ.get("TOKEN_ESTIMATE_RATIO", "1.0"))

_TOKENS_PER_MESSAGE = 4       # 每条消息的 role / 分隔符 / 特殊 token 固定开销
_TOKENS_FOR_REPLY_PRIMER = 2  # 末尾 assistant 前缀


class _HFTokenizerAdapter:
    """把 HF `tokenizers.Tokenizer` 包成 tiktoken 那样 `.encode(text) -> list[int]`。

    只为了让 `count_tokens` 能用一条代码路径应付两种编码器，不必到处 isinstance。
    """

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self.name = _TOKENIZER_SPEC

    def encode(self, text: str):
        # add_special_tokens=False：BOS/EOS 这些由 _TOKENS_PER_MESSAGE 统一算，
        # 否则每条消息会多计一份，估算又会系统性偏高
        return self._tokenizer.encode(text, add_special_tokens=False).ids


@lru_cache(maxsize=1)
def _get_encoder():
    """按 TOKENIZER_SPEC 加载编码器，三级回退：HF 词表 → tiktoken → None(粗估)。

    延迟加载，避免仅 import 本模块就触发下载；任何一步失败都往下回退，
    不让"算个 token 数"把 demo 弄挂。
    """
    spec = _TOKENIZER_SPEC

    if spec.startswith(("hf:", "file:")):
        try:
            from tokenizers import Tokenizer

            if spec.startswith("file:"):
                path = spec[len("file:"):]
            else:
                from huggingface_hub import hf_hub_download

                repo = spec[len("hf:"):]
                try:
                    # 先只查本地缓存：离线时立刻回退，不卡在联网重试上
                    path = hf_hub_download(repo, "tokenizer.json", local_files_only=True)
                except Exception:
                    path = hf_hub_download(repo, "tokenizer.json")  # 首次下载
            return _HFTokenizerAdapter(Tokenizer.from_file(path))
        except Exception:
            pass  # 没装包 / 没网 / 文件坏了，都往下回退

    try:
        import tiktoken
    except ImportError:
        return None
    for name in (spec, "cl100k_base"):
        try:
            return tiktoken.get_encoding(name)
        except Exception:
            continue
    return None


def encoder_name() -> str:
    """当前实际生效的编码器名，打日志/排查用：`hf:...` / `cl100k_base` / `rough`。"""
    enc = _get_encoder()
    if enc is None:
        return "rough"
    return getattr(enc, "name", "unknown")


def _rough_token_count(text: str) -> int:
    """三级兜底的最后一层（词表、tiktoken 都不可用时）：非 ASCII 约 1 字 1 token，
    ASCII 约 4 字符 1 token。"""
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ascii_chars = len(text) - non_ascii
    return non_ascii + (ascii_chars + 3) // 4


def count_tokens(text_or_messages: str | list[dict]) -> int:
    """估算一段文本 / 一组 messages 会占多少 token（DeepSeek-V3 词表 + 校准系数）。

    Args:
        text_or_messages: 字符串，或 OpenAI 格式的消息列表
            `[{"role": str, "content": str}, ...]`。传列表时会额外计入每条
            消息的固定开销（见 `_TOKENS_PER_MESSAGE`），更接近真实 prompt_tokens。

    Returns:
        估算的 token 数（乘以 `TOKEN_ESTIMATE_RATIO`，非精确值，说明见上方注释）。
    """
    enc = _get_encoder()
    if isinstance(text_or_messages, str):
        texts = [text_or_messages]
        overhead = 0
    else:
        texts = [str(m.get("content") or "") for m in text_or_messages]
        overhead = _TOKENS_PER_MESSAGE * len(texts) + _TOKENS_FOR_REPLY_PRIMER

    if enc is None:
        estimate = overhead + sum(_rough_token_count(t) for t in texts)
    else:
        estimate = overhead + sum(len(enc.encode(t)) for t in texts)
    return int(estimate * _TOKEN_ESTIMATE_RATIO)


# ============================================================
# 核心方法：模拟非流式响应
# ============================================================

def chat(
    messages: list[dict],
    *,
    temperature: float = 0,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    stream: bool = False,
    client=None,
    model: str | None = None,
) -> "Message":
    """调用 LLM，返回完整的 Message 对象（内部强制流式收集）。

    参数：
        stream: 是否逐字打印 content（打字机效果）。默认 False 静默收集。

    返回值：
        Message 对象，属性：
        - msg.content: str | None  （完整文本）
        - msg.tool_calls: list[ToolCall] | None
          每个 ToolCall 有：tc.id, tc.function.name, tc.function.arguments

    用法：
        # 静默收集（工具调用等中间步骤）
        msg = chat(messages, tools=TOOLS_SCHEMA)

        # 逐字打印（最终回答）
        msg = chat(messages, stream=True)
    """
    if client is None:
        client = make_client()
    if model is None:
        model = get_model_id()

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,  # 强制流式（接口限制）
        "stream_options": {"include_usage": True},
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    response = client.chat.completions.create(**kwargs)
    return _collect_stream(response, print_content=stream)


# 别名，语义更清晰
chat_stream = lambda messages, **kwargs: chat(messages, stream=True, **kwargs)


# ============================================================
# 内部实现：流式收集
# ============================================================

def _collect_stream(response, print_content: bool = False) -> "Message":
    """流式收集所有 chunk，拼装为完整的 Message 对象。"""
    content_parts: list[str] = []
    tool_calls_map: dict[int, dict] = {}
    usage = None  # 最后一个 chunk 可能携带 usage 信息

    for chunk in response:
        if hasattr(chunk, "usage") and chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content:
            content_parts.append(delta.content)
            if print_content:
                print(delta.content, end="", flush=True)

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {"id": "", "name": "", "arguments": ""}
                if tc_delta.id:
                    tool_calls_map[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_calls_map[idx]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls_map[idx]["arguments"] += tc_delta.function.arguments

    if print_content and content_parts:
        print()  # 换行

    return Message(
        content="".join(content_parts) if content_parts else None,
        tool_calls=[
            ToolCall(
                id=tool_calls_map[i]["id"],
                function=Function(
                    name=tool_calls_map[i]["name"],
                    arguments=tool_calls_map[i]["arguments"],
                ),
            )
            for i in sorted(tool_calls_map)
        ] if tool_calls_map else None,
        usage=usage,
    )


# ============================================================
# 数据类：模拟 OpenAI 非流式响应的 message 结构
# ============================================================

class Function:
    """tool_call.function"""
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments

    def __repr__(self):
        return f"Function(name={self.name!r}, arguments={self.arguments!r})"


class ToolCall:
    """单个 tool_call"""
    def __init__(self, id: str, function: Function):
        self.id = id
        self.function = function

    def __repr__(self):
        return f"ToolCall(id={self.id!r}, function={self.function!r})"


class Message:
    """模拟 ChatCompletionMessage"""
    def __init__(self, content: Optional[str], tool_calls: Optional[list] = None,
                 usage=None):
        self.content = content
        self.tool_calls = tool_calls
        self.usage = usage

    def __repr__(self):
        if self.tool_calls:
            return f"Message(content={self.content!r}, tool_calls={self.tool_calls!r})"
        return f"Message(content={self.content!r})"
