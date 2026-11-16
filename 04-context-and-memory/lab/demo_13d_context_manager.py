"""
Demo 13d · 产出：ContextManager 类
======================================

配套：第 13 节《上下文管理：从上下文到 ContextManager》

把"何时压缩（防爆炸）+ 怎么压缩（防变笨）+ 尽量少压缩（保缓存）"
三件事统一封装成一个可复用的类，而不是三个互不知道对方存在的函数。

依赖 demo_13b_strategies.py 里的 strategy_hierarchical 做实际压缩，
用 demo_13a 的同一段对话逐轮喂入，观察 ContextManager 什么时候触发压缩、
什么时候什么都不做（append-only，为缓存命中率让路）。

运行方式：
    cd /workspace/04-context-and-memory/lab/
    source /workspace/.venv/bin/activate
    python demo_13d_context_manager.py
"""

from _env import chat, count_tokens
from demo_13a_explosion import (
    FOLLOW_UP_INSTRUCTION,
    OPENING_USER,
    SYSTEM_PROMPT,
)
from demo_13b_strategies import _print_row, extract_key_facts_llm, strategy_hierarchical


class ContextManager:
    """统一管理三件事：何时压缩（防爆炸）、怎么压缩（防变笨）、尽量少压缩（保缓存）。

    不是把截断/摘要/分层三个函数简单包一层，而是显式解决前面讲的核心矛盾：
    压缩能防爆炸，但会打断缓存前缀——所以"压缩判断"本身必须是节流的、
    有阈值意识的，而不是"消息数变多就无脑压一次"。

    对外只有两个方法：
        - `add_messages(*messages)`：追加新消息（写入），内部自动判断要不要压缩。
        - `messages()`：读取当前应该发给模型的消息列表（纯读取，不触发任何副作用）。

    `ContextManager` 自己持有 messages 这份状态——调用方不需要（也不应该）
    自己拿着一份 messages 列表来回传进传出，这正是"统一封装成一个类"的意义：
    状态和"状态该怎么变"的规则放在一起，而不是让调用方代劳。

    设计要点：
        1. **Append-only**：两次压缩之间，只往内部列表尾部追加，不改写
           已经发送过的历史，让这段前缀能尽量长时间地被缓存命中。
        2. **按 token 阈值触发，而非数消息条数 / 字符数**：窗口上限、计费、
           缓存全都按 token 算，一条消息可能 5 个字也可能 5000 字，数条数
           完全反映不出真实压力。只有 `count_tokens(messages)` 超过
           `max_tokens` 才压缩，避免像"每轮都摘要"那样把缓存命中率打成 0%
           （原因见讲义第 1 节的前缀缓存三条规则）。
        3. **压缩策略用分层 + LLM 事实抽取**：压缩这一下必须把活干好——单独
           抽一次关键事实，不随早期对话一起被摘要掉，解决"防变笨"。

    Attributes:
        keep_last: 压缩时保留的最近消息条数，这些消息不会被摘要化。
        max_tokens: 触发压缩的 token 阈值，超过此值才会压缩。

    Example:
        >>> ctx = ContextManager(keep_last=6, max_tokens=2000)
        >>> ctx.add_messages({"role": "system", "content": "你是一个资深产品顾问。"})
        >>> ctx.add_messages({"role": "user", "content": "我准备 6 月去北京玩 7 天。"})
        >>> reply = llm.chat(ctx.messages())
        >>> ctx.add_messages({"role": "assistant", "content": reply})
    """

    def __init__(self, keep_last: int = 6, max_tokens: int = 2000) -> None:
        self._messages: list[dict] = []
        self._keep_last = keep_last if keep_last > 0 else 6
        self._max_tokens = max_tokens if max_tokens > 0 else 2000
        self._last_compressed = False  # 上一次 add_messages 调用是否真的触发了压缩

    @property
    def keep_last(self) -> int:
        return self._keep_last

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def last_compressed(self) -> bool:
        """上一次调用 `add_messages` 是否真的触发了压缩，仅用于观测/打日志。"""
        return self._last_compressed

    def _should_compress(self, messages: list[dict]) -> bool:
        """阈值触发判断：只有真正接近窗口压力时才压缩，不是消息一多就压。

        用 `_env.count_tokens` 估算当前 messages 会占多少 token，而不是数消息
        条数或字符数——模型读的是 token，窗口上限也是 token。注意这是发请求
        之前的本地估算；API 返回的 usage.prompt_tokens 是事后真实值，来不及
        用来做"要不要压缩"的决策。

        内部实现细节，外部不需要关心，也不应该自己拿这个判断结果去决定
        要不要调 `_compress`——那样每个调用方都要重写一遍判断逻辑，
        正是这个类存在的理由就是不让这种事发生。
        """
        return count_tokens(messages) > self._max_tokens

    def _compress(self, messages: list[dict]) -> list[dict]:
        """执行一次分层压缩（事实抽取显式走 LLM，见 demo_13b）。"""
        return strategy_hierarchical(
            messages, keep_last=self._keep_last, fact_extractor=extract_key_facts_llm
        )

    def add_messages(self, *new_messages: dict) -> None:
        """追加一条或多条新消息，内部自动判断要不要压缩——这是唯一的写入口。

        只要调用这一个方法，剩下的事（该不该压、怎么压）全部由
        `ContextManager` 自己完成，调用方不需要关心。
        """
        self._messages.extend(new_messages)
        if self._should_compress(self._messages):
            self._last_compressed = True
            self._messages = self._compress(self._messages)
        else:
            self._last_compressed = False

    def messages(self) -> list[dict]:
        """读取当前应该发给模型的消息列表——这是唯一的读取口，纯读取、
        不会触发压缩或任何其他副作用（副作用只发生在 `add_messages` 里）。

        返回的是内部列表的浅拷贝，调用方对返回值的修改不会影响内部状态。
        """
        return list(self._messages)


def run_context_manager_demo(n_rounds: int = 20) -> None:
    """用 13a 的同一段对话跑一遍，展示 ContextManager 什么时候压、什么时候不压。"""
    print("ContextManager 实际运行：同一段对话，按 token 阈值压缩，而不是每轮都压")
    print("=" * 78)

    ctx = ContextManager(keep_last=6, max_tokens=600)
    print(f"压缩阈值：{ctx.max_tokens:,} tokens（估算，窗口上限也按 token 算）\n")

    widths = [4, 8, 12, 14, 10]
    _print_row(widths, ["轮次", "条数", "tokens(估)", "prompt_tokens", "是否压缩"])
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))

    # 13a 完全相同的开局：system + 首轮 user
    ctx.add_messages(
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": OPENING_USER},
    )

    for i in range(1, n_rounds + 1):
        # 一轮里有两次写入（user / assistant），任一次触发压缩都算本轮压过
        compressed = False

        ctx.add_messages({"role": "user", "content": FOLLOW_UP_INSTRUCTION})
        compressed = compressed or ctx.last_compressed

        # 和 13a 一样发请求，只是发之前 ContextManager 可能已经压过一次了
        msg = chat(ctx.messages(), temperature=0, max_tokens=150)
        ctx.add_messages({"role": "assistant", "content": msg.content or ""})
        compressed = compressed or ctx.last_compressed

        usage = msg.usage
        _print_row(
            widths,
            [
                str(i),
                str(len(ctx.messages())),
                f"{count_tokens(ctx.messages()):,}",
                f"{usage.prompt_tokens:,}",
                "✅ 压缩了" if compressed else "-",
            ],
        )

    final_messages = ctx.messages()
    print(f"\n💡 最终 {len(final_messages)} 条 / 约 {count_tokens(final_messages):,} tokens。")
    print("   ContextManager 只在估算 token 数超过阈值时才压缩（防爆炸），")
    print("   压缩时用 LLM 事实抽取保住关键事实，包括关键词抓不到的隐性约束（防变笨），")
    print("   两次压缩之间 append-only、前缀不变（尽量保住缓存命中率）。")


if __name__ == "__main__":
    run_context_manager_demo()
