"""
Demo 13a · 上下文爆炸
======================

配套：第 13 节《上下文管理：从上下文到 ContextManager》

用最朴素的方式实现"多轮对话"——每一轮都把【从第一轮开始的全部历史】
原样重新发给模型，不做任何管理。

模拟一段对话：每一轮都是同样的指令——"总结一下目前的讨论 + 补充 3 条新建议"。

观察两件事：
  1. prompt_tokens 随轮数的增长曲线——这就是"没有上下文管理"的真实代价。
  2. 缓存命中率随轮数的变化——因为每轮都是"上一轮完整前缀 + 新增内容"，
     天然构成可复用的公共前缀，网关的 Prompt Caching 会识别并命中这部分。
     命中率越高，重复内容计费越便宜，但窗口占用一个 token 都不会少。

运行方式：
    cd /workspace/04-context-and-memory/lab/
    source /workspace/.venv/bin/activate
    python demo_13a_explosion.py
"""

import unicodedata

from _env import chat


# ============================================================
# 小工具：按"显示宽度"对齐表格列（中文字符按 2 格算，避免表头和数据错位）
# ============================================================
def _visual_width(s: str) -> int:
    """计算字符串在等宽终端里的显示宽度（中文字符占 2 格）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _rjust_visual(s: str, width: int) -> str:
    """按显示宽度右对齐（兼容中英文混排）。"""
    return " " * max(width - _visual_width(s), 0) + s


def _print_row(widths: list[int], values: list[str]) -> None:
    print("  ".join(_rjust_visual(v, w) for w, v in zip(widths, values)))


# ============================================================
# 上下文爆炸 —— 多轮对话里 prompt_tokens 是怎么涨起来的
# ============================================================
# 每一轮都发同样的指令：这正是真实场景里最常见的模式——
# 用户不断追问、模型不断回答，但没有人主动清理历史。
FOLLOW_UP_INSTRUCTION = "请用 2-3 句话总结一下我们目前讨论的内容，并补充 3 条新的建议。"


def run_context_explosion(n_rounds: int = 20) -> list[dict]:
    """跑 n_rounds 轮"重复指令"对话，记录并【逐行流式】打印每一轮真实的 prompt_tokens。

    关键点：messages 列表只增不减——每一轮都在上一轮的基础上
    再追加一对 (user, assistant)，然后把【完整历史】原样重新发给模型。
    这就是最朴素、也是生产里最常见的"无管理"实现方式。

    每一轮的 API 调用结束后，立刻打印该轮的表格行——不等全部跑完再统一出表，
    这样能实时看到 prompt_tokens 是怎么一轮一轮涨起来的。

    额外统计缓存命中率：因为每一轮的 messages 都是"上一轮的完整前缀 + 新增内容"，
    天然构成可复用的公共前缀，网关的 Prompt Caching 机制会识别并命中这部分——
    缓存命中率越高，说明"重复发送的历史"被识别出来了，计费/延迟上会打折，
    但注意：这不代表 token 总量不再增长，窗口占用依然在持续膨胀。
    """
    messages = [
        {"role": "system", "content": "你是一个资深产品顾问，帮用户打磨一个新产品创意。"},
        {"role": "user", "content": "我们在讨论一个随身空气质量监测器的创意，先给点初步建议。"},
    ]

    widths = [4, 13, 13, 8, 10, 10, 12]
    _print_row(
        widths,
        ["轮次", "messages 条数", "prompt_tokens", "本轮新增", "缓存命中", "命中率", "total_tokens"],
    )
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))

    records = []
    prev_prompt_tokens = None
    for i in range(1, n_rounds + 1):
        # 不管上一轮聊了什么，每一轮都追加同样的指令——
        # 但因为历史在变长，模型每次"总结"的内容也会越来越长。
        messages.append({"role": "user", "content": FOLLOW_UP_INSTRUCTION})

        msg = chat(messages, temperature=0, max_tokens=150)
        messages.append({"role": "assistant", "content": msg.content or ""})

        usage = msg.usage
        delta = "-" if prev_prompt_tokens is None else f"+{usage.prompt_tokens - prev_prompt_tokens:,}"

        # 缓存命中 token 数：优先读 prompt_tokens_details.cached_tokens（OpenAI 标准字段），
        # 部分网关（如本例的 DeepSeek 兼容网关）还会额外给 prompt_cache_hit_tokens，取较大值兜底。
        cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        hit_rate = f"{cached / usage.prompt_tokens:.0%}" if usage.prompt_tokens else "-"

        _print_row(
            widths,
            [
                str(i),
                str(len(messages)),
                f"{usage.prompt_tokens:,}",
                delta,
                f"{cached:,}",
                hit_rate,
                f"{usage.total_tokens:,}",
            ],
        )
        prev_prompt_tokens = usage.prompt_tokens

        records.append(
            {
                "round": i,
                "n_messages": len(messages),
                "prompt_tokens": usage.prompt_tokens,
                "total_tokens": usage.total_tokens,
                "cached_tokens": cached,
            }
        )
    return records


def summarize_explosion(records: list[dict]) -> int:
    """打印膨胀倍数，并线性外推"按这个速度，撞到 128K 窗口还要几轮"。

    返回最后一轮的 prompt_tokens。
    """
    first, last = records[0], records[-1]
    growth = last["prompt_tokens"] / max(first["prompt_tokens"], 1)
    print(
        f"\n💡 第 1 轮 prompt = {first['prompt_tokens']:,} tokens"
        f" → 第 {last['round']} 轮 prompt = {last['prompt_tokens']:,} tokens"
        f"，涨了 {growth:.0f} 倍——而这还只是一个「反复总结」的简单任务。"
    )

    if last["round"] > first["round"]:
        delta_per_round = (last["prompt_tokens"] - first["prompt_tokens"]) / (last["round"] - first["round"])
        if delta_per_round > 0:
            rounds_to_128k = first["round"] + (128_000 - first["prompt_tokens"]) / delta_per_round
            print(
                f"   按这个增速线性外推，大约到第 {rounds_to_128k:.0f} 轮就会撞到 128K 窗口"
                "（GPT-4o / DeepSeek-V3 的典型上限）。"
            )

    last_hit_rate = last["cached_tokens"] / max(last["prompt_tokens"], 1)
    print(
        f"   最后一轮缓存命中率 {last_hit_rate:.0%}：因为每轮都是「上一轮完整前缀 + 新增内容」，"
        "重复的历史前缀被网关的 Prompt Caching 识别命中了，计费/延迟会打折——"
        "但命中的只是“省钱”，窗口占用并不会因此减少。"
    )
    return last["prompt_tokens"]


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    print("实验 · 上下文爆炸：20 轮「重复指令」对话，看 prompt_tokens 怎么涨")
    print("=" * 70)
    records = run_context_explosion(n_rounds=20)
    summarize_explosion(records)

    print("\n💡 下一步（demo_13b_strategies.py）：怎么在撞墙之前，主动把这座「大山」压下去")


if __name__ == "__main__":
    main()
