"""
Demo 13c · 真实验证"压缩税"：压缩会打断缓存前缀
====================================================

配套：第 13 节《上下文管理：从上下文到 ContextManager》

核心矛盾：压缩能防爆炸，但会打断缓存前缀。本 demo 用真实 API 调用验证
这件事——跑一段"重复指令"对话（缓存命中率本应随轮数一路爬升），在第 N
轮主动触发一次摘要压缩（strategy_summary，来自 demo_13b_strategies.py），
观察缓存命中率是不是从高位直接摔回 0%，然后要重新从零爬升。

运行方式：
    cd /workspace/04-context-and-memory/lab/
    source /workspace/.venv/bin/activate
    python demo_13c_cache_tax.py
"""

from _env import chat
from demo_13b_strategies import _print_row, strategy_summary


FOLLOW_UP_INSTRUCTION = "请用 2-3 句话总结一下我们目前讨论的内容，并补充 3 条新的建议。"


def run_compaction_cache_tax_experiment(
    n_rounds: int = 12, compress_at_round: int = 6, keep_last: int = 4
) -> None:
    """用真实 API 调用证明"压缩 = 缓存一次性清零"。

    跑 n_rounds 轮"重复指令"对话（和 demo_13a_explosion.py 一样的套路，
    缓存命中率会随轮数一路爬升），但在第 `compress_at_round` 轮结束后，
    主动调用 strategy_summary 对历史做一次摘要压缩（改写前缀）。

    观察重点：压缩发生的那一轮之后，缓存命中率是不是从高位直接摔回 0%，
    然后要重新从零开始爬升——这就是前面讲的"压缩税"在真实数据上的样子。
    """
    print(f"压缩税实验：第 {compress_at_round} 轮后主动触发一次摘要压缩，看缓存命中率怎么变")
    print("=" * 78)

    messages = [
        {"role": "system", "content": "你是一个资深产品顾问，帮用户打磨一个新产品创意。"},
        {"role": "user", "content": "我们在讨论一个随身空气质量监测器的创意，先给点初步建议。"},
    ]

    widths = [4, 13, 10, 10, 18]
    _print_row(widths, ["轮次", "prompt_tokens", "缓存命中", "命中率", "备注"])
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))

    for i in range(1, n_rounds + 1):
        messages.append({"role": "user", "content": FOLLOW_UP_INSTRUCTION})
        msg = chat(messages, temperature=0, max_tokens=150)
        messages.append({"role": "assistant", "content": msg.content or ""})

        usage = msg.usage
        cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        hit_rate = f"{cached / usage.prompt_tokens:.0%}" if usage.prompt_tokens else "-"

        note = ""
        if i == compress_at_round:
            messages = strategy_summary(messages, keep_last=keep_last)
            note = "← 本轮后触发压缩"

        _print_row(widths, [str(i), f"{usage.prompt_tokens:,}", f"{cached:,}", hit_rate, note])

    print(
        f"\n💡 观察第 {compress_at_round + 1} 轮：压缩重写了前缀后，缓存命中率是不是直接摔回接近 0%，"
        "然后要重新从头爬升——这就是「压缩税」：省了窗口空间，但付出的代价是缓存清零重算。"
    )


if __name__ == "__main__":
    run_compaction_cache_tax_experiment()
