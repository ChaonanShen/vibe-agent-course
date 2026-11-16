"""
Demo 13d · 产出：ContextManager 类
======================================

配套：第 13 节《上下文管理：从上下文到 ContextManager》

把"何时压缩（防爆炸）+ 怎么压缩（防变笨）+ 尽量少压缩（保缓存）"
三件事统一封装成一个可复用的类，而不是三个互不知道对方存在的函数。

依赖 demo_13b_strategies.py 里的 strategy_hierarchical 做实际压缩，
用一段 30 轮历史模拟逐轮喂入，观察 ContextManager 什么时候触发压缩、
什么时候什么都不做（append-only，为缓存命中率让路）。

运行方式：
    cd /workspace/04-context-and-memory/lab/
    source /workspace/.venv/bin/activate
    python demo_13d_context_manager.py
"""

from demo_13b_strategies import (
    _print_row,
    build_fake_history,
    extract_key_facts_llm,
    fact_check,
    strategy_hierarchical,
)


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
        2. **阈值触发，而非每轮触发**：只有消息数超过 `max_messages` 才压缩，
           避免像"每轮都摘要"那样把缓存命中率打成 0%（demo_13c 已验证代价）。
        3. **压缩策略用分层 + LLM 事实抽取**：压缩这一下必须把活干好——用
           `extract_key_facts_llm`（而非关键词匹配）单独抽出关键事实，
           不随早期对话一起被摘要掉，解决"防变笨"，也是生产级实现方式。

    Attributes:
        keep_last: 压缩时保留的最近消息条数，这些消息不会被摘要化。
        max_messages: 触发压缩的阈值，超过此值才会压缩。

    Example:
        >>> ctx = ContextManager(keep_last=6, max_messages=20)
        >>> ctx.add_messages({"role": "system", "content": "你是旅行助手。"})
        >>> ctx.add_messages({"role": "user", "content": "我准备 6 月去北京玩 7 天。"})
        >>> reply = llm.chat(ctx.messages())
        >>> ctx.add_messages({"role": "assistant", "content": reply})
    """

    def __init__(self, keep_last: int = 6, max_messages: int = 20) -> None:
        self._messages: list[dict] = []
        self._keep_last = keep_last if keep_last > 0 else 6
        self._max_messages = max_messages if max_messages > 0 else 20
        self._last_compressed = False  # 上一次 add_messages 调用是否真的触发了压缩

    @property
    def keep_last(self) -> int:
        return self._keep_last

    @property
    def max_messages(self) -> int:
        return self._max_messages

    @property
    def last_compressed(self) -> bool:
        """上一次调用 `add_messages` 是否真的触发了压缩，仅用于观测/打日志。"""
        return self._last_compressed

    def _should_compress(self, messages: list[dict]) -> bool:
        """阈值触发判断：只有真正接近窗口压力时才压缩，不是消息一多就压。

        内部实现细节，外部不需要关心，也不应该自己拿这个判断结果去决定
        要不要调 `_compress`——那样每个调用方都要重写一遍判断逻辑，
        正是这个类存在的理由就是不让这种事发生。
        """
        return len(messages) > self._max_messages

    def _compress(self, messages: list[dict]) -> list[dict]:
        """执行一次分层压缩。

        事实抽取用 `extract_key_facts_llm`（生产级做法），而不是
        demo_13b 里作为教学对比基线的关键词匹配版本。
        """
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


def run_context_manager_demo() -> None:
    """用 ContextManager 跑一遍 30 轮历史，展示它什么时候压缩、什么时候不压缩。"""
    print("ContextManager 实际运行：按需压缩，而不是每轮都压")
    print("=" * 78)

    ctx = ContextManager(keep_last=6, max_messages=20)
    history = build_fake_history(n_rounds=30)

    # 模拟逐轮喂入：先写入 system，再逐轮 add_messages，看什么时候触发压缩
    ctx.add_messages(history[0])
    widths = [4, 10, 12]
    _print_row(widths, ["步骤", "messages条数", "是否压缩"])
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))

    for i, m in enumerate(history[1:], 1):
        ctx.add_messages(m)
        _print_row(
            widths,
            [
                str(i),
                str(len(ctx.messages())),
                "✅ 压缩了" if ctx.last_compressed else "-",
            ],
        )

    final_messages = ctx.messages()
    has_aller, has_spicy, has_budget = fact_check(final_messages)
    print(f"\n💡 最终 messages {len(final_messages)} 条，过敏事实{'✅ 保住' if has_aller else '❌ 丢失'}，"
          f"吃辣这条隐性事实{'✅ 保住' if has_spicy else '❌ 丢失'}，"
          f"预算事实{'✅ 保住' if has_budget else '❌ 丢失'}。")
    print("   ContextManager 只在真正超过阈值时才压缩（防爆炸），")
    print("   压缩时用 LLM 事实抽取保住关键事实，包括关键词抓不到的隐性约束（防变笨），")
    print("   两次压缩之间 append-only、前缀不变（尽量保住缓存命中率）。")


if __name__ == "__main__":
    run_context_manager_demo()
