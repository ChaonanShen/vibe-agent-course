"""
Demo 14c · 产出：Memory 类（把两层记忆封装成一个可复用组件）
==============================================================

配套：第 14 节《记忆系统：历史会话 + Profile 档案卡》的收口。

和 13d 对称：13d 把"上下文该怎么管"封装成 ContextManager，14c 把"两层记忆
该怎么用"封装成 Memory。到了 demo_15，AgentV2 = ContextManager + Memory 两个
现成组件一拼即成，不用自己再分别拿着 profile、history 两个对象来回编排。

Memory 不发明任何新机制，它只做"门面"，收口两件最容易写乱的事：

  1. 写入时机统一
     两层的写入时机本就一致——都在"会话结束"那一刻。于是
     `commit_session()` 一次把两层都写了：
       · Profile 覆盖更新（抽字段 → 单值覆盖 / 多值并集）
       · History 追加一条本次会话摘要（embed 入库）
     调用方不用再记得"先更 profile、再 commit_session"。

  2. 持久化统一
     `save()` / `load()` 一次搞定两层，底层仍是两个物理文件：
       · history_store.json —— 摘要 + **向量**（语义检索要用）
       · profile_store.json —— 纯结构化字段（整卡注入，**无向量**）

但**读取接口刻意不统一**——这正是本章主线，门面不该把它抹平：
    · profile_card()      Profile 整卡，无 query —— 它该一直生效，不检索
    · search(query, k) History 按问题语义召回 top-k —— 要检索
两层数据形态不同，检索策略就必须不同；强行合成一个方法只会两边都做不好。

运行方式：
    cd /workspace/04-context-and-memory/lab/
    source /workspace/.venv/bin/activate
    python demo_14c_memory.py
"""

import os

from demo_14a_history import STORE_PATH as HISTORY_PATH
from demo_14a_history import History, SessionEntry
from demo_14b_profile import PROFILE_PATH, Profile


# ============================================================
# Memory：两层长期记忆的统一门面
# ============================================================
class Memory:
    """两层长期记忆的统一门面：对外一个 Memory，对内 History + Profile 各司其职。

    为什么是"门面"而不是"把两层揉成一层"：
        历史会话是非结构自由文本、只增不改 → 语义召回（search）；
        Profile 是结构化稳定事实、要覆盖更新 → 整卡注入（profile_card）。
        两者数据形态不同、检索策略就必须不同。Memory 的价值是把
        "什么时候写、怎么持久化"这些**编排**收口到一处，而不是抹掉两层差异。

    读（对话前）：
        - profile_card():        Profile 整卡（每轮一致生效，不检索）
        - search(query, k):   History 按问题召回 top-k
    写（预置 / 对话后）：
        - add_summary(s):                 直接追加一条历史摘要（预置往期记忆用）
        - commit_session(msgs): 会话结束，两层一起写回
    持久化：
        - save() / load():    两层一起落盘 / 还原（底层两个文件）

    Attributes:
        profile: 用户结构化档案卡（第 ① 层）。
        history: 历史会话向量库（第 ② 层）。
    """

    def __init__(
        self,
        profile: Profile | None = None,
        history: History | None = None,
        history_path: str = HISTORY_PATH,
        profile_path: str = PROFILE_PATH,
    ) -> None:
        """初始化 Memory。

        Args:
            profile: 档案卡，传 `None` 时新建空 `Profile()`。
            history: 历史会话向量库，传 `None` 时新建空 `History()`。
            history_path: 历史会话向量库的落盘路径（默认复用 14a 的 STORE_PATH）。
            profile_path: 档案卡的落盘路径（默认复用 14b 的 PROFILE_PATH）。
        """
        self.profile = profile or Profile()
        self.history = history or History()
        self._history_path = history_path
        self._profile_path = profile_path

    # ---------- 读：对话前 ----------
    def profile_card(self) -> str:
        """Profile 整卡渲染（委托 Profile）。没有 query——它该一直生效，不检索。"""
        return self.profile.profile_card()

    def search(self, query: str, k: int = 3) -> list[tuple[float, SessionEntry]]:
        """按语义召回 top-k 条历史会话（委托 History）。"""
        return self.history.search(query, k=k)

    # ---------- 写：预置 / 对话后 ----------
    def add_summary(self, summary: str) -> None:
        """直接追加一条历史会话摘要（委托 History），常用于预置往期记忆。"""
        self.history.add_summary(summary)

    def commit_session(self, messages: list[dict]) -> tuple[dict, str]:
        """会话结束时一次写回两层：Profile 覆盖更新 + History 追加一条摘要。

        这正是门面的意义——两层写入时机本就一致（会话结束），调用方只需一句
        `memory.commit_session(...)`，不必记得先更 profile 再 commit_session。

        Returns:
            `(profile_delta, history_summary)`，仅用于演示打印看清写了什么。
        """
        delta = self.profile.commit_session(messages)
        summary = self.history.commit_session(messages)
        return delta, summary

    # ---------- 持久化：两层一起 ----------
    def save(self) -> None:
        """两层一起落盘：History → 带向量的 JSON，Profile → 纯字段 JSON。"""
        self.history.save(self._history_path)
        self.profile.save(self._profile_path)

    def load(self) -> None:
        """两层一起还原；文件不存在时保持为空，不报错（首次运行的冷启动）。"""
        if os.path.exists(self._history_path):
            self.history.load(self._history_path)
        self.profile = Profile.load(self._profile_path)


# ============================================================
# 演示：Memory 单独怎么用（预置 → 写回 → 落盘 → 新实例加载 → 两种读法）
# ============================================================
def run_memory_demo() -> None:
    """展示 Memory 这个组件的完整生命周期，对标 13d 的 ContextManager 演示。"""
    print("=" * 60)
    print("【Memory 组件：两层记忆的统一门面】")
    print("=" * 60)

    # 1) 建一个 Memory，预置两条往期会话摘要（History 层）
    memory = Memory()
    memory.add_summary("用户上次咨询周末活动，因花粉过敏选择了室内博物馆。")
    memory.add_summary("用户问过朝阳区附近有哪些安静的咖啡馆。")

    # 2) 一次会话结束：一句话同时写回两层
    session = [
        {"role": "user", "content": "我对花粉过敏，每天预算 500 元，常住北京朝阳，喜欢逛博物馆。"},
        {"role": "assistant", "content": "好的，已经帮你记下来了。"},
    ]
    delta, summary = memory.commit_session(session)
    print("\n一句 commit_session 写回两层：")
    print(f"  ① Profile 覆盖更新，增量 = {delta}")
    print(f"  ② History 追加摘要      = {summary}")

    # 3) 落盘：一次 save 搞定两层（底层两个文件）
    memory.save()
    print("\n💾 一次 save 落盘两层：history_store.json（带向量）+ profile_store.json（纯字段）")

    # 4) 模拟"下次启动"：全新 Memory，一次 load 还原两层
    reader = Memory()
    reader.load()
    print("\n新实例一次 load 还原两层，但两种读法各用其所——")
    print("① profile_card()：Profile 整卡，不检索（该一直生效）")
    print(reader.profile_card())
    print("\n② search(query)：History 语义召回，要带 query")
    for score, e in reader.search("周末去哪玩比较好", k=2):
        print(f"  [{score:.3f}] {e.summary}")

    print("\n👀 Memory 把『写时机 + 持久化』收口成统一入口，")
    print("   却保留了两层各自的读法——这就是 demo_15 里 AgentV2 要用的记忆组件。")


if __name__ == "__main__":
    run_memory_demo()
