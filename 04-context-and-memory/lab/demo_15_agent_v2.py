"""
Demo 15 · Agent v2：把最小 Agent Loop 接入 ContextManager + 两层记忆
=====================================================================

把第 2 章的朴素 Agent Loop 升级为 v2，接入本章（第 4 章）两个现成组件：

  ① ContextManager（第 13 节·产出类 demo_13d）
     —— 每轮调 LLM 前按 token 阈值自动压缩 messages，防止上下文爆炸

  ② Memory（第 14 节·产出类 demo_14c）
     —— 两层长期记忆的统一门面，对内 Profile + History 各司其职：
        · Profile 档案卡：结构化稳定事实，整卡注入 system（memory.profile_card）
        · 历史会话向量库：按 user_input 语义召回往期摘要（memory.search）
        · 会话结束一句 memory.commit_session() 同时写回两层

AgentV2 自己不再分别拿着 profile、history 两个对象来回编排——它只持有一个
ContextManager 和一个 Memory，正是"把能力沉淀成可复用组件"的意义。

跑一段多轮旅行规划，直观看到：
  - ContextManager 在第几轮触发压缩
  - Profile 里的稳定偏好与关键事实（过敏 / 预算）每轮都在，不会被压缩冲掉
  - 历史会话召回把"相关的往期上下文"按需捞回来

运行方式：
    cd /workspace/04-context-and-memory/lab/
    source /workspace/.venv/bin/activate
    python demo_15_agent_v2.py
"""

from _env import chat, get_model_id
from demo_13d_context_manager import ContextManager
from demo_14b_profile import Profile
from demo_14c_memory import Memory

# ContextManager 的完整实现见 demo_13d_context_manager.py（第 13 节的产出）：
# 它统一处理"何时压缩（防爆炸）+ 怎么压缩（防变笨）+ 尽量少压缩（保缓存命中率）"
# 三件事。Memory 见 demo_14c_memory.py（第 14 节的产出）：它把两层记忆
# （Profile 档案卡 + 历史会话向量库）封装成统一门面。AgentV2 直接复用这两个组件。


# ============================================================
# Agent v2：组合 ContextManager + Memory
# ============================================================
class AgentV2:
    """增强版 Agent，整合上下文压缩与两层长期记忆。

    相比第 2 章的朴素 Agent Loop，v2 新增两件事：

    1. 上下文压缩：messages 超 token 阈值时由 ContextManager 自动压缩，
       避免上下文爆炸（第 13 节产出组件）。
    2. 长期记忆：持有一个 Memory（第 14 节产出组件），它内部管着两层——
       · Profile 整卡注入 system，让稳定偏好与关键事实每轮一致生效、永不漏召回；
       · 历史会话按 user_input 语义召回相关往期摘要，临时注入 system。

    读写分工：
        - 对话前（读）：memory.profile_card() 整卡注入 + memory.search() 召回 top-k；
        - 对话后（写）：见 `end_session()`——一句 memory.commit_session()
          同时写回两层（Profile 覆盖、History 追加摘要），数据源是未经压缩的
          会话原文 `self._session_log`，而非 ctx 压缩后的历史；
        - 会话历史全权交给 `self.ctx`（ContextManager）持有，AgentV2 只通过
          `ctx.add_messages()` 写、`ctx.messages()` 读，压缩判断和历史状态始终绑定。

    为什么要额外留一份 `_session_log`：
        ctx.messages() 是"发给模型的上下文"，会被压缩（早期 user 原文被摘要冲掉）；
        而长期记忆写回要基于"本次会话的完整原话"。两者数据源不同，必须分开持有，
        否则预算 5000 这类早期关键事实会在压缩后被抽取不到，Profile 更新就漏了。

    Attributes:
        system_prompt: 系统提示词（只读）。
        memory: 两层长期记忆门面（Profile + 历史会话）。
        ctx: 上下文管理器，持有并压缩当前会话 messages。
        messages: 当前会话消息历史（只读，读自 `self.ctx.messages()`）。
    """

    def __init__(
        self,
        system_prompt: str = "你是一个有用的助手。",
        memory: Memory | None = None,
        ctx_manager: ContextManager | None = None,
    ) -> None:
        """初始化 AgentV2。

        Args:
            system_prompt: 系统提示词，决定 Agent 的角色与行为准则。
            memory: 两层长期记忆门面，传 `None` 时新建空 `Memory()`。
            ctx_manager: 上下文压缩管理器，传 `None` 时用默认
                `ContextManager(keep_last=6, max_tokens=2000)`。
        """
        self._system_prompt = system_prompt
        self.memory = memory or Memory()
        self.ctx = ctx_manager or ContextManager()
        self.ctx.add_messages({"role": "system", "content": system_prompt})
        # 本次会话的完整原文，专供会话结束时抽取长期记忆用。
        # 必须和 self.ctx 分开：ctx 里的历史会被压缩（早期 user 原文会被摘要冲掉），
        # 而 Profile/History 写回要基于「本次说过的原话」，不能用压缩后的残缺上下文。
        self._session_log: list[dict] = []

    @property
    def system_prompt(self) -> str:
        """只读系统提示词，防止与历史里的 system 消息不一致。"""
        return self._system_prompt

    @property
    def messages(self) -> list[dict]:
        """当前会话消息历史，读自 `self.ctx.messages()`（可能已被压缩）。"""
        return self.ctx.messages()

    def _build_system(self, user_input: str) -> str:
        """拼这一轮注入的 system：角色 + Profile 整卡 + 召回的历史会话。

        两层读法不同，正好对应 Memory 暴露的两个方法：
        - Profile：memory.profile_card() 整卡注入，不检索（它该一直生效）；
        - 历史会话：memory.search() 按当前问题召回 top-k，过滤相似度 < 0.5 的噪声。
        """
        blocks = [self.system_prompt]

        card = self.memory.profile_card()
        if card and card != "（档案卡为空）":
            blocks.append(f"【用户档案卡（稳定偏好与约束，务必遵守）】\n{card}")

        recalled = [(s, e) for s, e in self.memory.search(user_input, k=3) if s > 0.5]
        if recalled:
            block = "\n".join(f"- {e.summary}" for _, e in recalled)
            blocks.append(f"【相关的历史会话（仅供参考）】\n{block}")

        return "\n\n".join(blocks)

    def chat(self, user_input: str) -> tuple[str, int, int]:
        """处理一轮对话，返回 `(reply, prompt_tokens, total_tokens)`。

        流程：
          1. 读历史（self.ctx.messages()）；
          2. 把 Profile 整卡 + 召回的历史会话拼进 system（仅本轮有效，不写入历史）；
          3. 调 LLM；
          4. 只把 user / assistant 原文写回 ctx（不含临时注入的记忆），
             压缩判断在 `add_messages` 内部按 token 阈值自动完成。
        """
        history = self.ctx.messages()

        # 1~2. 用两层记忆重建 system，其余历史消息（非 system）原样跟上
        injected = [{"role": "system", "content": self._build_system(user_input)}]
        injected += [m for m in history[1:] if m["role"] != "system"]
        injected.append({"role": "user", "content": user_input})

        # 3. 调 LLM
        msg = chat(
            messages=injected,
            temperature=0.3,
            max_tokens=400,
            model=get_model_id(),
        )
        reply = msg.content or ""
        usage = msg.usage

        # 4. 只写回原文，交给 ContextManager 统一管理（压缩判断在内部完成）
        turn = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": reply},
        ]
        self.ctx.add_messages(*turn)
        # 同步留一份到会话原文日志：这份不参与压缩，供 end_session 抽取记忆
        self._session_log.extend(turn)
        if self.ctx.last_compressed:
            print(f"  ⚙️  上下文估算超过 {self.ctx.max_tokens:,} tokens，触发压缩")

        prompt_tk = usage.prompt_tokens if usage else 0
        total_tk = usage.total_tokens if usage else 0
        return reply, prompt_tk, total_tk

    def end_session(self) -> None:
        """会话结束时的写回：一句话同时更新两层记忆。

        Memory 把"两层写时机一致"这件事收口了——这里不用再分别
        更 profile、追加 history，一句 `commit_session` 全搞定。

        注意传的是 `self._session_log` 而不是 `self.messages`：后者是 ctx 压缩后的
        上下文，早期 user 原文（如"预算 5000"）可能已被摘要冲掉，拿它抽 Profile 会
        漏字段。长期记忆写回必须基于本次会话的完整原话，与上下文压缩解耦。
        """
        self.memory.commit_session(self._session_log)


# ============================================================
# 跑一段多轮对话，观察压缩触发 + 两层记忆的注入效果
# ============================================================
def run_demo() -> None:
    print("=" * 60)
    print("【Agent v2】多轮旅行规划：压缩 + Profile + 历史会话")
    print("=" * 60)

    agent = AgentV2(system_prompt="你是旅行规划师。回答简洁、可执行。")

    # 预置两层记忆（模拟"往期会话已经沉淀下来的东西"），全部挂在 agent.memory 上
    # 第 ① 层：用户的稳定偏好与关键事实，整卡注入、每轮都在
    agent.memory.profile = Profile(
        allergies=["花粉"],
        budget="每日 500 元",
        preferences=["喜欢博物馆"],
        city="北京朝阳",
    )
    # 第 ② 层：往期会话摘要，按问题召回
    agent.memory.add_summary("用户上次问过朝阳区附近安静的咖啡馆。")
    agent.memory.add_summary("用户不喜欢人多嘈杂的网红景点。")

    questions = [
        "我准备 6 月去北京玩 7 天。预算 5000 元, 喜欢天安门",
        "第 1 天有什么推荐？",
        "第 2 天呢？",
        "第 3 天去户外好不好？",     # ⭐ 档案卡里的"花粉过敏"应影响回答
        "第 4 天就在城里逛吧。",
        "第 5 天有没有什么文化体验？",
        "顺便推荐 2 个安静的咖啡馆。",  # ⭐ 应召回历史会话里的"安静咖啡馆/不喜欢嘈杂"
        "总结一下整体预算。",          # ⭐ 档案卡里的"每日 500 元"应影响回答
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n--- 第 {i} 轮 ---")
        print(f"用户: {q}")
        reply, prompt_tk, total_tk = agent.chat(q)
        print(f"助手: {reply[:150]}{'...' if len(reply) > 150 else ''}")
        print(f"      [prompt={prompt_tk} total={total_tk} messages={len(agent.messages)}]")

    print("\n\n" + "=" * 60)
    print("📝 会话结束 → 两层记忆写回")
    print("=" * 60)
    agent.end_session()

    print("\n① Profile 档案卡（覆盖更新后）：")
    print(agent.memory.profile_card())
    print(f"\n② 历史会话向量库（现有 {len(agent.memory.history.entries)} 条摘要）：")
    for e in agent.memory.history.entries:
        print(f"  - {e.summary}")
    print("\n   下次会话时：档案卡整卡注入，相关的历史摘要按问题召回。")


if __name__ == "__main__":
    run_demo()

    print("\n\n" + "=" * 60)
    print("🎯 v2 Agent 跟第 2 章朴素版的区别")
    print("=" * 60)
    print("  只多持有两个现成组件，就有了'跨会话的记忆'：")
    print("  ① ContextManager（13d）：messages 超 token 阈值时主动压缩（截断/摘要/分层）")
    print("  ② Memory（14c）：两层记忆统一门面，对内 Profile + 历史会话各司其职")
    print("     · Profile：稳定偏好/关键事实整卡注入，每轮一致生效、覆盖更新、天然免去重")
    print("     · 历史会话：每轮按 user_input 召回相关往期摘要，会话结束追加一条")
    print("     · 一句 memory.commit_session() 同时写回两层")
    print()
    print("  下一步（第 6/7 章）：把 ContextManager + Memory 接进真实的 Coding Agent。")
