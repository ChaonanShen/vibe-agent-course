<p align="center">
    <img src="../assets/banner-04-context-memory.svg" alt="Context & Memory">
</p>

# 第 4 章 · 上下文与记忆

本章目标：解决 Agent 的两个关键问题 —— 对话过长导致上下文爆炸（token 超限）、会话结束后遗忘用户偏好（无长期记忆）。
产出：`my-agent/context.py`（上下文管理器）和 `my-agent/memory.py`（向量记忆系统）。



## 本章要点

- 上下文工程要同时解决三个互相牵制的问题：防爆炸（撞窗口）、提缓存命中率（省钱）、防变笨（关键信息被埋）——压缩能防爆炸，却会打断缓存前缀，这是核心矛盾
- 三种治理策略对比：截断（只留头尾，简单但丢早期事实）、摘要（LLM 压缩中间段，省 token 但有「压缩税」）、分层（system / 关键事实 / 早期摘要 / 最近原文分层保留，兼顾防爆炸与防变笨）
- `ContextManager`：阈值触发压缩 + 分层策略 + 两次压缩之间 append-only，统一协调三个问题而非单独优化一个
- 向量记忆系统：用 embedding 将事实转为向量，通过余弦相似度检索相关长期记忆，实现跨会话的"读→写"闭环
- Agent v2 实战整合：将 `ContextManager` 和 `Memory` 接入 Agent Loop，对比朴素版的 token 消耗差异



## 内容结构

| 节 | 内容 | 讲义 | Demo |
|----|------|------|------|
| 13 | 上下文管理：原理、问题与 ContextManager | [13-context-management.md](./lecture/13-context-management.md) | `demo_13a_explosion.py` / `demo_13b_strategies.py` / `demo_13c_cache_tax.py` / `demo_13d_context_manager.py` |
| 14 | Memory 系统 | [14-memory-types.md](./lecture/14-memory-types.md) | `demo_14_memory.py` |



## 运行示例

在云原生开发环境中执行：

```bash
cd 04-context-and-memory/lab
python demo_13a_explosion.py        # 上下文爆炸实验
python demo_13b_strategies.py       # 三种压缩策略对比（截断/摘要/分层）
python demo_13c_cache_tax.py        # 真实验证压缩打断缓存命中率
python demo_13d_context_manager.py  # 产出 ContextManager 类
python demo_14_memory.py
```



## 作业指引

本章作业需要在配套仓库中完成，按以下步骤操作：

> [!TIP]
>
> 📦 **作业仓库**：<https://cnb.cool/opencamp/learning-agent/vibe-agent-exercise>

**操作流程**：

1. **Fork** 作业仓库到自己的账号下
2. 在 `my-agent/` 目录中完成核心代码产出，在 `exercises/` 目录中完成配套填空题
3. **提交 PR**，CI 流水线（`.cnb.yml`）会自动运行测评
4. 全部测试通过（全绿）即视为作业通过检测 ✅，作业完成



## 本章产出

在作业仓库中补全 `my-agent/context.py` 和 `my-agent/memory.py`，执行以下命令让测试通过：

```bash
cd my-agent
python tests/test_context.py
python tests/test_memory.py
```



## 配套练习

作业仓库的 `exercises/04-context-and-memory/` — 3 道代码填空题（运行 `python exercises/04-context-and-memory/ex_*.py` 看到 ✅ PASSED 即通过）。
