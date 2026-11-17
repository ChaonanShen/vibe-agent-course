"""
Demo 20 · Skills 加载原理全景演示
=================================

目标：用最少的代码把讲义第 20 节的「两级渐进式加载」跑给你看，
不依赖 LLM / 网络，直接 `python demo_20_skill_loading.py` 即可。

演示四块：
  1. Level 1 —— 启动扫盘，只取 (name, brief) 索引（省 token）
  2. system prompt 注入 —— 把索引拼进去，模型开局就知道有哪些 skill
  3. 两个工具 —— list_skills / load_skill，让 LLM 能自己加载（路径 A）
  4. 两条运行路径 —— A: LLM 自主调 load_skill；B: 用户 @name 本地拦截

跑法：
    python demo_20_skill_loading.py
"""

from __future__ import annotations

from skill_loader import SkillLoader


SKILLS_DIR = "/root/.agents/skills"


# ============================================================
# 1. 两个工具：把 Loader 暴露给 LLM（讲义 4.2）
#    —— Skills 机制本身，正是用两个 Tool 实现的
# ============================================================

def make_skill_tools(loader: SkillLoader):
    """返回 list_skills / load_skill 两个可调用函数。"""

    def list_skills() -> str:
        """列出所有可用 skill（name + 一句话）。不确定有哪些能力时调用。"""
        lines = [f"- {name}: {brief}" for name, brief in loader.list_index()]
        return "\n".join(lines) if lines else "(没有可用 skill)"

    def load_skill(name: str) -> str:
        """按名字加载某个 skill 的完整规范（SKILL.md 全文）。"""
        skill = loader.load(name)
        return f"# Skill: {skill.name}\n---\n{skill.body}"

    return list_skills, load_skill


# ============================================================
# 2. 启动时把 Level 1 索引拼进 system prompt（讲义 4.3）
# ============================================================

BASE_SYSTEM_PROMPT = "你是一个 Coding Agent，用工具完成用户的编程任务。"

SKILLS_INDEX_TEMPLATE = """

# 可用 Skills（领域规范，命中时用 load_skill 读全文）
{index}
"""


def build_system_prompt(loader: SkillLoader) -> str:
    index = loader.list_index()
    if not index:
        return BASE_SYSTEM_PROMPT
    index_text = "\n".join(f"- **{name}**: {brief}" for name, brief in index)
    # 常驻开销只有这几十 token：只装"目录"，不装"正文"
    return BASE_SYSTEM_PROMPT + SKILLS_INDEX_TEMPLATE.format(index=index_text)


# ============================================================
# 3. 两条运行路径（讲义 4.4）
# ============================================================

def route_a_llm_autoload(loader: SkillLoader, hit_name: str) -> str:
    """路径 A：LLM 自主判断命中 → 生成 tool_call: load_skill(name)。

    这里用"假的 LLM"：直接模拟它决定调用 load_skill。
    真实场景中，是模型读到 system prompt 的索引后自己产出这个 tool_call。
    """
    _, load_skill = make_skill_tools(loader)
    tool_result = load_skill(hit_name)  # 全文回填到 role=tool 消息
    # 真实链路：这段 tool_result 会作为 role=tool 塞回 messages，模型再据此生成答案
    return tool_result


def route_b_at_mention(loader: SkillLoader, user_input: str) -> str:
    """路径 B：用户显式 @name，在调用 LLM 之前本地拦截（快捷指令）。

    语法：
        /skills        → 本地直接列出所有 skill，不走 LLM
        @name <任务>   → 本地加载 skill 全文，拼成"前置规范 + 任务"再送 Agent
        其它            → 原样透传给 Agent
    """
    list_skills, _ = make_skill_tools(loader)

    if user_input.startswith("/skills"):
        return list_skills()

    if user_input.startswith("@"):
        # 语法：@name <任务>
        rest = user_input[1:].strip()
        name, _, task = rest.partition(" ")
        skill = loader.load(name)
        return (
            f"[按以下规范完成]\n{skill.body}\n\n[任务]\n{task.strip()}"
        )

    return user_input  # 原样透传


# ============================================================
# 主流程：把四块串起来打印
# ============================================================

def main():
    loader = SkillLoader(SKILLS_DIR)

    print("=" * 60)
    print("① Level 1：启动扫盘，只取 (name, brief) 索引（省 token）")
    print("=" * 60)
    index = loader.list_index()
    for name, brief in index:
        print(f"  - {name}: {brief}")

    # 从索引里动态取一个 skill 名来做后面的演示（真实目录里有啥用啥）
    sample = index[0][0] if index else None

    print("\n" + "=" * 60)
    print("② system prompt 注入：开局只装'目录'，不装'正文'")
    print("=" * 60)
    print(build_system_prompt(loader))

    print("\n" + "=" * 60)
    print("③ 路径 B：用户 /skills（本地拦截，不走 LLM）")
    print("=" * 60)
    print(route_b_at_mention(loader, "/skills"))

    if sample:
        print("\n" + "=" * 60)
        print(f"④ 路径 B：用户 @{sample} <任务>（本地拦截，拼前置规范）")
        print("=" * 60)
        rewritten = route_b_at_mention(loader, f"@{sample} 帮我按这个规范处理一下")
        print(rewritten)

        print("\n" + "=" * 60)
        print(f"⑤ 路径 A：LLM 自主命中 → load_skill('{sample}')（Level 2 读全文）")
        print("=" * 60)
        print(route_a_llm_autoload(loader, sample))

    print("\n" + "=" * 60)
    print("⑥ 安全演示：非法名字 / 目录穿越会被挡下")
    print("=" * 60)
    for bad in ["../secret", "add docstring", "not-exist"]:
        try:
            loader.load(bad)
            print(f"  加载 {bad!r} → 居然成功了（不应该）")
        except Exception as e:
            print(f"  加载 {bad!r} → 被挡下: {type(e).__name__}: {e}")

    print("\n💡 对照讲义：")
    print("   Level 1 = 只读 name+brief 装进 system prompt（上面①②）")
    print("   Level 2 = 命中才 load() 读全文（上面④⑤）")
    print("   路径 A 经 load_skill 工具回填 role=tool；路径 B 由 @name 本地拦截直达")


if __name__ == "__main__":
    main()
