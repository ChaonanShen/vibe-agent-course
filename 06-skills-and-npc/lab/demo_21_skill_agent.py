"""
Demo 21 · 一个真能跑的 Skill Agent
===================================

目标：把第 20 节的原理接上真实 LLM，变成一个**能对话的 agent**——
你说一句话，LLM 自己判断该用哪个 skill，自己调用 load_skill 读规范，
再按规范回答你。这就是讲义里的「路径 A：LLM 自主加载」。

两条路径都做了：
  - 路径 A（自动）：普通提问 → LLM 自己决定调 list_skills / load_skill
  - 路径 B（显式）：/skills 本地列出；@name <任务> 本地拦截拼前置规范

依赖：
  - 环境变量：BASE_URL + API_KEY（或 CNB 环境下的 CNB_API_ENDPOINT/CNB_REPO_SLUG + CNB_TOKEN）
  - pip install openai

跑法：
    # 交互式对话
    python demo_21_skill_agent.py

    # 跑一条预设任务（非交互，方便一次性看效果）
    python demo_21_skill_agent.py "我要给 CNB 的 issue 发一条评论，该用什么命令？"
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys

from _env import chat
from skill_loader import SkillLoader


SKILLS_DIR = "/root/.agents/skills"

# bash 工具的命令白名单：本 demo 只允许跑 cnb CLI（skill 规范里用到的"手"）。
# 这样既能真正执行命令，又避免 agent 被诱导去跑任意危险命令。
ALLOWED_CMDS = ("cnb",)


# ============================================================
# 1. 工具层：把 Loader 暴露成 LLM 可调用的工具
#    - list_skills / load_skill：Skills 机制本身（"把规范读进来"）
#    - bash：真正执行命令的"手"（讲义：Skill 定标准，Tool 落地执行）
# ============================================================

class SkillTools:
    def __init__(self, loader: SkillLoader):
        self.loader = loader

    def list_skills(self) -> str:
        """列出所有可用 skill（name + 一句话）。"""
        lines = [f"- {n}: {b}" for n, b in self.loader.list_index()]
        return "\n".join(lines) if lines else "(没有可用 skill)"

    def load_skill(self, name: str) -> str:
        """加载某个 skill 的 SKILL.md 全文。"""
        skill = self.loader.load(name)
        return f"# Skill: {skill.name}\n---\n{skill.body}"

    def bash(self, command: str) -> str:
        """执行一条 shell 命令（受白名单限制），返回 stdout/stderr。"""
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return f"命令解析失败: {e}"
        if not argv or argv[0] not in ALLOWED_CMDS:
            return (f"拒绝执行：本 demo 只允许运行 {ALLOWED_CMDS} 开头的命令，"
                    f"收到的是 {argv[0] if argv else '(空)'!r}")
        try:
            # 不使用 shell=True，避免命令注入；参数已按白名单校验
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "命令超时（30s）"
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        tail = f"\n[stderr]\n{err}" if err else ""
        return (out + tail).strip() or f"(命令无输出，exit={proc.returncode})"

    # --- 分发 + OpenAI tools schema ---
    def invoke(self, name: str, args: dict) -> str:
        try:
            if name == "list_skills":
                return self.list_skills()
            if name == "load_skill":
                return self.load_skill(args["name"])
            if name == "bash":
                return self.bash(args["command"])
            return f"未知工具: {name}"
        except Exception as e:  # 把异常当工具结果回填，让模型自己纠正
            return f"调用 {name} 失败: {type(e).__name__}: {e}"

    @staticmethod
    def schemas() -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_skills",
                    "description": "列出所有可用 skill（name + 一句话摘要）。"
                                   "不确定有哪些能力、或要挑一个 skill 时先调用它。",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "load_skill",
                    "description": "按名字加载某个 skill 的完整规范（SKILL.md 全文）。"
                                   "命中某个 skill 后调用它拿到详细规范，再据此行动。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "skill 的名字（= 目录名）"}
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "执行一条 shell 命令并返回输出。用它来真正运行 skill "
                                   "规范里写到的命令（如 cnb ...）。注意：只能执行 cnb 命令。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "完整的 shell 命令"}
                        },
                        "required": ["command"],
                    },
                },
            },
        ]


# ============================================================
# 2. system prompt：启动只注入 Level 1 索引（省 token）
# ============================================================

def build_system_prompt(loader: SkillLoader) -> str:
    base = (
        "你是一个助手。你有一批「Skills」（领域规范包）可用，下面是它们的索引。\n"
        "工作方式：\n"
        "1. 回答涉及某能力时，**先用 load_skill 读取对应 skill 的完整规范**；\n"
        "2. 规范里若给出了命令（如 cnb ...），**用 bash 工具真正执行它**，再根据真实输出回答；\n"
        "3. 严格按规范与命令的实际输出作答，不要凭空编造命令、字段或结果。\n"
        "若规范里没有直接对应的命令，可以用 `cnb --help` / `cnb <module> --help` 探索可用命令。\n"
        "若不清楚有哪些 skill，先调用 list_skills。"
    )
    index = loader.list_index()
    if not index:
        return base
    idx = "\n".join(f"- {n}: {b}" for n, b in index)
    return base + "\n\n# 可用 Skills（仅索引，全文用 load_skill 读）\n" + idx


# ============================================================
# 3. Agent Loop：让 LLM 自主调用工具（路径 A）
# ============================================================

def run_agent(messages: list[dict], tools: SkillTools, max_iters: int = 8) -> str:
    for turn in range(1, max_iters + 1):
        # chat() 内部强制流式并把 tool_calls 收集回完整 Message
        msg = chat(
            messages,
            tools=tools.schemas(),
            tool_choice="auto",
            temperature=0,
        )

        if not msg.tool_calls:
            answer = msg.content or ""
            messages.append({"role": "assistant", "content": answer})
            return answer

        # 回填 assistant 的 tool_calls
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        # 逐个执行工具，结果回填到 role=tool
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            print(f"  🔧 LLM 调用工具: {tc.function.name}({args})")
            result = tools.invoke(tc.function.name, args)
            # 命令输出可能很长，回填前截断，保护上下文
            if len(result) > 4000:
                result = result[:4000] + "\n...(输出已截断)"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "⚠️ 达到最大工具轮次仍未给出最终回答"


# ============================================================
# 4. 对话入口：路径 B 本地拦截 + 路径 A 交给 agent
# ============================================================

def handle_turn(user_input: str, messages: list[dict], tools: SkillTools,
                loader: SkillLoader) -> str:
    text = user_input.strip()

    # 路径 B-1：/skills 本地直接列出，不走 LLM
    if text.startswith("/skills"):
        return tools.list_skills()

    # 路径 B-2：@name <任务> 本地加载全文，拼成"前置规范 + 任务"再送 LLM
    if text.startswith("@"):
        name, _, task = text[1:].strip().partition(" ")
        try:
            skill = loader.load(name)
        except Exception as e:
            return f"加载 @{name} 失败: {type(e).__name__}: {e}"
        print(f"  📎 本地拦截：已加载 skill «{name}» 全文作为前置规范")
        messages.append({
            "role": "user",
            "content": f"[请严格按以下规范完成]\n{skill.body}\n\n[任务]\n{task.strip()}",
        })
        return run_agent(messages, tools)

    # 路径 A：普通提问，交给 agent，LLM 自己决定要不要调 skill 工具
    messages.append({"role": "user", "content": text})
    return run_agent(messages, tools)


def main():
    loader = SkillLoader(SKILLS_DIR)
    tools = SkillTools(loader)
    messages = [{"role": "system", "content": build_system_prompt(loader)}]

    print(f"📚 已加载 skills 索引（{len(loader.list_index())} 个），目录: {SKILLS_DIR}")

    # 命令行带参数 → 跑一条预设任务（非交互）
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(f"\n🧑 你: {q}")
        print(f"\n🤖 助手:\n{handle_turn(q, messages, tools, loader)}")
        return

    # 否则进入交互式 REPL
    print("进入对话。命令：/skills 列出全部；@name <任务> 指定 skill；exit 退出。\n")
    while True:
        try:
            q = input("🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 👋")
            break
        if not q:
            continue
        if q in ("exit", "quit", ":q"):
            print("再见 👋")
            break
        print(f"\n🤖 助手:\n{handle_turn(q, messages, tools, loader)}\n")


if __name__ == "__main__":
    main()
