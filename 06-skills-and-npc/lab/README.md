# Lab · Skills 加载原理 + 能跑的 Skill Agent

把第 20 节的 **「两级渐进式加载」** 既讲清楚、又真能跑：

- `demo_20_skill_loading.py`：**不依赖 LLM / 网络**，纯本地把原理打印出来。
- `demo_21_skill_agent.py`：**接真实 LLM**，一个能对话的 agent——你说话，它自己判断该用哪个
  skill、自己调 `load_skill` 读规范再回答。

扫描的 skills 目录是平台真实目录：`/root/.agents/skills`（里面是 `cnb-*` 系列 skill）。

## 文件

| 文件 | 作用 |
|------|------|
| `skill_loader.py` | 核心：`SkillLoader`（扫描 / 索引 / 读全文 / 防目录穿越） |
| `demo_20_skill_loading.py` | 原理全景（离线）：两级加载、两个工具、prompt 注入、两条路径 |
| `demo_21_skill_agent.py` | 能跑的 Skill Agent：Agent Loop + LLM 自主调用 skill 工具 |
| `_env.py` | LLM 配置 + 流式客户端封装（接口只支持流式，内部自动收集 tool_calls） |

## 1. 离线看原理（无需任何配置）

```bash
cd 06-skills-and-npc/lab
python demo_20_skill_loading.py
```

- **Level 1（省 token）**：启动只扫出 `(name, brief)` 索引拼进 system prompt，能力再多常驻也只几十 token。
- **Level 2（命中才付费）**：某 skill 被命中时才 `load()` 读 `SKILL.md` 全文。
- **两个工具** `list_skills` / `load_skill`：Skills 机制本身正是用 Tool 实现的。
- **两条路径**：A（LLM 自主调 `load_skill`）/ B（`@name` 本地拦截拼前置规范）。
- **安全**：非法名字 / `../` 目录穿越会被 `SkillLoader` 挡下。

## 2. 跑真实 Agent（需要 LLM 环境变量）

需要 `BASE_URL + API_KEY`，或 CNB 环境下的 `CNB_API_ENDPOINT + CNB_REPO_SLUG + CNB_TOKEN`。

```bash
# 在仓库根目录用 uv（已带 openai 依赖）
uv run python 06-skills-and-npc/lab/demo_21_skill_agent.py "我要给 CNB 的 issue 发一条评论，该用什么命令？"

# 交互式对话
uv run python 06-skills-and-npc/lab/demo_21_skill_agent.py
```

交互里支持：

- 普通提问 → **路径 A**：LLM 自己决定是否调 `list_skills` / `load_skill`
- `/skills` → 本地直接列出全部 skill（不走 LLM）
- `@name <任务>` → **路径 B**：本地加载该 skill 全文当前置规范再送 LLM
- `exit` 退出

实测效果（路径 A）：

```
🧑 你: 我要给 CNB 的 issue 发一条评论，该用什么命令？
  🔧 LLM 调用工具: load_skill({'name': 'cnb-api'})
🤖 助手: 用 `cnb issues comment --body '评论内容'` ……（按 skill 全文规范作答，不编造）
```

## 换个目录看

两个 demo 顶部的 `SKILLS_DIR` 改成任意放了 `SKILL.md` 的目录即可——
**不用改一行逻辑**，新目录里的 skill 就自动出现在索引里。这就是 Skills「零代码热插拔」。
