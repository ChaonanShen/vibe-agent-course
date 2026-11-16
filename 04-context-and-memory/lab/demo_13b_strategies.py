"""
Demo 13b · 三种上下文压缩策略对比（截断 / 摘要 / 分层）
========================================================

配套：第 13 节《上下文管理：从上下文到 ContextManager》

针对"上下文爆炸"问题，给出三种治理手段，构造一段故意埋了"早期事实"和
"近期事实"的 30 轮对话历史，横向对比三种压缩策略：谁丢了早期事实、
谁靠运气、谁稳稳保住。

本文件可独立运行，也被 demo_13c_cache_tax.py / demo_13d_context_manager.py
导入复用（build_fake_history / strategy_* 系列函数）。

运行方式：
    cd /workspace/04-context-and-memory/lab/
    source /workspace/.venv/bin/activate
    python demo_13b_strategies.py
"""

import json
import unicodedata

from _env import chat


# ============================================================
# 小工具：按"显示宽度"对齐表格列（中文字符按 2 格算）
# ============================================================
def _visual_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _rjust_visual(s: str, width: int) -> str:
    return " " * max(width - _visual_width(s), 0) + s


def _print_row(widths: list[int], values: list[str]) -> None:
    print("  ".join(_rjust_visual(v, w) for w, v in zip(widths, values)))


# ============================================================
# 模拟历史：30 轮对话（包含 1 个早期重要事实 + 1 个晚期重要事实）
# ============================================================
def build_fake_history(n_rounds: int = 30) -> list[dict]:
    """构造一段模拟的多轮对话历史，用于测试不同上下文管理策略的效果。

    对话结构：
        1. system 消息：设定角色为旅行助手。
        2. 第 1-2 轮（早期重要事实，关键词能命中）：用户透露"对花粉过敏"，
           容易被截断策略丢弃、或被摘要策略压缩丢失。
        3. 第 3 轮（早期隐性事实，关键词命中不了）：用户说"吃辣一般，太辣受不了"，
           不含任何 `KEY_FACT_KEYWORDS` 关键词，但明显是一条硬约束——
           用来暴露"关键词匹配"这种简化做法的真实盲区。
        4. 中间填充轮：由预定义的问答模板循环填充，模拟大量"噪音"信息。
        5. 倒数第二轮（晚期重要事实）：用户补充"每天预算 500 元"，
           通常能被所有策略保留，作为对照组。

    Args:
        n_rounds: 期望生成的对话轮数，默认 30。

    Returns:
        扁平的消息列表，格式为 `[{"role": str, "content": str}, ...]`。
    """
    n_rounds = n_rounds if n_rounds > 0 else 30

    msgs = [{"role": "system", "content": "你是个旅行助手。"}]

    msgs += [
        {"role": "user", "content": "我准备 6 月去北京玩 7 天。"},
        {"role": "assistant", "content": "好的，6 月北京天气合适。先问您几个问题..."},
        {"role": "user", "content": "重要：我对花粉过敏，户外活动要避开花期。"},  # ⭐ 埋点 1（关键词能命中）
        {"role": "assistant", "content": "明白，会注意避开花季景点。"},
        {"role": "user", "content": "我吃辣一般般，太辣的菜可能受不了，点餐帮我注意一下。"},  # ⭐ 埋点 2（关键词命中不了）
        {"role": "assistant", "content": "好的，推荐餐厅会优先选微辣或不辣的。"},
    ]

    _filler_qa = [
        ("故宫门票需要提前预约吗？", "建议提前在官网预约，现场票经常售罄，尤其旺季。"),
        ("从故宫到颐和园怎么走方便？", "地铁 4 号线直达北宫门站，大约 40 分钟。"),
        ("北京 6 月天气怎么样？", "6 月已经偏热，白天 30°C 左右，注意防晒补水。"),
        ("有没有适合拍照的胡同？", "南锣鼓巷、杨梅竹斜街都不错，建议早上去人少。"),
        ("长城哪个段落人少一点？", "慕田峪比八达岭人少很多，景色也更好。"),
        ("天坛几点开门？", "公园 6:00 开门，祈年殿等核心景点 8:00 开放。"),
        ("北京烤鸭哪家好？", "全聚德、便宜坊是老字号，四季民福性价比也不错。"),
        ("需要带现金吗？", "基本不用，手机支付覆盖几乎所有场景。"),
        ("798 艺术区值得去吗？", "喜欢当代艺术的话很推荐，周末还有各种展览和集市。"),
        ("早晚高峰地铁挤吗？", "8-9 点和 17-19 点非常拥挤，建议错峰出行。"),
        ("什刹海晚上有什么玩的？", "可以坐游船、逛酒吧街，夏天夜晚很热闹。"),
        ("国家博物馆需要身份证吗？", "是的，需要身份证或护照，提前预约可以走快速通道。"),
        ("簋街有什么推荐菜？", "麻辣小龙虾是招牌，很多店营业到凌晨。"),
        ("从大兴机场到市区怎么走？", "乘坐地铁最实惠，非高峰期最快；打车约 100-400 元，看路况和车型。"),
        ("圆明园和颐和园选一个？", "颐和园更完整更好逛，圆明园偏历史遗迹感。"),
        ("有没有免费的景点？", "奥林匹克公园、前门大街、琉璃厂都免费，值得逛。"),
        ("北京的出租车好打吗？", "路边招手不容易，建议用打车软件。"),
        ("王府井小吃街还值得去吗？", "本地人很少去，更推荐牛街或护国寺小吃。"),
        ("后海白天好看还是晚上好看？", "各有味道，白天安静看景，晚上热闹泡吧。"),
        ("香山现在有红叶吗？", "6 月没有红叶，夏天爬山看绿野也不错。"),
        ("鸟巢可以进去参观吗？", "可以的，买票进内部参观，水立方也挨在一起。"),
        ("北京的公共交通用什么卡？", "亿通行 App 扫码就行，地铁公交都能用。"),
        ("景山公园值得爬吗？", "很值得！山顶俯瞰故宫全景是经典机位。"),
        ("北大清华能进去逛吗？", "需要预约，一般提前在公众号上申请参观名额。"),
        ("回程想带点特产，推荐什么？", "稻香村糕点、张一元茶叶、内联升布鞋都挺有特色。"),
    ]
    for index in range(n_rounds - 5):
        question, answer = _filler_qa[index % len(_filler_qa)]
        msgs += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

    msgs += [
        {"role": "user", "content": "对了，我的预算是每天 500 元。"},  # ⭐ 埋点 3（晚期，通常都能保住）
        {"role": "assistant", "content": "我会把每天行程控制在预算内。"},
    ]

    return msgs


# ============================================================
# 策略 ①截断
# ============================================================
def strategy_truncate(messages: list[dict], keep_last: int = 6) -> list[dict]:
    """截断策略：保留所有 system 消息 + 最近 N 轮对话，中间直接砍掉。"""
    keep_last = keep_last if keep_last > 0 else 6
    sys_msgs = [m for m in messages if m["role"] == "system"]
    non_sys = [m for m in messages if m["role"] != "system"]
    recent = non_sys[-keep_last:] if keep_last < len(non_sys) else non_sys
    return sys_msgs + recent


# ============================================================
# 策略 ②摘要
# ============================================================
def strategy_summary(messages: list[dict], keep_last: int = 6) -> list[dict]:
    """摘要压缩策略：system + LLM 摘要（中间段）+ 最近 N 轮原文。

    注意：这个函数每次调用都会用 LLM **重新生成一段新的摘要文本**去替换
    中间历史——即便语义不变，字面内容也几乎必然和上一次不同。这意味着
    调用这个函数 = 主动改写前缀 = 下一次请求无法再复用之前积累的缓存。
    demo_13c_cache_tax.py 会用真实数据验证这一点。
    """
    keep_last = keep_last if keep_last > 0 else 6

    non_sys = [m for m in messages if m["role"] != "system"]
    if len(non_sys) <= keep_last:
        return messages

    sys_msgs = [m for m in messages if m["role"] == "system"]
    middle = non_sys[:-keep_last]
    recent = non_sys[-keep_last:]

    if not middle:
        return sys_msgs + recent

    middle_text = "\n".join(f"{m['role']}: {m['content']}" for m in middle)
    summary_prompt = (
        "请把下面这段对话压缩成一段 100 字以内的中文摘要，"
        "**特别保留任何用户偏好、约束、关键事实**：\n\n" + middle_text
    )
    msg = chat([{"role": "user", "content": summary_prompt}], temperature=0, max_tokens=150)
    summary = msg.content or ""

    return (
        sys_msgs
        + [{"role": "system", "content": f"[早期对话摘要]\n{summary}"}]
        + recent
    )


# ============================================================
# 策略 ③分层 —— 关键事实怎么抽，提供两种实现
# ============================================================

# ---- ③-a 关键词匹配：教学用的简化基线，不是生产做法 ----
KEY_FACT_KEYWORDS = {"过敏", "预算", "重要", "对了", "一定要", "不要"}


def _contains_key_fact(text: str) -> bool:
    return any(kw in text for kw in KEY_FACT_KEYWORDS)


def extract_key_facts_keyword(messages: list[dict]) -> list[str]:
    """基于关键词匹配提取关键事实——**教学简化版，真实生产不会这么做**。

    致命盲区：只要用户没说中 `KEY_FACT_KEYWORDS` 里的词，再硬的约束也抓不到。
    比如"我吃辣一般，太辣受不了"——明显是一条硬约束，但一个关键词都不命中，
    关键词版本会直接漏掉它（下方 `run_strategy_comparison` 会用真实数据验证）。

    留着这个函数主要是作对比基线：让你亲眼看到"看起来能用"和"真正好用"
    之间的差距有多大，而不是直接告诉你"这样不行"。
    """
    facts = []
    for m in messages:
        if m["role"] == "user" and _contains_key_fact(m["content"]):
            facts.append(m["content"].strip())
    return facts


# ---- ③-b LLM 抽取：生产级做法 ----
FACT_EXTRACTION_PROMPT = """请阅读下面的多轮对话，找出其中用户透露的**硬约束 / 偏好 / 关键事实**
（例如：过敏、忌口、预算、时间限制、明确的喜好或厌恶）。

**严格输出 JSON 数组**，每项是一句精炼后的事实描述，没有就输出 `[]`。
不要输出任何其他文字、不要用 ```json 包裹。

对话：
{transcript}

输出："""


def extract_key_facts_llm(messages: list[dict]) -> list[str]:
    """用一次独立的 LLM 调用做事实抽取——生产级做法，本质是一个 Minion。

    Minion：职责收窄到极致的"一次性子 Agent"——接收组装好的 context，
    执行一次，吐出结构化结果，没有多轮循环、不维护跨调用的状态。跟第 3 章
    Plan-and-Solve 里 Planner/Aggregator 的角色完全同构：**收窄职责，一次
    调用只做一件事**，输出结构化 JSON，而不是让模型自由发挥，也不需要像
    ReAct 那样走 Thought-Action-Observation 的循环。

    相比关键词匹配，这是语义理解而非字面匹配，能抓住"我吃辣一般，太辣受不了"
    这类没有命中任何关键词、但显然是硬约束的句子。

    生产环境的两个常见优化（这里为了简化没做，但值得知道）：
        1. Minion 做的是收窄任务，通常会路由到更便宜/更快的模型，而不是用
           驱动主 Agent 循环的那个前沿模型——Claude Code 的分层 compaction、
           SelfCompact 都是这个思路。
        2. 真实生产会用 `response_format={"type": "json_object"}` 或
           pydantic + structured output 强约束输出，而不是手动找 `[` 和 `]`。
    """
    transcript = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages if m["role"] == "user"
    )
    msg = chat(
        [{"role": "user", "content": FACT_EXTRACTION_PROMPT.format(transcript=transcript)}],
        temperature=0,
        max_tokens=300,
    )
    raw = (msg.content or "").strip()

    # 容错解析：和第 11 节 Plan-and-Solve 的 Planner 输出解析同一套思路——
    # 模型偶尔会用 ```json 包裹，或者在数组前后多写几句话。
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        facts = json.loads(raw[start : end + 1])
        return [str(f).strip() for f in facts if str(f).strip()]
    except (json.JSONDecodeError, ValueError):
        return []


def strategy_hierarchical(
    messages: list[dict],
    keep_last: int = 6,
    fact_extractor=extract_key_facts_keyword,
) -> list[dict]:
    """分层压缩策略：system + 关键事实槛 + 早期摘要 + 最近 N 轮原文。

    结构：
        [原始 system]       —— 永不丢失
        [关键事实槛]         —— 单独抽出，跨轮持久化，解决"防变笨"
        [早期对话摘要]       —— 委托 strategy_summary 压缩，解决"防爆炸"
        [最近 N 轮原文]      —— 保持精度

    Args:
        fact_extractor: 事实抽取函数，默认用关键词版（`extract_key_facts_keyword`）
            保持简单确定、不额外调用 LLM；想要生产级效果请传入
            `extract_key_facts_llm`。两者接口一致，可直接替换。
    """
    sys_msgs = [m for m in messages if m["role"] == "system"]
    summarized = strategy_summary(messages, keep_last)

    facts = fact_extractor(messages)
    fact_block = "\n".join(f"- {f}" for f in facts)
    if fact_block:
        return (
            sys_msgs
            + [{"role": "system", "content": f"[关键事实，永远记住]\n{fact_block}"}]
            + [m for m in summarized if m not in sys_msgs]
        )
    return summarized


def stats(messages: list[dict]) -> dict:
    chars = sum(len(m["content"]) for m in messages)
    return {"count": len(messages), "chars": chars}


def fact_check(messages: list[dict]) -> tuple[bool, bool, bool]:
    """检查三个埋点事实是否还在：过敏(早期/关键词能命中)、吃辣(早期/关键词命中不了)、预算(近期)。"""
    text = " ".join(m["content"] for m in messages)
    return "过敏" in text, "辣" in text, "预算" in text


def run_strategy_comparison() -> None:
    """四种策略横向对比——处理后的规模 vs 三个埋点事实是否还在。

    注意：这张表里「吃辣(隐性/早期)」这一列只能看出"最终组装出的消息里
    是否还留有相关文字"，**不能**直接当作"事实抽取器抓没抓到"的证据——
    比如 ③a/③b 都走了 `strategy_summary` 生成摘要，LLM 写摘要时偶尔会
    自己提到"辣"字，这会让这一列呈假阳性。真正干净的证据见下面
    `run_extraction_comparison`：直接对比两个抽取函数的原始输出。
    """
    history = build_fake_history(n_rounds=30)
    print(f"历史 messages：{len(history)} 条 / {sum(len(m['content']) for m in history)} 字符\n")

    keep_last = 6
    strategies = {
        "不处理（基线）": history,
        f"① 截断（最近 {keep_last} 轮）": strategy_truncate(history, keep_last=keep_last),
        f"② 摘要（摘要 + 最近 {keep_last} 轮）": strategy_summary(history, keep_last=keep_last),
        f"③a 分层（关键词抽取）": strategy_hierarchical(
            history, keep_last=keep_last, fact_extractor=extract_key_facts_keyword
        ),
        f"③b 分层（LLM 抽取）": strategy_hierarchical(
            history, keep_last=keep_last, fact_extractor=extract_key_facts_llm
        ),
    }

    widths = [24, 8, 8, 10, 14, 10]
    _print_row(widths, ["策略", "条数", "字符数", "过敏(早期)", "吃辣(隐性/早期)", "预算(近期)"])
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for name, msgs in strategies.items():
        s = stats(msgs)
        has_aller, has_spicy, has_budget = fact_check(msgs)
        _print_row(
            widths,
            [
                name,
                str(s["count"]),
                str(s["chars"]),
                "✅" if has_aller else "❌",
                "✅" if has_spicy else "❌",
                "✅" if has_budget else "❌",
            ],
        )

    print("\n💡 截断靠位置，摘要靠运气，分层靠算法——关键事实能不能保住，取决于工程方案，而非碰巧。")
    print("💡 「吃辣」这一列不够干净（可能被摘要文本意外带出），下面单独对比两个抽取器的原始输出：")


def run_extraction_comparison() -> None:
    """直接对比两个事实抽取函数的原始输出——绕开 strategy_summary 这类
    会"意外带出"关键词的中间步骤，给出一份干净、不会被污染的证据。

    这里只调用 `extract_key_facts_keyword` / `extract_key_facts_llm` 本身，
    不涉及摘要、不涉及最终组装的 messages，看的就是"抽取器到底抽到了什么"。
    """
    history = build_fake_history(n_rounds=30)

    keyword_facts = extract_key_facts_keyword(history)
    llm_facts = extract_key_facts_llm(history)

    print("关键词版抽取结果（extract_key_facts_keyword）：")
    for f in keyword_facts:
        print(f"  - {f}")
    if not keyword_facts:
        print("  （空，一条都没抓到）")

    print("\nLLM 版抽取结果（extract_key_facts_llm）：")
    for f in llm_facts:
        print(f"  - {f}")
    if not llm_facts:
        print("  （空）")

    spicy_in_keyword = any("辣" in f for f in keyword_facts)
    spicy_in_llm = any("辣" in f for f in llm_facts)
    print('\n💡 「我吃辣一般，太辣的菜可能受不了」—— 不含任何 KEY_FACT_KEYWORDS 关键词：')
    print(f"   关键词版：{'✅ 抓到了（意外？检查关键词表）' if spicy_in_keyword else '❌ 完全漏掉——这就是关键词匹配的真实盲区'}")
    print(f"   LLM   版：{'✅ 理解语义，正确抓住' if spicy_in_llm else '❌ 也没抓到（可重试或检查 prompt）'}")


if __name__ == "__main__":
    run_strategy_comparison()
    print("\n" + "=" * 78 + "\n")
    run_extraction_comparison()
