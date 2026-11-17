"""
Demo 14a · 记忆第一版：历史会话向量库
=======================================

配套：第 14 节《记忆系统：历史会话 + Profile 档案卡》的第一版。

跨会话记忆最直觉的做法：把聊过的每次会话压成一条摘要存下来，下次用当前问题
按语义召回相关的往期会话。

  - 存"说过的话"：每次会话压成一条摘要，再 embed 入库（只增不改）
  - **持久化到本地**：摘要 + 向量一起落盘（JSON），下次直接加载，不用重新 embed
  - 取用时用当前问题做**向量召回 top-k**（最小本地 RAG：加载向量库 → 检索）

向量检索在这里用得其所——摘要是自由文本、条数会一直涨，关键词匹配接不住，
必须靠 embedding + 余弦。落盘 + 加载 + 召回这一套，就是一个麻雀虽小的本地 RAG。

但它有两个硬伤，正是这两个硬伤引出了第二版 Profile 档案卡（见 demo_14b_profile.py）：
  1. 该一直生效的偏好时有时无——召回是概率性 top-k，稳定偏好/习惯/领域关键事实
     未必每次都排进来（一致性问题）；
  2. 偏好变了没法更新——只增不改，预算 500 和 800 会同时存在（准确性问题）。

运行方式：
    cd /workspace/04-context-and-memory/lab/
    source /workspace/.venv/bin/activate
    python demo_14a_history.py
"""

import json
import os
from dataclasses import dataclass

import numpy as np
from openai import OpenAI

from _env import chat, get_api_key, get_base_url


# 单独建一个 client：embeddings 端点和 chat 端点共用同一组凭证
_emb_client = OpenAI(base_url=get_base_url(), api_key=get_api_key())

# 注意：模型名以你的 LLM gateway 实际暴露的为准
# CNB / 混元一般支持 hunyuan-embedding；OpenAI 兼容就用 text-embedding-3-small
EMBEDDING_MODEL = "hunyuan-embedding"

# 本地持久化文件：摘要 + 向量一起落盘，就是一个最小的本地向量库
STORE_PATH = os.path.join(os.path.dirname(__file__), "history_store.json")


def embed(text: str) -> np.ndarray:
    """调用 embedding API 将文本转换为语义向量。

    通过 OpenAI 兼容的 embeddings 接口，把任意自然语言文本编码为一个固定
    维度的浮点向量。语义相近的文本在向量空间里距离更近（余弦相似度更高）。

    注意：只有「历史会话」层用到它——Profile 是结构化字段，整卡注入，
    不需要 embedding。这正是分层的意义：别拿一种机制硬套两种数据。

    Args:
        text: 待编码的文本内容，长度由 API 端点的上下文窗口决定。

    Returns:
        形状为 `(d,)` 的 float32 向量，`d` 是 embedding 模型的输出维度
        （随模型而定，hunyuan-embedding 约 1024 或 1536）。
    """
    resp = _emb_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量之间的余弦相似度。

    衡量的是方向上的接近程度，与向量长度无关。值域 [-1, 1]，越接近 1 语义越近。
    分母加 `1e-9` 防止零向量导致的除零错误。

        cos(a, b) = (a · b) / (|a| × |b|)
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ============================================================
# 历史会话向量库（非结构化，向量召回）
# ============================================================
SESSION_SUMMARY_SYSTEM = """请把下面这次对话压缩成**一句话要点**（50 字以内），
保留用户这次最关心的问题和结论，用于日后按语义检索。只输出这句话本身。
"""


@dataclass
class SessionEntry:
    """一条历史会话记录：一次会话的要点摘要 + 它的语义向量。"""

    summary: str
    vec: np.ndarray


class History:
    """历史会话向量库：每次会话压成一条摘要存起来，按语义召回。

    这是最直觉的第一版记忆：把聊过的存下来、下次按相关性捞回来。
    向量检索在这里用得其所——摘要是自由文本、条数会一直涨，关键词匹配接不住，
    必须靠 embedding + 余弦。

    但它有两个硬伤（见第 14 节讲义），正是这两个硬伤引出了 Profile 档案卡：
        1. 该一直生效的偏好时有时无——召回是概率性 top-k，稳定偏好/习惯/领域
           关键事实未必每次都排进来（一致性问题）；
        2. 偏好变了没法更新——只增不改，预算 500 和 800 会同时存在（准确性问题）。
    """

    def __init__(self) -> None:
        self._entries: list[SessionEntry] = []

    @property
    def entries(self) -> tuple[SessionEntry, ...]:
        return tuple(self._entries)

    def add_summary(self, summary: str) -> None:
        """直接存一条现成的摘要（文本 → 向量 → 入库）。"""
        summary = summary.strip()
        if summary:
            self._entries.append(SessionEntry(summary=summary, vec=embed(summary)))

    def add_session(self, messages: list[dict]) -> str:
        """会话结束时调用：先用 LLM 把整段对话压成一条要点，再 embed 入库。

        存摘要而非逐轮原文：省空间、召回更干净（一次会话 = 一个检索单位）。

        Returns:
            这次存入的摘要文本（便于演示打印）。
        """
        transcript = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in messages
            if m["role"] in ("user", "assistant")
        )
        msg = chat(
            [
                {"role": "system", "content": SESSION_SUMMARY_SYSTEM},
                {"role": "user", "content": transcript},
            ],
            temperature=0,
            max_tokens=120,
        )
        summary = (msg.content or "").strip()
        self.add_summary(summary)
        return summary

    def search(self, query: str, k: int = 3) -> list[tuple[float, SessionEntry]]:
        """按语义召回 top-k 条历史会话。

        注意排序用显式 `key=lambda x: x[0]`：分数打平时不会去比较第二个元素
        （SessionEntry 默认不可比较，靠 tuple 默认比较会 TypeError）。
        """
        if not self._entries:
            return []
        q = embed(query)
        scored = [(cosine(q, e.vec), e) for e in self._entries]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    def save(self, path: str) -> None:
        """持久化到 JSON：向量从 ndarray 转 list 存入。"""
        data = [{"summary": e.summary, "vec": e.vec.tolist()} for e in self._entries]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """从 JSON 还原：list 转回 float32 ndarray。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._entries = [
            SessionEntry(summary=d["summary"], vec=np.array(d["vec"], dtype=np.float32))
            for d in data
        ]


# ============================================================
# 演示：建库落盘 → 重新加载 → 向量召回（最小本地 RAG 闭环）
# ============================================================
def demo_persistence() -> None:
    """走通本地 RAG 闭环：建库 → 落盘 → 换新实例加载 → 向量召回。

    关键点：加载时直接读回摘要 + 向量，**已有条目不用重新 embed**，只有
    查询串需要现算一次。这正是持久化向量库相对"每次重建"的价值。
    """
    print("=" * 60)
    print("【历史会话向量库：建库落盘 → 重新加载 → 向量召回】")
    print("=" * 60)

    # 1) 建库并写入磁盘（真实场景里这些摘要是历次会话结束时陆续追加的）
    writer = History()
    past_summaries = [
        "用户咨询周末北京户外活动，因花粉过敏最终选择了室内博物馆。",
        "用户想给橘猫买自动喂食器，纠结 300 元档和 600 元档的差别。",
        "用户计划下月去上海出差三天，问了高铁和酒店的性价比。",
        "用户问咖啡机怎么选，强调只喝美式、不要带奶泡功能。",
    ]
    for s in past_summaries:
        print(f"  + 入库: {s}")
        writer.add_summary(s)
    writer.save(STORE_PATH)
    size_kb = os.path.getsize(STORE_PATH) / 1024
    print(f"\n已把 {len(writer.entries)} 条会话（摘要+向量）落盘：")
    print(f"  {STORE_PATH}  ({size_kb:.1f} KB)")

    # 2) 模拟"下次启动"：一个全新的空实例，从磁盘加载
    reader = History()
    print(f"\n新实例加载前：{len(reader.entries)} 条")
    reader.load(STORE_PATH)
    print(f"新实例加载后：{len(reader.entries)} 条（向量已随文件还原，无需重新 embed）")

    # 3) 直接在加载好的库上做向量召回（本地 RAG 检索）
    for query in ["周末去哪玩比较好", "有没有养宠物相关的事", "出差相关"]:
        print(f"\n查询: {query}")
        for score, e in reader.search(query, k=2):
            print(f"  [{score:.3f}] {e.summary}")

    print("\n👀 持久化 → 加载 → 召回：这套就是一个最小的本地 RAG。")
    print("   生产里把 JSON 换成 FAISS / Qdrant / pgvector，接口思路完全一致。")
    print("\n   但它是『概率性 top-k + 只增不改』，有两个硬伤：")
    print("   · 稳定偏好/习惯/领域关键事实可能排不进 top-k 被漏掉（一致性）")
    print("   · 偏好变了新旧值还会并存打架（准确性）")
    print("   这两个硬伤，正是 demo_14b_profile.py 里 Profile 档案卡要解决的。")


if __name__ == "__main__":
    demo_persistence()