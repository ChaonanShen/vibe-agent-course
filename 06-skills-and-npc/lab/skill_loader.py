"""
SkillLoader · Skills 加载的核心（对应讲义第 20 节「两级渐进式加载」）
=====================================================================

一句话：Skill = 一份 SKILL.md（YAML frontmatter 元数据 + Markdown 正文规范）。
加载分两级，核心就为了「省 token」：

    Level 1（启动时，常驻几十 token）
        扫描 skills/ 目录，每个 skill 只取 name + 一句话 brief，拼成索引
        → 注入 system prompt，让模型"知道有哪些能力"

    Level 2（运行时，命中才付费）
        某个 skill 被命中时，才真正读它的 SKILL.md 全文注入上下文

本文件只实现 Loader 本身（不依赖任何 LLM / 网络），纯本地文件操作。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Skill:
    """一个被完整加载的 skill（Level 2 产物）。"""
    name: str       # 唯一标识（= 目录名）
    brief: str      # 一句话摘要
    body: str       # SKILL.md 全文


# 合法 skill 名：字母/数字/下划线/连字符，挡掉 ../ 之类的目录穿越
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


# ============================================================
# frontmatter / brief 解析
# ============================================================

def extract_brief(text: str) -> str:
    """从 SKILL.md 文本里提取一句话摘要 brief。

    优先级：
      1. YAML frontmatter 里的 description 字段
      2. 没有 frontmatter 时，退化为正文第一个"非标题非空行"
    """
    # 1) 尝试解析 frontmatter（文件以 --- 开头）
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            front = text[3:end]
            for line in front.splitlines():
                if line.strip().lower().startswith("description:"):
                    return line.split(":", 1)[1].strip()

    # 2) 退化：正文第一个非标题非空行
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith("---"):
            return s
    return "(无描述)"


# ============================================================
# SkillLoader：扫描 + 索引 + 按需读全文
# ============================================================

class SkillLoader:
    """负责三件事：
      - list_index(): Level 1，只读元数据生成 (name, brief) 索引（惰性 + 缓存）
      - load(name):   Level 2，命中某 skill 才读 SKILL.md 全文
      - 安全：白名单校验名字 + 校验路径不逃出 root（防目录穿越）
    """

    def __init__(self, skills_dir: str):
        self.root = os.path.abspath(skills_dir)
        self._index: list[tuple[str, str]] | None = None  # 缓存，首次访问才填充
        self._dirs: dict[str, str] = {}                   # name → 实际目录绝对路径

    # ---- Level 1：只读元数据，生成一句话索引 ----
    def list_index(self) -> list[tuple[str, str]]:
        """返回 [(name, brief), ...]。首次调用才真正扫盘（惰性）。"""
        if self._index is None:
            self._index = self._scan()
        return self._index

    def _scan(self) -> list[tuple[str, str]]:
        result: dict[str, tuple[str, str]] = {}
        if not os.path.isdir(self.root):
            return []

        for name in os.listdir(self.root):
            child = os.path.join(self.root, name)
            if not os.path.isdir(child):
                continue

            skill_md = os.path.join(child, "SKILL.md")
            if os.path.isfile(skill_md):
                brief = extract_brief(_read_head(skill_md))  # 只读文件头若干 KB
                result[name] = (name, brief)
                self._dirs[name] = child
            else:
                # 兼容"嵌套 skill 包"：child/skills/<inner>/SKILL.md
                nested = os.path.join(child, "skills")
                if os.path.isdir(nested):
                    for inner in os.listdir(nested):
                        inner_dir = os.path.join(nested, inner)
                        inner_md = os.path.join(inner_dir, "SKILL.md")
                        if os.path.isfile(inner_md):
                            brief = extract_brief(_read_head(inner_md))
                            result[inner] = (inner, brief)
                            self._dirs[inner] = inner_dir

        return sorted(result.values(), key=lambda x: x[0])

    # ---- Level 2：命中某个 skill 才读全文 ----
    def load(self, name: str) -> Skill:
        """读取某个 skill 的 SKILL.md 全文。"""
        if not _NAME_RE.match(name):
            raise ValueError(f"非法 skill 名: {name!r}")

        self.list_index()  # 确保 _dirs 已填充
        skill_dir = self._dirs.get(name)
        if skill_dir is None:
            raise KeyError(f"未找到 skill: {name!r}")

        # 防目录穿越：加载目录必须仍在 root 之内
        real = os.path.realpath(skill_dir)
        if os.path.commonpath([real, self.root]) != self.root:
            raise PermissionError(f"路径逃逸: {name!r}")

        with open(os.path.join(real, "SKILL.md"), "r", encoding="utf-8") as f:
            text = f.read()
        return Skill(name=name, brief=extract_brief(text), body=text)


def _read_head(path: str, max_bytes: int = 4096) -> str:
    """只读文件头若干字节——Level 1 提取 brief 不需要读全文。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read(max_bytes)
