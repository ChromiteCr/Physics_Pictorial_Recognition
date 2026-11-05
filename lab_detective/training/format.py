"""训练/评测样本格式（docs/ARCHITECTURE.md §2.5）与泄漏检查。

服务器端 server/common.py 只认这个格式。每行一个 JSON：
  id, task, source, messages[{role, content}], meta{experiment_id, split, seed, ...}
content 可以是字符串，也可以是 [{"type": "image", "image": "images/x.jpg"}, {"type": "text", "text": "..."}]。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

TASKS = ("diagnose", "design", "recognize", "guide", "explain", "card", "knowledge")
SOURCES = ("engine", "template", "photo", "distill-deepseek", "distill-hy4")
ROLES = ("system", "user", "assistant")


@dataclass
class Example:
    id: str
    task: str
    source: str
    messages: list[dict]
    meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        assert self.task in TASKS, self.task
        assert self.source in SOURCES, self.source
        assert self.messages and self.messages[-1]["role"] == "assistant", "最后一条必须是 assistant"
        for m in self.messages:
            assert m["role"] in ROLES, m["role"]
            c = m["content"]
            if isinstance(c, list):
                for part in c:
                    assert part.get("type") in ("text", "image"), part
            else:
                assert isinstance(c, str)

    def to_dict(self) -> dict:
        return {"id": self.id, "task": self.task, "source": self.source,
                "messages": self.messages, "meta": self.meta}

    @staticmethod
    def from_dict(d: dict) -> "Example":
        return Example(d["id"], d["task"], d["source"], d["messages"], d.get("meta", {}))

    def prompt_text(self) -> str:
        """除最后一条 assistant 以外的全部文本（泄漏检查用）。"""
        parts = []
        for m in self.messages[:-1]:
            c = m["content"]
            if isinstance(c, str):
                parts.append(c)
            else:
                parts.extend(p.get("text", "") or p.get("image", "") for p in c)
        return "\n".join(parts)

    def images(self) -> list[str]:
        out = []
        for m in self.messages:
            c = m["content"]
            if isinstance(c, list):
                out.extend(p["image"] for p in c if p.get("type") == "image")
        return out


def content_hash(ex: Example) -> str:
    """只看输入（提示 + 图片路径），用于发现"同一道题既在训练集又在测试集"。"""
    return hashlib.sha256(ex.prompt_text().encode("utf-8")).hexdigest()


def write_jsonl(path: str | Path, examples: Iterable[Example]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            ex.validate()
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Example]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Example.from_dict(json.loads(line))


def leakage_report(train: Iterable[Example], tests: Iterable[Example]) -> dict:
    """按 id 和输入内容哈希双重检查测试样本是否混进训练集。"""
    train = list(train)
    tests = list(tests)
    ids = {e.id for e in train}
    hashes = {content_hash(e): e.id for e in train}
    id_hits = [t.id for t in tests if t.id in ids]
    hash_hits = [(t.id, hashes[h]) for t in tests if (h := content_hash(t)) in hashes]
    return {"n_train": len(train), "n_test": len(tests), "id_overlap": id_hits,
            "content_overlap": hash_hits, "clean": not id_hits and not hash_hits}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
