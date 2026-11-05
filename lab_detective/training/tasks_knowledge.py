"""知识问答的训练样本：从 knowledge/ 的知识点图谱生成"误差原因 / 错误概念 → 知识点和前置知识点"的问答。

三类题，每类 3 种问法：
  A 原因 → 知识点：训练实验的每个可用原因（不含扣下的原因），答案 = related_kps 的前 3 个 + 它们的直接前置。
  B 错误概念 → 知识点：知识点表里的 misconceptions（按症状或说法提问），答案 = 所属知识点 + 它的直接前置。
  C 知识点 → 前置知识点：答案 = learning_path 里离它最近的至多 5 个（保持先学在前的顺序）。
只用"训练可用"的知识点：没有关联实验的通用知识点、关联了训练实验的知识点、以及训练原因 / 训练实验卡用到的知识点；
只和留出实验有关的知识点（单摆、杠杆、平抛……）不出题；题面提到留出实验的错误概念也不出题。
题库是确定的；训练 / 验证按题目键的 crc32 分（每 20 题 1 题进验证，约 5%），各自按基础种子打乱后按序号取。
"""

from __future__ import annotations

import json
import zlib
from functools import lru_cache

import numpy as np

from .. import knowledge
from ..labsim.registry import EXPERIMENTS, get_experiment
from ..tutor.templates import compose
from . import prompts
from .format import Example
from .splits import BASE_SEED, HELD_OUT_EXPERIMENTS, TRAIN_EXPERIMENTS, allowed_causes, example_seed

MAX_KP = 3
MAX_PRE = 5
HELD_OUT_WORDS = ("单摆", "摆球", "摆长", "平抛", "杠杆", "阿特伍德", "自由落体")

Q_CAUSE = (
    "在「{exp}」中，如果出现“{cause}”（{desc}），这和哪些知识点有关？要先掌握哪些前置知识？",
    "学生做「{exp}」时，误差原因是“{cause}”。请指出相关的知识点和需要先复习的前置知识。",
    "“{cause}”是「{exp}」里的一个常见误差原因。它涉及哪些知识点？学之前要会什么？",
)
Q_MISC = (
    "学生出现了这样的情况：{symptom}。这反映了什么错误认识？和哪个知识点有关？应先复习哪些前置知识？",
    "有学生认为：{text}。这种想法错在哪里，属于哪个知识点？要先掌握什么？",
    "批改实验报告时发现：{symptom}。请指出相关的知识点和前置知识。",
)
Q_PRE = (
    "要学好「{kp}」，需要先掌握哪些知识点？",
    "「{kp}」的前置知识点有哪些？请按先学后学的顺序列出。",
    "学生想弄懂「{kp}」，应该先复习什么？",
)


def held_out_names() -> list[str]:
    names = []
    for eid in HELD_OUT_EXPERIMENTS:
        names.append(EXPERIMENTS[eid].name_zh)
        try:
            names.append(knowledge.load_card(eid)["name_zh"])
        except KeyError:
            pass
    return [n for n in dict.fromkeys(names) if n]


def _mentions_held_out(text: str) -> bool:
    return any(w in text for w in HELD_OUT_WORDS) or any(n in text for n in held_out_names())


@lru_cache(maxsize=1)
def train_kps() -> frozenset[str]:
    """训练可用的知识点 id。"""
    ok: set[str] = set()
    for k in knowledge.all_kps():
        if not k["experiments"] or any(e in TRAIN_EXPERIMENTS for e in k["experiments"]):
            ok.add(k["id"])
    for eid in TRAIN_EXPERIMENTS:
        try:
            ok.update(knowledge.load_card(eid).get("knowledge_points", []))
        except KeyError:
            pass
        for cid in allowed_causes(get_experiment(eid)):
            ok.update(knowledge.related_kps(eid, cid)[:MAX_KP])
    return frozenset(c for c in map(knowledge.canonical, ok) if c)


def _kp_item(kp_id: str) -> dict:
    return {"id": kp_id, "name": knowledge.kp(kp_id)["name_zh"]}


def _direct_pre(kp_ids: list[str]) -> list[str]:
    """几个知识点的直接前置（去掉它们自己），按全图的先学顺序排，至多 MAX_PRE 个。"""
    g = knowledge.graph()
    pre = [p for k in kp_ids for p in g.get(k, []) if p not in kp_ids]
    order = {k: i for i, k in enumerate(g)}
    seen = list(dict.fromkeys(pre))
    path_pos: dict[str, int] = {}
    for k in kp_ids:
        for i, p in enumerate(knowledge.learning_path(k)):
            path_pos.setdefault(p, i)
    return sorted(seen, key=lambda p: (path_pos.get(p, 10**6), order.get(p, 0)))[:MAX_PRE]


def _names(ids: list[str]) -> str:
    return "、".join(f"「{knowledge.kp(i)['name_zh']}」" for i in ids)


def _answer(kps: list[str], pre: list[str], head: str) -> str:
    parts = [head, f"相关知识点：{_names(kps)}。"]
    parts.append(f"先要掌握{_names(pre)}。" if pre else "它没有前置知识点，可以直接学。")
    return compose(parts, prompts.EXPLAIN_LIMIT)


@lru_cache(maxsize=1)
def question_pool() -> tuple[dict, ...]:
    """确定的题库：每题 {key, question, target, experiment_id, kind}。"""
    allowed = train_kps()
    pool: list[dict] = []
    # A：原因 → 知识点
    for eid in TRAIN_EXPERIMENTS:
        exp = get_experiment(eid)
        try:
            ename = knowledge.load_card(eid)["name_zh"]
        except KeyError:
            ename = exp.name_zh
        for cid in allowed_causes(exp):
            c = exp.cause(cid)
            kps = [k for k in knowledge.related_kps(eid, cid) if k in allowed][:MAX_KP]
            if not kps:
                continue
            pre = _direct_pre(kps)
            tgt = {"knowledge_points": [_kp_item(k) for k in kps], "prerequisites": [_kp_item(p) for p in pre],
                   "answer": _answer(kps, pre, "数据偏离只来自随机误差，没有系统误差。" if c.mechanism == "none"
                                     else f"“{c.name_zh}”是一种系统误差的来源。")}
            for v, q in enumerate(Q_CAUSE):
                pool.append({"key": f"cause|{cid}|{v}", "kind": "cause", "experiment_id": eid,
                             "question": q.format(exp=ename, cause=c.name_zh, desc=c.description_zh.rstrip("。")),
                             "target": tgt})
    # B：错误概念 → 知识点
    for k in knowledge.all_kps():
        if k["id"] not in allowed:
            continue
        for m in k.get("misconceptions", []):
            if _mentions_held_out(m.get("text", "") + m.get("symptom", "")):
                continue
            pre = _direct_pre([k["id"]])
            tgt = {"knowledge_points": [_kp_item(k["id"])], "prerequisites": [_kp_item(p) for p in pre],
                   "answer": _answer([k["id"]], pre, f"错误认识：{m['text'].rstrip('。')}。")}
            for v, q in enumerate(Q_MISC):
                pool.append({"key": f"misc|{m['id']}|{v}", "kind": "misconception", "experiment_id": "knowledge",
                             "question": q.format(symptom=m["symptom"].rstrip("。"), text=m["text"].rstrip("。")),
                             "target": tgt})
    # C：知识点 → 前置知识点
    for k in knowledge.all_kps():
        if k["id"] not in allowed:
            continue
        path = knowledge.learning_path(k["id"])[-MAX_PRE:]
        tgt = {"knowledge_points": [_kp_item(k["id"])], "prerequisites": [_kp_item(p) for p in path],
               "answer": compose([f"先学{_names(path)}，再学「{k['name_zh']}」。" if path
                                  else f"「{k['name_zh']}」没有前置知识点，可以直接学。"], prompts.EXPLAIN_LIMIT)}
        for v, q in enumerate(Q_PRE):
            pool.append({"key": f"pre|{k['id']}|{v}", "kind": "prerequisite", "experiment_id": "knowledge",
                         "question": q.format(kp=k["name_zh"]), "target": tgt})
    return tuple(pool)


def split_of(item: dict) -> str:
    """按题目（不含问法序号）分：同一题的 3 种问法同在训练或同在验证。"""
    base = item["key"].rsplit("|", 1)[0]
    return "val" if zlib.crc32(base.encode("utf-8")) % 20 == 0 else "train"


@lru_cache(maxsize=4)
def ordered_pool(split: str, base_seed: int = BASE_SEED) -> tuple[dict, ...]:
    items = [it for it in question_pool() if split_of(it) == split]
    perm = np.random.default_rng([int(base_seed), 4, 1 if split == "train" else 2]).permutation(len(items))
    return tuple(items[int(i)] for i in perm)


def pool_size(split: str, base_seed: int = BASE_SEED) -> int:
    return len(ordered_pool(split, base_seed))


def build_example(index: int, split: str, base_seed: int | None = None) -> Example | None:
    """题库按序号取一题；序号超出题库大小时返回 None。"""
    base = BASE_SEED if base_seed is None else base_seed
    pool = ordered_pool(split, base)
    if index >= len(pool):
        return None
    it = pool[index]
    messages = prompts.build_knowledge_messages(it["question"])
    meta = {"experiment_id": it["experiment_id"], "split": split, "seed": example_seed("knowledge", split, index, base),
            "kind": it["kind"], "question_key": it["key"], "prompt_version": prompts.PROMPT_VERSION}
    key = it["key"].replace("|", "-")
    ex = Example(f"know-{split}-{index:06d}-{key}", "knowledge", "template",
                 messages + [{"role": "assistant", "content": json.dumps(it["target"], ensure_ascii=False)}], meta)
    ex.validate()
    return ex
