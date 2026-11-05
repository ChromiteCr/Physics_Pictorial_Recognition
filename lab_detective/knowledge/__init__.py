"""实验卡 + 知识点图谱（初中物理、高中必修一/必修二/选择性必修一的力学部分，以及测量科学）。

数据都在 knowledge/data/ 下：
  knowledge_points.json   知识点列表（含前置知识点、常见错误概念、关联实验）
  cards/<实验 id>.json    实验卡（各实验作者分别维护，缺失的卡片会被跳过）
  cause_kp_map.json       误差原因 → 知识点：按机制标签（mechanism）和按具体原因（cause）

graph() 的方向：知识点 → 它的前置知识点（与 JSON 里的 prerequisites 一致）。
知识点可带 aliases（别名）：kp()/related_kps()/learning_path() 接受别名并返回正式 id；
卡片里用了别名时 validate_all 会提示改成正式 id。
labsim 注册表只在需要时才导入（related_kps、validate_all），导入失败不会让加载器崩溃。
"""

from __future__ import annotations

import json
from functools import lru_cache
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

from .schema import (
    EXPERIMENT_IDS,
    MECHANISM_TAGS,
    validate_card,
    validate_cause_map,
    validate_kp_list,
)

DATA = Path(__file__).parent / "data"
CARDS = DATA / "cards"

__all__ = [
    "load_card", "all_cards", "kp", "all_kps", "graph", "related_kps", "learning_path", "canonical",
    "validate_all", "card_errors", "clear_cache",
]


def _read(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 加载 ───────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _kps() -> tuple[dict, ...]:
    return tuple(_read(DATA / "knowledge_points.json"))


@lru_cache(maxsize=1)
def _kp_index() -> dict[str, dict]:
    return {k["id"]: k for k in _kps()}


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    """别名 → 正式 id（与正式 id 冲突的别名忽略，由校验报告）。"""
    ids = set(_kp_index())
    out: dict[str, str] = {}
    for k in _kps():
        for a in k.get("aliases", []):
            if a not in ids:
                out.setdefault(a, k["id"])
    return out


def canonical(kp_id: str) -> str | None:
    """正式 id 原样返回，别名换成正式 id，不认识的返回 None。"""
    if kp_id in _kp_index():
        return kp_id
    return _alias_index().get(kp_id)


@lru_cache(maxsize=1)
def _cause_map() -> dict:
    p = DATA / "cause_kp_map.json"
    return _read(p) if p.exists() else {"mechanism": {}, "cause": {}}


@lru_cache(maxsize=1)
def _scan_cards() -> tuple[dict[str, dict], dict[str, str]]:
    """读所有卡片。返回 (能解析的卡片, 解析失败的文件 -> 错误)。半成品卡片不会让加载器崩溃。"""
    ok: dict[str, dict] = {}
    bad: dict[str, str] = {}
    if not CARDS.is_dir():
        return ok, bad
    for p in sorted(CARDS.glob("*.json")):
        try:
            card = _read(p)
        except (OSError, ValueError) as e:
            bad[p.name] = f"{type(e).__name__}: {e}"
            continue
        if not isinstance(card, dict):
            bad[p.name] = "顶层不是 JSON 对象"
            continue
        ok[p.stem] = card
    return ok, bad


def clear_cache() -> None:
    """数据文件改动后（例如别的作者刚写好卡片）重新读取。"""
    for f in (_kps, _kp_index, _alias_index, _cause_map, _scan_cards):
        f.cache_clear()


def load_card(exp_id: str) -> dict:
    """读一张实验卡；卡片不存在或无法解析时抛 KeyError。返回副本，调用方改了也不影响缓存。"""
    ok, bad = _scan_cards()
    if exp_id in ok:
        return json.loads(json.dumps(ok[exp_id]))
    if f"{exp_id}.json" in bad:
        raise KeyError(f"实验卡 {exp_id}.json 无法解析：{bad[exp_id + '.json']}")
    raise KeyError(f"没有实验卡：{exp_id}")


def all_cards() -> dict[str, dict]:
    """所有能解析的实验卡：{实验 id: 卡片}。缺失或损坏的卡片跳过。"""
    ok, _ = _scan_cards()
    return {k: json.loads(json.dumps(v)) for k, v in ok.items()}


def kp(kp_id: str) -> dict:
    """按 id（或别名）取知识点。"""
    cid = canonical(kp_id)
    if cid is None:
        raise KeyError(f"没有知识点：{kp_id}")
    return json.loads(json.dumps(_kp_index()[cid]))


def all_kps() -> list[dict]:
    return json.loads(json.dumps(list(_kps())))


def graph() -> dict[str, list[str]]:
    """邻接表：知识点 id → 前置知识点 id 列表。"""
    return {k["id"]: list(k["prerequisites"]) for k in _kps()}


# ── 查询 ───────────────────────────────────────────────────────────────


def _dedup(ids) -> list[str]:
    seen: set[str] = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _cause_mechanism(labsim_id: str, cause_id: str) -> str | None:
    try:
        from ..labsim.registry import get_experiment

        return get_experiment(labsim_id).cause(cause_id).mechanism
    except Exception:  # 注册表里没有这个实验/原因，或某个并行实验文件导入失败
        return None


def related_kps(exp_id: str, cause_id: str | None = None) -> list[str]:
    """与实验（以及给定的误差原因）相关的知识点 id，去重并按相关性排序。

    不给原因：实验卡的 knowledge_points（无卡片时用 experiments 含该实验的知识点）→ 其余关联知识点。
    给了原因：实验的核心知识点（卡片第一个）→ 原因专属知识点（卡片 cause_kps + 映射文件 cause 段）→
    原因机制对应的测量科学知识点 → 卡片其余知识点 → 其余关联知识点。取前几个就能拿到与原因最相关的。
    别名换成正式 id，不认识的 id 丢弃。
    """
    by_exp = [k["id"] for k in _kps() if exp_id in k["experiments"]]
    ok, _ = _scan_cards()
    card = ok.get(exp_id, {})
    lst = lambda v: [i for i in v if isinstance(i, str)] if isinstance(v, list) else []  # noqa: E731
    core: list[str] = lst(card.get("knowledge_points")) or by_exp
    ids: list[str] = core[:1] if cause_id else list(core)
    if cause_id:
        cks = card.get("cause_kps")
        ids += lst(cks.get(cause_id)) if isinstance(cks, dict) else []
        cmap = _cause_map()
        ids += lst(cmap.get("cause", {}).get(cause_id))
        labsim_id = card.get("labsim_id") if isinstance(card.get("labsim_id"), str) else exp_id
        mech = _cause_mechanism(labsim_id, cause_id)
        if mech is not None:
            ids += lst(cmap.get("mechanism", {}).get(mech))
        ids += core
    ids += by_exp
    return _dedup(c for c in map(canonical, ids) if c is not None)


def learning_path(kp_id: str) -> list[str]:
    """kp_id 的全部（直接和间接）前置知识点，按拓扑序排列：先学的在前，不含 kp_id 本身。"""
    g = graph()
    start = canonical(kp_id)
    if start is None:
        raise KeyError(f"没有知识点：{kp_id}")
    seen: set[str] = set()  # 祖先集合
    stack = [start]
    while stack:
        for p in g[stack.pop()]:
            if p in g and p not in seen:
                seen.add(p)
                stack.append(p)
    pos = {k: i for i, k in enumerate(g)}  # 同一层按 JSON 顺序，结果确定
    ts = TopologicalSorter({k: [p for p in g[k] if p in g] for k in seen})
    ts.prepare()
    out: list[str] = []
    while ts.is_active():
        ready = sorted(ts.get_ready(), key=pos.__getitem__)
        out.extend(ready)
        ts.done(*ready)
    return out


# ── 校验 ───────────────────────────────────────────────────────────────


def _kp_problems() -> list[str]:
    probs: list[str] = []
    kps = list(_kps())
    probs += validate_kp_list(kps)
    ids = [k.get("id") for k in kps]
    known = set(ids)
    for i in sorted({i for i in ids if ids.count(i) > 1}):
        probs.append(f"知识点 id 重复：{i}")
    aliases = [a for k in kps for a in k.get("aliases", [])]
    for a in sorted({a for a in aliases if aliases.count(a) > 1}):
        probs.append(f"知识点别名重复：{a}")
    for a in sorted(set(aliases) & known):
        probs.append(f"知识点别名与正式 id 冲突：{a}")
    mcs = [m.get("id") for k in kps for m in k.get("misconceptions", [])]
    for m in sorted({m for m in mcs if mcs.count(m) > 1}):
        probs.append(f"错误概念 id 重复：{m}")
    for k in kps:
        for p in k.get("prerequisites", []):
            if p not in known:
                probs.append(f"{k.get('id')}: 前置知识点不存在 {p}")
            if p == k.get("id"):
                probs.append(f"{k.get('id')}: 前置知识点是自己")
    try:
        TopologicalSorter({k["id"]: [p for p in k["prerequisites"] if p in known] for k in kps}).prepare()
    except CycleError as e:
        probs.append(f"知识点图有环：{' → '.join(e.args[1])}")
    covered = {e for k in kps for e in k.get("experiments", [])}
    for e in EXPERIMENT_IDS:
        if e not in covered:
            probs.append(f"没有任何知识点关联实验 {e}")
    return probs


def _cause_map_problems() -> list[str]:
    m = _cause_map()
    probs = validate_cause_map(m)
    for sec in ("mechanism", "cause"):
        for key, ids in (m.get(sec) or {}).items():
            probs += _kp_ref_problems(f"cause_kp_map.json: {sec}.{key}", ids)
    for tag in MECHANISM_TAGS:
        if tag not in (m.get("mechanism") or {}):
            probs.append(f"cause_kp_map.json: 缺少机制标签 {tag}")
    return probs


def _registry():
    """惰性导入 labsim 注册表。返回 (EXPERIMENTS, 错误信息)。"""
    try:
        from ..labsim.registry import EXPERIMENTS

        return EXPERIMENTS, None
    except Exception as e:  # 某个并行实验文件有语法错误等
        return None, f"labsim 注册表导入失败：{type(e).__name__}: {e}"


def _kp_ref_problems(where: str, ids) -> list[str]:
    out = []
    for i in ids if isinstance(ids, list) else []:
        c = canonical(i) if isinstance(i, str) else None
        if c is None:
            out.append(f"{where}: 知识点不存在 {i}")
        elif c != i:
            out.append(f"{where}: {i} 是别名，请改成正式 id {c}")
    return out


def card_errors(exp_id: str, card: dict | None = None, registry=None) -> list[str]:
    """一张卡片的全部问题：结构、id 一致性、知识点是否存在、原因是否在 labsim 注册表里。"""
    where = f"cards/{exp_id}.json"
    if card is None:
        try:
            card = load_card(exp_id)
        except KeyError as e:
            return [f"{where}: {e.args[0]}"]
    probs = validate_card(card, where)
    if not isinstance(card, dict):
        return probs
    if card.get("id") != exp_id:
        probs.append(f"{where}: id={card.get('id')!r} 与文件名不一致")
    if exp_id not in EXPERIMENT_IDS:
        probs.append(f"{where}: {exp_id} 不在 12 个实验之列")
    probs += _kp_ref_problems(f"{where}: knowledge_points", card.get("knowledge_points", []))
    causes = set(card.get("common_causes", []))
    for cid, ids in (card.get("cause_kps") or {}).items():
        if cid not in causes:
            probs.append(f"{where}: cause_kps 里的 {cid} 不在 common_causes 中")
        probs += _kp_ref_problems(f"{where}: cause_kps.{cid}", ids)
    labsim_id = card.get("labsim_id", exp_id)
    if labsim_id != exp_id:
        probs.append(f"{where}: labsim_id={labsim_id!r} 与实验 id 不一致")
    if registry is None:
        registry, err = _registry()
        if err:
            return probs + [f"{where}: {err}"]
    exp = registry.get(labsim_id)
    if exp is None:
        probs.append(f"{where}: labsim 注册表里没有实验 {labsim_id}")
        return probs
    reg_causes = set(exp.cause_ids())
    for cid in card.get("common_causes", []):
        if cid not in reg_causes:
            probs.append(f"{where}: 原因 {cid} 不在 labsim 实验 {labsim_id} 的原因库里")
    cmap = (_cause_map().get("mechanism") or {})
    for c in exp.causes:
        if c.mechanism not in cmap:
            probs.append(f"{where}: 原因 {c.id} 的机制 {c.mechanism} 在 cause_kp_map.json 里没有映射")
    return probs


def validate_all() -> list[str]:
    """检查全部数据，返回问题清单（空列表 = 没问题）。"""
    probs = _kp_problems()
    probs += _cause_map_problems()
    for cid in (_cause_map().get("cause") or {}):
        if cid.split(".")[0] not in EXPERIMENT_IDS:
            probs.append(f"cause_kp_map.json: cause.{cid} 的实验前缀未知")
    ok, bad = _scan_cards()
    for name, err in bad.items():
        probs.append(f"cards/{name}: 无法解析（{err}）")
    registry, err = _registry()
    if err:
        probs.append(err)
        registry = {}
    for exp_id, card in ok.items():
        probs += card_errors(exp_id, card, registry)
    return probs
