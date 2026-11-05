"""实验前设计（规划）任务的训练样本：twin.py / uncertainty.py / atlas.py → 目标 JSON。

提示里给的（全部是引擎算好的数字或学生可见的信息）：实验卡要点、计划的方案、仪器的分度值和随机读数误差、
孪生对这台装置的预期读数（参数从先验抽一次，等于"一台具体的装置"）、标准方案下斜率的不确定度、
候选原因（编号随机）、追加测量菜单（编号随机）。不给：不确定度来源的份额、图谱结果。
目标：
  expected         照抄孪生预期读数（第一个读数；均值按分辨率多保留一位，标准差 1~2 位有效数字）
  dominant_source  uncertainty.budget 里对斜率贡献最大的那一类（random / resolution）和读数名称
  recommend        计划只测 1 组 → 改用标准方案的点数和范围；标准方案斜率相对不确定度 > 5% → 点数加倍（≤ 12）；
                   repeat_each = 随机读数误差的份额 > 分辨率的份额
  unresolved       atlas.atlas_table(方案, resolve=True) 里两两判别正确率 < 0.9 的原因对（< 0.75 记 confused，
                   其余 partial；按正确率从低到高，最多 6 对）；measurement = 最能分开这一对的追加测量，
                   正确率提升 < 0.1 时为 null
  explanation      ≤ 80 字
程序化实例占 70%（训练实验只有两种方案，提示变化少）；训练实验的图谱结果按 (实验, 方案) 缓存。
"""

from __future__ import annotations

import json
import warnings
import zlib

import numpy as np

from ..atlas import atlas_table
from ..facts import dec_for_res, dec_for_unc
from ..labsim.core import Experiment
from ..tutor.templates import compose
from ..twin import instantiate
from ..uncertainty import budget
from . import prompts
from .format import Example
from .splits import allowed_causes, example_rng, example_seed
from .tasks_diagnose import sample_experiment

DESIGN_GENERIC_FRACTION = 0.7
P_SINGLE = 0.2
ATLAS_N = 32  # 每个原因模拟的份数（每对 64 份）
BUDGET_N = 120
REL_LIMIT = 0.05
MAX_POINTS = 12
MAX_PAIRS = 6
MIN_GAIN = 0.1
CONFUSED_BELOW = 0.75
SEPARABLE_AT = 0.9

_ATLAS_CACHE: dict[tuple, list[dict]] = {}
_BUDGET_CACHE: dict[str, dict] = {}


def _round_dec(v: float, dec: int) -> float:
    r = round(float(v), max(dec, 0))
    return 0.0 if r == 0 else r


def _atlas(exp: Experiment, protocol: str, causes, generic: bool, rng: np.random.Generator) -> list[dict]:
    key = (exp.id, protocol, tuple(causes))
    if not generic and key in _ATLAS_CACHE:
        return _ATLAS_CACHE[key]
    arng = rng if generic else np.random.default_rng(zlib.crc32(f"{exp.id}|{protocol}|atlas".encode()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = atlas_table(exp, [protocol], ATLAS_N, arng, cause_ids=causes, resolve=True)
    if not generic:
        _ATLAS_CACHE[key] = rows
    return rows


def _budget(exp: Experiment, generic: bool, rng: np.random.Generator) -> dict:
    if not generic and exp.id in _BUDGET_CACHE:
        return _BUDGET_CACHE[exp.id]
    brng = rng if generic else np.random.default_rng(zlib.crc32(f"{exp.id}|budget".encode()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = budget(exp, "standard", n=BUDGET_N, rng=brng)
    if not generic:
        _BUDGET_CACHE[exp.id] = b
    return b


def expected_rows(exp: Experiment, protocol: str, rng: np.random.Generator) -> list[dict]:
    """一台具体的装置：干扰量和共享线性参数从先验各抽一次（视为已知），原因 = 只有随机误差。"""
    ro = exp.random_only_id
    params = {"nuisance": {k: float(p.draw(rng)) for k, p in exp.nuisance.items()},
              "linear": {p.name: p.draw(rng) for p in exp.linear_params(ro)}}
    tw = instantiate(exp, params, ro, n_draws=1, seed=0)
    out = []
    for row in tw.predict_rows(protocol):
        r = row["readings"][0]
        dec = dec_for_res(r["resolution"]) + 1
        sd_dec = dec_for_unc(r["sd"])
        out.append({"setting": row["setting"], "reading": r["label"],
                    "value": _round_dec(r["mean"], dec), "sd": _round_dec(r["sd"], sd_dec if sd_dec is not None else dec),
                    "unit": r["unit"]})
    return out


def dominant_source(b: dict) -> dict:
    """斜率不确定度的主要来源：按类型汇总份额，取大的一类；读数名称取该类里份额最大的（相近时取靠前的）。"""
    items = (b.get("slope") or {}).get("sources") or []
    if not items:
        src = (b.get("sources") or [{}])[0]
        rd = src.get("label_zh", "读数")
        return {"reading": rd, "type": "random" if src.get("sigma", 0) ** 2 >= src.get("resolution", 0) ** 2 / 12
                else "resolution"}
    by_type = {"random": 0.0, "resolution": 0.0}
    for it in items:
        by_type[it["type"]] += it["share"]
    typ = max(by_type, key=lambda k: (by_type[k], k == "random"))
    same = [it for it in items if it["type"] == typ]
    top = max(it["share"] for it in same)
    order = {s["label_zh"]: i for i, s in enumerate(b.get("sources") or [])}
    pick = min((it for it in same if it["share"] >= 0.8 * top), key=lambda it: order.get(_reading_of(it), 0))
    return {"reading": _reading_of(pick), "type": typ}


def _reading_of(item: dict) -> str:
    lab = item["label_zh"]
    for suf in ("的随机读数误差", "的读数分辨率"):
        if lab.endswith(suf):
            return lab[: -len(suf)]
    return lab


def recommend(exp: Experiment, protocol: str, b: dict) -> dict:
    plan = exp.protocols["standard"].plan
    main = plan[0][0]
    settings = [s for k, s in plan if k == main]
    n_std = len(settings)
    rel = (b.get("slope") or {}).get("rel_sd")
    n = n_std
    if rel is not None and rel > REL_LIMIT:
        n = min(MAX_POINTS, 2 * n_std)
    shares = {"random": 0.0, "resolution": 0.0}
    for it in (b.get("slope") or {}).get("sources") or []:
        shares[it["type"]] += it["share"]
    return {"n_points": int(n), "range": [float(min(settings)), float(max(settings))],
            "unit": exp.obs_kinds[main].setting_unit, "repeat_each": bool(shares["random"] > shares["resolution"])}


def budget_line(b: dict) -> str:
    """标准方案下斜率的相对不确定度（名义参数下的引擎预算；不写份额）。"""
    s = b.get("slope")
    if not s or s.get("sd") is None:
        return "引擎预算：按标准方案无法拟合直线，给不出斜率的不确定度。"
    if s.get("rel_sd") is None:
        return "引擎预算：斜率接近 0，不谈相对不确定度。"
    return f"引擎预算：按标准方案测量，斜率的相对标准不确定度约 {s['rel_sd'] * 100:.1f}%（名义参数下）。"


def unresolved(rows: list[dict], disp: dict[str, str], menu_disp: dict[str, str]) -> list[dict]:
    """图谱行 → 分不开的原因对。按正确率的点估计分级（每对只有 2·ATLAS_N 份，Wilson 区间太宽）：
    < CONFUSED_BELOW 记 confused，< SEPARABLE_AT 记 partial，其余算能分开、不列出。"""
    bad = [r for r in rows if r["acc"] < SEPARABLE_AT]
    bad.sort(key=lambda r: (r["acc"], r["cause_a"], r["cause_b"]))
    out = []
    for r in bad[:MAX_PAIRS]:
        a, b = sorted((disp[r["cause_a"]], disp[r["cause_b"]]), key=lambda s: int(s[1:]))
        m = r.get("best_item")
        gain = (r.get("best_item_acc", 0.0) - r["acc"]) if m else 0.0
        out.append({"pair": [a, b], "status": "confused" if r["acc"] < CONFUSED_BELOW else "partial",
                    "measurement": menu_disp[m] if m and gain >= MIN_GAIN else None})
    return out


def design_explanation(exp: Experiment, protocol: str, rec: dict, dom: dict, unres: list[dict],
                       names: dict[str, str], menu_labels: dict[str, str]) -> str:
    what = "随机读数误差" if dom["type"] == "random" else "读数分辨率"
    parts = [f"只测 1 组看不出规律，建议改测 {rec['n_points']} 组。" if protocol == "single"
             else f"建议测 {rec['n_points']} 组。"]
    if unres:
        u = unres[0]
        s = f"「{names[u['pair'][0]]}」和「{names[u['pair'][1]]}」不好分开"
        parts.append(s + (f"，可以{menu_labels[u['measurement']]}。" if u["measurement"] else "。"))
    else:
        parts.append("各原因在这个方案下都能分开。")
    parts.append(f"主要误差来源是{dom['reading']}的{what}，"
                 + ("宜多读几次取平均。" if rec["repeat_each"] else "重复读数帮助不大。"))
    return compose(parts, prompts.EXPLAIN_LIMIT)


def build_example(index: int, split: str, base_seed: int | None = None,
                  generic_fraction: float = DESIGN_GENERIC_FRACTION) -> Example:
    kw = {} if base_seed is None else {"base": base_seed}
    seed = example_seed("design", split, index, **kw)
    rng = example_rng("design", split, index, **kw)
    exp, family, generic = sample_experiment(rng, generic_fraction)
    protocol = "single" if rng.random() < P_SINGLE and "single" in exp.protocols else "standard"
    causes = allowed_causes(exp)
    rows_exp = expected_rows(exp, protocol, rng)
    b = _budget(exp, generic, rng)
    arows = _atlas(exp, protocol, causes, generic, rng)
    messages, disp, menu_disp = prompts.design_prompt(exp, protocol, rows_exp, budget_line(b), causes, rng)
    rec = recommend(exp, protocol, b)
    dom = dominant_source(b)
    unres = unresolved(arows, disp, menu_disp)
    names = {d: exp.cause(c).name_zh for c, d in disp.items()}
    labels = {d: exp.menu_item(m).label_zh for m, d in menu_disp.items()}
    target = {
        "expected": [{"setting": r["setting"], "value": r["value"], "sd": r["sd"], "unit": r["unit"]} for r in rows_exp],
        "dominant_source": dom,
        "recommend": rec,
        "unresolved": unres,
        "explanation": design_explanation(exp, protocol, rec, dom, unres, names, labels),
    }
    meta = {"experiment_id": family, "split": split, "seed": seed, "generic": generic, "protocol": protocol,
            "causes": [{"id": d, "name": names[d]} for d in disp.values()],
            "id_map": {d: c for c, d in disp.items()}, "menu_map": {d: m for m, d in menu_disp.items()},
            "slope_rel_sd": (b.get("slope") or {}).get("rel_sd"), "prompt_version": prompts.PROMPT_VERSION}
    if generic:
        meta["instance_id"] = exp.id
    ex = Example(f"design-{family}-{split}-{index:06d}", "design", "engine",
                 messages + [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}], meta)
    ex.validate()
    return ex

