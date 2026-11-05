"""实验目录：整理好的实验（起点）+"其他实验"（由物理组件组合）。不 import streamlit。

实验键（session_state 里只存字符串，不存对象）：
  - 整理好的实验：labsim 注册表里的 id，例如 "hooke"；
  - 其他实验："other:" + spec 的 JSON（键排序），同一份 spec 永远得到同一个孪生（固定种子，缓存）。

"其他实验"优先交给物理组件层 lab_detective.physics（并行开发中，懒加载）：
模块里有 experiment_from_spec(spec) 就用它；模块不存在或没有这个入口时退回 labsim.generic.GenericExperiment。
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import math
import re

import numpy as np

OTHER = "__other__"  # 手动选择下拉框里"其他实验"这一项
OTHER_PREFIX = "other:"

FAMILIES: dict[str, str] = {
    "proportional": "正比：y = a·x",
    "linear": "线性：y = a·x + b",
    "inverse": "反比：y = a / x",
    "square": "平方：y = a·x²",
    "sqrt": "开方：y = a·√x",
    "power": "幂：y = a·xⁿ",
}


# ── 整理好的实验 ───────────────────────────────────────────────────────


def registry() -> dict:
    from lab_detective.labsim.registry import EXPERIMENTS

    return EXPERIMENTS


@functools.lru_cache(maxsize=1)
def _cards() -> dict:
    try:
        from lab_detective import knowledge

        return knowledge.all_cards()
    except Exception:  # noqa: BLE001  知识库不可用时只用注册表里的名字
        return {}


def card_for(exp_id: str) -> dict | None:
    c = _cards().get(exp_id)
    return json.loads(json.dumps(c)) if c else None


def curated() -> list[dict]:
    """整理好的实验列表 [{id, name, grade}]：按识别字母表的顺序，字母表里没有的排在后面。"""
    reg = registry()
    try:
        from lab_detective.vision.letters import TABLE

        order = [e.experiment_id for e in TABLE if e.experiment_id]
    except Exception:  # noqa: BLE001
        order = []
    ids = [i for i in order if i in reg] + sorted(i for i in reg if i not in order)
    out = []
    for i in ids:
        card = _cards().get(i) or {}
        out.append({"id": i, "name": card.get("name_zh") or reg[i].name_zh, "grade": card.get("grade", "")})
    return out


# ── 其他实验：由物理组件组合 ───────────────────────────────────────────


def component_layer():
    """物理组件层（lab_detective.physics）；不存在时返回 None。"""
    try:
        return importlib.import_module("lab_detective.physics")
    except ImportError:
        return None


def other_key(spec: dict) -> str:
    return OTHER_PREFIX + json.dumps(spec, sort_keys=True, ensure_ascii=False)


def is_other(key: str | None) -> bool:
    return bool(key) and key.startswith(OTHER_PREFIX)


def spec_of(key: str) -> dict:
    return json.loads(key[len(OTHER_PREFIX):])


@functools.lru_cache(maxsize=32)
def _build_other(spec_json: str):
    spec = json.loads(spec_json)
    layer = component_layer()
    hook = getattr(layer, "experiment_from_spec", None) if layer is not None else None
    if callable(hook):
        return hook(spec)
    from lab_detective.labsim.generic import GenericExperiment

    return GenericExperiment(spec, np.random.default_rng(0))  # 固定种子：重跑页面时原因清单不变


def get_experiment(key: str):
    """实验键 → Experiment（整理好的实验取注册表里的单例，其他实验按 spec 构造并缓存）。"""
    if is_other(key):
        return _build_other(key[len(OTHER_PREFIX):])
    return registry()[key]


def backend_name(key: str) -> str:
    if not is_other(key):
        return "整理好的实验孪生"
    layer = component_layer()
    return "物理组件层" if layer is not None and callable(getattr(layer, "experiment_from_spec", None)) \
        else "通用关系族 + 通用误差算子"


def pseudo_card(exp, spec: dict) -> dict:
    """其他实验没有实验卡：按 spec 拼一张最小的卡（导师引导语要用目的、变量、公式）。"""
    xv, yv = spec["x_var"], spec["y_var"]
    fam = spec.get("relation_family", "proportional")
    return {
        "id": exp.id, "name_zh": exp.name_zh, "grade": "",
        "goal": f"研究{yv['name']}与{xv['name']}的关系（{FAMILIES.get(fam, fam)}）。",
        "formula": FAMILIES.get(fam, fam).split("：")[-1],
        "variables": [
            {"symbol": xv.get("symbol", "x"), "name": xv["name"], "unit": xv.get("unit", ""), "role": "control"},
            {"symbol": yv.get("symbol", "y"), "name": yv["name"], "unit": yv.get("unit", ""), "role": "measured"},
        ],
        "apparatus": [t for t in (xv.get("tool"), yv.get("tool")) if t],
        "steps": [], "knowledge_points": [], "relation_family": fam,
    }


_SPLIT = re.compile(r"[,\s，、;；]+")


def parse_settings(text: str) -> list[float]:
    """"10, 20 30；40" → [10, 20, 30, 40]；认不出的片段报 ValueError。"""
    out = []
    for tok in _SPLIT.split((text or "").strip()):
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError as e:
            raise ValueError(f"认不出的取值：{tok}") from e
        if not math.isfinite(v):
            raise ValueError(f"取值不是有限数：{tok}")
        out.append(v)
    return out


def make_spec(name: str, x: dict, y: dict, family: str, settings, exponent: float | None = None,
              slope_known: float | None = None, slope_guess: float | None = None) -> dict:
    """表单 → spec（并做物理上的合法性检查，出错抛 ValueError，信息给学生看）。

    x / y：{name, symbol, unit, tool, resolution}；settings：控制量取值；
    slope_known：理论斜率已知时填（例如已知密度），否则 None；slope_guess：斜率的粗略估计（未知时用来定先验）。
    """
    if family not in FAMILIES:
        raise ValueError(f"未知的关系类型：{family}")
    xs = sorted(set(float(v) for v in settings))
    if len(xs) < 3:
        raise ValueError("至少要 3 个不同的控制量取值，才能看出关系和误差")
    q = {"proportional": 1, "linear": 1, "inverse": -1, "square": 2, "sqrt": 0.5}.get(family, exponent)
    if q is None:
        raise ValueError("幂关系要填指数 n")
    if (q < 0 or q != int(q)) and min(xs) <= 0:
        raise ValueError("这种关系要求控制量都大于 0")
    for v, who in ((x, "自变量"), (y, "因变量")):
        if not str(v.get("name", "")).strip():
            raise ValueError(f"请填写{who}的名称")
        if not float(v.get("resolution", 0)) > 0:
            raise ValueError(f"{who}的仪器分辨率要大于 0")
    clean = lambda v, sym: {  # noqa: E731
        "name": str(v["name"]).strip(), "symbol": str(v.get("symbol") or sym).strip(),
        "unit": str(v.get("unit", "")).strip(), "tool": str(v.get("tool") or "测量仪器").strip(),
        "resolution": float(v["resolution"]),
    }
    nm = (name or "").strip() or f"探究{y['name']}与{x['name']}的关系"
    uid = hashlib.sha1(nm.encode("utf-8")).hexdigest()[:6]
    spec = {
        "id": f"other_{uid}", "name": nm, "name_en": "custom setup",
        "x_var": clean(x, "x"), "y_var": clean(y, "y"),
        "relation_family": family, "settings": xs,
        "ideal_slope": float(slope_known) if slope_known else None,
    }
    if family == "power":
        spec["exponent"] = float(exponent)
    if not slope_known and slope_guess:
        spec["slope_guess"] = float(slope_guess)
    return spec


def _sig3(v: float) -> float:
    return float(f"{v:.3g}")


def calibrate_other(key: str, observations) -> str:
    """其他实验、斜率未知：用已测的主测量数据粗拟合斜率（和线性族的截距），作为先验中心，返回新的实验键。

    通用孪生的斜率先验是"估计值 ± 25%"；学生没填估计时默认 1，和真实量级可能差几个数量级。
    拟合值取 3 位有效数字，数据小改时键不变（缓存稳定）。不满足条件时原样返回。
    """
    if not is_other(key):
        return key
    spec = spec_of(key)
    if spec.get("ideal_slope") is not None:
        return key
    pts = [(o.setting, o.values[0]) for o in observations if o.kind == "main" and o.values]
    if len({p[0] for p in pts}) < 2:
        return key
    fam = spec.get("relation_family", "proportional")
    q = {"proportional": 1, "linear": 1, "inverse": -1, "square": 2, "sqrt": 0.5}.get(fam, spec.get("exponent"))
    try:
        X = np.array([p[0] for p in pts], float) ** float(q)
    except (TypeError, ValueError):
        return key
    y = np.array([p[1] for p in pts], float)
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        return key
    if fam == "linear":
        a, b = np.polyfit(X, y, 1)
        spec["intercept_guess"] = _sig3(b) if b != 0 else 0.0
    else:
        a = float(X @ y / (X @ X)) if X @ X > 0 else 0.0
    if not math.isfinite(a) or a == 0:
        return key
    spec["slope_guess"] = _sig3(a)
    return other_key(spec)
