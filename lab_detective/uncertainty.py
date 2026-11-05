"""实验前的不确定度预算与线性传播。

budget：在名义参数（先验均值）、"只有随机误差"的条件下，按方案蒙特卡洛模拟 n 组数据，
用特征面板的直线拟合求斜率和截距，报告它们的标准差，以及每个读数的"分辨率"和"随机误差"
各贡献多少（逐个关掉一个来源，看方差减少多少）。

取整的处理（重要）：名义参数是固定的，若直接取整，没有噪声时每次结果都一样，分辨率的贡献会算成 0。
实际上学生并不知道真值落在刻度的哪个位置，所以每组模拟给真值加一个在 ±d/2 内均匀分布的随机相位，
取整后再减掉（减性抖动）。这样取整误差严格服从 U(−d/2, d/2)、与随机噪声独立，
方差正好是引擎用的 σ² + d²/12，各来源的方差可以相加。
"""

from __future__ import annotations

import math
import re
from typing import Callable, Mapping, Sequence

import numpy as np

from .labsim.core import Dataset, Experiment, Observation
from .panel import compute_panel
from .twin import nominal_truth, noise_free_dataset, resolve_plan

# ── 单位 ──────────────────────────────────────────────────────────────

_UNIT_RE = re.compile(r"[（(]([^()（）]*)[)）]\s*$")


def unit_of(label: str) -> str:
    """从"伸长量 x (cm)"这样的坐标轴标签里取出单位。"""
    m = _UNIT_RE.search(label or "")
    return m.group(1).strip() if m else ""


def _wrap(u: str) -> str:
    return f"({u})" if any(ch in u for ch in "/· *") else u


def slope_unit(x_label: str, y_label: str) -> str:
    """斜率的单位 = 纵轴单位 / 横轴单位（同单位之比无量纲；除以 1/m 写成 ·m）。"""
    xu, yu = unit_of(x_label), unit_of(y_label)
    if xu == yu:
        return ""
    if not xu:
        return yu
    if xu.startswith("1/"):
        inner = xu[2:].strip()
        inner = inner[1:-1] if inner.startswith("(") and inner.endswith(")") else inner
        return f"{_wrap(yu)}·{_wrap(inner)}" if yu else inner
    if not yu:
        return f"1/{_wrap(xu)}"
    return f"{_wrap(yu)}/{_wrap(xu)}"


def with_unit(text: str, unit: str) -> str:
    return f"{text} {unit}" if unit else text


# ── 线性传播 ──────────────────────────────────────────────────────────


def propagate(f: Callable, values, sds, rel_step: float = 1e-3) -> dict:
    """数值线性传播（输入相互独立）：u(f)² = Σ (∂f/∂x_i · u_i)²，偏导用中心差分。

    values / sds：同为字典（f 按关键字调用）或同为序列（f 按位置调用）。
    返回 {value, sd, terms: [{name, value, sd, partial, var, share}]}。
    """
    if isinstance(values, Mapping):
        names = list(values)
        x0 = np.array([float(values[k]) for k in names])
        u = np.array([float(sds.get(k, 0.0)) for k in names])

        def call(x):
            return float(f(**dict(zip(names, x))))
    else:
        x0 = np.array([float(v) for v in values])
        u = np.array([float(s) for s in sds])
        names = [f"x{i}" for i in range(x0.size)]

        def call(x):
            return float(f(*x))

    f0 = call(x0)
    terms, var = [], 0.0
    for i in range(x0.size):
        h = u[i] * rel_step if u[i] > 0 else 1e-6 * max(abs(x0[i]), 1.0)
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        d = (call(xp) - call(xm)) / (2 * h)
        v = (d * u[i]) ** 2
        var += v
        terms.append({"name": names[i], "value": float(x0[i]), "sd": float(u[i]), "partial": float(d), "var": float(v)})
    for t in terms:
        t["share"] = t["var"] / var if var > 0 else 0.0
    return {"value": f0, "sd": math.sqrt(var), "terms": terms}


# ── 实验前不确定度预算 ────────────────────────────────────────────────


def _fit(exp: Experiment, ds: Dataset) -> tuple[float, float]:
    p = compute_panel(exp.linearise(ds))
    return p.slope, p.intercept


def _f(v) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def budget(exp: Experiment, protocol="standard", n: int = 400, rng: np.random.Generator | None = None) -> dict:
    """实验前不确定度预算（JSON 可序列化，中文标签）。

    返回 slope / intercept 两项，每项含：名义值、蒙特卡洛标准差、公式（加权最小二乘）标准差、
    相对不确定度，以及 sources：每个读数的分辨率 / 随机误差各自的方差贡献和占比。
    控制量（如钩码质量）按模型视为准确，不计入。
    """
    rng = np.random.default_rng(0) if rng is None else rng
    plan = resolve_plan(exp, protocol)
    pid = protocol if isinstance(protocol, str) else "custom"
    plabel = exp.protocols[pid].label_zh if pid in exp.protocols else "自定义方案"
    truth = nominal_truth(exp)
    nominal = noise_free_dataset(exp, truth, plan, meta={"protocol": pid})
    lin0 = exp.linearise(nominal)
    p0 = compute_panel(lin0)
    out = {
        "experiment_id": exp.id, "experiment_name": exp.name_zh, "protocol": pid, "protocol_label": plabel,
        "n_mc": int(n), "x_label": lin0.x_label, "y_label": lin0.y_label,
        "nominal": {"cause": truth["cause"], "nuisance": truth["nuisance"], "linear": truth["linear"]},
        "notes": ["控制量（横轴设定值）按模型视为准确，不计入不确定度。",
                  "取整按真值在刻度内的位置未知处理（均匀分布），即分辨率贡献 d²/12。"],
    }
    if not np.isfinite(p0.slope):
        out.update(slope=None, intercept=None, sources=[],
                   summary_zh=f"「{plabel}」的数据点不足两个不同的横轴值，无法拟合直线，不能给出斜率和截距的不确定度。")
        return out

    # 每个读数值的均值、σ、d，以及它属于哪个"来源"
    means, sig, res, src_of = [], [], [], []
    sources: list[dict] = []
    src_index: dict[tuple[str, str], int] = {}
    for i, (kind, s) in enumerate(plan):
        mean, sg = exp._mean_and_sigma(truth, kind, np.array([float(s)]), np.array([float(i)]))
        d = exp.resolutions(kind)
        for j, rd in enumerate(exp.obs_kinds[kind].readings):
            means.append(float(mean[0, j]))
            sig.append(float(sg[j]))
            res.append(float(d[j]))
            src_of.append((kind, rd.key))
            if (kind, rd.key) not in src_index:
                src_index[(kind, rd.key)] = len(sources)
                sources.append({"kind": kind, "reading": rd.key, "label_zh": rd.label_zh, "unit": rd.unit,
                                "sigma": float(sg[j]), "resolution": float(d[j])})
    m, sig, res = np.array(means), np.array(sig), np.array(res)
    shape = [len(exp.obs_kinds[k].readings) for k, _ in plan]

    # 公共随机数：所有开关组合共用同一组噪声和相位，差值更稳
    E = rng.standard_normal((n, m.size))
    Ph = rng.uniform(-0.5, 0.5, (n, m.size)) * res

    def run(sig_on: np.ndarray, res_on: np.ndarray) -> np.ndarray:
        vals = m + E * (sig * sig_on)
        rmask = res_on & (res > 0)
        dd = np.where(res > 0, res, 1.0)
        vals = np.where(rmask, np.round((vals + Ph) / dd) * dd - Ph, vals)
        fits = np.empty((n, 2))
        for t in range(n):
            obs, k = [], 0
            for (kind, s), r in zip(plan, shape):
                obs.append(Observation(kind, float(s), tuple(float(v) for v in vals[t, k:k + r])))
                k += r
            fits[t] = _fit(exp, Dataset(exp.id, obs, None, {"protocol": pid}))
        return fits

    on = np.ones(m.size, bool)
    full = run(on, on)
    var_full = np.nanvar(full, axis=0, ddof=1)
    # 逐个关掉一个来源（某读数的 σ 或 d），方差的减少量 = 这个来源的贡献
    contribs = []  # (来源序号, 类型, 方差减少 (2,))
    for si, src in enumerate(sources):
        mask = np.array([src_of[q] == (src["kind"], src["reading"]) for q in range(m.size)])
        for typ, val in (("random", src["sigma"]), ("resolution", src["resolution"])):
            if val <= 0:
                continue
            sig_on = on & ~mask if typ == "random" else on
            res_on = on & ~mask if typ == "resolution" else on
            v_off = np.nanvar(run(sig_on, res_on), axis=0, ddof=1)
            contribs.append((si, typ, np.maximum(var_full - v_off, 0.0)))

    xu_label, yu = lin0.x_label, unit_of(lin0.y_label)
    units = {"slope": slope_unit(xu_label, lin0.y_label), "intercept": yu}
    labels = {"slope": "斜率", "intercept": "截距"}
    nominal_vals = {"slope": p0.slope, "intercept": p0.intercept}
    formula_sd = {"slope": p0.slope_se, "intercept": p0.intercept_se}
    for qi, q in enumerate(("slope", "intercept")):
        tot = sum(c[2][qi] for c in contribs)
        items = []
        for si, typ, dv in contribs:
            src = sources[si]
            what = "随机读数误差" if typ == "random" else "读数分辨率"
            val = src["sigma"] if typ == "random" else src["resolution"]
            items.append({
                "id": f"{src['kind']}.{src['reading']}.{typ}",
                "label_zh": f"{src['label_zh']}的{what}",
                "type": typ, "value": val, "unit": src["unit"],
                "var": float(dv[qi]), "share": float(dv[qi] / tot) if tot > 0 else 0.0,
            })
        items.sort(key=lambda d: -d["share"])
        sd = float(math.sqrt(var_full[qi]))
        nv = float(nominal_vals[q])
        nv = 0.0 if abs(nv) < 1e-6 * sd else nv  # 去掉浮点残差（如 7e-15）
        out[q] = {
            "label_zh": labels[q], "unit": units[q], "nominal": _f(nv),
            "mean_mc": _f(np.nanmean(full[:, qi])), "sd": _f(sd), "sd_formula": _f(formula_sd[q]),
            # 接近 0 的量不谈相对不确定度
            "rel_sd": _f(sd / abs(nv)) if abs(nv) > sd else None, "sources": items,
        }
    out["sources"] = sources
    s = out["slope"]
    top = "、".join(f"{it['label_zh']}（{it['share'] * 100:.0f}%）" for it in s["sources"][:2] if it["share"] > 0)
    rel = f"，相对 {s['rel_sd'] * 100:.1f}%" if s["rel_sd"] is not None else ""
    it = out["intercept"]
    slope_txt = with_unit(f"{s['nominal']:.4g}", s["unit"])
    sd_txt = with_unit(f"{s['sd']:.2g}", s["unit"])
    isd_txt = with_unit(f"{it['sd']:.2g}", it["unit"])
    out["summary_zh"] = (
        f"按「{plabel}」测量，斜率预计 {slope_txt}，标准不确定度约 {sd_txt}{rel}；截距的标准不确定度约 {isd_txt}。"
        + (f"主要来源：{top}。" if top else "")
    )
    return out
