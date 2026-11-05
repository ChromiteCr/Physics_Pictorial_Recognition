"""界面背后的纯逻辑（不 import streamlit）：数据表 ↔ 观测、逐行实时对照、模拟数据、事实单与幽灵曲线、
测量范围建议、可辨识性摘要、修正后复测对照、知识点。所有数字都由 lab_detective 的物理代码算出。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from lab_detective.labsim.core import Dataset, Experiment, Observation

Z_TENSION = 2.0
Z_INCONSISTENT = 3.0


def band_of(z) -> str | None:
    """|z| < 2 一致，2~3 有张力，> 3 不一致（与 facts.band_of 相同）。"""
    if z is None or not math.isfinite(float(z)):
        return None
    a = abs(float(z))
    return "consistent" if a < Z_TENSION else ("tension" if a <= Z_INCONSISTENT else "inconsistent")


# ── 方案与数据表 ─────────────────────────────────────────────────────


def plan_for(exp: Experiment, protocol: str) -> list[tuple[str, float]]:
    return [(str(k), float(s)) for k, s in exp.protocols[protocol].plan]


def kinds_in(plan: Sequence[tuple[str, float]]) -> list[str]:
    out: list[str] = []
    for k, _ in plan:
        if k not in out:
            out.append(k)
    return out


def has_setting(exp: Experiment, kind: str) -> bool:
    return bool(exp.obs_kinds[kind].setting_label_zh)


def column_labels(exp: Experiment, kind: str) -> dict[str, str]:
    """数据表列名 → 中文表头（带单位）。列名就是读数的 key，控制量列叫 setting。"""
    ok = exp.obs_kinds[kind]
    out = {}
    if has_setting(exp, kind):
        u = f"（{ok.setting_unit}）" if ok.setting_unit else ""
        out["setting"] = f"{ok.setting_label_zh}{u}"
    for r in ok.readings:
        out[r.key] = f"{r.label_zh}（{r.unit}）" if r.unit else r.label_zh
    return out


def template_df(exp: Experiment, kind: str, settings: Sequence[float]) -> pd.DataFrame:
    """空数据表：控制量按方案预填，读数留空。"""
    cols = ["setting"] + [r.key for r in exp.obs_kinds[kind].readings]
    rows = [{"setting": float(s), **{r.key: np.nan for r in exp.obs_kinds[kind].readings}} for s in settings]
    return pd.DataFrame(rows, columns=cols).astype(float)


def obs_to_df(exp: Experiment, kind: str, observations: Sequence[Observation]) -> pd.DataFrame:
    keys = [r.key for r in exp.obs_kinds[kind].readings]
    rows = [{"setting": o.setting, **dict(zip(keys, o.values))} for o in observations if o.kind == kind]
    return pd.DataFrame(rows, columns=["setting"] + keys).astype(float)


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def df_to_observations(exp: Experiment, kind: str, df: pd.DataFrame | None) -> tuple[list[Observation], int]:
    """数据表 → 完整的观测（控制量和所有读数都填了的行），以及填了一半的行数。"""
    if df is None or len(df) == 0:
        return [], 0
    keys = [r.key for r in exp.obs_kinds[kind].readings]
    need_setting = has_setting(exp, kind)
    out, partial = [], 0
    for _, row in df.iterrows():
        s = _num(row.get("setting")) if need_setting else (_num(row.get("setting")) or 0.0)
        vals = [_num(row.get(k)) for k in keys]
        filled = [v is not None for v in vals]
        if s is not None and all(filled):
            out.append(Observation(kind, float(s), tuple(float(v) for v in vals)))
        elif any(filled):
            partial += 1
    return out, partial


# ── 逐行实时对照：孪生按前面几行校准后预测这一行 ─────────────────────────


def predictive(exp: Experiment, prev: Sequence[Observation], kind: str, setting: float,
               cause_id: str | None = None) -> list[tuple[float, float]]:
    """在"只有随机误差"（或给定原因）下，用前面已测的行更新孪生的参数，预测下一行每个读数的 (均值, 标准差)。

    线性参数（如 1/k）按高斯后验解析传播，干扰量在网格上按后验加权（与引擎同一套积分）；
    标准差 = 参数不确定性 ⊕ 仪器读数误差（含分辨率 d²/12）。第一行没有数据可用，就是实验前的孪生预测。
    """
    from lab_detective.engine import _stack, build_grid, linear_marginal

    cid = cause_id or exp.random_only_id
    kinds = {o.kind for o in prev} | {kind}
    grid = build_grid(exp, cid, kinds)
    lin = exp.linear_params(cid)
    L, P = len(lin), grid.size
    m0 = np.array([p.mean for p in lin], float)
    s0 = np.array([p.sd for p in lin], float)
    if prev:
        v, c, U, sd = _stack(exp, list(prev), grid, cid)
        fit = linear_marginal(v, c, U, sd, m0, s0)
        lp = fit.log_lik + grid.logw
        th, cov = fit.theta_mean, fit.theta_cov
    else:
        lp = grid.logw.copy()
        th = np.tile(m0, (P, 1))
        cov = np.tile(np.diag(s0**2), (P, 1, 1)) if L else np.zeros((P, 0, 0))
    w = np.exp(lp - np.max(lp))
    w /= w.sum()
    r = len(exp.obs_kinds[kind].readings)
    c1, U1 = exp.predict(kind, np.array([float(setting)]), np.array([float(len(prev))]), grid.params, cid)
    c1 = np.broadcast_to(np.asarray(c1, float), (P, 1, r))[:, 0, :]
    if L:
        U1 = np.broadcast_to(np.asarray(U1, float), (P, 1, r, L))[:, 0]  # (P, r, L)
        mean_p = c1 + np.einsum("prl,pl->pr", U1, th)
        var_p = np.einsum("prl,plk,prk->pr", U1, cov, U1)
    else:
        mean_p, var_p = c1, np.zeros_like(c1)
    sd_rd = np.broadcast_to(exp.total_sd(kind, grid.params, cid), (P, 1, r))[:, 0, :]
    var_p = var_p + sd_rd**2
    mean = w @ mean_p
    var = w @ (var_p + mean_p**2) - mean**2
    return [(float(mean[j]), float(math.sqrt(max(var[j], 0.0)))) for j in range(r)]


HINT = {
    "consistent": "和孪生一致，继续下一组。",
    "tension": "有些偏离：再读一次，确认仪器静止、视线与刻度平齐。",
    "inconsistent": "明显不符：先查这一行有没有读错、抄错（单位、小数点）；读数没问题的话，"
                    "可能有系统误差，测完后到「④ 诊断」找原因。",
}


def live_checks(exp: Experiment, observations: Sequence[Observation], cause_id: str | None = None) -> list[dict]:
    """每一行和"用前面各行校准过的孪生"对照：预测、z、色带、提示（格式与 LabSession.record_check 一致）。"""
    out = []
    for i, o in enumerate(observations):
        try:
            preds = predictive(exp, observations[:i], o.kind, o.setting, cause_id)
        except Exception as e:  # noqa: BLE001  个别实验的某种观测不支持时只记录、不阻塞
            out.append({"index": i, "kind": o.kind, "setting": o.setting, "value": o.values[0] if o.values else None,
                        "predicted": None, "uncertainty": None, "z": None, "band": None,
                        "hint": f"这一行无法和孪生对照（{type(e).__name__}）。"})
            continue
        readings = exp.obs_kinds[o.kind].readings
        zs = [(v - m) / s if s > 0 else 0.0 for v, (m, s) in zip(o.values, preds)]
        j = int(np.argmax(np.abs(zs))) if zs else 0
        band = band_of(zs[j]) if zs else None
        out.append({
            "index": i, "kind": o.kind, "setting": o.setting,
            "value": o.values[j], "unit": readings[j].unit, "reading": readings[j].label_zh,
            "predicted": preds[j][0], "uncertainty": preds[j][1], "z": zs[j], "band": band,
            "per_reading": [{"label": rd.label_zh, "unit": rd.unit, "value": v, "predicted": m, "sd": s, "z": z}
                            for rd, v, (m, s), z in zip(readings, o.values, preds, zs)],
            "hint": HINT.get(band, ""),
        })
    return out


def fmt(v, sig: int = 4) -> str:
    if v is None or not math.isfinite(float(v)):
        return "—"
    v = float(v)
    if v == 0:
        return "0"
    mag = math.floor(math.log10(abs(v)))
    if mag >= 6 or mag <= -5:
        return f"{v:.{sig - 1}e}"
    dec = max(0, sig - 1 - mag)
    s = f"{v:.{dec}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def checks_table(checks: list[dict]) -> pd.DataFrame:
    zh = {"consistent": "一致", "tension": "有张力", "inconsistent": "不一致", None: "—"}
    rows = []
    for c in checks:
        u = c.get("unit") or ""
        rows.append({
            "序号": c["index"] + 1, "设定": fmt(c["setting"]),
            "读数": f"{fmt(c['value'])} {u}".strip() if c.get("value") is not None else "—",
            "孪生预测": (f"{fmt(c['predicted'])} ± {fmt(c['uncertainty'], 2)} {u}".strip()
                       if c.get("predicted") is not None else "—"),
            "z": fmt(c["z"], 2) if c.get("z") is not None else "—",
            "判断": zh.get(c.get("band"), "—"), "提示": c.get("hint", ""),
        })
    return pd.DataFrame(rows)


# ── 模拟数据（演示和测试用）──────────────────────────────────────────


def simulate(exp: Experiment, cause_id: str, protocol: str, seed: int) -> Dataset:
    return exp.simulate(np.random.default_rng(seed), cause_id, protocol, seed=seed)


def simulate_more(exp: Experiment, truth: dict, observations: Sequence[Observation], plan, seed: int) -> list[Observation]:
    """按记下的真值续测（追加测量），只返回新的几行。"""
    ds = Dataset(exp.id, list(observations), truth, {})
    ext = exp.extend(np.random.default_rng(seed), ds, list(plan))
    return ext.observations[len(observations):]


def simulate_fixed(exp: Experiment, truth: dict, protocol: str, seed: int) -> Dataset:
    """修正后的复测：原因换成"只有随机误差"，器材本身的参数（共用线性参数、干扰量）保持不变。"""
    shared = {p.name: truth["linear"][p.name] for p in exp.linear if p.name in truth.get("linear", {})}
    return exp.simulate(np.random.default_rng(seed), exp.random_only_id, protocol,
                        nuisance=truth.get("nuisance") or None, linear=shared or None, seed=seed)


# ── 事实单、幽灵曲线、数据点 ───────────────────────────────────────────


def dataset(exp: Experiment, observations: Sequence[Observation], protocol: str) -> Dataset:
    return Dataset(exp.id, list(observations), None, {"protocol": protocol})


def fact_sheet(exp: Experiment, ds: Dataset) -> dict:
    from lab_detective.facts import build_fact_sheet

    return build_fact_sheet(exp, ds, rng=np.random.default_rng(0), n_samples=96)


def ghosts(exp: Experiment, ds: Dataset, top_k: int = 3) -> list[dict]:
    from lab_detective.twin import ghost_curves

    return ghost_curves(exp, ds, top_k=top_k)


def data_points(exp: Experiment, ds: Dataset) -> dict:
    lin = exp.linearise(ds)
    x, y, sy = (np.asarray(a, float) for a in (lin.x, lin.y, lin.sy))
    ok = np.isfinite(x) & np.isfinite(y)
    return {"x": x[ok].tolist(), "y": y[ok].tolist(), "sy": np.broadcast_to(sy, x.shape)[ok].tolist(),
            "x_label": lin.x_label, "y_label": lin.y_label}


def number(fs: dict, name: str):
    for it in fs.get("numbers") or []:
        if it.get("name") == name:
            return it.get("value")
    return None


# ── 实验设计：测量范围、可辨识性 ─────────────────────────────────────


def range_advice(exp: Experiment, protocol: str) -> dict:
    """按方案给出控制量范围、预计读数范围、最小读数与分辨率之比，以及会弯曲的原因（不要超出范围）。"""
    from lab_detective.twin import instantiate

    plan = plan_for(exp, protocol)
    kind = plan[0][0]
    ok = exp.obs_kinds[kind]
    rd = ok.readings[0]
    rows = instantiate(exp).predict_rows(plan)
    means = [r["readings"][0]["mean"] for r in rows if r["kind"] == kind]
    settings = [s for k, s in plan if k == kind]
    absm = [abs(m) for m in means if math.isfinite(m) and m != 0]
    ratio = (min(absm) / rd.resolution) if absm and rd.resolution > 0 else None
    return {
        "setting_label": ok.setting_label_zh, "setting_unit": ok.setting_unit,
        "lo": min(settings), "hi": max(settings), "n": len(settings),
        "reading_label": rd.label_zh, "reading_unit": rd.unit, "resolution": rd.resolution,
        "pred_lo": min(means) if means else None, "pred_hi": max(means) if means else None,
        "min_ratio": ratio,
        "rel_round_pct": (100.0 / (ratio * math.sqrt(12))) if ratio else None,
        "bend_causes": [c.name_zh for c in exp.causes if c.mechanism == "bend"],
        "menu": [(m.label_zh, m.minutes) for m in exp.menu],
    }


def separability(exp: Experiment, protocol: str, n: int = 60, seed: int = 0) -> list[dict]:
    """按方案模拟：哪些原因两两分不开（confused）或只能部分分开（partial），以及哪一项追加测量能分开它们。"""
    from lab_detective.atlas import atlas_table

    rows = atlas_table(exp, [protocol], n, np.random.default_rng(seed), resolve=True)
    labels = {m.id: m.label_zh for m in exp.menu}
    out = []
    for r in rows:
        if r["status"] == "separable":
            continue
        out.append({**r, "best_label": labels.get(r.get("best_item"), r.get("best_item"))})
    out.sort(key=lambda r: (r["status"] != "confused", r["acc"]))
    return out


# ── 修正后复测 ───────────────────────────────────────────────────────


def rerun_compare(fs_before: dict, fs_after: dict, random_only_id: str) -> dict:
    """事实单里的"修正后预计截距"与复测实测截距对照，并看复测后最可能的原因。"""
    pb, pse = number(fs_before, "rerun.intercept"), number(fs_before, "rerun.intercept_se")
    pan = fs_after.get("panel") or {}
    b, bse = pan.get("intercept"), pan.get("intercept_se")
    z = None
    if None not in (pb, pse, b, bse) and math.isfinite(b) and math.isfinite(bse):
        den = math.hypot(float(pse), float(bse))
        z = (float(b) - float(pb)) / den if den > 0 else None
    ranked = (fs_after.get("diagnosis") or {}).get("ranked") or []
    top = ranked[0] if ranked else {}
    band = band_of(z)
    if top.get("cause_id") == random_only_id and float(top.get("prob") or 0) >= 0.5 and band != "inconsistent":
        verdict = "fixed"  # 复测数据最像"只有随机误差"，且与修正后的预测不矛盾
    elif top.get("cause_id") == random_only_id:
        verdict = "likely"  # 最像只有随机误差，但把握不足一半
    else:
        verdict = "not_fixed"
    return {"pred_intercept": pb, "pred_se": pse, "intercept": b, "intercept_se": bse, "z": z,
            "band": band, "top": top, "verdict": verdict, "agreement": fs_after.get("agreement") or {}}


# ── 知识点 ───────────────────────────────────────────────────────────


def kp_names() -> dict[str, str]:
    try:
        from lab_detective import knowledge

        return {k["id"]: k["name_zh"] for k in knowledge.all_kps()}
    except Exception:  # noqa: BLE001
        return {}


def related_kps(exp: Experiment, cause_id: str | None, limit: int = 4) -> list[str]:
    """与实验、最可能原因相关的知识点；其他实验没有实验卡时按原因的机制取测量科学知识点。"""
    try:
        from lab_detective import knowledge
    except Exception:  # noqa: BLE001
        return []
    ids: list[str] = []
    try:
        ids = knowledge.related_kps(exp.id, cause_id)
    except Exception:  # noqa: BLE001
        ids = []
    if not ids and cause_id:
        try:
            mech = exp.cause(cause_id).mechanism
            cmap = knowledge._cause_map()  # 原因 → 知识点映射（数据文件）
            raw = list(cmap.get("mechanism", {}).get(mech, [])) + ["kp.systematic_error", "kp.random_error"]
            ids = [c for c in (knowledge.canonical(i) for i in raw) if c]
        except Exception:  # noqa: BLE001
            ids = []
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:limit]


def kp_detail(kp_id: str) -> dict | None:
    try:
        from lab_detective import knowledge

        d = knowledge.kp(kp_id)
        d["path"] = knowledge.learning_path(kp_id)
        return d
    except Exception:  # noqa: BLE001
        return None
