"""事实单 FactSheet：引擎输出的唯一出口，解释器、声明校验器和界面共用（格式见 docs/ARCHITECTURE.md §2.1）。

数字纪律：事实单里出现在文字中的每个数字都经过同一个格式化函数，按显示的样子取整，
并登记到 numbers（名字、取整后的值、单位）。校验器只认 numbers 里的数字。
取整规则：有不确定度的量，不确定度保留 1 位有效数字（首位是 1、2 时保留 2 位），数值取到同一位；
z 值保留 1 位小数；概率保留 2 位小数（百分数取整）；p 值保留 2 位有效数字。
"""

from __future__ import annotations

import importlib
import inspect
import math
import re
from typing import Mapping, Sequence

import numpy as np

from .engine import posterior
from .labsim.core import Dataset, Experiment
from .panel import Panel, compute_panel
from .twin import noise_free_dataset, nominal_truth
from .uncertainty import slope_unit, unit_of, with_unit

BAND_ZH = {"consistent": "一致", "tension": "有张力", "inconsistent": "不一致"}
TOP_N = 4


def band_of(z) -> str | None:
    """|z| < 2 一致，2~3 有张力，> 3 不一致。"""
    if z is None or not math.isfinite(float(z)):
        return None
    a = abs(float(z))
    return "consistent" if a < 2 else ("tension" if a <= 3 else "inconsistent")


# ── 数字登记 ──────────────────────────────────────────────────────────


def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def dec_for_unc(u: float) -> int | None:
    """不确定度的显示位数：1 位有效数字，首位是 1、2 时保留 2 位。"""
    if not _finite(u) or u <= 0:
        return None
    e = math.floor(math.log10(u))
    lead = u / 10**e
    return (1 if lead < 3 else 0) - e


def dec_for_sig(v: float, sig: int = 3) -> int:
    if not _finite(v) or v == 0:
        return sig - 1
    return sig - 1 - math.floor(math.log10(abs(v)))


def dec_for_res(res: float) -> int:
    return max(0, math.ceil(-math.log10(res) - 1e-9)) if res and res > 0 else 2


def plain(v: float) -> str:
    """原样显示（学生记录的读数、控制量）。"""
    return f"{float(v):g}"


class Numbers:
    """数字登记表：add 返回显示字符串，同时记下取整后的值。同名只登记一次。"""

    def __init__(self):
        self.items: list[dict] = []
        self.text: dict[str, str] = {}

    def _put(self, name: str, value: float, unit: str, text: str) -> str:
        if name not in self.text:
            self.items.append({"name": name, "value": value, "unit": unit})
            self.text[name] = text
        return self.text[name]

    def add(self, name: str, value, unit: str = "", dec: int | None = None, sig: int = 3,
            signed: bool = False) -> str | None:
        if name in self.text:
            return self.text[name]
        if not _finite(value):
            return None
        value = float(value)
        dec = dec_for_sig(value, sig) if dec is None else dec
        rv = round(value, dec)
        rv = 0.0 if rv == 0 else rv
        d = max(dec, 0)
        text = f"{rv:+.{d}f}" if signed and rv != 0 else f"{rv:.{d}f}"
        return self._put(name, int(rv) if d == 0 else rv, unit, text)

    def raw(self, name: str, value, unit: str = "") -> str | None:
        if not _finite(value):
            return None
        return self._put(name, float(value), unit, plain(value))

    def p(self, name: str, p) -> str | None:
        if not _finite(p):
            return None
        p = float(p)
        if p < 1e-4:
            return self._put(name, 0.0001, "", "<0.0001")
        return self.add(name, p, "", dec=max(2, dec_for_sig(p, 2)))

    def value(self, name: str):
        for it in self.items:
            if it["name"] == name:
                return it["value"]
        return None


_NUM_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(分钟|[a-zA-Zμ°%]+)?")


def label_numbers(nums: Numbers, prefix: str, text: str) -> None:
    """把固定文字（方案名、菜单项、修正方法）里写死的数字也登记进去。"""
    for i, m in enumerate(_NUM_UNIT_RE.finditer(text or "")):
        nums.raw(f"{prefix}[{i}]", float(m.group(1)), m.group(2) or "")


def _clean(o):
    """转成严格 JSON：numpy → Python，NaN/inf → None，tuple → list。"""
    if isinstance(o, Mapping):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


# ── 面板数字与证据 ────────────────────────────────────────────────────


def _panel_numbers(nums: Numbers, p: Panel, units: dict, ideal_slope) -> dict:
    t: dict[str, str | None] = {}
    t["n"] = nums.add("n_points", p.n, "", dec=0)
    ds = dec_for_unc(p.slope_se)
    t["slope"] = nums.add("slope", p.slope, units["slope"], dec=ds)
    t["slope_se"] = nums.add("slope_se", p.slope_se, units["slope"], dec=ds)
    di = dec_for_unc(p.intercept_se)
    t["intercept"] = nums.add("intercept", p.intercept, units["y"], dec=di, signed=True)
    t["intercept_se"] = nums.add("intercept_se", p.intercept_se, units["y"], dec=di)
    t["intercept_z"] = nums.add("intercept_z", p.intercept_z, "", dec=1)
    if ideal_slope:
        dr = dec_for_unc(p.slope_se / abs(ideal_slope)) if _finite(p.slope_se) else None
        t["slope_ratio"] = nums.add("slope_ratio", p.slope_ratio, "", dec=dr)
        t["slope_ratio_z"] = nums.add("slope_ratio_z", p.slope_ratio_z, "", dec=1)
    t["curvature_z"] = nums.add("curvature_z", p.curvature_z, "", dec=1)
    t["chi2_dof"] = nums.add("chi2_dof", p.chi2_dof, "", dec=2)
    t["scatter_p_high"] = nums.p("scatter_p_high", p.scatter_p_high)
    t["scatter_p_low"] = nums.p("scatter_p_low", p.scatter_p_low)
    t["drift_rho"] = nums.add("drift_rho", p.drift_rho, "", dec=2)
    t["drift_p"] = nums.p("drift_p", p.drift_p)
    return t


def _sig_word(z) -> str:
    return "显著" if _finite(z) and abs(float(z)) > 2 else "不显著"


def _evidence(cause, p: Panel, t: dict, units: dict, ideal_slope) -> str:
    """按原因对应的面板量写证据（只引用面板数字，不写原因的"标准特征"标签）。"""
    if not p.sufficient:
        return "数据点少于 3 个，面板无法判断"
    sigs = [s for s in cause.signatures if s != "none"]
    parts: list[str] = []
    seen: set[str] = set()
    for s in sigs:
        q = s.split("_")[0]
        if q in seen:
            continue
        seen.add(q)
        if q == "intercept" and t["intercept"] is not None:
            parts.append(f"截距 {with_unit(t['intercept'], units['y'])}（z={t['intercept_z']}，{_sig_word(p.intercept_z)}）")
        elif q == "slope":
            if ideal_slope and t.get("slope_ratio") is not None:
                parts.append(f"斜率比 {t['slope_ratio']}（z={t['slope_ratio_z']}，{_sig_word(p.slope_ratio_z)}）")
            elif t["slope"] is not None:
                parts.append(f"斜率 {with_unit(t['slope'], units['slope'])}（没有理论斜率可比，标准数据单独看不出比例误差）")
        elif q == "curve":
            if t["curvature_z"] is not None:
                parts.append(f"曲率 z={t['curvature_z']}（{_sig_word(p.curvature_z)}）")
            else:
                parts.append("数据点少于 4 个，无法检验弯曲")
        elif q == "scatter" and t["chi2_dof"] is not None:
            if s == "scatter_deficit":
                pv, word = t["scatter_p_low"], "偏小"
                sig = _finite(p.scatter_p_low) and p.scatter_p_low < 0.05
            else:
                pv, word = t["scatter_p_high"], "偏大"
                sig = _finite(p.scatter_p_high) and p.scatter_p_high < 0.05
            parts.append(f"χ²/自由度 = {t['chi2_dof']}（p={pv}，散布{word}{'显著' if sig else '不显著'}）")
        elif q == "drift":
            if t["drift_rho"] is not None:
                parts.append(f"残差随试次的秩相关 ρ={t['drift_rho']}（p={t['drift_p']}）")
            else:
                parts.append("数据点少于 5 个，无法检验漂移")
    if parts:
        return "；".join(parts)
    if cause.mechanism == "none":
        flagged = [s for s in p.signatures if s != "none"]
        base = f"截距 z={t['intercept_z']}，χ²/自由度 = {t['chi2_dof']}"
        if not flagged:
            return base + "，偏离都在随机误差范围内"
        return base + "，面板出现了显著的系统偏离"
    return "标准数据看不出这个原因，要靠追加测量来区分"


# ── 一致性、下一次测量、修正 ──────────────────────────────────────────


def _agreement(exp: Experiment, ds: Dataset, p: Panel, lin, units: dict, nums: Numbers) -> dict:
    custom = getattr(exp, "agreement", None)
    if callable(custom):
        a = dict(custom(ds, p))
        band = a.get("band") or band_of(a.get("z"))
    else:
        ideal = lin.ideal_slope
        if ideal:
            a = {"quantity": "斜率比（实测/理论）", "value": p.slope_ratio,
                 "uncertainty": p.slope_se / abs(ideal) if _finite(p.slope_se) else None,
                 "unit": "", "z": p.slope_ratio_z}
        else:
            a = {"quantity": "截距", "value": p.intercept, "uncertainty": p.intercept_se,
                 "unit": units["y"], "z": p.intercept_z}
        # 设计 §5.2：少于 3 个点时不单独显示 z 色带
        band = band_of(a["z"]) if p.sufficient else None
    d = dec_for_unc(a.get("uncertainty")) if _finite(a.get("uncertainty")) else None
    unit = a.get("unit", "")
    nums.add("agreement.value", a.get("value"), unit, dec=d)
    nums.add("agreement.uncertainty", a.get("uncertainty"), unit, dec=d)
    nums.add("agreement.z", a.get("z"), "", dec=1)
    return {
        "quantity": a.get("quantity", ""),
        "value": nums.value("agreement.value"),
        "uncertainty": nums.value("agreement.uncertainty"),
        "unit": unit,
        "z": nums.value("agreement.z"),
        "band": band,
    }


def _get(o, key, default=None):
    if isinstance(o, Mapping):
        return o.get(key, default)
    return getattr(o, key, default)


def _first_number(v):
    if isinstance(v, Mapping):
        for k in ("value", "mean"):
            if k in v:
                return _first_number(v[k])
        if "readings" in v:
            return _first_number(v["readings"])
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return _first_number(v[0]) if len(v) else None
    return float(v) if _finite(v) else None


def _norm_predicted(exp: Experiment, item_id: str, predicted) -> tuple[list[tuple[str, float, str]], int | None]:
    """MenuScore.predicted 可能是字典或列表，统一成 [(cause_id, 值, 单位)]。"""
    unit, dec = "", None
    try:
        kind = exp.menu_item(item_id).plan[0][0]
        rd = exp.obs_kinds[kind].readings[0]
        unit, dec = rd.unit, dec_for_res(rd.resolution)
    except (KeyError, IndexError):
        pass
    pairs = []
    if isinstance(predicted, Mapping):
        pairs = list(predicted.items())
    elif isinstance(predicted, (list, tuple)):
        for it in predicted:
            if isinstance(it, Mapping) and "cause_id" in it:
                pairs.append((it["cause_id"], it))
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                pairs.append((it[0], it[1]))
    out = []
    for cid, v in pairs:
        u = v.get("unit", unit) if isinstance(v, Mapping) else unit
        val = _first_number(v)
        if val is not None:
            out.append((str(cid), val, u))
    return out, dec


def _next_measurement(exp: Experiment, ds: Dataset, n_samples: int, rng, nums: Numbers, cause_ids=None):
    """返回 (是否可用, 内容)。eig 模块不存在时不可用（事实单里省略这一项）。"""
    try:
        recommend = importlib.import_module("lab_detective.eig").recommend
    except (ImportError, AttributeError):
        return False, None
    kw = {}
    if cause_ids is not None and "cause_ids" in inspect.signature(recommend).parameters:
        kw["cause_ids"] = tuple(cause_ids)  # 和诊断用同一组候选原因
    scores = list(recommend(exp, ds, n_samples, rng, **kw) or [])
    if not scores:
        return True, None
    best = scores[0]
    item_id = str(_get(best, "item_id", _get(best, "id", "")))
    label = _get(best, "label_zh", "") or ""
    label_numbers(nums, "next.label", label)
    nums.add("next.eig_bits", _get(best, "eig_bits"), "bit", dec=2)
    nums.raw("next.minutes", _get(best, "minutes"), "分钟")
    preds, dec = _norm_predicted(exp, item_id, _get(best, "predicted", []))
    predicted = []
    for cid, val, u in preds:
        nums.add(f"next.predicted.{cid}", val, u, dec=dec, sig=3)
        try:
            name = exp.cause(cid).name_zh
        except KeyError:
            name = cid
        label_numbers(nums, f"cause_name.{cid}", name)
        predicted.append({"cause_id": cid, "name": name, "value": nums.value(f"next.predicted.{cid}"), "unit": u})
    sep = _get(best, "separates", [])
    return True, {
        "id": item_id, "label": label, "eig_bits": nums.value("next.eig_bits"), "predicted": predicted,
        "minutes": nums.value("next.minutes"), "separates": _clean(list(sep) if sep is not None else []),
    }


def _rerun_protocol(exp: Experiment, ds: Dataset) -> str | None:
    pid = ds.meta.get("protocol")
    if pid in exp.protocols:
        return pid
    if "standard" in exp.protocols:
        return "standard"
    return next(iter(exp.protocols), None)


def _fix(exp: Experiment, ds: Dataset, post, top: str, units: dict, nums: Numbers) -> dict:
    cause = exp.cause(top)
    label_numbers(nums, "fix.text", cause.fix_zh)
    out = {"cause_id": top, "text": cause.fix_zh, "predicted_rerun": ""}
    pid = _rerun_protocol(exp, ds)
    if top == exp.random_only_id or pid is None:
        return out
    # 修正后的装置 = "只有随机误差"，共用的线性参数（如 1/k）和干扰量取当前原因的 MAP
    cp = post.per_cause[post.cause_ids.index(top)]
    mp = cp.map_params()
    mp.pop("mag", None)
    ml = cp.map_linear(exp)
    shared = {p.name: ml[p.name] for p in exp.linear if p.name in ml}
    truth = nominal_truth(exp, exp.random_only_id, {"nuisance": mp, "linear": shared})
    lin = exp.linearise(noise_free_dataset(exp, truth, exp.protocols[pid].plan, meta={"protocol": pid}))
    p = compute_panel(lin)
    if not np.isfinite(p.slope):
        return out
    di = dec_for_unc(p.intercept_se)
    b = nums.add("rerun.intercept", p.intercept, units["y"], dec=di, signed=True)
    ub = nums.add("rerun.intercept_se", p.intercept_se, units["y"], dec=di)
    if lin.ideal_slope:
        dr = dec_for_unc(p.slope_se / abs(lin.ideal_slope))
        a = nums.add("rerun.slope_ratio", p.slope_ratio, "", dec=dr)
        ua = nums.add("rerun.slope_ratio_se", p.slope_se / abs(lin.ideal_slope), "", dec=dr)
        slope_txt = f"斜率比 {a} ± {ua}"
    else:
        ds_ = dec_for_unc(p.slope_se)
        a = nums.add("rerun.slope", p.slope, units["slope"], dec=ds_)
        ua = nums.add("rerun.slope_se", p.slope_se, units["slope"], dec=ds_)
        slope_txt = f"斜率 {with_unit(f'{a} ± {ua}', units['slope'])}"
    if lin.ideal_intercept == 0:
        ideal_txt = "图线应通过原点"
    else:
        ideal = nums.add("rerun.ideal_intercept", lin.ideal_intercept, units["y"], dec=di)
        ideal_txt = f"理论截距 {with_unit(ideal, units['y'])}"
    out["predicted_rerun"] = f"改正后按原方案重做，预计截距 {with_unit(f'{b} ± {ub}', units['y'])}（{ideal_txt}），{slope_txt}。"
    return out


def _estimates(exp: Experiment, post, top: str, kinds: set[str], nums: Numbers) -> None:
    """首要原因的 MAP 参数估计（例如漏算的挂钩质量、原长读数偏差），讲解时可能引用。"""
    cp = post.per_cause[post.cause_ids.index(top)]
    i = cp.map_index()
    for j, lp in enumerate(exp.linear_params(top)):
        sd = math.sqrt(max(float(cp.fit.theta_cov[i, j, j]), 0.0))
        d = dec_for_unc(sd)
        nums.add(f"estimate.{lp.name}", cp.fit.theta_mean[i, j], lp.unit, dec=d)
        nums.add(f"estimate.{lp.name}_sd", sd, lp.unit, dec=d)
    cause = exp.cause(top)
    mp = cp.map_params()
    if cause.magnitude.kind != "point":
        nums.add("estimate.magnitude", mp["mag"], cause.magnitude_unit, sig=2)
    for n in exp.relevant_nuisance(top, kinds):
        if exp.nuisance[n].kind != "point":
            nums.add(f"estimate.{n}", mp[n], "", sig=3)


# ── 主入口 ────────────────────────────────────────────────────────────


def build_fact_sheet(exp: Experiment, ds: Dataset, rng: np.random.Generator | None = None,
                     n_samples: int = 96, cause_ids: Sequence[str] | None = None) -> dict:
    """引擎 → 事实单（JSON 可序列化，NaN 已换成 null）。"""
    rng = np.random.default_rng(0) if rng is None else rng
    nums = Numbers()
    label_numbers(nums, "experiment_name", exp.name_zh)
    pid = ds.meta.get("protocol", "custom")
    if pid in exp.protocols:
        label_numbers(nums, "protocol_label", exp.protocols[pid].label_zh)

    table = []
    for i, o in enumerate(ds.observations):
        ok = exp.obs_kinds[o.kind]
        nums.raw(f"data[{i}].setting", o.setting, ok.setting_unit)
        for rd, v in zip(ok.readings, o.values):
            nums.raw(f"data[{i}].{rd.key}", v, rd.unit)
        table.append({"kind": o.kind, "setting": o.setting, "values": list(o.values)})

    lin = exp.linearise(ds)
    panel = compute_panel(lin)
    units = {"x": unit_of(lin.x_label), "y": unit_of(lin.y_label),
             "slope": slope_unit(lin.x_label, lin.y_label)}
    t = _panel_numbers(nums, panel, units, lin.ideal_slope)
    agreement = _agreement(exp, ds, panel, lin, units, nums)

    post = posterior(exp, ds, cause_ids)
    ranked = []
    for cid, prob in post.ranked()[:TOP_N]:
        cause = exp.cause(cid)
        label_numbers(nums, f"cause_name.{cid}", cause.name_zh)
        nums.add(f"prob.{cid}", prob, "", dec=2)
        p2 = nums.value(f"prob.{cid}")
        nums.add(f"prob_pct.{cid}", round(p2 * 100), "%", dec=0)
        ranked.append({"cause_id": cid, "name": cause.name_zh, "prob": p2,
                       "evidence": _evidence(cause, panel, t, units, lin.ideal_slope)})
    misfit_p = post.misfit_p
    nums.p("misfit_p", misfit_p)
    diagnosis = {"ranked": ranked, "not_in_list": bool(_finite(misfit_p) and misfit_p < 0.01),
                 "misfit_p": nums.value("misfit_p")}

    top = post.top()
    _estimates(exp, post, top, ds.kinds(), nums)
    fix = _fix(exp, ds, post, top, units, nums)
    has_next, nxt = _next_measurement(exp, ds, n_samples, rng, nums, cause_ids)

    try:
        kps = list(importlib.import_module("lab_detective.knowledge").load_card(exp.id)["knowledge_points"])
    except (ImportError, AttributeError, KeyError, LookupError, OSError, ValueError, TypeError):
        kps = []

    fs = {
        "experiment_id": exp.id,
        "experiment_name": exp.name_zh,
        "protocol": pid,
        "data": {"n_points": len(ds.observations), "table": table},
        "panel": panel.to_dict(),
        "agreement": agreement,
        "diagnosis": diagnosis,
    }
    if has_next:
        fs["next_measurement"] = nxt
    fs["fix"] = fix
    fs["knowledge_points"] = kps
    fs["numbers"] = nums.items
    return _clean(fs)


# ── 给界面用的 Markdown ───────────────────────────────────────────────


def _fmt_unc(v, u) -> tuple[str, str]:
    d = dec_for_unc(u) if _finite(u) else None
    if d is None:
        return (plain(v) if _finite(v) else "—"), ""
    d = max(d, 0)
    return f"{float(v):.{d}f}", f"{float(u):.{d}f}"


def to_markdown_zh(fs: dict) -> str:
    """事实单 → 中文 Markdown（界面展示用，只用事实单里已有的数字）。"""
    L: list[str] = [f"### 事实单：{fs.get('experiment_name', fs.get('experiment_id', ''))}", ""]
    data = fs.get("data", {})
    L.append(f"- 方案：{fs.get('protocol', '')}；共 {data.get('n_points', 0)} 条记录")
    a = fs.get("agreement") or {}
    if a.get("value") is not None:
        v, u = _fmt_unc(a["value"], a.get("uncertainty"))
        unit = f" {a['unit']}" if a.get("unit") else ""
        s = f"- 一致性：{a.get('quantity', '')} = {v}" + (f" ± {u}" if u else "") + unit
        if a.get("z") is not None:
            s += f"，z = {a['z']:.1f}"
        if a.get("band"):
            s += f"（{BAND_ZH.get(a['band'], a['band'])}）"
        L.append(s)
    dg = fs.get("diagnosis", {})
    L += ["", "**嫌疑人看板**", "", "| 原因 | 概率 | 证据 |", "| --- | --- | --- |"]
    for r in dg.get("ranked", []):
        L.append(f"| {r['name']} | {round(r['prob'] * 100)}% | {r['evidence']} |")
    if dg.get("not_in_list"):
        mp = dg.get("misfit_p")
        ptxt = "<0.0001" if mp is not None and mp <= 0.0001 else (f"{mp:g}" if mp is not None else "—")
        L += ["", f"> 注意：数据和清单里所有原因都对不上（失配 p={ptxt}），可能存在清单以外的原因。"]
    nx = fs.get("next_measurement")
    if nx:
        L += ["", f"**下一次测量**：{nx['label']}"
              + (f"（约 {nx['minutes']:g} 分钟，" if nx.get("minutes") is not None else "（")
              + f"信息增益 {nx['eig_bits']:.2f} bit）"]
        for pr in nx.get("predicted", []):
            L.append(f"- 如果是「{pr.get('name', pr['cause_id'])}」：预计 {pr['value']:g} {pr['unit']}".rstrip())
    elif "next_measurement" in fs:
        L += ["", "**下一次测量**：菜单里没有能进一步区分这些原因的测量。"]
    fx = fs.get("fix") or {}
    if fx.get("text"):
        L += ["", f"**修正方法**：{fx['text']}"]
        if fx.get("predicted_rerun"):
            L.append(f"- {fx['predicted_rerun']}")
    if fs.get("knowledge_points"):
        L += ["", "**相关知识点**：" + "、".join(fs["knowledge_points"])]
    return "\n".join(L)
