"""孪生实例化：实验 + 参数 → 预期读数（带 ±sd 带）、无噪声数据集、幽灵曲线。

约定（与 labsim/core.py 一致）：
- 读数均值 = c + U @ θ；θ 是线性参数（高斯先验），其余是干扰量和原因幅度。
- 用户给定的参数视为已知（不确定度为 0）；没给的线性参数按高斯先验解析传播，
  没给的干扰量/幅度从先验抽样（蒙特卡洛）。
- "无噪声"数据直接取均值、不取整（取整会让稠密曲线变成台阶）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .labsim.core import Dataset, Experiment, MenuItem, Observation, Prior

# ── 参数 ──────────────────────────────────────────────────────────────


def prior_mean(prior: Prior) -> float:
    """先验均值（按引擎用的网格加权，和引擎的离散化一致）。"""
    pts, lw = prior.grid()
    w = np.exp(lw - np.max(lw))
    return float(np.sum(pts * w) / np.sum(w))


def _split_params(exp: Experiment, cause_id: str, params: Mapping | None) -> tuple[dict, dict, float | None]:
    """把参数字典拆成 (干扰量, 线性参数, 幅度)。

    支持两种写法：扁平 {干扰量名/线性参数名/'mag': 值}，或真值格式
    {'nuisance': {...}, 'linear': {...}, 'magnitude': 值}。未知键报错（防止拼错）。
    """
    nuis: dict[str, float] = {}
    lin: dict[str, float] = {}
    mag: float | None = None
    if not params:
        return nuis, lin, mag
    lin_names = {p.name for p in exp.linear_params(cause_id)}
    for k, v in params.items():
        if k in ("cause", "cause_id", "seed", "world", "noise_scale"):
            continue
        if k == "nuisance":
            for kk, vv in (v or {}).items():
                if kk not in exp.nuisance:
                    raise KeyError(f"未知干扰量 {kk}")
                nuis[kk] = float(vv)
        elif k == "linear":
            for kk, vv in (v or {}).items():
                if kk in lin_names:
                    lin[kk] = float(vv)  # 别的原因的线性参数（如 MAP 里带来的）直接忽略
        elif k in ("mag", "magnitude"):
            mag = None if v is None else float(v)
        elif k in exp.nuisance:
            nuis[k] = float(v)
        elif k in lin_names:
            lin[k] = float(v)
        elif any(k == p.name for c in exp.causes for p in c.linear):
            continue  # 属于另一个原因的线性参数，对本原因无影响
        else:
            raise KeyError(f"未知参数 {k}")
    return nuis, lin, mag


def nominal_truth(exp: Experiment, cause_id: str | None = None, params: Mapping | None = None) -> dict:
    """真值字典 {cause, magnitude, nuisance, linear}：给定的用给定值，其余取先验均值。"""
    cause_id = cause_id or exp.random_only_id
    nuis, lin, mag = _split_params(exp, cause_id, params)
    cause = exp.cause(cause_id)
    return {
        "cause": cause_id,
        "magnitude": prior_mean(cause.magnitude) if mag is None else mag,
        "nuisance": {k: nuis.get(k, prior_mean(p)) for k, p in exp.nuisance.items()},
        "linear": {p.name: lin.get(p.name, p.mean) for p in exp.linear_params(cause_id)},
    }


def resolve_plan(exp: Experiment, plan) -> tuple[tuple[str, float], ...]:
    """plan 可以是：方案 id、菜单项 id、MenuItem、Protocol 或 [(观测种类, 控制量), ...]。"""
    if isinstance(plan, str):
        if plan in exp.protocols:
            return tuple(exp.protocols[plan].plan)
        return tuple(exp.menu_item(plan).plan)
    if isinstance(plan, MenuItem) or hasattr(plan, "plan"):
        return tuple(plan.plan)
    return tuple((str(k), float(s)) for k, s in plan)


def noise_free_dataset(
    exp: Experiment,
    truth: dict,
    plan: Sequence[tuple[str, float]],
    orders: Sequence[float] | None = None,
    meta: dict | None = None,
) -> Dataset:
    """按真值生成"无噪声、不取整"的观测（读数 = 均值）。orders 可为小数（插值的试次序号）。"""
    orders = list(range(len(plan))) if orders is None else list(orders)
    obs = []
    for (kind, s), o in zip(plan, orders):
        mean, _ = exp._mean_and_sigma(truth, kind, np.array([float(s)]), np.array([float(o)]))
        obs.append(Observation(kind, float(s), tuple(float(v) for v in mean[0])))
    t = dict(truth, noise_scale=0.0, world="twin")
    return Dataset(exp.id, obs, t, dict(meta or {}))


# ── 孪生 ──────────────────────────────────────────────────────────────


@dataclass
class Twin:
    """一台实例化的实验装置。nuisance / linear / magnitude 里只放"已知"的值。"""

    exp: Experiment
    cause_id: str
    nuisance: dict = field(default_factory=dict)
    linear: dict = field(default_factory=dict)
    magnitude: float | None = None
    n_draws: int = 400
    seed: int = 0

    def truth(self) -> dict:
        """名义真值（没给的取先验均值），可直接交给 exp.observe / noise_free_dataset。"""
        return nominal_truth(
            self.exp, self.cause_id,
            {"nuisance": self.nuisance, "linear": self.linear, "magnitude": self.magnitude},
        )

    def _draw_params(self) -> dict[str, np.ndarray]:
        """非线性参数：已知的取定值，没给的从先验抽样；全是定值时只用 1 个样本。"""
        exp = self.exp
        cause = exp.cause(self.cause_id)
        free = [k for k, p in exp.nuisance.items() if k not in self.nuisance and p.kind != "point"]
        mag_free = self.magnitude is None and cause.magnitude.kind != "point"
        P = self.n_draws if (free or mag_free) else 1
        rng = np.random.default_rng(self.seed)
        params: dict[str, np.ndarray] = {}
        for k, p in exp.nuisance.items():
            if k in self.nuisance:
                params[k] = np.full((P, 1), float(self.nuisance[k]))
            elif p.kind == "point":
                params[k] = np.full((P, 1), p.a)
            else:
                params[k] = np.asarray(p.draw(rng, P), float).reshape(P, 1)
        if self.magnitude is not None:
            params["mag"] = np.full((P, 1), float(self.magnitude))
        elif cause.magnitude.kind == "point":
            params["mag"] = np.full((P, 1), cause.magnitude.a)
        else:
            params["mag"] = np.asarray(cause.magnitude.draw(rng, P), float).reshape(P, 1)
        return params

    def predict_rows(self, plan, start_order: int = 0) -> list[dict]:
        """每个试次的预期读数：mean、sd_reading（仪器：sqrt(σ²+d²/12)）、
        sd_param（参数未知带来的）、sd（两者合成）、lo/hi = mean ∓ sd。"""
        exp = self.exp
        plan = resolve_plan(exp, plan)
        params = self._draw_params()
        P = params["mag"].shape[0]
        lin = exp.linear_params(self.cause_id)
        m0 = np.array([self.linear.get(p.name, p.mean) for p in lin], float)
        s0 = np.array([0.0 if p.name in self.linear else p.sd for p in lin], float)
        rows = []
        for i, (kind, s) in enumerate(plan):
            ok = exp.obs_kinds[kind]
            r = len(ok.readings)
            c, U = exp.predict(kind, np.array([float(s)]), np.array([float(start_order + i)]),
                               params, self.cause_id)
            c = np.broadcast_to(np.asarray(c, float), (P, 1, r))[:, 0, :]  # (P, r)
            if lin:
                U = np.broadcast_to(np.asarray(U, float), (P, 1, r, len(lin)))[:, 0]  # (P, r, L)
                mean_p = c + U @ m0
                var_lin = (U**2) @ (s0**2)
            else:
                mean_p, var_lin = c, np.zeros_like(c)
            mean = mean_p.mean(axis=0)
            var_param = var_lin.mean(axis=0) + (mean_p.var(axis=0) if P > 1 else 0.0)
            sd_rd = np.broadcast_to(exp.total_sd(kind, params, self.cause_id), (P, 1, r))[:, 0, :]
            var_rd = (sd_rd**2).mean(axis=0)
            sd = np.sqrt(var_rd + var_param)
            readings = []
            for j, rd in enumerate(ok.readings):
                readings.append({
                    "key": rd.key, "label": rd.label_zh, "unit": rd.unit, "resolution": rd.resolution,
                    "mean": float(mean[j]), "sd": float(sd[j]),
                    "sd_reading": float(np.sqrt(var_rd[j])), "sd_param": float(np.sqrt(var_param[j])),
                    "lo": float(mean[j] - sd[j]), "hi": float(mean[j] + sd[j]),
                })
            rows.append({
                "order": start_order + i, "kind": kind, "kind_label": ok.label_zh,
                "setting": float(s), "setting_label": ok.setting_label_zh, "setting_unit": ok.setting_unit,
                "readings": readings,
            })
        return rows

    def predict_table(self, protocol="standard") -> list[dict]:
        """实验前的预期读数表（按方案逐行）。"""
        return self.predict_rows(protocol, 0)

    def noise_free(self, plan, orders=None) -> Dataset:
        return noise_free_dataset(self.exp, self.truth(), resolve_plan(self.exp, plan), orders)


def instantiate(exp: Experiment, params: Mapping | None = None, cause_id: str | None = None,
                n_draws: int = 400, seed: int = 0) -> Twin:
    """实验 + 参数 → 孪生。params 里可带 'cause'；默认原因是"只有随机误差"。"""
    params = dict(params or {})
    cause_id = cause_id or params.get("cause") or params.get("cause_id") or exp.random_only_id
    exp.cause(cause_id)  # 不存在就抛 KeyError
    nuis, lin, mag = _split_params(exp, cause_id, params)
    return Twin(exp, cause_id, nuis, lin, mag, n_draws, seed)


def expected_readings(exp: Experiment, cause_id: str, params: Mapping | None, plan,
                      start_order: int = 0, seed: int = 0) -> list[dict]:
    """"测量之前先预测"：假如原因是 cause_id（参数为 params），这次测量的读数应该是多少。"""
    tw = instantiate(exp, params, cause_id, seed=seed)
    return tw.predict_rows(plan, start_order)


# ── 幽灵曲线 ──────────────────────────────────────────────────────────


def _dense_plan(exp: Experiment, ds: Dataset, n_grid: int, include_zero: bool):
    """把数据里决定横轴的那种观测换成稠密的控制量网格，其他观测原样保留。返回 (plan, orders)。"""
    lin = exp.linearise(ds)
    if len(lin.order) == 0:
        return None
    main_kind = ds.observations[int(lin.order[0])].kind
    idx = [i for i, o in enumerate(ds.observations) if o.kind == main_kind]
    s = np.array([ds.observations[i].setting for i in idx], float)
    if np.ptp(s) == 0:
        return None
    lo, hi = float(s.min()), float(s.max())
    if include_zero and lo > 0:
        lo = 0.0
    grid = np.linspace(lo, hi, n_grid)
    # 试次序号按控制量插值（漂移类原因的均值依赖试次顺序）
    us = np.unique(s)
    uo = np.array([np.mean([i for i, si in zip(idx, s) if si == u]) for u in us])
    gorder = np.interp(grid, us, uo) if us.size > 1 else np.full(grid.size, uo[0])
    plan, orders, done = [], [], False
    for i, o in enumerate(ds.observations):
        if o.kind == main_kind:
            if not done:
                plan += [(main_kind, float(g)) for g in grid]
                orders += [float(g) for g in gorder]
                done = True
            continue
        plan.append((o.kind, o.setting))
        orders.append(float(i))
    return plan, orders


def _curve(exp: Experiment, truth: dict, plan, orders, meta) -> tuple[np.ndarray, np.ndarray, object]:
    with np.errstate(all="ignore"):  # 稠密网格延伸到 0 时，有的实验会出现 1/0，下面统一滤掉
        lin = exp.linearise(noise_free_dataset(exp, truth, plan, orders, meta))
    x, y = np.asarray(lin.x, float), np.asarray(lin.y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    k = np.argsort(x[ok])
    return x[ok][k], y[ok][k], lin


def ghost_curves(exp: Experiment, ds: Dataset, post=None, top_k: int = 3, n_grid: int = 61,
                 include_zero: bool = True) -> list[dict]:
    """每个高概率原因在其 MAP 参数下"应该"画出的曲线（面板坐标），用于和学生数据叠画。

    横轴范围 = 学生数据的范围（include_zero 时向 0 延伸，让截距类原因的偏离一眼可见）。
    """
    if post is None:
        from .engine import posterior

        post = posterior(exp, ds)
    xd = np.asarray(exp.linearise(ds).x, float)
    xd = xd[np.isfinite(xd)]
    lo, hi = (float(xd.min()), float(xd.max())) if xd.size else (-np.inf, np.inf)
    if include_zero and xd.size:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    tol = 1e-9 * max(hi - lo, 1e-300) if xd.size else 0.0
    order = np.argsort(-post.probs)[:top_k]
    dense = _dense_plan(exp, ds, n_grid, include_zero)
    own_plan = [(o.kind, o.setting) for o in ds.observations]
    out = []
    for j in order:
        cp = post.per_cause[int(j)]
        cid = post.cause_ids[int(j)]
        mp = cp.map_params()
        mag = mp.pop("mag")
        truth = {"cause": cid, "magnitude": mag, "nuisance": mp, "linear": cp.map_linear(exp)}
        x = y = None
        lin = None
        if dense is not None:
            try:
                x, y, lin = _curve(exp, truth, dense[0], dense[1], ds.meta)
                keep = (x >= lo - tol) & (x <= hi + tol)
                x, y = x[keep], y[keep]
            except Exception:  # 稠密网格超出某些实验 linearise 的适用范围时退回数据点
                x = None
        if x is None or x.size < 2:
            x, y, lin = _curve(exp, truth, own_plan, range(len(own_plan)), ds.meta)
        out.append({
            "cause_id": cid,
            "name": exp.cause(cid).name_zh,
            "prob": float(post.probs[int(j)]),
            "params": {"magnitude": float(mag), "nuisance": {k: float(v) for k, v in mp.items()},
                       "linear": {k: float(v) for k, v in truth["linear"].items()}},
            "x": [float(v) for v in x],
            "y": [float(v) for v in y],
            "x_label": lin.x_label,
            "y_label": lin.y_label,
        })
    return out
