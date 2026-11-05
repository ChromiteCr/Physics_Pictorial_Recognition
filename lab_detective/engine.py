"""贝叶斯诊断引擎：给定一组实验数据，算每个候选原因的后验概率。

做法（见 labsim/core.py 的建模约定）：
- 非线性参数（干扰量、故障幅度）放在网格上，按先验权重求和；
- 线性参数（高斯先验）解析积分，用线性高斯模型的边际似然闭式；
- 原因 c 的证据 Z_c = Σ_网格 先验权重 × ∫ 似然 dθ；后验 ∝ 原因先验 × Z_c。
同时给出"原因不在我的清单里"用的失配统计量（最佳拟合的 χ² 和 p 值）。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

from .labsim.core import Dataset, Experiment, Observation

LOG_2PI = math.log(2 * math.pi)


@dataclass
class CauseGrid:
    """一个原因的非线性参数网格。params: 名字 -> (P, 1)；logw: (P,) 归一化 log 先验权重。"""

    cause_id: str
    names: tuple[str, ...]
    params: dict[str, np.ndarray]
    logw: np.ndarray

    @property
    def size(self) -> int:
        return self.logw.size


def build_grid(exp: Experiment, cause_id: str, kinds: set[str]) -> CauseGrid:
    cause = exp.cause(cause_id)
    names = list(exp.relevant_nuisance(cause_id, kinds))
    priors = [exp.nuisance[n] for n in names]
    names.append("mag")
    priors.append(cause.magnitude)
    grids = [p.grid() for p in priors]
    pts = [g[0] for g in grids]
    lws = [g[1] for g in grids]
    mesh = np.meshgrid(*pts, indexing="ij")
    wmesh = np.meshgrid(*lws, indexing="ij")
    params = {n: m.reshape(-1, 1) for n, m in zip(names, mesh)}
    logw = np.sum([w.reshape(-1) for w in wmesh], axis=0)
    # 先验没有用到的干扰量取中值（只为让 predict 拿得到一个数，不影响似然）
    for n, p in exp.nuisance.items():
        if n not in params:
            g, lw = p.grid()
            mid = float(np.sum(g * np.exp(lw)))  # 先验均值处的网格点
            params[n] = np.full((logw.size, 1), g[int(np.argmin(np.abs(g - mid)))])
    return CauseGrid(cause_id, tuple(names), params, logw)


def _stack(exp: Experiment, observations: Sequence[Observation], grid: CauseGrid, cause_id: str):
    """把所有观测压平成 (v, c, U, sd)：v (N,), c (P,N), U (P,N,L), sd (P,N)。"""
    L = len(exp.linear_params(cause_id))
    P = grid.size
    vs, cs, Us, sds = [], [], [], []
    by_kind: dict[str, list[int]] = {}
    for i, o in enumerate(observations):
        by_kind.setdefault(o.kind, []).append(i)
    for kind, idx in by_kind.items():
        settings = np.array([observations[i].setting for i in idx], dtype=float)
        order = np.array(idx, dtype=float)
        vals = np.array([observations[i].values for i in idx], dtype=float)  # (n, r)
        n, r = vals.shape
        c, U = exp.predict(kind, settings, order, grid.params, cause_id)
        c = np.broadcast_to(np.asarray(c, dtype=float), (P, n, r)).reshape(P, n * r)
        if L:
            U = np.broadcast_to(np.asarray(U, dtype=float), (P, n, r, L)).reshape(P, n * r, L)
        else:
            U = np.zeros((P, n * r, 0))
        sd = np.broadcast_to(exp.total_sd_at(kind, settings, grid.params, cause_id), (P, n, r)).reshape(P, n * r)
        vs.append(vals.reshape(-1))
        cs.append(c)
        Us.append(U)
        sds.append(sd)
    return np.concatenate(vs), np.concatenate(cs, 1), np.concatenate(Us, 1), np.concatenate(sds, 1)


@dataclass
class LinearFit:
    log_lik: np.ndarray  # (P,) 对线性参数积分后的 log 边际似然
    theta_mean: np.ndarray  # (P, L) 线性参数后验均值
    theta_cov: np.ndarray  # (P, L, L)
    chi2: np.ndarray  # (P,) 在后验均值处的残差 χ²


def linear_marginal(
    v: np.ndarray,
    c: np.ndarray,
    U: np.ndarray,
    sd: np.ndarray,
    prior_mean: np.ndarray,
    prior_sd: np.ndarray,
    noise_scale: float = 1.0,
) -> LinearFit:
    """线性高斯模型 v = c + U θ + ε，ε ~ N(0, sd²)，θ ~ N(m0, diag(s0²))，对 θ 积分。

    v: (N,) 或 (S, N)（S 个模拟结果一起算）；c, sd: (P, N)；U: (P, N, L)。
    返回的数组在 v 为 (S, N) 时多一维放在最前面。
    """
    sd = sd * noise_scale
    w = 1.0 / sd**2  # (P, N)
    r0 = v[..., None, :] - c  # (..., P, N)
    lognorm = -np.sum(np.log(sd), axis=-1) - 0.5 * sd.shape[-1] * LOG_2PI  # (P,)
    L = U.shape[-1]
    if L == 0:
        chi2 = np.sum(r0**2 * w, axis=-1)
        return LinearFit(
            -0.5 * chi2 + lognorm,
            np.zeros(r0.shape[:-1] + (0,)),
            np.zeros(c.shape[:1] + (0, 0)),
            chi2,
        )
    lam0 = 1.0 / prior_sd**2  # (L,)
    A = np.einsum("pnl,pn,pnk->plk", U, w, U) + np.diag(lam0)  # (P, L, L)
    Ainv = np.linalg.inv(A)
    b = np.einsum("pnl,pn,...pn->...pl", U, w, r0) + lam0 * prior_mean  # (..., P, L)
    theta = np.einsum("plk,...pk->...pl", Ainv, b)
    quad = np.sum(r0**2 * w, axis=-1) + np.sum(lam0 * prior_mean**2) - np.sum(b * theta, axis=-1)
    _, logdetA = np.linalg.slogdet(A)
    log_lik = -0.5 * quad - 0.5 * logdetA + 0.5 * np.sum(np.log(lam0)) + lognorm
    resid = r0 - np.einsum("pnl,...pl->...pn", U, theta)
    chi2 = np.sum(resid**2 * w, axis=-1)
    return LinearFit(log_lik, theta, Ainv, chi2)


@dataclass
class CausePosterior:
    cause_id: str
    log_evidence: float
    grid: CauseGrid
    log_post: np.ndarray  # (P,) 网格点上的归一化 log 后验
    fit: LinearFit

    def map_index(self) -> int:
        return int(np.argmax(self.log_post))

    def map_params(self) -> dict[str, float]:
        i = self.map_index()
        out = {k: float(v[i, 0]) for k, v in self.grid.params.items()}
        return out

    def map_linear(self, exp: Experiment) -> dict[str, float]:
        i = self.map_index()
        lin = exp.linear_params(self.cause_id)
        return {p.name: float(self.fit.theta_mean[i, j]) for j, p in enumerate(lin)}


@dataclass
class Posterior:
    experiment_id: str
    cause_ids: tuple[str, ...]
    probs: np.ndarray
    log_evidence: np.ndarray
    per_cause: list[CausePosterior]
    n_values: int
    best_chi2: float = float("nan")
    dof: int = 0
    misfit_p: float = float("nan")

    def ranked(self) -> list[tuple[str, float]]:
        order = np.argsort(-self.probs)
        return [(self.cause_ids[i], float(self.probs[i])) for i in order]

    def top(self) -> str:
        return self.cause_ids[int(np.argmax(self.probs))]

    def prob(self, cause_id: str) -> float:
        return float(self.probs[self.cause_ids.index(cause_id)])

    def entropy(self) -> float:
        p = self.probs[self.probs > 0]
        return float(-np.sum(p * np.log(p)))


def cause_prior(exp: Experiment, cause_ids: Sequence[str], p_random: float | None = None) -> np.ndarray:
    """默认原因先验：均匀；给 p_random 时"只有随机误差"占 p_random，其余平分。"""
    k = len(cause_ids)
    if p_random is None:
        return np.full(k, 1.0 / k)
    pr = np.full(k, (1.0 - p_random) / max(k - 1, 1))
    if exp.random_only_id in cause_ids:
        pr[list(cause_ids).index(exp.random_only_id)] = p_random
    return pr / pr.sum()


def posterior(
    exp: Experiment,
    ds: Dataset,
    cause_ids: Sequence[str] | None = None,
    prior: np.ndarray | None = None,
    noise_scale: float = 1.0,
) -> Posterior:
    cause_ids = tuple(cause_ids) if cause_ids is not None else exp.cause_ids()
    prior = cause_prior(exp, cause_ids) if prior is None else np.asarray(prior, dtype=float)
    kinds = ds.kinds()
    per: list[CausePosterior] = []
    logZ = np.empty(len(cause_ids))
    for j, cid in enumerate(cause_ids):
        grid = build_grid(exp, cid, kinds)
        v, c, U, sd = _stack(exp, ds.observations, grid, cid)
        lin = exp.linear_params(cid)
        fit = linear_marginal(
            v,
            c,
            U,
            sd,
            np.array([p.mean for p in lin]),
            np.array([p.sd for p in lin]),
            noise_scale,
        )
        lp = fit.log_lik + grid.logw
        z = np.logaddexp.reduce(lp)
        logZ[j] = z
        per.append(CausePosterior(cid, float(z), grid, lp - z, fit))
    lpost = np.log(prior) + logZ
    lpost -= np.logaddexp.reduce(lpost)
    probs = np.exp(lpost)
    n_values = int(sum(len(o.values) for o in ds.observations))
    post = Posterior(exp.id, cause_ids, probs, logZ, per, n_values)
    _misfit(exp, post)
    return post


def _misfit(exp: Experiment, post: Posterior) -> None:
    """最佳原因、最佳网格点处的残差 χ² 与 p 值（"原因不在我的清单里"的判据）。"""
    best = post.per_cause[int(np.argmax(post.probs))]
    i = int(np.argmin(best.fit.chi2))
    chi2 = float(best.fit.chi2[i])
    n_par = len(exp.linear_params(best.cause_id)) + sum(
        1 for n in best.grid.names if exp_grid_dim(exp, best.cause_id, n) > 1
    )
    dof = max(post.n_values - n_par, 1)
    post.best_chi2 = chi2
    post.dof = dof
    post.misfit_p = float(stats.chi2.sf(chi2, dof))


def exp_grid_dim(exp: Experiment, cause_id: str, name: str) -> int:
    prior = exp.cause(cause_id).magnitude if name == "mag" else exp.nuisance[name]
    return prior.n if prior.kind != "point" else 1
