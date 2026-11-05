"""期望信息增益（EIG）测量顾问与模拟闭环。

对菜单里每一项追加测量，估计"做了它之后，原因后验平均能确定多少"（比特）：
- 网格按"已有数据的种类 ∪ 这项测量的种类"建（新种类可能带来新的干扰量维度，如称挂钩引入挂钩质量）；
  已有数据下 log 后验比最大值低 20 以上的网格点剪掉；
- 从当前联合后验里分层抽 S 组（原因、网格点、线性参数 θ）；θ 取该网格点上的高斯条件后验；
- 用 Experiment.observe 模拟这项测量的读数（加噪声、按分辨率取整），不读数据集里的真值；
- 把"已有数据 + 新读数"一次压平（c、U、sd 与样本无关），v 取 (S, N)，调 engine.linear_marginal
  一次算完所有样本的新证据；
- EIG = Σ_s 权重 × KL(新后验 ‖ 旧后验)，每一项 ≥ 0，期望等于原因后验熵的期望降幅。

闭环：同一个种子下，基础数据和后续读数走"数据流"，EIG 的蒙特卡洛和随机选择走"规划流"，
所以不同策略在同一种子下看到的是同一份基础数据（配对比较用）。
"""

from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from .engine import CauseGrid, Posterior, _stack, build_grid, cause_prior, linear_marginal, posterior
from .labsim.core import Dataset, Experiment, MenuItem, Observation

LN2 = math.log(2.0)
PRUNE_NATS = 20.0  # 剪掉 log 后验比最大值低 20 以上的网格点 / 原因
MAX_ELEMS = 1_000_000  # 每块 S·P·N 的上限（float64 约 8 MB），控制内存
MIN_PROB = 1e-4  # 后验低于它的原因不抽样（对 EIG 的贡献不超过 p·ln(1/p)）
NO_SEPARATION_BITS = 0.01  # EIG 低于它时认为这项测量分不开任何嫌疑人
ADAPTIVE = ("eig", "eig_per_min", "random")

Plan = tuple[tuple[str, float], ...]


@dataclass
class MenuScore:
    item_id: str
    label_zh: str
    minutes: float
    eig_bits: float
    eig_per_min: float
    predicted: dict = field(default_factory=dict)  # cause_id -> {"mean", "sd", "unit"}（这项测量的第一个读数）
    separates: list = field(default_factory=list)  # 预测分布差别最大的那些原因

    def to_dict(self) -> dict:
        return asdict(self)


# ── 方案解析 ────────────────────────────────────────────────────────────


def as_item(exp: Experiment, item: MenuItem | str) -> MenuItem:
    return item if isinstance(item, MenuItem) else exp.menu_item(item)


def repeat_plan(exp: Experiment, base_plan: Plan) -> Plan:
    """"简单重复"：重复基础方案里主测量的中间那一次（胡克定律标准方案 → 150 g）。"""
    kind0 = base_plan[0][0]
    main = [e for e in base_plan if e[0] == kind0]
    return (tuple(main[len(main) // 2]),)


def _plan_id(exp: Experiment, plan: Plan, default: str) -> str:
    for m in exp.menu:
        if tuple(m.plan) == tuple(plan):
            return m.id
    return default


def protocol_key(protocol) -> str:
    return protocol if isinstance(protocol, str) else repr(tuple(tuple(e) for e in protocol))


def _segments(exp: Experiment, protocol) -> list:
    """方案 → [基础 plan, 步骤...]；步骤是静态 plan 或自适应记号（eig / eig_per_min / random）。

    protocol 可以是方案 id、"standard+weigh_hanger+eig" 这样的组合，或显式的 (种类, 控制量) 序列。
    "repeat" 和 "best_fixed" 在这里就解析成静态 plan。
    """
    if not isinstance(protocol, str):
        return [tuple((str(k), float(s)) for k, s in protocol)]
    parts = protocol.split("+")
    base = tuple(exp.protocols[parts[0]].plan)
    segs: list = [base]
    for i, tok in enumerate(parts[1:], start=1):
        if tok in ADAPTIVE:
            segs.append(tok)
        elif tok == "repeat":
            segs.append(repeat_plan(exp, base))
        elif tok == "best_fixed":
            prefix = "+".join(parts[:i])
            if any(t in ADAPTIVE for t in parts[1:i]):
                raise ValueError("best_fixed 之前不能有自适应步骤")
            segs.append(tuple(exp.menu_item(_best_fixed_cached(exp, prefix)).plan))
        else:
            segs.append(tuple(exp.menu_item(tok).plan))
    return segs


def protocol_plan(exp: Experiment, protocol) -> Plan:
    """静态方案的完整 plan（含 "+菜单项"）；带自适应步骤时报错。"""
    plan: list = []
    for seg in _segments(exp, protocol):
        if isinstance(seg, str):
            raise ValueError(f"方案 {protocol!r} 含自适应步骤，请用 simulate_protocol")
        plan.extend(seg)
    return tuple(plan)


def _split_rng(rng: np.random.Generator) -> tuple[np.random.Generator, np.random.Generator]:
    s = rng.integers(0, 2**62, size=2)
    return np.random.default_rng(int(s[0])), np.random.default_rng(int(s[1]))


# ── 批量证据（同一观测布局的一批数据一次算完）──────────────────────────────


@dataclass
class _Stacked:
    perm: np.ndarray  # 压平后第 j 列对应"按观测顺序排列的第 perm[j] 个读数"
    c: np.ndarray  # (P, N)
    U: np.ndarray  # (P, N, L)
    sd: np.ndarray  # (P, N)
    m0: np.ndarray  # (L,)
    s0: np.ndarray  # (L,)


def _stacked(exp: Experiment, template: Sequence[Observation], grid: CauseGrid, cause_id: str) -> _Stacked:
    """用读数编号代替读数值调 engine._stack，取回列顺序（_stack 按种类分组，新旧读数会交错）。"""
    coded, k = [], 0
    for o in template:
        r = len(o.values)
        coded.append(Observation(o.kind, o.setting, tuple(float(k + j) for j in range(r))))
        k += r
    v, c, U, sd = _stack(exp, coded, grid, cause_id)
    lin = exp.linear_params(cause_id)
    return _Stacked(
        np.rint(v).astype(int),
        np.ascontiguousarray(c),
        np.ascontiguousarray(U),
        np.ascontiguousarray(sd),
        np.array([p.mean for p in lin], dtype=float),
        np.array([p.sd for p in lin], dtype=float),
    )


def _log_evidence(st: _Stacked, V: np.ndarray, logw: np.ndarray) -> np.ndarray:
    """V: (S, M) 按观测顺序排列的读数 → (S,) 每组数据的 log 证据（分块控制内存）。"""
    Vs = V[:, st.perm]
    S = Vs.shape[0]
    P, N = st.c.shape
    step = max(1, MAX_ELEMS // max(P * N, 1))
    out = np.empty(S)
    for a in range(0, S, step):
        fit = linear_marginal(Vs[a : a + step], st.c, st.U, st.sd, st.m0, st.s0)
        out[a : a + step] = logsumexp(fit.log_lik + logw, axis=-1)
    return out


def _layout(obs: Sequence[Observation]) -> tuple:
    return tuple((o.kind, o.setting, len(o.values)) for o in obs)


def batch_log_evidence(
    exp: Experiment, datasets: Sequence[Dataset], cause_ids: Sequence[str] | None = None
) -> np.ndarray:
    """一批数据集各原因的 log 证据 (n, K)，与 engine.posterior(...).log_evidence 逐个算的结果一致。"""
    cause_ids = tuple(cause_ids) if cause_ids is not None else exp.cause_ids()
    out = np.empty((len(datasets), len(cause_ids)))
    groups: dict[tuple, list[int]] = {}
    for i, ds in enumerate(datasets):
        groups.setdefault(_layout(ds.observations), []).append(i)
    for idx in groups.values():
        template = datasets[idx[0]].observations
        V = np.array([[v for o in datasets[i].observations for v in o.values] for i in idx], dtype=float)
        kinds = {o.kind for o in template}
        for j, cid in enumerate(cause_ids):
            grid = build_grid(exp, cid, kinds)
            out[idx, j] = _log_evidence(_stacked(exp, template, grid, cid), V, grid.logw)
    return out


def batch_probs(
    exp: Experiment,
    datasets: Sequence[Dataset],
    cause_ids: Sequence[str] | None = None,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    cause_ids = tuple(cause_ids) if cause_ids is not None else exp.cause_ids()
    prior = cause_prior(exp, cause_ids) if prior is None else np.asarray(prior, dtype=float)
    lp = np.log(prior) + batch_log_evidence(exp, datasets, cause_ids)
    return np.exp(lp - logsumexp(lp, axis=1, keepdims=True))


# ── EIG ─────────────────────────────────────────────────────────────────


@dataclass
class _CauseState:
    """某原因在"已有数据"下、按扩展种类建的剪枝网格上的后验。"""

    cause_id: str
    grid: CauseGrid
    log_z: float  # 剪枝网格上的 log 证据
    log_post: np.ndarray  # (P',) 归一化
    theta_mean: np.ndarray  # (P', L)
    theta_cov: np.ndarray  # (P', L, L)


def _prune(cid, grid, log_lik, theta_mean, theta_cov) -> _CauseState:
    lp = log_lik + grid.logw
    keep = lp >= lp.max() - PRUNE_NATS
    pg = CauseGrid(cid, grid.names, {k: a[keep] for k, a in grid.params.items()}, grid.logw[keep])
    lz = float(logsumexp(lp[keep]))
    return _CauseState(cid, pg, lz, lp[keep] - lz, theta_mean[keep], theta_cov[keep])


def _cause_state(exp: Experiment, ds: Dataset, cid: str, kinds: set[str], post: Posterior | None) -> _CauseState:
    names = tuple(exp.relevant_nuisance(cid, kinds)) + ("mag",)
    if post is not None and cid in post.cause_ids:
        cp = post.per_cause[post.cause_ids.index(cid)]
        if cp.grid.names == names:  # 网格没变，直接用引擎算好的拟合
            return _prune(cid, cp.grid, cp.fit.log_lik, cp.fit.theta_mean, cp.fit.theta_cov)
    grid = build_grid(exp, cid, kinds)
    v, c, U, sd = _stack(exp, ds.observations, grid, cid)
    lin = exp.linear_params(cid)
    fit = linear_marginal(v, c, U, sd, np.array([p.mean for p in lin]), np.array([p.sd for p in lin]))
    return _prune(cid, grid, fit.log_lik, fit.theta_mean, fit.theta_cov)


def _sample_gaussian(rng: np.random.Generator, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """mean (n, L), cov (n, L, L) → (n, L)。用特征分解，协方差数值上略不对称/半正定也稳。"""
    if mean.shape[-1] == 0:
        return mean
    cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
    vals, vecs = np.linalg.eigh(cov)
    z = rng.standard_normal(mean.shape) * np.sqrt(np.clip(vals, 0.0, None))
    return mean + np.einsum("nij,nj->ni", vecs, z)


def _predictive(st: _Stacked, state: _CauseState, M0: int) -> tuple[np.ndarray, np.ndarray]:
    """新读数（按观测顺序）的预测均值和标准差：网格点上的高斯混合的矩。"""
    cols = np.argsort(st.perm)[M0:]
    c, U, sd = st.c[:, cols], st.U[:, cols, :], st.sd[:, cols]
    mu = c + np.einsum("pnl,pl->pn", U, state.theta_mean)
    var = sd**2 + np.einsum("pnl,plk,pnk->pn", U, state.theta_cov, U)
    w = np.exp(state.log_post)[:, None]
    w = w / w.sum()
    mean = np.sum(w * mu, axis=0)
    second = np.sum(w * (var + mu**2), axis=0)
    return mean, np.sqrt(np.maximum(second - mean**2, 0.0))


def _bhattacharyya(m1, s1, m2, s2) -> float:
    v1, v2 = np.maximum(s1**2, 1e-300), np.maximum(s2**2, 1e-300)
    d = 0.25 * (m1 - m2) ** 2 / (v1 + v2) + 0.5 * np.log((v1 + v2) / (2.0 * np.sqrt(v1 * v2)))
    return float(np.sum(d))


def _separates(cause_ids, pb, moments, eig_bits) -> list[str]:
    """先验上可信（p ≥ 0.02）的原因两两比较预测分布（Bhattacharyya 重叠），挑出差别最大的那几对。"""
    if eig_bits < NO_SEPARATION_BITS:
        return []
    K = len(cause_ids)
    plaus = [j for j in range(K) if pb[j] >= 0.02]
    if len(plaus) < 2:
        plaus = list(np.argsort(-pb)[:2])
    pairs = []
    for i, a in enumerate(plaus):
        for b in plaus[i + 1 :]:
            sep = 1.0 - math.exp(-_bhattacharyya(*moments[a], *moments[b]))
            pairs.append((pb[a] * pb[b] * sep, sep, a, b))
    if not pairs:
        return []
    top = max(pairs)
    if top[1] < 0.1:
        return []
    chosen = {j for score, _, a, b in pairs if score >= 0.5 * top[0] for j in (a, b)}
    return [cause_ids[j] for j in sorted(chosen, key=lambda j: -pb[j])]


def _setup(exp, cause_ids, prior, post):
    if post is not None and cause_ids is None:
        cause_ids = post.cause_ids
    cause_ids = tuple(cause_ids) if cause_ids is not None else exp.cause_ids()
    if prior is not None:
        log_prior = np.log(np.asarray(prior, dtype=float))
    elif post is not None and tuple(post.cause_ids) == cause_ids:
        with np.errstate(divide="ignore"):
            log_prior = np.log(post.probs) - post.log_evidence  # 从后验反推所用的原因先验
    else:
        log_prior = np.log(cause_prior(exp, cause_ids))
    return cause_ids, log_prior - logsumexp(log_prior)


def _evaluate_item(exp, ds, item, cause_ids, log_prior, n_samples, rng, post, cache):
    """返回 (eig_bits, predicted, separates)。"""
    kinds = ds.kinds() | {k for k, _ in item.plan}
    K = len(cause_ids)
    states = []
    for cid in cause_ids:
        key = (cid, tuple(exp.relevant_nuisance(cid, kinds)))
        if key not in cache:
            cache[key] = _cause_state(exp, ds, cid, kinds, post)
        states.append(cache[key])

    # 已有数据下的原因后验（与更新后用同一套剪枝网格，无信息的测量严格给 0）
    lpb = log_prior + np.array([s.log_z for s in states])
    lpb -= logsumexp(lpb)
    pb = np.exp(lpb)
    active = np.flatnonzero(lpb >= lpb.max() - PRUNE_NATS)

    n0 = len(ds.observations)
    template = list(ds.observations) + [
        Observation(k, float(s), tuple(0.0 for _ in exp.obs_kinds[k].readings)) for k, s in item.plan
    ]
    M0 = sum(len(o.values) for o in ds.observations)
    stacks = [_stacked(exp, template, states[j].grid, cause_ids[j]) for j in range(K)]
    moments = [_predictive(stacks[j], states[j], M0) for j in range(K)]
    unit = exp.obs_kinds[item.plan[0][0]].readings[0].unit
    predicted = {
        cause_ids[j]: {"mean": float(moments[j][0][0]), "sd": float(moments[j][1][0]), "unit": unit}
        for j in range(K)
    }

    # 分层抽样：每个原因的样本数 ∝ √p（至少 2 个），权重 = p / 样本数
    pa = np.exp(lpb[active] - logsumexp(lpb[active]))
    samp = [(j, p) for j, p in zip(active, pa) if p >= MIN_PROB]
    root = np.sqrt([p for _, p in samp])
    counts = np.maximum(2, np.round(n_samples * root / root.sum()).astype(int))
    base_vals = np.array([v for o in ds.observations for v in o.values], dtype=float)
    new_vals, weights = [], []
    for (j, p), n_j in zip(samp, counts):
        st, cid = states[j], cause_ids[j]
        g = st.grid
        w = np.exp(st.log_post)
        idx = rng.choice(g.size, size=n_j, p=w / w.sum())
        theta = _sample_gaussian(rng, st.theta_mean[idx], st.theta_cov[idx])
        lin = exp.linear_params(cid)
        for t, i in enumerate(idx):
            truth = {
                "cause": cid,
                "magnitude": float(g.params["mag"][i, 0]),
                "nuisance": {k: float(g.params[k][i, 0]) for k in exp.nuisance},
                "linear": {q.name: float(theta[t, l]) for l, q in enumerate(lin)},
            }
            obs = exp.observe(rng, truth, item.plan, start_order=n0)
            new_vals.append([v for o in obs for v in o.values])
        weights.extend([p / n_j] * n_j)
    V = np.hstack([np.broadcast_to(base_vals, (len(new_vals), M0)), np.array(new_vals, dtype=float)])

    # 所有样本一起更新：每个原因一次 linear_marginal
    lz1 = np.column_stack([_log_evidence(stacks[j], V, states[j].grid.logw) for j in active])
    lp1 = log_prior[active] + lz1
    lp1 -= logsumexp(lp1, axis=1, keepdims=True)
    lb = lpb[active] - logsumexp(lpb[active])
    kl = np.sum(np.exp(lp1) * (lp1 - lb), axis=1)
    eig = max(float(np.dot(weights, kl)) / LN2, 0.0)
    return eig, predicted, _separates(cause_ids, pb, moments, eig)


def expected_information_gain(
    exp: Experiment,
    ds: Dataset,
    item: MenuItem | str,
    n_samples: int = 128,
    rng: np.random.Generator | None = None,
    cause_ids: Sequence[str] | None = None,
    prior: np.ndarray | None = None,
    post: Posterior | None = None,
) -> float:
    """做完 item 之后原因后验熵的期望降幅（比特，≥ 0）。post 可传入已算好的引擎后验以省时间。"""
    rng = np.random.default_rng(0) if rng is None else rng
    cause_ids, log_prior = _setup(exp, cause_ids, prior, post)
    eig, _, _ = _evaluate_item(exp, ds, as_item(exp, item), cause_ids, log_prior, n_samples, rng, post, {})
    return eig


def recommend(
    exp: Experiment,
    ds: Dataset,
    n_samples: int = 128,
    rng: np.random.Generator | None = None,
    cause_ids: Sequence[str] | None = None,
    prior: np.ndarray | None = None,
) -> list[MenuScore]:
    """给菜单里每一项打分，按 EIG 从大到小排序（同分时用时短的在前）。"""
    rng = np.random.default_rng(0) if rng is None else rng
    post = posterior(exp, ds, cause_ids, prior)
    cause_ids, log_prior = _setup(exp, cause_ids, prior, post)
    cache: dict = {}
    out = []
    for item in exp.menu:
        eig, predicted, sep = _evaluate_item(exp, ds, item, cause_ids, log_prior, n_samples, rng, post, cache)
        per_min = eig / item.minutes if item.minutes > 0 else float("inf")
        out.append(MenuScore(item.id, item.label_zh, float(item.minutes), eig, per_min, predicted, sep))
    out.sort(key=lambda m: (-m.eig_bits, m.minutes))
    return out


# ── 选择策略与模拟闭环 ──────────────────────────────────────────────────


def _choose(exp, ds, strategy, plan_rng, n_samples, protocol) -> tuple[str, Plan, float | None]:
    """按策略选下一项测量 → (项目 id, plan, 预测 EIG 或 None)。"""
    if strategy in ("eig", "eig_per_min"):
        scores = recommend(exp, ds, n_samples, plan_rng)
        key = (lambda m: m.eig_bits) if strategy == "eig" else (lambda m: m.eig_per_min)
        best = max(scores, key=key)  # 同分时保留排序靠前的
        return best.item_id, tuple(exp.menu_item(best.item_id).plan), best.eig_bits
    if strategy == "repeat":
        plan = repeat_plan(exp, _segments(exp, protocol)[0])
        return _plan_id(exp, plan, "repeat"), plan, None
    if strategy == "random":
        m = exp.menu[int(plan_rng.integers(len(exp.menu)))]
        return m.id, tuple(m.plan), None
    if strategy == "best_fixed":
        m = exp.menu_item(_best_fixed_cached(exp, protocol))
        return m.id, tuple(m.plan), None
    if strategy.startswith("fixed:"):
        m = exp.menu_item(strategy.split(":", 1)[1])
        return m.id, tuple(m.plan), None
    raise ValueError(f"unknown strategy {strategy}")


def simulate_protocol(
    exp: Experiment,
    rng: np.random.Generator,
    cause_id: str,
    protocol="standard",
    magnitude: float | None = None,
    n_samples: int = 64,
) -> Dataset:
    """按（可能含自适应步骤的）方案模拟一份数据，例如 "standard+eig"、"standard+random"。"""
    data_rng, plan_rng = _split_rng(rng)
    segs = _segments(exp, protocol)
    plan, i = list(segs[0]), 1
    while i < len(segs) and not isinstance(segs[i], str):
        plan.extend(segs[i])
        i += 1
    ds = exp.simulate(data_rng, cause_id, tuple(plan), magnitude=magnitude)
    for seg in segs[i:]:
        if isinstance(seg, str):
            _, seg, _ = _choose(exp, ds, seg, plan_rng, n_samples, protocol)
        ds = exp.extend(data_rng, ds, seg)
    ds.meta["protocol"] = protocol_key(protocol)
    return ds


def closed_loop(
    exp: Experiment,
    rng: np.random.Generator,
    cause_id: str,
    protocol="standard",
    strategy: str = "eig",
    rounds: int = 1,
    magnitude: float | None = None,
    n_samples: int = 96,
) -> dict:
    """模拟闭环：基础方案 → 每轮按策略追加一项测量 → 记录真实原因的后验。

    strategy: eig | eig_per_min | repeat | random | fixed:<item_id> | best_fixed。
    返回的列表都含第 0 轮（只有基础方案）。
    """
    data_rng, plan_rng = _split_rng(rng)
    ds = exp.simulate(data_rng, cause_id, protocol_plan(exp, protocol), magnitude=magnitude)
    ds.meta["protocol"] = protocol_key(protocol)

    rec = {"p_true": [], "top": [], "top1": [], "entropy_bits": []}

    def log(post):
        rec["p_true"].append(post.prob(cause_id))
        rec["top"].append(post.top())
        rec["top1"].append(post.top() == cause_id)
        rec["entropy_bits"].append(post.entropy() / LN2)

    log(posterior(exp, ds))
    chosen, eigs, minutes = [], [], 0.0
    for _ in range(rounds):
        item_id, plan, eig = _choose(exp, ds, strategy, plan_rng, n_samples, protocol)
        ds = exp.extend(data_rng, ds, plan)
        log(posterior(exp, ds))
        chosen.append(item_id)
        eigs.append(eig)
        try:
            minutes += exp.menu_item(item_id).minutes
        except KeyError:
            pass
    return {
        "experiment_id": exp.id,
        "cause_id": cause_id,
        "protocol": protocol_key(protocol),
        "strategy": strategy,
        "rounds": rounds,
        "magnitude": float(ds.truth["magnitude"]),
        "chosen": chosen,
        "eig_bits": eigs,
        "minutes": minutes,
        **rec,
    }


def best_fixed_item(exp: Experiment, protocol, rng: np.random.Generator, n_cases: int = 60) -> str:
    """校准集上平均最能提高真实原因后验的那一项固定追加测量（最强的非自适应基线）。

    校准集：原因轮流取（各原因数量相同），基础数据按 protocol 模拟；每项测量用同一串随机数续测（公共随机数）。
    """
    plan = protocol_plan(exp, protocol)
    cids = exp.cause_ids()
    datasets, truth, seeds = [], [], []
    for i in range(n_cases):
        cid = cids[i % len(cids)]
        datasets.append(exp.simulate(rng, cid, plan))
        truth.append(i % len(cids))
        seeds.append(int(rng.integers(0, 2**62)))
    rows = np.arange(n_cases)
    best, best_key = None, None
    for item in exp.menu:
        ext = [exp.extend(np.random.default_rng(s), ds, item.plan) for ds, s in zip(datasets, seeds)]
        score = float(np.mean(batch_probs(exp, ext, cids)[rows, truth]))
        key = (score, -item.minutes)
        if best_key is None or key > best_key:
            best, best_key = item.id, key
    return best


_BEST_FIXED: dict[tuple, str] = {}


def _best_fixed_cached(exp: Experiment, protocol, n_cases: int = 60) -> str:
    """独立的固定种子校准集（crc32，不受评测随机流影响），按 (实验, 方案) 缓存。"""
    key = (exp.id, protocol_key(protocol), n_cases)
    if key not in _BEST_FIXED:
        seed = zlib.crc32(f"{exp.id}|{protocol_key(protocol)}|best_fixed".encode())
        _BEST_FIXED[key] = best_fixed_item(exp, protocol, np.random.default_rng(seed), n_cases)
    return _BEST_FIXED[key]
