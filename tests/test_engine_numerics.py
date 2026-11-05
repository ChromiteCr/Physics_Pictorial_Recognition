"""引擎证据（线性参数解析积分）与暴力数值积分的对照。"""
import numpy as np
import pytest
from scipy import stats

from lab_detective.engine import build_grid, _stack, linear_marginal, posterior
from lab_detective.labsim import get_experiment


def _brute_log_evidence(exp, ds, cause_id, n=4001):
    """对每个线性参数在 ±8σ 内做稠密网格求和（最多 2 个线性参数），非线性网格照常求和。"""
    grid = build_grid(exp, cause_id, ds.kinds())
    v, c, U, sd = _stack(exp, ds.observations, grid, cause_id)
    lin = exp.linear_params(cause_id)
    axes = []
    for p in lin:
        m = 401 if len(lin) == 2 else n
        axes.append(np.linspace(p.mean - 8 * p.sd, p.mean + 8 * p.sd, m))
    mesh = np.meshgrid(*axes, indexing="ij")
    thetas = np.stack([g.reshape(-1) for g in mesh], axis=1)  # (T, L)
    dvol = np.prod([a[1] - a[0] for a in axes])
    logprior = sum(stats.norm.logpdf(thetas[:, j], p.mean, p.sd) for j, p in enumerate(lin))
    out = []
    for i in range(grid.size):
        mu = c[i][None, :] + thetas @ U[i].T  # (T, N)
        ll = np.sum(stats.norm.logpdf(v[None, :], mu, sd[i][None, :]), axis=1)
        out.append(np.logaddexp.reduce(ll + logprior) + np.log(dvol))
    return float(np.logaddexp.reduce(np.array(out) + grid.logw))


@pytest.mark.parametrize("exp_id,cause", [
    ("hooke", "hooke.random_only"),        # 1 个线性参数
    ("hooke", "hooke.zero_offset"),        # 2 个线性参数
    ("hooke", "hooke.reading_scatter"),    # 线性 + 非线性网格
])
def test_analytic_evidence_matches_brute_force(exp_id, cause):
    exp = get_experiment(exp_id)
    ds = exp.simulate(np.random.default_rng(11), cause)
    post = posterior(exp, ds)
    analytic = post.per_cause[post.cause_ids.index(cause)].log_evidence
    brute = _brute_log_evidence(exp, ds, cause)
    assert analytic == pytest.approx(brute, abs=1e-3)
