"""评测用统计：精确 McNemar、Wilson 区间、精确符号检验、配对 bootstrap、TOST、Holm、Brier、ECE。

全部纯 numpy/scipy，结果可直接写进 JSON。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """配对正确/错误的精确 McNemar（双侧）。b01 = A 对 B 错，b10 = A 错 B 对。"""
    a = np.asarray(a_correct, bool)
    b = np.asarray(b_correct, bool)
    assert a.shape == b.shape
    b01 = int(np.sum(a & ~b))
    b10 = int(np.sum(~a & b))
    n = b01 + b10
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b01, b10), n, 0.5)))
    return {"a_only": b01, "b_only": b10, "discordant": n, "p_value": p,
            "diff": float(a.mean() - b.mean()) if a.size else float("nan")}


def sign_test(diffs: Sequence[float]) -> dict:
    """跨实验的精确符号检验（双侧），零差值丢弃。"""
    d = np.asarray(diffs, float)
    d = d[d != 0]
    n = d.size
    k = int(np.sum(d > 0))
    p = 1.0 if n == 0 else float(min(1.0, 2 * min(stats.binom.cdf(k, n, 0.5), stats.binom.sf(k - 1, n, 0.5))))
    return {"n": n, "positive": k, "p_value": p}


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                     seed: int = 0, clusters: Sequence | None = None) -> dict:
    """均值差 a − b 的配对 bootstrap 区间；给 clusters 时按簇重抽样。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = a - b
    if clusters is None:
        idx = rng.integers(0, d.size, size=(n_boot, d.size))
        boots = d[idx].mean(axis=1)
    else:
        cl = np.asarray(clusters)
        uniq = np.unique(cl)
        groups = [d[cl == u] for u in uniq]
        boots = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.integers(0, len(groups), len(groups))
            boots[i] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)]}


def tost_paired(a: Sequence[float], b: Sequence[float], margin: float, alpha: float = 0.05) -> dict:
    """配对等价检验：90% 区间完全落在 ±margin 内即等价。"""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    se = d.std(ddof=1) / math.sqrt(n) if n > 1 else float("inf")
    tcrit = stats.t.ppf(1 - alpha, max(n - 1, 1))
    lo, hi = d.mean() - tcrit * se, d.mean() + tcrit * se
    return {"diff": float(d.mean()), "ci90": [float(lo), float(hi)], "margin": margin,
            "equivalent": bool(lo > -margin and hi < margin)}


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    reject = np.zeros(p.size, bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (p.size - rank):
            reject[i] = True
        else:
            break
    return reject.tolist()


def brier(prob_true: Sequence[float]) -> float:
    """多类 Brier 的简化：只用真类概率，(1 − p_true)² 的均值（完整版需要全概率向量，见 brier_full）。"""
    p = np.asarray(prob_true, float)
    return float(np.mean((1 - p) ** 2))


def brier_full(probs: np.ndarray, true_idx: Sequence[int]) -> float:
    probs = np.asarray(probs, float)
    y = np.zeros_like(probs)
    y[np.arange(len(true_idx)), np.asarray(true_idx)] = 1
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def log_score(prob_true: Sequence[float], eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(prob_true, float), eps, 1)
    return float(np.mean(-np.log(p)))


def ece(confidence: Sequence[float], correct: Sequence[bool], n_bins: int = 15) -> dict:
    c = np.asarray(confidence, float)
    y = np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
        if m.sum() == 0:
            continue
        gap = abs(c[m].mean() - y[m].mean())
        total += m.mean() * gap
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "conf": float(c[m].mean()), "acc": float(y[m].mean())})
    return {"ece": float(total), "bins": rows}


def chance_product_of_marginals(true_labels: Sequence, pred_labels: Sequence) -> float:
    """独立假设下的机会准确率 = Σ P(label) × P(pred)（换表对照 K3 用）。"""
    t = list(true_labels)
    p = list(pred_labels)
    labels = set(t) | set(p)
    n = len(t)
    return float(sum((t.count(l) / n) * (p.count(l) / n) for l in labels))
