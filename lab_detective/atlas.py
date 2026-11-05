"""可辨识性图谱：某个测量方案下哪些原因能被分开，哪一次追加测量能把分不开的一对分开。

判别器用贝叶斯最优判别 = 引擎后验的 MAP；两两比较时只在这一对原因里比，两者先验各 1/2。
同一方案下所有模拟数据的观测布局相同，c、U、sd 与读数无关，整批数据一次算完（eig.batch_log_evidence）。

protocol 写法：方案 id（"standard"）、加追加测量（"standard+weigh_hanger"、"standard+repeat"）、
加自适应步骤（"standard+eig"、"standard+random"，逐例模拟，较慢）或显式 (种类, 控制量) 序列。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .eig import batch_log_evidence, protocol_key, simulate_protocol
from .labsim.core import Experiment

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间。"""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _simulate(exp: Experiment, protocol, cause_ids: Sequence[str], n: int, rng, n_samples: int = 64):
    """每个原因模拟 n 份数据 → (数据集列表, 真实原因下标)。"""
    datasets, truth = [], []
    for j, cid in enumerate(cause_ids):
        for _ in range(n):
            datasets.append(simulate_protocol(exp, rng, cid, protocol, n_samples=n_samples))
            truth.append(j)
    return datasets, np.array(truth)


def _map(logp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """逐行取最大；完全相同的并列随机挑一个（避免总偏向排在前面的原因）。"""
    best = logp.max(axis=1, keepdims=True)
    ties = logp >= best
    noise = rng.random(logp.shape)
    return np.argmax(np.where(ties, noise, -1.0), axis=1)


def _pair_correct(lz_a: np.ndarray, lz_b: np.ndarray, is_a: np.ndarray, rng) -> np.ndarray:
    d = lz_a - lz_b
    pick_a = np.where(d == 0, rng.random(d.shape) < 0.5, d > 0)
    return np.where(is_a, pick_a, ~pick_a)


def _summary(correct: np.ndarray) -> tuple[float, float, float]:
    k, n = int(correct.sum()), int(correct.size)
    lo, hi = wilson(k, n)
    return k / n, lo, hi


def confusion_matrix(
    exp: Experiment, protocol, n_per_cause: int, rng: np.random.Generator, cause_ids: Sequence[str] | None = None
) -> tuple[np.ndarray, tuple[str, ...]]:
    """K×K 计数：行 = 真实原因，列 = MAP 判成的原因（原因先验均匀）。"""
    cause_ids = tuple(cause_ids) if cause_ids is not None else exp.cause_ids()
    datasets, truth = _simulate(exp, protocol, cause_ids, n_per_cause, rng)
    pred = _map(batch_log_evidence(exp, datasets, cause_ids), rng)
    counts = np.zeros((len(cause_ids), len(cause_ids)), dtype=int)
    np.add.at(counts, (truth, pred), 1)
    return counts, cause_ids


def pairwise_accuracy(
    exp: Experiment, protocol, a: str, b: str, n: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """a、b 各模拟 n 份（共 2n 份），只在这一对里按等先验 MAP 判别 → (正确率, Wilson 下限, 上限)。"""
    datasets, truth = _simulate(exp, protocol, (a, b), n, rng)
    lz = batch_log_evidence(exp, datasets, (a, b))
    return _summary(_pair_correct(lz[:, 0], lz[:, 1], truth == 0, rng))


def resolving_measurement(
    exp: Experiment, protocol, a: str, b: str, n: int, rng: np.random.Generator
) -> list[tuple[str, float, float, float]]:
    """在 protocol 之后分别追加菜单里每一项，a/b 的判别正确率 → [(项目 id, 正确率, 下限, 上限)]。

    每份基础数据只模拟一次，各项追加测量用同一串随机数续测（公共随机数）。按正确率从高到低排序。
    """
    datasets, truth = _simulate(exp, protocol, (a, b), n, rng)
    seeds = rng.integers(0, 2**62, size=len(datasets))
    rows = []
    for item in exp.menu:
        ext = [exp.extend(np.random.default_rng(int(s)), ds, item.plan) for ds, s in zip(datasets, seeds)]
        lz = batch_log_evidence(exp, ext, (a, b))
        rows.append((item.id, *_summary(_pair_correct(lz[:, 0], lz[:, 1], truth == 0, rng))))
    minutes = {m.id: m.minutes for m in exp.menu}
    rows.sort(key=lambda r: (-r[1], minutes[r[0]]))
    return rows


def atlas_table(
    exp: Experiment,
    protocols: Sequence,
    n: int,
    rng: np.random.Generator,
    cause_ids: Sequence[str] | None = None,
    confused_below: float = 0.75,
    separable_at: float = 0.9,
    resolve: bool = False,
) -> list[dict]:
    """图谱表：每个方案 × 每对原因一行（可直接 json.dumps）。

    n = 每个原因模拟的份数（每对共 2n 份）。status：Wilson 上限 < confused_below 记"confused"（分不开），
    下限 ≥ separable_at 记"separable"，其余"partial"。resolve=True 时对没分开的对再找最能分开它们的追加测量。
    """
    cause_ids = tuple(cause_ids) if cause_ids is not None else exp.cause_ids()
    names = {c.id: c.name_zh for c in exp.causes}
    rows: list[dict] = []
    for protocol in protocols:
        datasets, truth = _simulate(exp, protocol, cause_ids, n, rng)
        lz = batch_log_evidence(exp, datasets, cause_ids)
        for ia in range(len(cause_ids)):
            for ib in range(ia + 1, len(cause_ids)):
                mask = (truth == ia) | (truth == ib)
                correct = _pair_correct(lz[mask, ia], lz[mask, ib], truth[mask] == ia, rng)
                acc, lo, hi = _summary(correct)
                status = "confused" if hi < confused_below else ("separable" if lo >= separable_at else "partial")
                a, b = cause_ids[ia], cause_ids[ib]
                row = {
                    "experiment_id": exp.id,
                    "protocol": protocol_key(protocol),
                    "cause_a": a,
                    "cause_b": b,
                    "name_a": names[a],
                    "name_b": names[b],
                    "n_total": int(correct.size),
                    "acc": float(acc),
                    "lo": float(lo),
                    "hi": float(hi),
                    "status": status,
                }
                if resolve and status != "separable":
                    item_id, r_acc, r_lo, r_hi = resolving_measurement(exp, protocol, a, b, n, rng)[0]
                    row.update(best_item=item_id, best_item_acc=float(r_acc),
                               best_item_lo=float(r_lo), best_item_hi=float(r_hi))
                rows.append(row)
    return rows
