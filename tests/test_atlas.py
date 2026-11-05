"""可辨识性图谱的检查：Wilson 区间、混淆矩阵、胡克定律里"称挂钩"能分开漏算挂钩与原长读偏。"""
import json

import numpy as np
import pytest

from lab_detective import atlas
from lab_detective.labsim import get_experiment

EXP = get_experiment("hooke")
H, Z, T = "hooke.hanger_omitted", "hooke.zero_offset", "hooke.total_length"


def test_wilson_interval():
    lo, hi = atlas.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = atlas.wilson(100, 100)
    assert hi == 1.0 and lo == pytest.approx(0.9630, abs=1e-3)


def test_confusion_matrix_counts():
    counts, ids = atlas.confusion_matrix(EXP, "standard", 40, np.random.default_rng(2))
    assert ids == EXP.cause_ids()
    assert counts.shape == (len(ids), len(ids))
    assert (counts.sum(axis=1) == 40).all()
    t = ids.index(T)
    assert counts[t, t] >= 38  # 记录总长：截距约 12 cm，标准方案下一眼就能认出


def test_weigh_hanger_resolves_hanger_vs_zero_offset():
    before = atlas.pairwise_accuracy(EXP, "standard", H, Z, 200, np.random.default_rng(0))
    after = atlas.pairwise_accuracy(EXP, "standard+weigh_hanger", H, Z, 200, np.random.default_rng(0))
    assert all(0.0 <= v <= 1.0 for v in before + after)
    assert before[1] <= before[0] <= before[2]
    assert after[0] > before[0] + 0.04
    assert after[0] >= 0.95


def test_resolving_measurement_ranks_menu():
    rows = atlas.resolving_measurement(EXP, "standard", H, Z, 150, np.random.default_rng(1))
    assert [r[0] for r in rows] != [] and {r[0] for r in rows} == {m.id for m in EXP.menu}
    assert rows[0][0] in {"weigh_hanger", "zero_check"}
    acc = {r[0]: r[1] for r in rows}
    assert acc["repeat_150"] < rows[0][2]  # 重复一次的正确率低于最佳项的置信下限


def test_atlas_table_rows():
    protocols = ["single", "standard", "standard+weigh_hanger"]
    rows = atlas.atlas_table(EXP, protocols, 120, np.random.default_rng(3))
    K = len(EXP.causes)
    assert len(rows) == len(protocols) * K * (K - 1) // 2
    json.dumps(rows, ensure_ascii=False)
    pick = {(r["protocol"], r["cause_a"], r["cause_b"]): r for r in rows}
    single, std, wh = (pick[(p, H, Z)] for p in protocols)
    assert single["status"] == "confused"  # 只读一组：斜率未知，什么也分不开
    assert std["acc"] < wh["acc"]  # 标准方案不如称挂钩之后好分
    assert wh["status"] == "separable"
    assert pick[("standard", H, T)]["status"] == "separable"
