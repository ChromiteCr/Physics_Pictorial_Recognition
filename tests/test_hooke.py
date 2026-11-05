"""胡克定律样板实验的解析检查和引擎检查。"""
import numpy as np
import pytest

from lab_detective.engine import posterior
from lab_detective.labsim import get_experiment
from lab_detective.panel import compute_panel

EXP = get_experiment("hooke")


def test_zero_fault_recovers_k():
    rng = np.random.default_rng(0)
    ds = EXP.simulate(rng, "hooke.random_only", linear={"inv_k": 0.04}, noise_scale=1e-6)
    lin = EXP.linearise(ds)
    panel = compute_panel(lin)
    # 斜率 = 100/k（cm/N），k = 25 N/m → 4.0；取整到 0.1 cm 带来的误差很小
    assert panel.slope == pytest.approx(4.0, rel=0.02)
    assert abs(panel.intercept) < 0.1


def test_large_hanger_mass_gives_predicted_intercept():
    rng = np.random.default_rng(1)
    ds = EXP.simulate(rng, "hooke.hanger_omitted", linear={"inv_k": 0.04},
                      nuisance={"hanger_g": 40.0}, noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    # 截距 = (100/k) × m_h g = 4.0 cm/N × 0.392 N = 1.568 cm
    assert panel.intercept == pytest.approx(4.0 * 0.04 * 9.8, abs=0.08)
    assert "intercept_pos" in panel.signatures


@pytest.mark.parametrize("cause,expect", [
    ("hooke.total_length", "intercept_pos"),
    ("hooke.beyond_limit", "curve_up"),
    ("hooke.reading_scatter", "scatter_excess"),
])
def test_large_faults_show_signature(cause, expect):
    rng = np.random.default_rng(2)
    mag = {"hooke.beyond_limit": 0.5, "hooke.reading_scatter": 0.6}.get(cause)
    hits = 0
    for _ in range(20):
        ds = EXP.simulate(rng, cause, magnitude=mag)
        hits += expect in compute_panel(EXP.linearise(ds)).signatures
    assert hits >= 15


def test_posterior_normalised_and_random_only_wins_on_clean_data():
    rng = np.random.default_rng(3)
    wins = 0
    for _ in range(30):
        post = posterior(EXP, EXP.simulate(rng, "hooke.random_only"))
        assert post.probs.sum() == pytest.approx(1.0)
        wins += post.top() == "hooke.random_only"
    assert wins >= 20  # 奥卡姆：没有故障时简单假设应占优


def test_weighing_the_hanger_separates_confounded_pair():
    """斜率自由时漏算挂钩（x 偏移）和原长读偏（y 偏移）只靠标准数据分不开；称挂钩之后能分开。"""
    rng = np.random.default_rng(4)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, "hooke.hanger_omitted", nuisance={"hanger_g": 8.0}, linear={"inv_k": 0.03})
        p0 = posterior(EXP, ds).prob("hooke.hanger_omitted")
        ds2 = EXP.extend(rng, ds, EXP.menu_item("weigh_hanger").plan)
        p1 = posterior(EXP, ds2).prob("hooke.hanger_omitted")
        before.append(p0)
        after.append(p1)
    assert np.mean(after) > np.mean(before) + 0.1
    assert np.mean(after) > 0.8


def test_total_length_is_identified():
    rng = np.random.default_rng(5)
    ok = sum(posterior(EXP, EXP.simulate(rng, "hooke.total_length")).top() == "hooke.total_length"
             for _ in range(10))
    assert ok >= 9


def test_dataset_roundtrip():
    from lab_detective.labsim import Dataset
    ds = EXP.simulate(np.random.default_rng(6), "hooke.zero_offset")
    assert Dataset.from_dict(ds.to_dict()).observations == ds.observations
