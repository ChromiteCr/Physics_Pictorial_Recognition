"""机械能守恒（小车 + 倾斜轨道）的解析检查和引擎检查。"""
import math
import time

import numpy as np
import pytest

from lab_detective.engine import posterior
from lab_detective.labsim import get_experiment
from lab_detective.labsim.experiments.energy import Energy
from lab_detective.panel import compute_panel

EXP = get_experiment("energy")


def _panel(ds, exp=EXP):
    return compute_panel(exp.linearise(ds))


def test_registered_with_expected_shape():
    assert not EXP.held_out
    assert EXP.causes[0].id == "energy.random_only"
    assert 4 <= len(EXP.causes) - 1 <= 6
    assert all(c.id.startswith("energy.") for c in EXP.causes)
    for m in EXP.menu:
        assert all(k in EXP.obs_kinds for k, _ in m.plan)


def test_zero_fault_recovers_unit_slope():
    rng = np.random.default_rng(0)
    ds = EXP.simulate(rng, "energy.random_only", noise_scale=1e-6)
    p = _panel(ds)
    # 理想 ½v² = g h：斜率 1、截距 0；0.1 ms 取整在 Δt ≈ 4.5 ms 处带来 ~2% 的 ½v² 误差
    assert p.slope == pytest.approx(1.0, abs=0.03)
    assert abs(p.intercept) < 0.02
    # 由斜率反推 g（课本取 9.8）
    assert p.slope * 9.8 == pytest.approx(9.8, rel=0.03)


@pytest.mark.parametrize("cause,kw,slope", [
    # k = d₀/d = 10/11 → 斜率 k²
    ("energy.flag_width", dict(linear={"w_err": 1.0}), (10.0 / 11.0) ** 2),
    # 斜率 sinθ
    ("energy.slant_length", dict(nuisance={"theta_deg": 20.0}), math.sin(math.radians(20.0))),
    # 斜率 1 − μ cotθ
    ("energy.track_friction", dict(magnitude=0.02, nuisance={"theta_deg": 15.0}),
     1.0 - 0.02 / math.tan(math.radians(15.0))),
    # 斜率 1/(1 + m_w/2M)
    ("energy.wheel_inertia", dict(magnitude=0.05), 1.0 / 1.05),
])
def test_slope_matches_derivation(cause, kw, slope):
    ds = EXP.simulate(np.random.default_rng(1), cause, noise_scale=1e-6, **kw)
    assert _panel(ds).slope == pytest.approx(slope, abs=0.03)


def test_intercepts_match_derivation():
    rng = np.random.default_rng(2)
    p = _panel(EXP.simulate(rng, "energy.wrong_height_point", magnitude=3.0, noise_scale=1e-6))
    assert p.intercept == pytest.approx(-9.8 * 0.03, abs=0.02)  # 截距 −gδ
    p = _panel(EXP.simulate(rng, "energy.push", magnitude=0.5, noise_scale=1e-6))
    assert p.intercept == pytest.approx(0.5 * 0.5**2, abs=0.02)  # 截距 +½v₀²


@pytest.mark.parametrize("cause,kw,expect", [
    ("energy.flag_width", dict(linear={"w_err": 1.0}), "slope_low"),
    ("energy.flag_width", dict(linear={"w_err": -1.0}), "slope_high"),
    ("energy.slant_length", dict(nuisance={"theta_deg": 20.0}), "slope_low"),
    ("energy.track_friction", dict(magnitude=0.02, nuisance={"theta_deg": 15.0}), "slope_low"),
    ("energy.wheel_inertia", dict(magnitude=0.07), "slope_low"),
    ("energy.wrong_height_point", dict(magnitude=3.0), "intercept_neg"),
    ("energy.wrong_height_point", dict(magnitude=-3.0), "intercept_pos"),
    ("energy.push", dict(magnitude=0.5), "intercept_pos"),
])
def test_large_faults_show_signature(cause, kw, expect):
    assert expect in EXP.cause(cause).signatures
    rng = np.random.default_rng(3)
    hits = sum(expect in _panel(EXP.simulate(rng, cause, **kw)).signatures for _ in range(20))
    assert hits >= 15


@pytest.mark.parametrize("cause,kw", [
    ("energy.slant_length", dict(nuisance={"theta_deg": 20.0})),
    ("energy.wrong_height_point", dict(magnitude=3.0)),
])
def test_engine_ranks_clear_fault_top(cause, kw):
    rng = np.random.default_rng(4)
    ok = sum(posterior(EXP, EXP.simulate(rng, cause, **kw)).top() == cause for _ in range(20))
    assert ok >= 14


def test_posterior_normalised_and_random_only_wins_on_clean_data():
    rng = np.random.default_rng(5)
    wins = 0
    for _ in range(20):
        post = posterior(EXP, EXP.simulate(rng, "energy.random_only"))
        assert post.probs.sum() == pytest.approx(1.0)
        wins += post.top() == "energy.random_only"
    assert wins >= 12


def _menu_gain(rng, cause, item, n=20, **kw):
    before, after = [], []
    for _ in range(n):
        ds = EXP.simulate(rng, cause, **kw)
        before.append(posterior(EXP, ds).prob(cause))
        ds2 = EXP.extend(rng, ds, EXP.menu_item(item).plan)
        after.append(posterior(EXP, ds2).prob(cause))
    return float(np.mean(before)), float(np.mean(after))


def test_coast_test_separates_friction_from_wheel_inertia():
    """倾角固定时摩擦和车轮转动都只让斜率变小、形式完全相同；水平滑行减速度 a = μg 只对摩擦不为零。"""
    b, a = _menu_gain(np.random.default_rng(6), "energy.track_friction", "coast_test",
                      magnitude=0.02, nuisance={"theta_deg": 15.0})
    assert a > b + 0.3
    assert a > 0.8


def test_caliper_separates_flag_width():
    b, a = _menu_gain(np.random.default_rng(7), "energy.flag_width", "caliper_flag", linear={"w_err": 0.5})
    assert a > b + 0.2
    assert a > 0.8


def test_string_release_separates_push_from_wrong_point():
    """推了一下（v² 多 v₀²）和高度少记（v² 多 2g|δ|）在主测量里函数形式完全一样；剪线释放后推力消失。"""
    b, a = _menu_gain(np.random.default_rng(8), "energy.push", "string_release", magnitude=0.4)
    assert a > b + 0.1
    assert a > 0.8


def test_posterior_is_fast():
    rng = np.random.default_rng(9)
    ds = EXP.simulate(rng, "energy.track_friction")
    posterior(EXP, ds)
    t0 = time.perf_counter()
    for _ in range(5):
        posterior(EXP, ds)
    assert (time.perf_counter() - t0) / 5 < 0.5


def test_ticker_profile_and_recorded_angle():
    exp = Energy(timer="ticker", angle_deg=15.0)
    assert "energy.flag_width" not in exp.cause_ids()
    assert "energy.slant_length" not in exp.cause_ids()  # 记录了倾角就不列"斜面距离当高度"
    assert exp.linear == ()
    rng = np.random.default_rng(10)
    ds = exp.simulate(rng, "energy.random_only", noise_scale=1e-6)
    assert compute_panel(exp.linearise(ds)).slope == pytest.approx(1.0, abs=0.04)
    ok = 0
    for _ in range(10):
        ds = exp.simulate(rng, "energy.wrong_height_point", magnitude=3.0)
        post = posterior(exp, ds)
        assert post.probs.sum() == pytest.approx(1.0)
        ok += post.top() == "energy.wrong_height_point"
    assert ok >= 7
    ds2 = exp.extend(rng, ds, exp.menu_item("height_check").plan)
    assert posterior(exp, ds2).probs.sum() == pytest.approx(1.0)
    # 默认实例不受影响；只给倾角时斜面距离原因也去掉
    assert "energy.flag_width" in EXP.cause_ids() and "energy.slant_length" in EXP.cause_ids()
    assert "energy.slant_length" not in Energy(angle_deg=20.0).cause_ids()


def test_single_protocol_runs():
    ds = EXP.simulate(np.random.default_rng(12), "energy.push", protocol="single")
    assert EXP.linearise(ds).x.size == 1
    assert posterior(EXP, ds).probs.sum() == pytest.approx(1.0)


def test_dataset_roundtrip():
    from lab_detective.labsim import Dataset
    ds = EXP.simulate(np.random.default_rng(11), "energy.push")
    assert Dataset.from_dict(ds.to_dict()).observations == ds.observations
