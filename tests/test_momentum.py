"""动量守恒实验（两车碰撞 + 光电门）的解析检查和引擎检查。"""
import time

import numpy as np
import pytest

from lab_detective.engine import posterior
from lab_detective.labsim import G as G_ACC, get_experiment
from lab_detective.labsim.experiments.momentum import Momentum
from lab_detective.panel import compute_panel

EXP = get_experiment("momentum")
STD = np.array([s for _, s in EXP.protocols["standard"].plan])


def _top_or_tie(post, cause_id, tol=1e-9):
    """cause_id 的后验不低于最大值（与标准方案测不出的原因打平也算赢）。"""
    return post.prob(cause_id) >= post.probs.max() - tol


def test_registry_and_cause_ids():
    assert not EXP.held_out
    ids = EXP.cause_ids()
    assert ids[0] == "momentum.random_only"
    assert all(i.startswith("momentum.") for i in ids)
    assert 4 <= len(ids) - 1 <= 6
    for c in EXP.causes:
        assert c.source == "assumption"


def test_zero_fault_recovers_conservation():
    ds = EXP.simulate(np.random.default_rng(0), "momentum.random_only", noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    # 只剩 0.1 ms 取整误差：p'/p = 1
    assert panel.slope == pytest.approx(1.0, abs=0.01)
    assert abs(panel.intercept) < 0.002


def test_ideal_sticky_times_follow_mass_ratio():
    c, _ = EXP.predict("main", STD, np.arange(5), {"mag": np.zeros((1, 1))}, "momentum.random_only")
    # m_A v₁ = (m_A + m_B) v₂ ⇒ Δt₂ = Δt₁ (m_A + m_B)/m_A = Δt₁ × 500/300
    assert np.allclose(c[0, :, 0], STD * 500.0 / 300.0)


def test_mass_entry_gives_predicted_slope():
    ds = EXP.simulate(np.random.default_rng(1), "momentum.mass_entry", magnitude=40.0, noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    # p'/p = (m_A + m_B)/(m_A + m_B + δ) = 500/540
    assert panel.slope_ratio == pytest.approx(500.0 / 540.0, abs=0.01)


def test_timing_offset_ratio_formula():
    tau = 2.0
    ds = EXP.simulate(np.random.default_rng(2), "momentum.timing_offset", linear={"tau": tau},
                      noise_scale=1e-6)
    lin = EXP.linearise(ds)
    k = 500.0 / 300.0
    expect = 1.0 / (1.0 - (k - 1.0) * tau / (k * STD))
    assert np.allclose(lin.y / lin.x, expect, atol=0.004)


def test_wheel_inertia_effective_mass():
    """车轮转动惯量：守恒的是 Σ(m + ΣI/R²)v，按天平质量算的 p'/p = (M/m_A)·(m_A+J)/(M+2J)。"""
    ex = Momentum(wheel_inertia_g=10.0)
    c, _ = ex.predict("main", STD, np.arange(5), {"mag": np.zeros((1, 1))}, "momentum.random_only")
    ratio = (500.0 / 300.0) * STD / c[0, :, 0]
    assert np.allclose(ratio, (500.0 / 300.0) * 310.0 / 520.0)


def test_flag_width_is_undetectable_in_standard_protocol():
    """宽度误差把 p 和 p' 按同一比例缩放，且 Δt₂/Δt₁ 与 w 无关：碰撞数据与"只有随机误差"完全一样。"""
    assert EXP.cause("momentum.flag_width").signatures == ("none",)
    mags = np.array([[-0.15], [0.05], [0.15]])
    c0, _ = EXP.predict("main", STD, np.arange(5), {"mag": np.zeros((1, 1))}, "momentum.random_only")
    c1, _ = EXP.predict("main", STD, np.arange(5), {"mag": mags}, "momentum.flag_width")
    assert np.allclose(c1, c0)


@pytest.mark.parametrize("cause,mag,lin,expect", [
    ("momentum.track_tilt", 3.0, None, "slope_low"),     # 下坡：慢的几次 p' 偏大最多
    ("momentum.track_tilt", -3.0, None, "slope_high"),   # 上坡：与摩擦同号
    ("momentum.friction", 0.003, None, "slope_high"),
    ("momentum.mass_entry", 40.0, None, "slope_low"),
    ("momentum.mass_entry", -40.0, None, "slope_high"),
    ("momentum.timing_offset", None, {"tau": 2.5}, "slope_high"),
    ("momentum.timing_offset", None, {"tau": -2.5}, "slope_low"),
])
def test_large_faults_show_signature(cause, mag, lin, expect):
    assert expect in EXP.cause(cause).signatures
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(20):
        ds = EXP.simulate(rng, cause, magnitude=mag, linear=lin)
        hits += expect in compute_panel(EXP.linearise(ds)).signatures
    assert hits >= 15


@pytest.mark.parametrize("cause,mag,lin", [
    ("momentum.track_tilt", 3.0, None),
    ("momentum.mass_entry", 40.0, None),
    ("momentum.timing_offset", None, {"tau": 2.5}),
])
def test_engine_ranks_large_fault_top(cause, mag, lin):
    rng = np.random.default_rng(5)
    ok = sum(posterior(EXP, EXP.simulate(rng, cause, magnitude=mag, linear=lin)).top() == cause
             for _ in range(20))
    assert ok >= 14


def test_posterior_normalised_and_random_only_wins_on_clean_data():
    """遮光片宽度误差与"只有随机误差"的似然完全相同（物理上测不出），打平时算 random_only 赢。"""
    rng = np.random.default_rng(6)
    wins = 0
    for _ in range(20):
        post = posterior(EXP, EXP.simulate(rng, "momentum.random_only"))
        assert post.probs.sum() == pytest.approx(1.0)
        wins += _top_or_tie(post, "momentum.random_only")
    assert wins >= 12


def test_reverse_coast_separates_friction_from_uphill_tilt():
    """正向碰撞里"上坡"和"摩擦"都是恒定减速，分不开；反向单独推一次：坡变成下坡，摩擦仍然减速。"""
    rng = np.random.default_rng(7)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, "momentum.friction", magnitude=0.003)
        before.append(posterior(EXP, ds).prob("momentum.friction"))
        ds2 = EXP.extend(rng, ds, EXP.menu_item("coast_reverse").plan)
        after.append(posterior(EXP, ds2).prob("momentum.friction"))
    assert np.mean(after) > np.mean(before) + 0.3
    assert np.mean(after) > 0.8


def test_caliper_reveals_flag_width():
    rng = np.random.default_rng(8)
    before, after = [], []
    for _ in range(10):
        ds = EXP.simulate(rng, "momentum.flag_width", magnitude=0.08)
        before.append(posterior(EXP, ds).prob("momentum.flag_width"))
        ds2 = EXP.extend(rng, ds, EXP.menu_item("caliper").plan)
        after.append(posterior(EXP, ds2).prob("momentum.flag_width"))
    assert np.mean(before) < 0.6
    assert np.mean(after) > 0.9


def test_posterior_is_fast():
    rng = np.random.default_rng(9)
    ds = EXP.simulate(rng, "momentum.track_tilt", magnitude=2.0)
    t0 = time.perf_counter()
    posterior(EXP, ds)
    assert time.perf_counter() - t0 < 0.5


def test_elastic_mode():
    ex = Momentum(collision="elastic", m_a_g=400.0, m_b_g=200.0)
    ds = ex.simulate(np.random.default_rng(10), "momentum.random_only", noise_scale=1e-6)
    panel = compute_panel(ex.linearise(ds))
    assert panel.slope == pytest.approx(1.0, abs=0.01)
    rng = np.random.default_rng(11)
    ok = sum(posterior(ex, ex.simulate(rng, "momentum.mass_entry", magnitude=40.0)).top()
             == "momentum.mass_entry" for _ in range(10))
    assert ok >= 7


def test_dataset_roundtrip():
    from lab_detective.labsim import Dataset
    ds = EXP.simulate(np.random.default_rng(12), "momentum.friction")
    assert Dataset.from_dict(ds.to_dict()).observations == ds.observations


def test_card_is_valid():
    from lab_detective.knowledge import card_errors, load_card
    assert card_errors("momentum") == []
    assert set(load_card("momentum")["common_causes"]) == set(EXP.cause_ids())


# ── 独立物理复核补充的回归测试 ─────────────────────────────────────────


def test_coast_gate_to_gate_time_carries_entry_timing_error():
    """单独滑行：T = (D/w)·Δt_in，入口挡光时间的 0.1 ms 误差被放大 40 倍进 T，不能只算米尺量 D 的 2 ms。"""
    sd = EXP.total_sd("solo_fwd", {"mag": np.zeros((1, 1))}, "momentum.random_only")
    s_in = 0.1**2 + 0.1**2 / 12.0
    expect = np.sqrt(2.0**2 + 40.0**2 * s_in + 0.1**2 / 12.0)
    assert np.ravel(sd)[1] == pytest.approx(expect)
    assert np.ravel(sd)[1] > 4.5


def test_wheel_inertia_reduces_tilt_acceleration():
    """带轮小车下坡：(m + ΣI/R²)·a = mg·sinθ ⇒ a = g·sinθ·m/(m + ΣI/R²)。"""
    ex = Momentum(wheel_inertia_g=20.0)
    tilt = 3.0  # mm/m
    c, _ = ex.predict("solo_fwd", np.array([20.0]), np.arange(1), {"mag": np.array([[tilt]])},
                      "momentum.track_tilt")
    v_in, v_out = 0.01 / 0.020, 0.01 / (c[0, 0, 0] / 1000.0)
    a = G_ACC * tilt / 1000.0 * 300.0 / 320.0
    assert v_out**2 - v_in**2 == pytest.approx(2 * a * 0.40, rel=1e-9)
    # 粘合后两车一起：a = g·sinθ·(m_A + m_B)/(m_A + m_B + 2ΣI/R²)
    c, _ = ex.predict("main", np.array([20.0]), np.arange(1), {"mag": np.array([[tilt]])}, "momentum.track_tilt")
    v1 = 0.01 / 0.020
    vc = np.sqrt(v1**2 + 2 * G_ACC * tilt / 1000.0 * 300.0 / 320.0 * 0.20)
    v_after = vc * 320.0 / 540.0
    v2 = np.sqrt(v_after**2 + 2 * G_ACC * tilt / 1000.0 * 500.0 / 540.0 * 0.20)
    assert c[0, 0, 0] == pytest.approx(1000.0 * 0.01 / v2, rel=1e-9)


# 各原因在标准方案下面板上出现过半的特征都必须写进 signatures（散布类不算：大弯曲时二次拟合不够造成的 χ² 偏大）
@pytest.mark.parametrize("cause,mag,lin", [
    ("momentum.track_tilt", 0.5, None), ("momentum.track_tilt", -0.5, None),
    ("momentum.track_tilt", 3.5, None), ("momentum.track_tilt", -3.5, None),
    ("momentum.friction", 0.0005, None), ("momentum.friction", 0.0035, None),
    ("momentum.mass_entry", 50.0, None), ("momentum.mass_entry", -50.0, None),
    ("momentum.timing_offset", None, {"tau": 2.5}), ("momentum.timing_offset", None, {"tau": -2.5}),
])
def test_frequent_panel_features_are_declared(cause, mag, lin):
    rng = np.random.default_rng(13)
    seen = {}
    for _ in range(20):
        for s in compute_panel(EXP.linearise(EXP.simulate(rng, cause, magnitude=mag, linear=lin))).signatures:
            seen[s] = seen.get(s, 0) + 1
    frequent = {s for s, k in seen.items() if k >= 10 and not s.startswith("scatter")}
    assert frequent <= set(EXP.cause(cause).signatures), (frequent, seen)


def test_large_friction_bends_down():
    rng = np.random.default_rng(14)
    hits = sum("curve_down" in compute_panel(EXP.linearise(EXP.simulate(rng, "momentum.friction", magnitude=0.0035))).signatures
               for _ in range(20))
    assert hits >= 15


def test_friction_and_uphill_tilt_share_the_posterior():
    """正向碰撞里摩擦与上坡给出同一个减速度、似然完全相同，谁排第一只由两者的先验密度决定
    （大 μ 时均匀先验的坡度占优）；引擎至少要把概率集中在这两者上。"""
    rng = np.random.default_rng(15)
    for _ in range(10):
        post = posterior(EXP, EXP.simulate(rng, "momentum.friction", magnitude=0.0035))
        assert post.prob("momentum.friction") + post.prob("momentum.track_tilt") > 0.95
