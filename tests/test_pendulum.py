"""单摆测 g（留出实验）的解析检查和引擎检查。"""
import math
import time

import numpy as np
import pytest

from lab_detective.engine import posterior
from lab_detective.labsim import get_experiment
from lab_detective.labsim.experiments.pendulum import K_STD, Pendulum, _period_factor
from lab_detective.panel import compute_panel

EXP = get_experiment("pendulum")
SLOPE0 = 4 * math.pi**2 / 9.8 * K_STD**2  # 含 5° 摆角修正的理想斜率


def _panel(ds, exp=EXP):
    return compute_panel(exp.linearise(ds))


def test_registered_as_far_held_out():
    assert EXP.held_out and EXP.transfer == "far"
    assert EXP.causes[0].id == "pendulum.random_only"
    assert 4 <= len(EXP.causes) - 1 <= 6
    for m in EXP.menu:
        assert all(k in EXP.obs_kinds for k, _ in m.plan)


def test_period_factor_exact_values():
    # 1/AGM(1, cos45°) = 2K(sin45°)/π = 1.18034（90° 摆角的经典值）
    assert _period_factor(math.pi / 2) == pytest.approx(1.180341, rel=1e-6)
    # 小角度展开 1 + θ²/16 + 11θ⁴/3072
    th = math.radians(20.0)
    assert _period_factor(th) == pytest.approx(1 + th**2 / 16 + 11 * th**4 / 3072, rel=1e-5)


def test_zero_fault_recovers_g():
    ds = EXP.simulate(np.random.default_rng(0), "pendulum.random_only",
                      linear={"inv_sqrt_g": 1 / math.sqrt(9.8)}, noise_scale=1e-6)
    p = _panel(ds)
    g = 4 * math.pi**2 * K_STD**2 / p.slope
    assert g == pytest.approx(9.8, rel=0.005)  # t 取整到 0.01 s
    assert abs(p.intercept) < 0.01


@pytest.mark.parametrize("cause,kw,slope_factor", [
    ("pendulum.large_amplitude", dict(magnitude=30.0), (_period_factor(math.radians(30.0)) / K_STD) ** 2),
    ("pendulum.miscount", dict(magnitude=-1.0), (29 / 30) ** 2),
    # 圆锥摆 T² 精确为 (4π²L/g)cosα，不含 5° 平面摆修正，所以相对 SLOPE0 还要除以 K_STD²
    ("pendulum.conical", dict(magnitude=20.0), math.cos(math.radians(20.0)) / K_STD**2),
])
def test_scale_faults_match_derivation(cause, kw, slope_factor):
    ds = EXP.simulate(np.random.default_rng(1), cause, linear={"inv_sqrt_g": 1 / math.sqrt(9.8)},
                      noise_scale=1e-6, **kw)
    assert _panel(ds).slope == pytest.approx(SLOPE0 * slope_factor, rel=0.004)


def test_top_of_bob_intercept_is_4pi2_r_over_g():
    ds = EXP.simulate(np.random.default_rng(2), "pendulum.top_of_bob", nuisance={"r_cm": 1.5},
                      linear={"inv_sqrt_g": 1 / math.sqrt(9.8)}, noise_scale=1e-6)
    assert _panel(ds).intercept == pytest.approx(SLOPE0 * 0.015, abs=0.008)


@pytest.mark.parametrize("cause,kw,expect", [
    ("pendulum.top_of_bob", dict(nuisance={"r_cm": 1.5}), ("intercept_pos",)),
    ("pendulum.large_amplitude", dict(magnitude=30.0), ("slope_high",)),
    ("pendulum.miscount", dict(magnitude=-1.0), ("slope_low",)),
    ("pendulum.miscount", dict(magnitude=1.0), ("slope_high",)),
    ("pendulum.conical", dict(magnitude=20.0), ("slope_low",)),
    ("pendulum.timing_offset", dict(linear={"tau": 0.6}), ("intercept_pos", "slope_high")),
    ("pendulum.timing_offset", dict(linear={"tau": -0.6}), ("intercept_neg", "slope_low")),
])
def test_large_faults_show_signature(cause, kw, expect):
    assert set(expect) <= set(EXP.cause(cause).signatures)
    rng = np.random.default_rng(3)
    hits = {e: 0 for e in expect}
    for _ in range(20):
        sig = _panel(EXP.simulate(rng, cause, **kw)).signatures
        for e in expect:
            hits[e] += e in sig
    assert min(hits.values()) >= 15, hits


@pytest.mark.parametrize("cause,kw", [
    ("pendulum.large_amplitude", dict(magnitude=30.0)),
    ("pendulum.top_of_bob", dict(nuisance={"r_cm": 1.5})),
])
def test_engine_ranks_clear_fault_top(cause, kw):
    rng = np.random.default_rng(4)
    ok = sum(posterior(EXP, EXP.simulate(rng, cause, **kw)).top() == cause for _ in range(20))
    assert ok >= 14


def test_posterior_normalised_and_random_only_wins_on_clean_data():
    rng = np.random.default_rng(5)
    wins = 0
    for _ in range(20):
        post = posterior(EXP, EXP.simulate(rng, "pendulum.random_only"))
        assert post.probs.sum() == pytest.approx(1.0)
        wins += post.top() == "pendulum.random_only"
    assert wins >= 12


def _menu_gain(rng, cause, item, n=20, **kw):
    before, after = [], []
    for _ in range(n):
        ds = EXP.simulate(rng, cause, **kw)
        before.append(posterior(EXP, ds).prob(cause))
        ds2 = EXP.extend(rng, ds, EXP.menu_item(item).plan)
        after.append(posterior(EXP, ds2).prob(cause))
    return float(np.mean(before)), float(np.mean(after))


def test_small_angle_release_separates_conical_from_miscount():
    """圆锥摆 cosα = (29/30)²（α ≈ 20.9°）时与"少数一次全振动"的 T² 比例只差 5° 摆角修正（0.1%），标准数据下奥卡姆因子偏向
    只有两个取值的"数错次数"；在竖直平面内小角度重新释放后圆锥摆的影响消失，数错次数的影响还在。"""
    alpha = math.degrees(math.acos((29 / 30) ** 2))
    b, a = _menu_gain(np.random.default_rng(6), "pendulum.conical", "small_angle", magnitude=alpha)
    assert a > b + 0.3
    assert a > 0.8


def test_count10_separates_timing_offset_from_top_of_bob():
    """计时偏差 τ 和摆长少量半径都主要抬高截距；只数 10 次时 τ/N 变成 3 倍，摆长误差不变。"""
    b, a = _menu_gain(np.random.default_rng(7), "pendulum.timing_offset", "count10", linear={"tau": 0.3})
    assert a > b + 0.1
    assert a > 0.9


def test_posterior_is_fast():
    rng = np.random.default_rng(8)
    ds = EXP.simulate(rng, "pendulum.conical")
    posterior(EXP, ds)
    t0 = time.perf_counter()
    for _ in range(5):
        posterior(EXP, ds)
    assert (time.perf_counter() - t0) / 5 < 0.5


def test_stopwatch_profile():
    exp = Pendulum(timer="stopwatch")
    assert exp.obs_kinds["main"].readings[0].sigma > 0.1
    rng = np.random.default_rng(9)
    ds = exp.simulate(rng, "pendulum.large_amplitude", magnitude=40.0)
    ds = exp.extend(rng, ds, exp.menu_item("count10").plan)
    post = posterior(exp, ds)
    assert post.probs.sum() == pytest.approx(1.0)
    # 默认实例不受影响
    assert EXP.obs_kinds["main"].readings[0].sigma < 0.05


def test_single_protocol_runs():
    ds = EXP.simulate(np.random.default_rng(11), "pendulum.top_of_bob", protocol="single")
    assert EXP.linearise(ds).x.size == 1
    assert posterior(EXP, ds).probs.sum() == pytest.approx(1.0)


def test_dataset_roundtrip():
    from lab_detective.labsim import Dataset
    ds = EXP.simulate(np.random.default_rng(10), "pendulum.miscount")
    assert Dataset.from_dict(ds.to_dict()).observations == ds.observations


# ── 独立核对：直接积分运动方程，不用任何周期公式 ─────────────────────────────


def _sphere_pendulum_period(L, a_deg, b_deg, n=3):
    """点质量球摆（刚性摆线）：r̈ = g − k r，k 由 |r| = L 定出；从摆角 a 处释放，
    侧向初速度给出短半轴角 b（b = a 时用精确圆锥摆速度），数 x 由负到正穿过 0 的平均周期。"""
    from scipy.integrate import solve_ivp
    g = 9.8
    a, b = math.radians(a_deg), math.radians(b_deg)
    v = math.sqrt(g * L * math.sin(a) * math.tan(a)) if a_deg == b_deg else math.sqrt(g / L) * L * b
    y0 = np.array([L * math.sin(a), 0.0, -L * math.cos(a), 0.0, v, 0.0])

    def f(t, y):
        r, u = y[:3], y[3:]
        k = (u @ u - g * r[2]) / (L * L)
        return np.concatenate([u, np.array([0.0, 0.0, -g]) - k * r])

    def cross(t, y):
        return y[0]
    cross.direction = 1
    T0 = 2 * math.pi * math.sqrt(L / g)
    sol = solve_ivp(f, (0, (n + 1.5) * T0), y0, events=cross, rtol=1e-11, atol=1e-12, method="DOP853")
    return float(np.mean(np.diff(sol.t_events[0])))


def _pred_T(cause, mag, kind="main"):
    """预测的单周期（L = 100 cm，去掉复摆修正）。"""
    from lab_detective.labsim.experiments.pendulum import R_NOM_CM
    truth = {"cause": cause, "magnitude": mag, "nuisance": {"r_cm": 1.0},
             "linear": {p.name: (1 / math.sqrt(9.8) if p.name == "inv_sqrt_g" else 0.0)
                        for p in EXP.linear_params(cause)}}
    mean, _ = EXP._mean_and_sigma(truth, kind, np.array([100.0]), np.array([0]))
    L_eff = 1.0 + 0.4 * (R_NOM_CM / 100.0) ** 2
    return float(mean[0, 0]) / EXP.n_osc * math.sqrt(1.0 / L_eff)


def test_amplitude_and_cone_periods_match_equations_of_motion():
    # 平面大摆角：数值积分与 AGM 公式一致
    assert _pred_T("pendulum.large_amplitude", 30.0) == pytest.approx(_sphere_pendulum_period(1.0, 30.0, 0.0), rel=1e-6)
    assert _pred_T("pendulum.random_only", 0.0) == pytest.approx(_sphere_pendulum_period(1.0, 5.0, 0.0), rel=1e-6)
    # 圆锥摆：周期 2π√(L cosα/g)，不能再乘 5° 平面摆修正（回归：曾多乘 K_STD，T² 偏大 0.1%）
    for alpha in (10.0, 20.0):
        assert _pred_T("pendulum.conical", alpha) == pytest.approx(_sphere_pendulum_period(1.0, alpha, alpha), rel=1e-6)


def test_thin_ellipse_changes_counted_period_only_at_order_b_squared():
    """5° 释放带一点侧向晃动（短半轴 1°）：按同一平面计数的周期与平面摆只差 ~b² 量级（< 2×10⁻⁴），可以不建模。"""
    planar = _sphere_pendulum_period(1.0, 5.0, 0.0)
    d1 = _sphere_pendulum_period(1.0, 5.0, 1.0) / planar - 1
    d2 = _sphere_pendulum_period(1.0, 5.0, 2.0) / planar - 1
    assert abs(d1) < 2e-4
    assert d2 / d1 == pytest.approx(4.0, rel=0.1)  # ∝ b²
