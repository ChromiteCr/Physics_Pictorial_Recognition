"""组件层的基础件：Affine 代数、组合规则、干扰量相关性、模板物理（单摆周期修正、平抛、杠杆、碰撞）。"""

import math

import numpy as np
import pytest

from lab_detective import facts
from lab_detective.labsim.core import G, MenuItem
from lab_detective.labsim.experiments.pendulum import _period_factor
from lab_detective.labsim.experiments.projectile import Projectile
from lab_detective.physics import Affine, Control, Setup, catalog, compose
from lab_detective.physics.components import Ball, Cart, Hanger, Lever, Spring
from lab_detective.physics.instruments import Photogate, Ruler, SpringScale
from lab_detective.physics.templates import Collision1D, ProjectileRamp, StaticSpring, TorqueBalance, period_factor

EXPS = {e.id: e.experiment() for e in catalog.CATALOG.values()}


# ── Affine ────────────────────────────────────────────────────────────


def test_affine_algebra_and_numpy_interop():
    a = Affine.lin("k", np.array([[1.0, 2.0]])) + 3.0
    b = 2.0 * a - Affine.lin("z") + np.array([[1.0, 1.0]])  # ndarray 在左边也要交给 Affine
    assert isinstance(b, Affine) and isinstance(np.array([2.0]) * a, Affine)
    c, U = (b / 2.0).to_cU(["k", "z"], (1, 2))
    np.testing.assert_allclose(c, [[3.5, 3.5]])
    np.testing.assert_allclose(U[0, :, 0], [1.0, 2.0])
    np.testing.assert_allclose(U[0, :, 1], [-0.5, -0.5])
    assert b.value({"k": 1.0, "z": 2.0}) == pytest.approx(np.array([[7.0, 9.0]]))
    with pytest.raises(TypeError):
        _ = a * Affine.lin("z")  # 两个含线性参数的量不能相乘
    with pytest.raises(TypeError):
        _ = 1.0 / a
    with pytest.raises(KeyError):
        a.to_cU(["z"], (1, 2))  # 用到本原因没有的线性参数


# ── 组合规则 ──────────────────────────────────────────────────────────


def _mini(**kw) -> Setup:
    base = dict(id="mini_spring", name_zh="弹簧", name_en="spring", template=StaticSpring(),
                parts=[Spring("spring"), Hanger("hanger", enabled=("omitted",)), Ruler("ruler")],
                instrument="ruler", control=Control("mass", "钩码质量", "mass", "g", 1e-3),
                standard=(50.0, 100.0, 150.0), single=100.0)
    base.update(kw)
    return Setup(**base)


def test_setup_validation():
    with pytest.raises(ValueError):
        _mini(id="a.b")
    with pytest.raises(ValueError):
        _mini(parts=[Spring("x"), Ruler("x")], instrument="x")
    with pytest.raises(ValueError):
        _mini(instrument="scale")
    with pytest.raises(KeyError):
        Spring("spring", enabled=("no_such",)).effects()
    with pytest.raises(KeyError):
        compose(_mini(exact_in_faults=("spring_nothing",)))
    with pytest.raises(ValueError):
        compose(_mini(extra_menu=(MenuItem("zero_check", "x", "x", 1.0, (("main", 0.0),)),)))


def test_exclude_and_alias():
    exp = compose(_mini(exclude=("ruler_total_length",), cause_alias={"hanger_omitted": "hook_left_out"}))
    ids = exp.cause_ids()
    assert "mini_spring.hook_left_out" in ids and "mini_spring.ruler_total_length" not in ids
    assert ids[0] == "mini_spring.random_only"


@pytest.mark.parametrize("eid", list(EXPS))
def test_relevant_nuisance_is_complete(eid):
    """引擎只把 relevant_nuisance 放进网格，其余干扰量取众数：它们必须真的不影响预测和噪声。"""
    exp = EXPS[eid]
    rng = np.random.default_rng(0)
    P = 3
    for cause in exp.causes:
        for kind in exp.obs_kinds:
            rel = set(exp.relevant_nuisance(cause.id, {kind}))
            par = {"mag": np.array([cause.magnitude.draw(rng) for _ in range(P)])[:, None]}
            for k, pr in exp.nuisance.items():
                par[k] = np.array([pr.draw(rng) for _ in range(P)])[:, None]
            if "block.mu_pull" in par:
                par["block.mu_pull"] = np.abs(par["block.mu_pull"])
            s = np.array([float(x) for x in exp.setup.standard[:3]])
            c1, U1 = exp.predict(kind, s, np.arange(3.0), par, cause.id)
            par2 = dict(par)
            for k, pr in exp.nuisance.items():
                if k not in rel:
                    par2[k] = np.array([pr.draw(rng) for _ in range(P)])[:, None]
            c2, U2 = exp.predict(kind, s, np.arange(3.0), par2, cause.id)
            np.testing.assert_allclose(c2, c1, err_msg=f"{cause.id}/{kind}")
            np.testing.assert_allclose(U2, U1, err_msg=f"{cause.id}/{kind}")
            np.testing.assert_allclose(exp.total_sd(kind, par2, cause.id), exp.total_sd(kind, par, cause.id))


@pytest.mark.parametrize("eid", list(EXPS))
def test_every_cause_changes_some_prediction(eid):
    """每个原因（取较大幅度）至少在一种观测里改变读数均值或散布，否则它在任何数据下都不可辨识。"""
    from lab_detective.twin import nominal_truth

    exp = EXPS[eid]
    s = np.array([float(exp.setup.standard[-1])])
    base = {k: exp._mean_and_sigma(nominal_truth(exp), k, s, np.zeros(1)) for k in exp.obs_kinds}
    for cause in exp.causes[1:]:
        t = nominal_truth(exp, cause.id)
        t["magnitude"] = float(np.max(cause.magnitude.grid()[0]))
        for p in cause.linear:
            t["linear"][p.name] = p.mean + 3.0 * p.sd
        moved = False
        for kind in exp.obs_kinds:
            m, sg = exp._mean_and_sigma(t, kind, s, np.zeros(1))
            moved |= not np.allclose(m, base[kind][0], rtol=1e-6) or not np.allclose(sg, base[kind][1])
        assert moved, cause.id


def test_friction_sync_keeps_denominator_mu_equal_to_linear_mu():
    exp = EXPS["friction_composed"]
    t = exp.draw_truth(np.random.default_rng(1), "friction_composed.pull_angle")
    assert t["nuisance"]["block.mu_pull"] == t["linear"]["block.mu"]


def test_fact_sheet_works_for_a_new_composition():
    exp = EXPS["system_energy"]
    ds = exp.simulate(np.random.default_rng(2), "system_energy.push", magnitude=0.5)
    fs = facts.build_fact_sheet(exp, ds, rng=np.random.default_rng(0), n_samples=32)
    assert fs["experiment_id"] == "system_energy"
    assert fs["diagnosis"]["ranked"][0]["cause_id"] == "system_energy.push"
    assert fs["numbers"]


# ── 模板物理 ──────────────────────────────────────────────────────────


def test_period_factor_matches_curated_and_series():
    th = np.radians([1.0, 5.0, 20.0, 45.0])
    np.testing.assert_allclose(period_factor(th), _period_factor(th), rtol=1e-14)
    assert period_factor(np.radians(5.0)) == pytest.approx(1 + np.radians(5.0) ** 2 / 16, rel=1e-5)


def test_projectile_ramp_matches_curated_random_only():
    setup = Setup(id="ramp", name_zh="平抛", name_en="ramp", template=ProjectileRamp(ball="ball", launch_h_cm=80.0),
                  parts=[Ball("ball", diameter_cm=2.0, enabled=()),
                         Ruler("ruler", role="reading", enabled=(), key_name="x", label="水平距离",
                               sigma=0.7 / math.sqrt(5))],
                  instrument="ruler", control=Control("height", "释放高度 H", "release height", "cm", 1e-2),
                  standard=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0), single=20.0)
    exp, cur = compose(setup), Projectile()
    s = np.array([5.0, 10.0, 20.0, 30.0])
    par = {"mag": np.zeros((1, 1))}
    c1, _ = cur.predict("main", s, np.arange(4.0), par, "projectile.random_only")
    c2, _ = exp.predict("main", s, np.arange(4.0), par, "ramp.random_only")
    np.testing.assert_allclose(c2, c1, rtol=1e-12)
    # x² = 4hH/(1+β)，β = 2/5
    np.testing.assert_allclose(c2[0, :, 0] ** 2, 4 * 80.0 * s / 1.4, rtol=1e-12)
    ds = exp.simulate(np.random.default_rng(0), "ramp.random_only", noise_scale=1e-6)
    assert exp.linearise(ds).ideal_slope == pytest.approx(4 * 80.0 / 1.4)


def test_torque_balance_and_unbalanced_lever():
    setup = Setup(id="lever_scale", name_zh="杠杆", name_en="lever",
                  template=TorqueBalance(lever="lever", load_g=200.0, load_arm_cm=10.0),
                  parts=[Lever("lever"), SpringScale("scale", orientation="vertical", enabled=("zero_shift",))],
                  instrument="scale", control=Control("length", "动力臂 L1", "arm", "cm", 1e-2),
                  standard=(5.0, 10.0, 15.0, 20.0, 25.0), single=10.0)
    exp = compose(setup)
    ds = exp.simulate(np.random.default_rng(0), "lever_scale.random_only", noise_scale=1e-6)
    for o in ds.observations:  # F1 = F2·L2/L1
        assert o.values[0] == pytest.approx(0.2 * G * 10.0 / o.setting, abs=0.051)
    p = facts.compute_panel(exp.linearise(ds))
    assert p.slope == pytest.approx(0.2 * G * 0.1, rel=0.03)
    ds2 = exp.simulate(np.random.default_rng(1), "lever_scale.lever_not_balanced", linear={"lever.m0": 0.05},
                       noise_scale=1e-6)
    p2 = facts.compute_panel(exp.linearise(ds2))
    assert p2.slope - p.slope == pytest.approx(0.05, abs=0.01)  # 附加力矩直接加到斜率 F2·L2 上
    # 水平调零、竖直使用：每个读数多 w，截距为正
    ds3 = exp.simulate(np.random.default_rng(2), "lever_scale.scale_zero_shift", linear={"scale.hook_weight": 0.2},
                       noise_scale=1e-6)
    assert facts.compute_panel(exp.linearise(ds3)).intercept == pytest.approx(0.2, abs=0.05)


def test_collision_conserves_momentum_with_wheel_inertia():
    setup = Setup(id="carts", name_zh="碰撞", name_en="collision",
                  template=Collision1D(cart_a="a", cart_b="b", gate="gate"),
                  parts=[Cart("a", mass_g=300.0), Cart("b", mass_g=200.0, enabled=()),
                         Photogate("gate", mode="speed", flag_mm=10.0, width_linear=False, enabled=())],
                  instrument="gate", control=Control("time", "碰前挡光时间 Δt₁", "gate-1 time", "ms", 1.0),
                  standard=(20.0, 25.0, 30.0, 35.0, 40.0), single=30.0)
    exp = compose(setup)
    s = np.array([20.0, 40.0])
    c, _ = exp.predict("main", s, np.arange(2.0), {"mag": np.zeros((1, 1))}, "carts.random_only")
    np.testing.assert_allclose(c[0, :, 0], s * 500.0 / 300.0)  # m_A v₁ = (m_A + m_B) v₂
    # A 车轮等效质量 J = ε·m_A：守恒的是 Σ(m + J)v
    c2, _ = exp.predict("main", s, np.arange(2.0), {"mag": np.full((1, 1), 0.05)}, "carts.a_wheel_inertia")
    np.testing.assert_allclose(c2[0, :, 0], s * (500.0 + 15.0) / (300.0 + 15.0))
    ds = exp.simulate(np.random.default_rng(0), "carts.random_only", noise_scale=1e-6)
    p = facts.compute_panel(exp.linearise(ds))
    assert p.slope == pytest.approx(1.0, abs=0.01)


def test_ball_drag_formula():
    """v² = (g/κ)(1 − e^{−2κh})：小 κ 时趋于 2gh。"""
    exp = EXPS["photogate_free_fall"]
    h = np.array([50.0, 100.0])
    par = {"mag": np.full((1, 1), 1e-6)}
    c, U = exp.predict("main", h, np.arange(2.0), par, "photogate_free_fall.air_drag")
    c0, U0 = exp.predict("main", h, np.arange(2.0), par, "photogate_free_fall.random_only")
    np.testing.assert_allclose(U, U0, rtol=1e-5)
    par = {"mag": np.full((1, 1), 0.1)}
    c, U = exp.predict("main", h, np.arange(2.0), par, "photogate_free_fall.air_drag")
    v2_ratio = (U0[0, :, 0, 0] / U[0, :, 0, 0]) ** 2  # Δt ∝ 1/v
    expect = (1 - np.exp(-2 * 0.1 * h / 100)) / 0.1 / (2 * h / 100)
    np.testing.assert_allclose(v2_ratio, expect, rtol=1e-9)
