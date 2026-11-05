"""平抛实验（斜槽滚下 → 槽口水平飞出）的解析检查和引擎检查。"""
import time

import numpy as np
import pytest

from lab_detective.engine import posterior
from lab_detective.labsim import G, get_experiment
from lab_detective.labsim.experiments.projectile import Projectile
from lab_detective.panel import compute_panel

EXP = get_experiment("projectile")


def test_registry_and_cause_ids():
    assert EXP.held_out and EXP.transfer == "far"
    ids = EXP.cause_ids()
    assert ids[0] == "projectile.random_only"
    assert all(i.startswith("projectile.") for i in ids)
    assert 4 <= len(ids) - 1 <= 6


def test_zero_fault_recovers_rolling_factor():
    """实心球纯滚：x² = (20/7)hH，斜率 / (4h) = 5/7（g 约掉）。"""
    ds = EXP.simulate(np.random.default_rng(0), "projectile.random_only", noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    assert panel.slope / (4.0 * EXP.h) == pytest.approx(5.0 / 7.0, rel=0.005)
    assert panel.slope_ratio == pytest.approx(1.0, abs=0.005)
    assert abs(panel.intercept) < 5.0  # cm²，只剩 0.1 cm 取整


def test_groove_track_rolling_factor():
    """双轨槽：接触线相距 2s，β = (2/5)R²/(R² − s²)，x² = 4hH/(1+β)。"""
    ex = Projectile(rail_gap_cm=1.2)  # R = 1 cm，s = 0.6 cm ⇒ β = 0.4/0.64 = 0.625
    assert ex.beta == pytest.approx(0.625)
    c, _ = ex.predict("main", np.array([20.0]), np.arange(1), {"mag": np.zeros((1, 1))}, "projectile.random_only")
    assert c[0, 0, 0] ** 2 == pytest.approx(4 * 80.0 * 20.0 / 1.625)


def test_tilted_lip_matches_independent_trajectory():
    """槽口上倾 5°：与按 y(t) = v₀sinθ·t − ½gt² = −h 直接解二次方程的结果一致。"""
    H, deg = 20.0, 5.0
    c, _ = EXP.predict("main", np.array([H]), np.arange(1), {"mag": np.array([[deg]])}, "projectile.end_tilted")
    v0 = np.sqrt(10.0 / 7.0 * G * H / 100.0)
    th = np.deg2rad(deg)
    roots = np.roots([-0.5 * G, v0 * np.sin(th), EXP.h / 100.0])
    t = roots[roots > 0][0]
    assert c[0, 0, 0] == pytest.approx(100.0 * v0 * np.cos(th) * t, rel=1e-9)
    # 小角度近似：Δx ≈ v₀²θ/g
    x0 = 100.0 * v0 * np.sqrt(2 * EXP.h / 100.0 / G)
    assert c[0, 0, 0] - x0 == pytest.approx(100.0 * v0**2 * th / G, rel=0.1)


@pytest.mark.parametrize("cause,mag,expect", [
    ("projectile.ramp_loss", 0.15, "slope_low"),
    ("projectile.height_wrong", 3.0, "slope_high"),
    ("projectile.height_wrong", -3.0, "slope_low"),
    ("projectile.end_tilted", 4.0, "slope_high"),
    ("projectile.end_tilted", -4.0, "slope_low"),
    ("projectile.slipping", 0.5, "slope_high"),
    ("projectile.landing_scatter", 2.0, "scatter_excess"),
    ("projectile.release_zero", 2.0, "intercept_pos"),
    ("projectile.release_zero", -2.0, "intercept_neg"),
])
def test_large_faults_show_signature(cause, mag, expect):
    assert expect in EXP.cause(cause).signatures
    rng = np.random.default_rng(2)
    hits = 0
    for _ in range(20):
        ds = EXP.simulate(rng, cause, magnitude=mag)
        hits += expect in compute_panel(EXP.linearise(ds)).signatures
    assert hits >= 15


@pytest.mark.parametrize("cause,mag", [
    ("projectile.ramp_loss", 0.15),
    ("projectile.slipping", 0.5),
    ("projectile.release_zero", 2.0),
    ("projectile.landing_scatter", 2.0),
])
def test_engine_ranks_large_fault_top(cause, mag):
    rng = np.random.default_rng(3)
    ok = sum(posterior(EXP, EXP.simulate(rng, cause, magnitude=mag)).top() == cause for _ in range(20))
    assert ok >= 14


def test_posterior_normalised_and_random_only_wins_on_clean_data():
    rng = np.random.default_rng(4)
    wins = 0
    for _ in range(20):
        post = posterior(EXP, EXP.simulate(rng, "projectile.random_only"))
        assert post.probs.sum() == pytest.approx(1.0)
        wins += post.top() == "projectile.random_only"
    assert wins >= 12


def test_exit_gate_separates_ramp_loss_from_wrong_height():
    """能量损失 4% 与"h 记大了 3 cm"都让 x² 整体缩小约 4%，标准数据分不开；
    槽口光电门直接测 v₀：能量损失使 v₀ 变小，量错 h 不影响 v₀。"""
    rng = np.random.default_rng(5)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, "projectile.ramp_loss", magnitude=0.04)
        before.append(posterior(EXP, ds).prob("projectile.ramp_loss"))
        ds2 = EXP.extend(rng, ds, EXP.menu_item("exit_gate").plan)
        after.append(posterior(EXP, ds2).prob("projectile.ramp_loss"))
    assert np.mean(after) > np.mean(before) + 0.3
    assert np.mean(after) > 0.8


def test_height_check_reveals_wrong_height():
    rng = np.random.default_rng(6)
    after = []
    for _ in range(10):
        ds = EXP.simulate(rng, "projectile.height_wrong", magnitude=3.2)
        ds2 = EXP.extend(rng, ds, EXP.menu_item("h_check").plan)
        after.append(posterior(EXP, ds2).prob("projectile.height_wrong"))
    assert np.mean(after) > 0.9


def test_posterior_is_fast():
    ds = EXP.simulate(np.random.default_rng(7), "projectile.end_tilted", magnitude=3.0)
    t0 = time.perf_counter()
    posterior(EXP, ds)
    assert time.perf_counter() - t0 < 0.5


def test_card_is_valid():
    from lab_detective.knowledge import card_errors, load_card
    assert card_errors("projectile") == []
    assert set(load_card("projectile")["common_causes"]) == set(EXP.cause_ids())


def test_dataset_roundtrip():
    from lab_detective.labsim import Dataset
    ds = EXP.simulate(np.random.default_rng(8), "projectile.slipping")
    assert Dataset.from_dict(ds.to_dict()).observations == ds.observations
