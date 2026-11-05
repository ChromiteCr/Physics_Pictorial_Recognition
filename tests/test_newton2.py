"""牛顿第二定律（小车 + 槽码）孪生的解析检查和引擎检查。"""
import time

import numpy as np
import pytest

from lab_detective.engine import posterior
from lab_detective.labsim import get_experiment
from lab_detective.labsim.core import G
from lab_detective.labsim.experiments.newton2 import Newton2
from lab_detective.panel import compute_panel

EXP = get_experiment("newton2")

LARGE = {
    "newton2.unbalanced": dict(linear={"f_unbal": -0.08}),
    "newton2.overbalanced": dict(linear={"f_over": 0.08}),
    "newton2.heavy_hanging": dict(),
    "newton2.string_angle": dict(magnitude=25.0),
    "newton2.pulley_drag": dict(magnitude=25.0, linear={"f_pulley": 0.035}),
}


def test_registry_conventions():
    assert not EXP.held_out and EXP.transfer is None
    assert EXP.causes[0].id == "newton2.random_only"
    assert all(c.id.startswith("newton2.") for c in EXP.causes)
    assert 4 <= len(EXP.causes) - 1 <= 6
    assert all(c.source == "assumption" for c in EXP.causes)
    assert 3 <= len(EXP.menu) <= 6
    assert set(EXP.protocols) >= {"standard", "single"}


def test_zero_fault_recovers_cart_mass():
    rng = np.random.default_rng(0)
    ds = EXP.simulate(rng, "newton2.random_only", noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    # 斜率 = 1/M，M = 0.5 kg → 2.0 kg⁻¹；a 记录到 0.01 m/s²，取整误差很小
    assert 1.0 / panel.slope == pytest.approx(0.5, rel=0.02)
    assert abs(panel.intercept) < 0.01


def test_heavy_hanging_follows_exact_connected_body_law():
    ds = EXP.simulate(np.random.default_rng(1), "newton2.heavy_hanging", noise_scale=1e-6)
    for o in ds.observations:
        m = o.setting / 1000.0
        assert o.values[0] == pytest.approx(m * G / (0.5 + m), abs=0.006)
    # m = 50 g 时比理想值 F/M 小 m/(M + m) ≈ 9%
    top = ds.observations[-1].values[0]
    assert top / (0.05 * G / 0.5) == pytest.approx(0.5 / 0.55, abs=0.01)


def test_unbalanced_gives_predicted_force_intercept():
    ds = EXP.simulate(np.random.default_rng(2), "newton2.unbalanced", linear={"f_unbal": -0.06}, noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    # a = (F + F_res)/M → F 轴截距 = −F_res = 0.06 N，a 轴截距 = F_res/M = −0.12 m/s²
    assert panel.intercept == pytest.approx(-0.12, abs=0.01)
    assert -panel.intercept / panel.slope == pytest.approx(0.06, abs=0.005)


@pytest.mark.parametrize("cause", list(LARGE))
def test_large_faults_show_declared_signature(cause):
    rng = np.random.default_rng(3)
    declared = set(EXP.cause(cause).signatures)
    hits = 0
    for _ in range(20):
        ds = EXP.simulate(rng, cause, **LARGE[cause])
        hits += bool(declared & set(compute_panel(EXP.linearise(ds)).signatures))
    assert hits >= 15


@pytest.mark.parametrize("cause", ["newton2.overbalanced", "newton2.heavy_hanging", "newton2.unbalanced"])
def test_engine_ranks_large_fault_top(cause):
    rng = np.random.default_rng(4)
    ok = sum(posterior(EXP, EXP.simulate(rng, cause, **LARGE[cause])).top() == cause for _ in range(20))
    assert ok >= 14


def test_posterior_normalised_and_random_only_wins_on_clean_data():
    rng = np.random.default_rng(5)
    wins = 0
    for _ in range(20):
        post = posterior(EXP, EXP.simulate(rng, "newton2.random_only"))
        assert post.probs.sum() == pytest.approx(1.0)
        wins += post.top() == "newton2.random_only"
    assert wins >= 12


def test_balance_check_separates_pulley_drag_from_unbalanced():
    """滑轮阻力和没平衡摩擦力在标准数据里都是 F 轴正截距；取下细线测小车自身加速度能分开它们。"""
    rng = np.random.default_rng(6)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, "newton2.pulley_drag", magnitude=20.0, linear={"f_pulley": 0.03})
        before.append(posterior(EXP, ds).prob("newton2.pulley_drag"))
        ds2 = EXP.extend(rng, ds, EXP.menu_item("balance_check").plan)
        after.append(posterior(EXP, ds2).prob("newton2.pulley_drag"))
    assert np.mean(after) > np.mean(before) + 0.2
    assert np.mean(after) > 0.85


def test_string_height_separates_string_angle_from_heavy_hanging():
    rng = np.random.default_rng(7)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, "newton2.string_angle", magnitude=22.0)
        before.append(posterior(EXP, ds).prob("newton2.string_angle"))
        ds2 = EXP.extend(rng, ds, EXP.menu_item("string_height").plan)
        after.append(posterior(EXP, ds2).prob("newton2.string_angle"))
    assert np.mean(after) > np.mean(before) + 0.3


def test_posterior_is_fast():
    rng = np.random.default_rng(8)
    ds = EXP.simulate(rng, "newton2.pulley_drag")
    t0 = time.perf_counter()
    for _ in range(5):
        posterior(EXP, ds)
    assert (time.perf_counter() - t0) / 5 < 0.5


def test_photogate_option_has_smaller_noise():
    pg = Newton2(timer="photogate")
    assert pg.obs_kinds["main"].readings[0].sigma < EXP.obs_kinds["main"].readings[0].sigma
    ds = pg.simulate(np.random.default_rng(9), "newton2.random_only", noise_scale=1e-6)
    assert 1.0 / compute_panel(pg.linearise(ds)).slope == pytest.approx(0.5, rel=0.005)
    with pytest.raises(ValueError):
        Newton2(timer="sundial")


def test_card_matches_module():
    from lab_detective import knowledge

    card = knowledge.load_card("newton2")
    assert knowledge.card_errors("newton2") == []
    assert set(card["common_causes"]) == set(EXP.cause_ids())
    assert card["labsim_id"] == "newton2"
