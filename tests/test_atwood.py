"""阿特伍德机孪生（留出实验）的解析检查和引擎检查。"""
import time

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from lab_detective.engine import posterior
from lab_detective.labsim import get_experiment
from lab_detective.labsim.core import G
from lab_detective.labsim.experiments.atwood import Atwood
from lab_detective.panel import compute_panel

EXP = get_experiment("atwood")
MT, H = 0.6, 0.8

LARGE = {
    "atwood.pulley_inertia": dict(magnitude=80.0),
    "atwood.pulley_friction": dict(magnitude=0.06),
    "atwood.string_mass": dict(magnitude=6.0),
    "atwood.timing_offset": dict(linear={"dt": -0.35}),
    "atwood.wrong_distance": dict(magnitude=18.0),
}


def test_registry_conventions():
    assert EXP.held_out and EXP.transfer == "near"
    assert EXP.causes[0].id == "atwood.random_only"
    assert all(c.id.startswith("atwood.") for c in EXP.causes)
    assert 4 <= len(EXP.causes) - 1 <= 6
    assert 3 <= len(EXP.menu) <= 6
    assert set(EXP.protocols) >= {"standard", "single"}


def test_zero_fault_recovers_total_mass():
    ds = EXP.simulate(np.random.default_rng(0), "atwood.random_only", noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    # 斜率 = 1/M_t = 1/0.6 kg；t 记录到 0.01 s，带来 < 1% 的取整误差
    assert 1.0 / panel.slope == pytest.approx(MT, rel=0.02)
    assert abs(panel.intercept) < 0.01


def test_pulley_inertia_adds_to_accelerated_mass():
    ds = EXP.simulate(np.random.default_rng(1), "atwood.pulley_inertia", magnitude=60.0, noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    assert 1.0 / panel.slope == pytest.approx(MT + 0.060, rel=0.02)


def test_string_mass_closed_form_matches_ode():
    """细线有质量时 ẍ = A + Bx 的解析解与直接数值积分一致。"""
    lam, dm = 0.005, 0.010
    params = {"mag": np.array([[lam * 1000]])}
    t_formula = EXP._fall_time("main", np.array([dm * 1000]), params, "atwood.string_mass")[0, 0]
    meff = MT + lam * (H + 0.8)

    def rhs(_, y):  # m1 一侧线长减 m2 一侧线长 = 2x − h
        return [y[1], (dm * G + lam * G * (2 * y[0] - H)) / meff]

    hit = lambda _, y: y[0] - H  # noqa: E731
    hit.terminal = True
    sol = solve_ivp(rhs, [0, 30], [0.0, 0.0], events=hit, rtol=1e-10, atol=1e-12)
    assert t_formula == pytest.approx(sol.t_events[0][0], rel=1e-6)
    # 一阶近似：a_测 ≈ (Δm g − (2/3)λgh)/M_eff
    a_meas = 2 * H / t_formula**2
    assert a_meas == pytest.approx((dm * G - 2 / 3 * lam * G * H) / meff, rel=0.03)


def test_timing_offset_enters_raw_time_linearly():
    ds = EXP.simulate(np.random.default_rng(2), "atwood.timing_offset", linear={"dt": 0.2}, noise_scale=1e-6)
    ideal = np.sqrt(2 * H / (np.array([o.setting for o in ds.observations]) / 1000 * G / MT))
    got = np.array([o.values[0] for o in ds.observations])
    assert np.allclose(got - ideal, 0.2, atol=0.006)


@pytest.mark.parametrize("cause", list(LARGE))
def test_large_faults_show_declared_signature(cause):
    rng = np.random.default_rng(3)
    declared = set(EXP.cause(cause).signatures)
    hits = 0
    for _ in range(20):
        ds = EXP.simulate(rng, cause, **LARGE[cause])
        hits += bool(declared & set(compute_panel(EXP.linearise(ds)).signatures))
    assert hits >= 15


@pytest.mark.parametrize("cause", ["atwood.pulley_friction", "atwood.timing_offset", "atwood.wrong_distance"])
def test_engine_ranks_large_fault_top(cause):
    rng = np.random.default_rng(4)
    ok = sum(posterior(EXP, EXP.simulate(rng, cause, **LARGE[cause])).top() == cause for _ in range(20))
    assert ok >= 14


def test_posterior_normalised_and_random_only_wins_on_clean_data():
    rng = np.random.default_rng(5)
    wins = 0
    for _ in range(20):
        post = posterior(EXP, EXP.simulate(rng, "atwood.random_only"))
        assert post.probs.sum() == pytest.approx(1.0)
        wins += post.top() == "atwood.random_only"
    assert wins >= 12


def test_thin_thread_separates_string_mass_from_pulley_friction():
    """粗绳的重力差和滑轮轴摩擦在固定 h 时都等效于恒定阻力；换细线后前者消失、后者不变。"""
    rng = np.random.default_rng(6)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, "atwood.string_mass", magnitude=5.0)
        before.append(posterior(EXP, ds).prob("atwood.string_mass"))
        ds2 = EXP.extend(rng, ds, EXP.menu_item("thin_thread").plan)
        after.append(posterior(EXP, ds2).prob("atwood.string_mass"))
    assert np.mean(after) > np.mean(before) + 0.2
    assert np.mean(after) > 0.85


def test_photogate_run_separates_timing_offset_from_pulley_inertia():
    rng = np.random.default_rng(7)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, "atwood.pulley_inertia", magnitude=50.0)
        before.append(posterior(EXP, ds).prob("atwood.pulley_inertia"))
        ds2 = EXP.extend(rng, ds, EXP.menu_item("photogate_run").plan)
        after.append(posterior(EXP, ds2).prob("atwood.pulley_inertia"))
    assert np.mean(after) > np.mean(before) + 0.15


def test_posterior_is_fast():
    ds = EXP.simulate(np.random.default_rng(8), "atwood.string_mass")
    t0 = time.perf_counter()
    for _ in range(5):
        posterior(EXP, ds)
    assert (time.perf_counter() - t0) / 5 < 0.5


def test_photogate_timer_option():
    pg = Atwood(timer="photogate")
    assert pg.obs_kinds["main"].readings[0].sigma < 0.01
    assert pg.cause("atwood.timing_offset").linear[0].sd < 0.05
    ds = pg.simulate(np.random.default_rng(9), "atwood.random_only", noise_scale=1e-6)
    assert 1.0 / compute_panel(pg.linearise(ds)).slope == pytest.approx(MT, rel=0.002)
    with pytest.raises(ValueError):
        Atwood(timer="hourglass")


def test_card_matches_module():
    from lab_detective import knowledge

    card = knowledge.load_card("atwood")
    assert knowledge.card_errors("atwood") == []
    assert set(card["common_causes"]) == set(EXP.cause_ids())
    assert card["labsim_id"] == "atwood"
