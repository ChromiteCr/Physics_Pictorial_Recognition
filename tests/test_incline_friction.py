"""斜面动摩擦因数（留出实验）的解析检查和引擎检查。"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from lab_detective.engine import posterior
from lab_detective.labsim import G, get_experiment
from lab_detective.labsim.experiments.incline_friction import InclineFriction
from lab_detective.panel import compute_panel

EXP = get_experiment("incline_friction")


def _main_mean(exp, cause, theta, mag, mu, kind="main"):
    c, U = exp.predict(kind, np.array([theta]), np.array([0]), {"mag": np.array([[mag]])}, cause)
    return float(c[0, 0, 0] + U[0, 0, 0, 0] * mu)


def test_held_out_flags():
    assert EXP.held_out and EXP.transfer == "near"


def test_zero_fault_recovers_mu_and_g():
    ds = EXP.simulate(np.random.default_rng(0), "incline_friction.random_only", linear={"mu": 0.3},
                      noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    # a 取整到 0.01 m/s²，斜率 ≈ g、μ = −截距/g
    assert panel.slope == pytest.approx(G, rel=0.005)
    assert -panel.intercept / G == pytest.approx(0.3, abs=0.005)
    assert not {"slope_high", "slope_low", "intercept_pos", "intercept_neg"} & set(panel.signatures)
    assert math.isnan(panel.intercept_z)  # 截距含待测 μ，不检验


def test_newton_second_law_and_mass_independence():
    mu, th = 0.3, 40.0
    a = _main_mean(EXP, "incline_friction.random_only", th, 0.0, mu)
    assert a == pytest.approx(G * (math.sin(math.radians(th)) - mu * math.cos(math.radians(th))))
    assert _main_mean(EXP, "incline_friction.random_only", th, 0.0, mu, "heavy") == pytest.approx(a)


def test_protractor_offset_is_exact_line():
    """δ 偏差下 a/cos θ_r = g(cos δ + μ sin δ) tan θ_r + g(sin δ − μ cos δ)：仍是直线，μ 却偏差约 δ。"""
    mu, d = 0.3, math.radians(4.0)
    x, y = [], []
    for th in (30.0, 35.0, 40.0, 45.0, 50.0):
        x.append(math.tan(math.radians(th)))
        y.append(_main_mean(EXP, "incline_friction.protractor_zero", th, 4.0, mu) / math.cos(math.radians(th)))
    slope, icpt = np.polyfit(x, y, 1)
    assert np.max(np.abs(np.polyval([slope, icpt], x) - y)) < 1e-9
    assert slope == pytest.approx(G * (math.cos(d) + mu * math.sin(d)))
    assert -icpt / G == pytest.approx(mu * math.cos(d) - math.sin(d))


def test_scale_causes_cancel_in_intercept_over_slope():
    """遮光片宽度、光电门间距只给 a 乘一个比例：μ = −截距/拟合斜率 不受影响。"""
    for cause, mag, s in [("incline_friction.flag_width", 0.05, 1.05**2),
                          ("incline_friction.gate_spacing", 2.0, 48.0 / 50.0)]:
        x = [math.tan(math.radians(t)) for t in (30.0, 40.0, 50.0)]
        y = [_main_mean(EXP, cause, t, mag, 0.3) / math.cos(math.radians(t)) for t in (30.0, 40.0, 50.0)]
        slope, icpt = np.polyfit(x, y, 1)
        assert slope == pytest.approx(s * G)
        assert -icpt / slope == pytest.approx(0.3)


@pytest.mark.parametrize("cause,mag,expect", [
    ("incline_friction.protractor_zero", -5.0, "slope_low"),
    ("incline_friction.flag_width", 0.06, "slope_high"),
    ("incline_friction.flag_width", -0.06, "slope_low"),
    ("incline_friction.gate_spacing", 3.0, "slope_low"),
    ("incline_friction.mu_load", 0.3, "slope_low"),
    ("incline_friction.release_scatter", 0.4, "scatter_excess"),
])
def test_large_faults_show_signature(cause, mag, expect):
    assert expect in EXP.cause(cause).signatures
    rng = np.random.default_rng(2)
    hits = sum(expect in compute_panel(EXP.linearise(EXP.simulate(rng, cause, magnitude=mag))).signatures
               for _ in range(20))
    assert hits >= 15


@pytest.mark.parametrize("cause,mag", [
    ("incline_friction.release_scatter", 0.3),
    ("incline_friction.flag_width", 0.06),  # 斜率偏大 12%：超出量角器偏差能造成的范围
])
def test_engine_ranks_large_fault_top(cause, mag):
    rng = np.random.default_rng(3)
    tops = sum(posterior(EXP, EXP.simulate(rng, cause, magnitude=mag)).top() == cause for _ in range(20))
    assert tops >= 14


def test_random_only_wins_on_clean_data():
    rng = np.random.default_rng(4)
    wins = 0
    for _ in range(20):
        post = posterior(EXP, EXP.simulate(rng, "incline_friction.random_only"))
        assert post.probs.sum() == pytest.approx(1.0)
        wins += post.top() == "incline_friction.random_only"
    assert wins >= 12


@pytest.mark.parametrize("cause,mag,item", [
    ("incline_friction.gate_spacing", 2.0, "remeasure_spacing"),  # 与 μ 随压力变化同为斜率偏小
    ("incline_friction.mu_load", 0.15, "heavy_block"),  # f = μN 时 a 与质量无关
    ("incline_friction.protractor_zero", -3.0, "measure_height"),  # h = L sin(θ + δ)
    ("incline_friction.flag_width", -0.04, "caliper_flag"),
])
def test_menu_item_separates_confounded_pair(cause, mag, item):
    rng = np.random.default_rng(5)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, cause, magnitude=mag)
        before.append(posterior(EXP, ds).prob(cause))
        after.append(posterior(EXP, EXP.extend(rng, ds, EXP.menu_item(item).plan)).prob(cause))
    assert np.mean(after) > np.mean(before) + 0.3
    assert np.mean(after) > 0.9


def test_posterior_speed():
    rng = np.random.default_rng(6)
    for cause in EXP.cause_ids():
        ds = EXP.simulate(rng, cause)
        t0 = time.perf_counter()
        posterior(EXP, ds)
        assert time.perf_counter() - t0 < 0.5


def test_ticker_tape_variant():
    tape = InclineFriction(timer="ticker_tape")
    assert "incline_friction.tape_drag" in tape.cause_ids()
    assert "incline_friction.flag_width" not in tape.cause_ids()
    assert all(k in tape.obs_kinds for m in tape.menu for k, _ in m.plan)
    # 阻力 f/m 使 a 小 f/m；质量加倍时影响减半
    a0 = _main_mean(tape, "incline_friction.random_only", 40.0, 0.0, 0.3)
    assert _main_mean(tape, "incline_friction.tape_drag", 40.0, 0.4, 0.3) == pytest.approx(a0 - 0.4)
    assert _main_mean(tape, "incline_friction.tape_drag", 40.0, 0.4, 0.3, "heavy") == pytest.approx(a0 - 0.2)
    rng = np.random.default_rng(7)
    hits = sum("slope_low" in compute_panel(tape.linearise(
        tape.simulate(rng, "incline_friction.tape_drag", magnitude=0.5))).signatures for _ in range(20))
    assert hits >= 15
    with pytest.raises(ValueError):
        InclineFriction(timer="stopwatch")


def test_menu_and_protocols_wellformed():
    assert EXP.causes[0].id == "incline_friction.random_only"
    assert 4 <= len(EXP.causes) - 1 <= 6
    assert {"standard", "single"} <= set(EXP.protocols)
    for c in EXP.causes:
        assert c.id.startswith("incline_friction.") and c.source == "assumption" and c.fix_zh
    for m in EXP.menu:
        assert all(k in EXP.obs_kinds for k, _ in m.plan)
    # 最小角下能由静止滑下：tan 30° 比 μ 先验 +3σ 大 25% 以上
    mu = EXP.linear[0]
    assert math.tan(math.radians(30.0)) > 1.25 * (mu.mean + 3 * mu.sd)


def test_card_matches_module():
    card = json.loads((Path(__file__).resolve().parents[1]
                       / "lab_detective/knowledge/data/cards/incline_friction.json").read_text(encoding="utf-8"))
    assert card["id"] == card["labsim_id"] == "incline_friction"
    assert card["grade"] == "高中" and card["relation_family"] == "linear"
    assert set(card["common_causes"]) == set(EXP.cause_ids())
    assert all(k.startswith("kp.") for k in card["knowledge_points"])
    assert set(card["yolo_cues"]) <= {"cart", "track", "string", "dynamometer", "wooden_block", "iron_block", "spring"}


def test_dataset_roundtrip():
    from lab_detective.labsim import Dataset
    ds = EXP.simulate(np.random.default_rng(8), "incline_friction.mu_load")
    assert Dataset.from_dict(ds.to_dict()).observations == ds.observations


def test_protractor_cause_matches_height_check():
    """量斜面高度只能查出量角器自身的零点偏差；桌面倾斜时从桌面量的高度仍给出 sin θ_r，
    所以原因描述里不能把桌面倾斜也算进来。"""
    c = EXP.cause("incline_friction.protractor_zero")
    assert "桌面" not in c.description_zh and "table" not in c.description_en
    c0, _ = EXP.predict("height", np.array([30.0]), np.array([0]), {"mag": np.array([[3.0]])}, c.id)
    assert float(c0[0, 0, 0]) == pytest.approx(100.0 * math.sin(math.radians(33.0)))
