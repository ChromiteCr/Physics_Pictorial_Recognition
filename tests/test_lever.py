"""杠杆平衡条件（留出、远迁移）的解析检查、面板特征检查和引擎检查。"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from lab_detective.engine import posterior
from lab_detective.labsim import get_experiment
from lab_detective.labsim.experiments.lever import Lever
from lab_detective.panel import compute_panel

EXP = get_experiment("lever")
CARD = Path(__file__).resolve().parents[1] / "lab_detective/knowledge/data/cards/lever.json"
TAU = 4 * 0.05 * 9.8 * 0.10  # F2·L2 = 0.196 N·m


def test_registered_as_far_held_out():
    assert EXP.held_out and EXP.transfer == "far"
    assert EXP.causes[0].id == "lever.random_only"
    assert 4 <= len(EXP.causes) - 1 <= 6
    assert all(c.source == "assumption" for c in EXP.causes)


def test_zero_fault_recovers_lever_law():
    rng = np.random.default_rng(0)
    ds = EXP.simulate(rng, "lever.random_only", linear={"load_factor": 1.0}, noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    # 斜率 = F2·L2 = 0.196 N·m；取整到 0.1 N 带来约 1% 的偏差
    assert panel.slope == pytest.approx(TAU, rel=0.03)
    assert abs(panel.intercept) < 0.1
    assert EXP.linearise(ds).ideal_slope == pytest.approx(TAU)


def test_pivot_offset_matches_derivation():
    """F1 = F2·(L2 − p)/(L1 + p)：p = 1 cm 时 L1 = 5 cm 处 F1 = 1.96 × 9/6 = 2.94 N。"""
    params = {"mag": np.array([[1.0]])}
    c, U = EXP.predict("main", np.array([5.0, 25.0]), np.array([0, 1]), params, "lever.pivot_offset")
    F = (c + U @ np.array([1.0]))[0, :, 0]
    assert F == pytest.approx([1.96 * 9 / 6, 1.96 * 9 / 26])


def test_scale_angle_matches_derivation():
    params = {"mag": np.array([[30.0]])}
    c, U = EXP.predict("main", np.array([10.0]), np.array([0]), params, "lever.scale_angle")
    assert (c + U @ np.array([1.0]))[0, 0, 0] == pytest.approx(1.96 / math.cos(math.radians(30)))


@pytest.mark.parametrize("cause,kw,expect", [
    ("lever.unbalanced", {"linear": {"offset_torque": 0.03}}, "slope_high"),
    ("lever.unbalanced", {"linear": {"offset_torque": -0.03}}, "slope_low"),
    ("lever.pivot_offset", {"magnitude": 1.0}, "slope_low"),
    ("lever.pivot_offset", {"magnitude": -1.0}, "slope_high"),
    ("lever.scale_angle", {"magnitude": 30.0}, "slope_high"),
    ("lever.pivot_friction", {"magnitude": 0.35}, "scatter_excess"),
    ("lever.scale_zero", {"linear": {"zero_shift": -0.3}}, "intercept_neg"),
    ("lever.scale_zero", {"linear": {"zero_shift": 0.3}}, "intercept_pos"),
])
def test_large_faults_show_signature(cause, kw, expect):
    assert expect in EXP.cause(cause).signatures
    rng = np.random.default_rng(2)
    hits = sum(expect in compute_panel(EXP.linearise(EXP.simulate(rng, cause, **kw))).signatures
               for _ in range(20))
    assert hits >= 15


@pytest.mark.parametrize("cause,kw", [
    ("lever.pivot_offset", {"magnitude": 1.0}),
    ("lever.pivot_friction", {"magnitude": 0.35}),
    ("lever.scale_zero", {"linear": {"zero_shift": -0.3}}),
])
def test_engine_ranks_large_detectable_fault_top(cause, kw):
    rng = np.random.default_rng(3)
    ok = sum(posterior(EXP, EXP.simulate(rng, cause, **kw)).top() == cause for _ in range(20))
    assert ok >= 14


def test_posterior_normalised_and_random_only_wins_on_clean_data():
    rng = np.random.default_rng(5)
    wins = 0
    for _ in range(30):
        post = posterior(EXP, EXP.simulate(rng, "lever.random_only"))
        assert post.probs.sum() == pytest.approx(1.0)
        wins += post.top() == "lever.random_only"
    assert wins >= 18


def test_moving_the_load_separates_angle_from_imbalance():
    """斜拉（各点同乘 1/cosα）和没调平（加恒定力矩 τ0）在标准方案下都只让斜率变大、分不开；
    把钩码移到 20 cm 后阻力矩加倍：斜拉的偏差跟着加倍，没调平的偏差不变。"""
    rng = np.random.default_rng(4)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, "lever.scale_angle", magnitude=20.0)
        before.append(posterior(EXP, ds).prob("lever.scale_angle"))
        ds2 = EXP.extend(rng, ds, EXP.menu_item("move_load").plan)
        after.append(posterior(EXP, ds2).prob("lever.scale_angle"))
    assert np.mean(after) > np.mean(before) + 0.15
    assert np.mean(after) > 0.7


def test_balance_check_identifies_unbalanced_lever():
    rng = np.random.default_rng(6)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, "lever.unbalanced", linear={"offset_torque": 0.013})
        before.append(posterior(EXP, ds).prob("lever.unbalanced"))
        ds2 = EXP.extend(rng, ds, EXP.menu_item("balance_check").plan)
        after.append(posterior(EXP, ds2).prob("lever.unbalanced"))
    assert np.mean(after) > np.mean(before) + 0.3


def test_posterior_is_fast():
    ds = EXP.simulate(np.random.default_rng(7), "lever.pivot_friction")
    t0 = time.perf_counter()
    posterior(EXP, ds)
    assert time.perf_counter() - t0 < 0.5


def test_instrument_is_configurable():
    coarse = Lever(scale_division=0.2)
    assert coarse.obs_kinds["main"].readings[0].resolution == 0.2
    assert EXP.obs_kinds["main"].readings[0].resolution == 0.1  # 不影响默认实例
    ds = coarse.simulate(np.random.default_rng(8), "lever.random_only")
    assert all(abs(v / 0.2 - round(v / 0.2)) < 1e-6 for o in ds.observations for v in o.values)
    with pytest.raises(ValueError):
        Lever(n_masses=6)  # 5 cm 处要 5.9 N，超量程


def test_card_matches_module():
    card = json.loads(CARD.read_text(encoding="utf-8"))
    assert card["id"] == card["labsim_id"] == "lever"
    assert card["common_causes"] == list(EXP.cause_ids())
    assert card["relation_family"] == "inverse"
    assert set(card["yolo_cues"]) <= {"cart", "track", "string", "dynamometer", "wooden_block",
                                      "iron_block", "spring"}
    assert all(k.startswith("kp.") for k in card["knowledge_points"])
