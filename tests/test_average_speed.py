"""测量平均速度（小车沿斜面下滑）的解析检查、特征检查和引擎检查。"""
import json
import time
from pathlib import Path

import numpy as np
import pytest

from lab_detective.engine import posterior
from lab_detective.labsim import Dataset, get_experiment
from lab_detective.labsim.experiments.average_speed import AverageSpeed
from lab_detective.panel import compute_panel

EXP = get_experiment("average_speed")
CARD = Path(__file__).resolve().parents[1] / "lab_detective/knowledge/data/cards/average_speed.json"

# "大"故障：反应时间 0.35 s、车长 15 cm、等效助跑 50 cm、额外计时误差 0.35 s
LARGE = {
    "average_speed.stop_lag": dict(linear={"stop_lag": 0.35}),
    "average_speed.start_late": dict(linear={"start_lag": 0.35}),
    "average_speed.start_early": dict(linear={"start_lead": 0.35}),
    "average_speed.cart_length": dict(magnitude=15.0),
    "average_speed.pushed": dict(magnitude=50.0),
    "average_speed.rushed_timing": dict(magnitude=0.35),
}


def test_ids_and_structure():
    assert EXP.id == "average_speed" and not EXP.held_out
    assert EXP.cause_ids()[0] == "average_speed.random_only"
    assert 4 <= len(EXP.causes) - 1 <= 6
    assert all(c.source == "assumption" for c in EXP.causes)
    assert 3 <= len(EXP.menu) <= 6
    assert {"standard", "single"} <= set(EXP.protocols)


def test_zero_fault_recovers_acceleration():
    rng = np.random.default_rng(0)
    ds = EXP.simulate(rng, "average_speed.random_only", linear={"beta": 2.0}, noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    # t = β√s，β = √(2/a) = 2.0 → a = 0.5 m/s²；停表取整到 0.01 s
    assert panel.slope == pytest.approx(2.0, rel=0.01)
    assert abs(panel.intercept) < 0.01
    assert 2.0 / panel.slope**2 == pytest.approx(0.5, rel=0.02)


def test_pushed_matches_run_up_formula():
    """推一下 ≡ 从更高 d0 处静止出发：t = β(√(s+d0) − √d0)。"""
    rng = np.random.default_rng(1)
    ds = EXP.simulate(rng, "average_speed.pushed", magnitude=20.0, linear={"beta": 2.0}, noise_scale=1e-6)
    s = np.array([o.setting for o in ds.observations]) / 100
    t = np.array([o.values[0] for o in ds.observations])
    assert np.allclose(t, 2.0 * (np.sqrt(s + 0.2) - np.sqrt(0.2)), atol=0.006)


def test_stop_lag_offset_value():
    rng = np.random.default_rng(2)
    ds = EXP.simulate(rng, "average_speed.stop_lag", linear={"beta": 2.0, "stop_lag": 0.25}, noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    assert panel.intercept == pytest.approx(0.25, abs=0.01)
    assert panel.slope == pytest.approx(2.0, rel=0.01)


@pytest.mark.parametrize("cause", sorted(LARGE))
def test_large_faults_show_signature(cause):
    rng = np.random.default_rng(3)
    declared = set(EXP.cause(cause).signatures)
    hits = 0
    for _ in range(20):
        ds = EXP.simulate(rng, cause, **LARGE[cause])
        hits += bool(declared & set(compute_panel(EXP.linearise(ds)).signatures))
    assert hits >= 15


@pytest.mark.parametrize("cause", ["average_speed.pushed", "average_speed.rushed_timing"])
def test_large_fault_ranked_top(cause):
    rng = np.random.default_rng(4)
    ok = sum(posterior(EXP, EXP.simulate(rng, cause, **LARGE[cause])).top() == cause for _ in range(20))
    assert ok >= 14


def test_posterior_normalised_and_random_only_wins_on_clean_data():
    rng = np.random.default_rng(5)
    wins = 0
    for _ in range(20):
        post = posterior(EXP, EXP.simulate(rng, "average_speed.random_only"))
        assert post.probs.sum() == pytest.approx(1.0)
        assert np.all(post.probs >= 0)
        wins += post.top() == "average_speed.random_only"
    assert wins >= 12


def test_solo_timing_separates_stop_lag_from_early_start():
    """"停表滞后"和"先按表后松手"在标准数据里完全同形（都是 t 多出一个常数）；
    让放车的同学自己开表后，开表提前消失、停表滞后仍在，两者分开。"""
    rng = np.random.default_rng(6)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, "average_speed.start_early", linear={"start_lead": 0.25})
        before.append(posterior(EXP, ds).prob("average_speed.start_early"))
        ds2 = EXP.extend(rng, ds, EXP.menu_item("solo_timing").plan)
        after.append(posterior(EXP, ds2).prob("average_speed.start_early"))
    assert np.mean(after) > np.mean(before) + 0.2
    assert np.mean(after) > 0.8


def test_distance_check_separates_cart_length_from_late_start():
    rng = np.random.default_rng(7)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, "average_speed.cart_length", magnitude=8.0)
        before.append(posterior(EXP, ds).prob("average_speed.cart_length"))
        ds2 = EXP.extend(rng, ds, EXP.menu_item("distance_check").plan)
        after.append(posterior(EXP, ds2).prob("average_speed.cart_length"))
    assert np.mean(after) > np.mean(before) + 0.2


def test_posterior_is_fast():
    rng = np.random.default_rng(8)
    ds = EXP.simulate(rng, "average_speed.cart_length")
    posterior(EXP, ds)
    t0 = time.perf_counter()
    for _ in range(5):
        posterior(EXP, ds)
    assert (time.perf_counter() - t0) / 5 < 0.5


def test_photogate_variant_smoke():
    exp = AverageSpeed(timer="photogate")
    assert exp.cause_ids() == ("average_speed.random_only", "average_speed.cart_length", "average_speed.pushed")
    assert all(k in exp.obs_kinds for m in exp.menu for k, _ in m.plan)
    assert exp.obs_kinds["main"].readings[0].resolution == pytest.approx(0.001)
    rng = np.random.default_rng(9)
    ds = exp.simulate(rng, "average_speed.pushed", magnitude=30.0)
    post = posterior(exp, ds)
    assert post.probs.sum() == pytest.approx(1.0)
    assert post.top() == "average_speed.pushed"
    with pytest.raises(ValueError):
        AverageSpeed(timer="sundial")


def test_card_consistent_with_module():
    card = json.loads(CARD.read_text(encoding="utf-8"))
    for key in ("id", "name_zh", "name_en", "grade", "goal", "principle", "formula", "variables",
                "apparatus", "steps", "data_table", "safety", "common_causes", "knowledge_points",
                "relation_family", "labsim_id", "yolo_cues"):
        assert key in card
    assert card["id"] == card["labsim_id"] == EXP.id
    assert card["grade"] == "初中"
    assert card["relation_family"] in ("proportional", "linear", "inverse", "square", "sqrt", "power")
    assert set(card["common_causes"]) == set(EXP.cause_ids())
    assert all(v["role"] in ("control", "measured", "derived") for v in card["variables"])
    assert all(k.startswith("kp.") for k in card["knowledge_points"])
    assert set(card["yolo_cues"]) <= {"cart", "track", "string", "dynamometer", "wooden_block", "iron_block", "spring"}


def test_dataset_roundtrip():
    ds = EXP.simulate(np.random.default_rng(10), "average_speed.stop_lag")
    assert Dataset.from_dict(ds.to_dict()).observations == ds.observations
