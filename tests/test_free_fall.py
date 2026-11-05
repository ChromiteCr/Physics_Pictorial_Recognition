"""自由落体（光电门测下落时间）的解析检查、特征检查和引擎检查。"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from lab_detective.engine import posterior
from lab_detective.labsim import G, Dataset, get_experiment
from lab_detective.labsim.experiments.free_fall import BETA0, FreeFall, _drag_factor
from lab_detective.panel import compute_panel

EXP = get_experiment("free_fall")
CARD = Path(__file__).resolve().parents[1] / "lab_detective/knowledge/data/cards/free_fall.json"

# "大"故障：延迟 20 ms、多记 2.5 cm、光束低 1.8 cm、乒乓球级阻力、高度误差 1.4 cm
LARGE = {
    "free_fall.release_delay": dict(linear={"release_delay": 0.02}),
    "free_fall.height_to_top": dict(magnitude=2.5),
    "free_fall.gate_mark": dict(magnitude=1.8),
    "free_fall.air_drag": dict(magnitude=0.15),
    "free_fall.gate_scatter": dict(magnitude=1.4),
}


def test_ids_and_structure():
    assert EXP.id == "free_fall" and EXP.held_out and EXP.transfer == "near"
    assert EXP.cause_ids()[0] == "free_fall.random_only"
    assert 4 <= len(EXP.causes) - 1 <= 6
    assert all(c.source == "assumption" for c in EXP.causes)
    assert 3 <= len(EXP.menu) <= 6
    assert {"standard", "single"} <= set(EXP.protocols)


def test_zero_fault_recovers_g():
    rng = np.random.default_rng(0)
    ds = EXP.simulate(rng, "free_fall.random_only", linear={"beta": BETA0}, noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    assert panel.slope_ratio == pytest.approx(1.0, abs=1e-3)
    assert 2.0 / panel.slope**2 == pytest.approx(G, rel=2e-3)
    assert abs(panel.intercept) < 2e-4


def test_drag_factor_limits():
    """阻力因子：κ→0 时为 1；小 κh 时 ≈ (1 + κh/6)(1 + ε/2)；与数值积分一致。"""
    h = np.array([0.2, 1.0])
    assert np.allclose(_drag_factor(np.array(1e-7), h), 1.0, atol=1e-6)
    k = 0.01
    approx = (1 + k * h / 6) * (1 + 0.5 * 8 * 0.02 / (3 * 0.47) * k)
    assert np.allclose(_drag_factor(np.array(k), h), approx, rtol=1e-4)
    # 数值积分 dv/dt = g' − κv²，到下落 1 m 的时间
    kappa, eps = 0.15, 8 * 0.02 / (3 * 0.47) * 0.15
    gp = G * (1 - eps)
    dt, v, y, t = 1e-5, 0.0, 0.0, 0.0
    while y < 1.0:
        a = gp - kappa * v * v
        v += a * dt
        y += v * dt
        t += dt
    assert t == pytest.approx(math.sqrt(2 / G) * _drag_factor(np.array(kappa), np.array([1.0]))[0], rel=1e-3)


def test_height_to_top_matches_formula():
    rng = np.random.default_rng(1)
    ds = EXP.simulate(rng, "free_fall.height_to_top", magnitude=2.0, linear={"beta": BETA0}, noise_scale=1e-6)
    h = np.array([o.setting for o in ds.observations]) / 100
    t = np.array([o.values[0] for o in ds.observations])
    assert np.allclose(t, BETA0 * np.sqrt(h - 0.02), atol=1.5e-4)


@pytest.mark.parametrize("cause", sorted(LARGE))
def test_large_faults_show_signature(cause):
    rng = np.random.default_rng(2)
    declared = set(EXP.cause(cause).signatures)
    hits = 0
    for _ in range(20):
        ds = EXP.simulate(rng, cause, **LARGE[cause])
        hits += bool(declared & set(compute_panel(EXP.linearise(ds)).signatures))
    assert hits >= 15


@pytest.mark.parametrize("cause", ["free_fall.release_delay", "free_fall.air_drag", "free_fall.height_to_top"])
def test_large_fault_ranked_top(cause):
    rng = np.random.default_rng(3)
    ok = sum(posterior(EXP, EXP.simulate(rng, cause, **LARGE[cause])).top() == cause for _ in range(20))
    assert ok >= 14


def test_posterior_normalised_and_random_only_wins_on_clean_data():
    rng = np.random.default_rng(4)
    wins = 0
    for _ in range(20):
        post = posterior(EXP, EXP.simulate(rng, "free_fall.random_only"))
        assert post.probs.sum() == pytest.approx(1.0)
        assert np.all(post.probs >= 0)
        wins += post.top() == "free_fall.random_only"
    assert wins >= 12


def test_height_check_separates_small_gate_mark_from_random():
    """光束只比读数位置低 3.5 mm 时，标准数据和"只有随机误差"难分；重量一次高度就分开了。"""
    rng = np.random.default_rng(5)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, "free_fall.gate_mark", magnitude=0.35)
        before.append(posterior(EXP, ds).prob("free_fall.gate_mark"))
        ds2 = EXP.extend(rng, ds, EXP.menu_item("height_check").plan)
        after.append(posterior(EXP, ds2).prob("free_fall.gate_mark"))
    assert np.mean(after) > np.mean(before) + 0.15
    assert np.mean(after) > 0.9


def test_gate_scatter_height_check_sigma_in_cm():
    """光电门位置读数散布：重量高度时门要重新放回、又读错一次，差值的标准差 = σ_h（cm，不换算成时间）。"""
    params = {"mag": np.array([[1.0]])}
    sd = FreeFall().reading_sigma("height_check", params, "free_fall.gate_scatter")
    assert np.ravel(sd)[0] == pytest.approx(math.sqrt(0.05**2 + 1.0**2))
    sd0 = FreeFall().reading_sigma("height_check", params, "free_fall.random_only")
    assert np.ravel(sd0)[0] == pytest.approx(0.05)


def test_gate_scatter_survives_realistic_height_check():
    """回归：真值是"光电门位置每次都读得不准"时，学生重量 60 cm 这组得到的差值是 N(0, σ_h)（手工注入，
    不用 extend，免得模拟器和引擎同错同对）。修正前这一步会把约四成诊断推向"量错起点/读在门框上沿"。"""
    from lab_detective.labsim import Observation

    rng = np.random.default_rng(12)
    before, after, top = [], [], 0
    for _ in range(20):
        ds = EXP.simulate(rng, "free_fall.gate_scatter", magnitude=1.0)
        before.append(posterior(EXP, ds).prob("free_fall.gate_scatter"))
        diff = round(rng.normal(0.0, 1.0) + rng.normal(0.0, 0.05), 1)
        hc = Observation("height_check", 60.0, (diff,))
        post = posterior(EXP, Dataset(ds.experiment_id, ds.observations + [hc], ds.truth, ds.meta))
        after.append(post.prob("free_fall.gate_scatter"))
        top += post.top() == "free_fall.gate_scatter"
    assert top >= 17
    assert np.mean(after) > np.mean(before) - 0.1


def test_paper_shim_removes_delay_only_for_magnet_cause():
    rng = np.random.default_rng(6)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, "free_fall.release_delay", linear={"release_delay": 0.004})
        before.append(posterior(EXP, ds).prob("free_fall.release_delay"))
        ds2 = EXP.extend(rng, ds, EXP.menu_item("paper_shim").plan)
        after.append(posterior(EXP, ds2).prob("free_fall.release_delay"))
    assert np.mean(after) > np.mean(before) + 0.05
    assert np.mean(after) > 0.95


def test_posterior_is_fast():
    rng = np.random.default_rng(7)
    ds = EXP.simulate(rng, "free_fall.air_drag")
    posterior(EXP, ds)
    t0 = time.perf_counter()
    for _ in range(5):
        posterior(EXP, ds)
    assert (time.perf_counter() - t0) / 5 < 0.5


def test_two_gate_variant_smoke():
    exp = FreeFall(start="gate")
    assert "free_fall.release_speed" in exp.cause_ids()
    assert "free_fall.release_delay" not in exp.cause_ids()
    assert exp.cause_ids()[0] == "free_fall.random_only"
    assert all(k in exp.obs_kinds for m in exp.menu for k, _ in m.plan)
    rng = np.random.default_rng(8)
    ds = exp.simulate(rng, "free_fall.release_speed", magnitude=1.0, linear={"beta": BETA0}, noise_scale=1e-6)
    h = np.array([o.setting for o in ds.observations]) / 100
    t = np.array([o.values[0] for o in ds.observations])
    assert np.allclose(t, BETA0 * (np.sqrt(h + 0.01) - 0.1), atol=1.5e-4)
    post = posterior(exp, exp.simulate(rng, "free_fall.release_speed", magnitude=0.5))
    assert post.probs.sum() == pytest.approx(1.0)
    assert post.top() == "free_fall.release_speed"
    with pytest.raises(ValueError):
        FreeFall(start="hand")


def test_card_consistent_with_module():
    card = json.loads(CARD.read_text(encoding="utf-8"))
    for key in ("id", "name_zh", "name_en", "grade", "goal", "principle", "formula", "variables",
                "apparatus", "steps", "data_table", "safety", "common_causes", "knowledge_points",
                "relation_family", "labsim_id", "yolo_cues"):
        assert key in card
    assert card["id"] == card["labsim_id"] == EXP.id
    assert card["grade"] == "高中"
    assert card["relation_family"] in ("proportional", "linear", "inverse", "square", "sqrt", "power")
    assert set(card["common_causes"]) == set(EXP.cause_ids())
    assert all(v["role"] in ("control", "measured", "derived") for v in card["variables"])
    assert all(k.startswith("kp.") for k in card["knowledge_points"])


def test_dataset_roundtrip():
    ds = EXP.simulate(np.random.default_rng(9), "free_fall.gate_mark")
    assert Dataset.from_dict(ds.to_dict()).observations == ds.observations
