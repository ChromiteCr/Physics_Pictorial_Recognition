"""滑动摩擦力实验的解析检查和引擎检查。"""
import json
import time
from pathlib import Path

import numpy as np
import pytest

from lab_detective.engine import posterior
from lab_detective.labsim import get_experiment
from lab_detective.labsim.experiments.friction import Friction
from lab_detective.panel import compute_panel

EXP = get_experiment("friction")
# 标准方案下有面板特征的原因；其余（斜拉、加速、木板倾斜）是纯比例，与 μ 混淆
DETECTABLE = ("friction.random_only", "friction.zero_vertical", "friction.reading_scatter")
UNDETECTABLE = ("friction.pull_angle", "friction.accelerating", "friction.board_tilt")


def test_zero_fault_recovers_mu():
    ds = EXP.simulate(np.random.default_rng(0), "friction.random_only", linear={"mu": 0.3}, noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    # F = 0.3 N 正好落在 0.1 N 刻度上，斜率就是 μ
    assert panel.slope == pytest.approx(0.3, abs=1e-6)
    assert abs(panel.intercept) < 1e-6


def test_zero_vertical_gives_minus_hook_weight_intercept():
    ds = EXP.simulate(np.random.default_rng(1), "friction.zero_vertical",
                      linear={"mu": 0.3, "hook_weight": 0.2}, noise_scale=1e-6)
    panel = compute_panel(EXP.linearise(ds))
    assert panel.intercept == pytest.approx(-0.2, abs=1e-6)
    assert panel.slope == pytest.approx(0.3, abs=1e-6)
    # 水平空载时指针在零刻度线上方 w
    obs = EXP.extend(np.random.default_rng(1), ds, EXP.menu_item("zero_h").plan).observations[-1]
    assert obs.values[0] == pytest.approx(-0.2, abs=1e-6)


def _mean(exp, cause, kind, W, mag, mu):
    params = {"mag": np.array([[mag]]), "mu_pull": np.array([[mu]])}
    c, U = exp.predict(kind, np.array([W]), np.array([0]), params, cause)
    lin = exp.linear_params(cause)
    theta = np.array([{"mu": mu}.get(p.name, 0.0) for p in lin])
    return float((np.broadcast_to(c, (1, 1, 1)) + np.broadcast_to(U, (1, 1, 1, len(lin))) @ theta)[0, 0, 0])


def test_pull_angle_physics():
    """F = μN/(cos α + μ sin α)：α = arctan μ 时最小（√(1+μ²) 倍），2·arctan μ 时回到 μN。"""
    mu, W = 0.3, 4.0
    a_min = np.degrees(np.arctan(mu))
    f_min = _mean(EXP, "friction.pull_angle", "main", W, a_min, mu)
    assert f_min == pytest.approx(mu * W / np.sqrt(1 + mu**2), rel=1e-9)
    for a in (5.0, 10.0, 25.0, 30.0):
        assert _mean(EXP, "friction.pull_angle", "main", W, a, mu) > f_min
    assert _mean(EXP, "friction.pull_angle", "main", W, 2 * a_min, mu) == pytest.approx(mu * W, rel=1e-9)
    assert _mean(EXP, "friction.pull_angle", "main", W, 45.0, mu) > mu * W  # 角度大时反而偏大
    # 固定测力计的方法与拉角无关
    assert _mean(EXP, "friction.pull_angle", "fixed", W, 30.0, mu) == pytest.approx(mu * W)


def test_accelerating_and_tilt_physics():
    mu, W = 0.3, 5.0
    # F = N(μ + a/g)
    assert _mean(EXP, "friction.accelerating", "main", W, 0.98, mu) == pytest.approx(W * (mu + 0.1))
    assert _mean(EXP, "friction.accelerating", "fixed", W, 0.98, mu) == pytest.approx(W * mu)
    # 正反两方向之差 = 2 W sin β，平均 = μ W cos β
    b = np.radians(4.0)
    up = _mean(EXP, "friction.board_tilt", "main", W, 4.0, mu)
    down = _mean(EXP, "friction.board_tilt", "reverse", W, 4.0, mu)
    assert up - down == pytest.approx(2 * W * np.sin(b))
    assert (up + down) / 2 == pytest.approx(mu * W * np.cos(b))


@pytest.mark.parametrize("cause,mag,lin,expect", [
    ("friction.zero_vertical", None, {"hook_weight": 0.3}, "intercept_neg"),
    ("friction.reading_scatter", 0.35, None, "scatter_excess"),
])
def test_large_faults_show_signature(cause, mag, lin, expect):
    assert expect in EXP.cause(cause).signatures
    rng = np.random.default_rng(2)
    hits = sum(expect in compute_panel(EXP.linearise(EXP.simulate(rng, cause, magnitude=mag, linear=lin))).signatures
               for _ in range(20))
    assert hits >= 15


@pytest.mark.parametrize("cause", UNDETECTABLE)
def test_pure_scale_causes_declared_undetectable(cause):
    """斜拉、加速、木板倾斜都让 F 仍与 N 成正比、过原点，只改变比例；
    斜率 μ 自由时标准方案的面板看不出（所以不做特征检验），只能靠菜单里的追加测量区分。"""
    assert EXP.cause(cause).signatures == ("none",)
    assert EXP.cause(cause).mechanism == "y_scale"


@pytest.mark.parametrize("cause,mag,lin", [
    ("friction.zero_vertical", None, {"hook_weight": 0.3}),
    ("friction.reading_scatter", 0.35, None),
])
def test_engine_ranks_large_detectable_fault_top(cause, mag, lin):
    rng = np.random.default_rng(3)
    tops = sum(posterior(EXP, EXP.simulate(rng, cause, magnitude=mag, linear=lin)).top() == cause
               for _ in range(20))
    assert tops >= 14


def test_random_only_on_clean_data():
    """干净数据：在有特征的原因里"只有随机误差"应占优；纯比例原因与它打平是物理上的不可辨识，不算错。"""
    rng = np.random.default_rng(4)
    wins, group_wins = 0, 0
    for _ in range(20):
        post = posterior(EXP, EXP.simulate(rng, "friction.random_only"))
        assert post.probs.sum() == pytest.approx(1.0)
        p = dict(zip(post.cause_ids, post.probs))
        wins += max(DETECTABLE, key=p.get) == "friction.random_only"
        group = p["friction.random_only"] + sum(p[c] for c in UNDETECTABLE)
        group_wins += group > max(p["friction.zero_vertical"], p["friction.reading_scatter"])
    assert wins >= 12
    assert group_wins >= 12


@pytest.mark.parametrize("cause,mag,item", [
    ("friction.accelerating", 0.6, "fixed_scale"),  # 木块静止，示数与木板快慢无关
    ("friction.board_tilt", 4.0, "reverse_pull"),  # 掉头后重力分力反向
])
def test_menu_item_separates_confounded_pair(cause, mag, item):
    rng = np.random.default_rng(5)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, cause, magnitude=mag)
        before.append(posterior(EXP, ds).prob(cause))
        after.append(posterior(EXP, EXP.extend(rng, ds, EXP.menu_item(item).plan)).prob(cause))
    assert np.mean(after) > np.mean(before) + 0.3
    assert np.mean(after) > 0.8


def test_horizontal_zero_check_confirms_zero_vertical():
    rng = np.random.default_rng(6)
    before, after = [], []
    for _ in range(20):
        ds = EXP.simulate(rng, "friction.zero_vertical", linear={"hook_weight": 0.15})
        before.append(posterior(EXP, ds).prob("friction.zero_vertical"))
        ds2 = EXP.extend(rng, ds, EXP.menu_item("zero_h").plan)
        after.append(posterior(EXP, ds2).prob("friction.zero_vertical"))
    assert np.mean(after) > np.mean(before) + 0.2


def test_posterior_speed():
    rng = np.random.default_rng(7)
    for cause in EXP.cause_ids():
        ds = EXP.simulate(rng, cause)
        t0 = time.perf_counter()
        posterior(EXP, ds)
        assert time.perf_counter() - t0 < 0.5


def test_instrument_option_and_registry_default():
    assert EXP.obs_kinds["main"].readings[0].resolution == 0.1
    coarse = Friction(scale_division=0.2)
    assert coarse.obs_kinds["main"].readings[0].resolution == 0.2
    ds = coarse.simulate(np.random.default_rng(8), "friction.random_only")
    assert all(abs(v / 0.2 - round(v / 0.2)) < 1e-9 for o in ds.observations for v in o.values)


def test_menu_and_protocols_wellformed():
    assert EXP.causes[0].id == "friction.random_only"
    assert 4 <= len(EXP.causes) - 1 <= 6
    assert {"standard", "single"} <= set(EXP.protocols)
    for c in EXP.causes:
        assert c.id.startswith("friction.") and c.source == "assumption" and c.fix_zh
    for m in EXP.menu:
        assert all(k in EXP.obs_kinds for k, _ in m.plan)


def test_card_matches_module():
    card = json.loads((Path(__file__).resolve().parents[1]
                       / "lab_detective/knowledge/data/cards/friction.json").read_text(encoding="utf-8"))
    assert card["id"] == card["labsim_id"] == "friction"
    assert card["grade"] == "初中" and card["relation_family"] == "proportional"
    assert set(card["common_causes"]) <= set(EXP.cause_ids())
    assert all(k.startswith("kp.") for k in card["knowledge_points"])
    assert set(card["yolo_cues"]) <= {"cart", "track", "string", "dynamometer", "wooden_block", "iron_block", "spring"}
    for v in card["variables"]:
        assert v["role"] in ("control", "measured", "derived")


def test_dataset_roundtrip():
    from lab_detective.labsim import Dataset
    ds = EXP.simulate(np.random.default_rng(9), "friction.pull_angle")
    ds2 = Dataset.from_dict(ds.to_dict())
    assert ds2.observations == ds.observations
    assert ds.truth["nuisance"]["mu_pull"] == ds.truth["linear"]["mu"]
