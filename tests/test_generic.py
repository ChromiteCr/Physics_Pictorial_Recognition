"""通用关系族（"其他实验"通道）和程序化任务族的检查。"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from lab_detective.engine import posterior
from lab_detective.labsim import EXPERIMENTS
from lab_detective.labsim.generic import (
    FAMILY_EXPONENT,
    GenericExperiment,
    law_keys,
    procedural_task_family,
    spec_from_card,
)
from lab_detective.panel import compute_panel

pytestmark = pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")

G = 9.8
# 单摆走"其他实验"通道（E5 真实实验）：T = 2π√(L/g)，sqrt 族；只在测试里用，不进程序化任务族
PEND = {"id": "pend", "name": "单摆测重力加速度", "x_var": {"name": "摆长", "symbol": "L", "tool": "刻度尺"},
        "y_var": {"name": "周期", "symbol": "T", "tool": "手机计时器"}, "units": {"x": "m", "y": "s"},
        "resolutions": {"x": 0.001, "y": 0.01}, "sigma": {"y": 0.02}, "relation_family": "sqrt",
        "ideal_slope": None, "slope_guess": 2.0, "x_range": [0.3, 1.0], "n_points": 6}
PEND_K = dict(PEND, id="pendk", ideal_slope=2 * math.pi / math.sqrt(G))
OHM = {"id": "ohm", "name": "探究电流与电压的关系", "x_var": {"name": "电压", "tool": "电压表"},
       "y_var": {"name": "电流", "tool": "电流表"}, "units": {"x": "V", "y": "A"},
       "resolutions": {"x": 0.1, "y": 0.02}, "sigma": {"y": 0.005}, "relation_family": "proportional",
       "ideal_slope": 0.1, "x_range": [0.5, 3.0], "n_points": 6}
HEAT = {"id": "heat", "name": "水温与加热时间", "x_var": "加热时间", "y_var": {"name": "水温", "tool": "温度计"},
        "units": {"x": "min", "y": "°C"}, "resolutions": {"x": 0.1, "y": 0.5}, "sigma": {"y": 0.3},
        "relation_family": "linear", "slope_guess": 3.0, "intercept_guess": 20.0, "intercept_sd": 5.0,
        "x_range": [1.0, 6.0]}
SPECS = {s["id"]: GenericExperiment(s) for s in (PEND, PEND_K, OHM, HEAT)}


def _large(exp, cid, sign):
    kw = exp.large[cid]
    if "magnitude" in kw:
        return {"magnitude": sign * kw["magnitude"]}
    return {"linear": {k: sign * v for k, v in kw["linear"].items()}}


def test_generic_is_not_auto_registered():
    assert GenericExperiment.id == ""
    assert not any(isinstance(e, GenericExperiment) for e in EXPERIMENTS.values())


def test_zero_fault_recovers_g_from_pendulum():
    exp = SPECS["pend"]
    ds = exp.simulate(np.random.default_rng(0), "pend.random_only",
                      linear={"slope": 2 * math.pi / math.sqrt(G)}, noise_scale=1e-6)
    panel = compute_panel(exp.linearise(ds))
    g = (2 * math.pi / panel.slope) ** 2  # T = (2π/√g)·√L
    assert g == pytest.approx(G, rel=0.02)
    assert abs(panel.intercept) < 0.02


def test_zero_fault_recovers_known_slope():
    exp = SPECS["ohm"]
    ds = exp.simulate(np.random.default_rng(1), "ohm.random_only", linear={"slope": 0.1}, noise_scale=1e-6)
    panel = compute_panel(exp.linearise(ds))
    # 电流表分度 0.02 A，读数 0.05~0.3 A，取整误差可达几个百分点，但在面板的不确定度之内
    assert panel.slope == pytest.approx(0.1, rel=0.1)
    assert abs(panel.slope_ratio_z) < 2 and abs(panel.intercept_z) < 2


def test_unidentifiable_causes_are_omitted():
    """截距未知时整体平移被截距吸收；斜率未知时比例误差被斜率吸收；linear 族时间零点也被截距吸收。"""
    heat = SPECS["heat"]
    assert set(heat.omitted) == {"x_offset", "y_offset", "scale"}
    assert np.isnan(heat.linearise(heat.simulate(np.random.default_rng(2), "heat.random_only")).ideal_intercept)
    assert "scale" in SPECS["pend"].omitted and "pendk.scale" in SPECS["pendk"].cause_ids()
    for exp in SPECS.values():
        assert exp.causes[0].id == exp.random_only_id
        assert all(c.id.startswith(exp.id + ".") for c in exp.causes)


CASES = [(sid, c.id, sign) for sid, e in SPECS.items() for c in e.causes[1:] for sign in e.fault_signs[c.id]
         if e.signature_by_sign[c.id][sign] != ("none",)]


@pytest.mark.parametrize("sid,cid,sign", CASES)
def test_large_faults_show_declared_signature(sid, cid, sign):
    exp = SPECS[sid]
    expect = exp.signature_by_sign[cid][sign]
    assert set(expect) <= set(exp.cause(cid).signatures)
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(20):
        sigs = compute_panel(exp.linearise(exp.simulate(rng, cid, **_large(exp, cid, sign)))).signatures
        hits += any(s in sigs for s in expect)
    assert hits >= 15


@pytest.mark.parametrize("sid,cid", [("ohm", "ohm.scale"), ("ohm", "ohm.bend"), ("pend", "pend.drift"),
                                     ("pend", "pend.scatter"), ("heat", "heat.bend"), ("pendk", "pendk.x_offset")])
def test_engine_ranks_large_detectable_fault_top(sid, cid):
    exp = SPECS[sid]
    rng = np.random.default_rng(4)
    ok = sum(posterior(exp, exp.simulate(rng, cid, **_large(exp, cid, 1))).top() == cid for _ in range(20))
    assert ok >= 14


@pytest.mark.parametrize("sid", list(SPECS))
def test_random_only_wins_on_clean_data(sid):
    exp = SPECS[sid]
    rng = np.random.default_rng(5)
    wins = 0
    for _ in range(30):
        post = posterior(exp, exp.simulate(rng, exp.random_only_id))
        assert post.probs.sum() == pytest.approx(1.0)
        wins += post.top() == exp.random_only_id
    assert wins >= 18


def test_zero_check_separates_x_offset_from_y_offset():
    """正比关系里 x 的零点错和 y 仪器没归零在主数据上完全一样（都是截距）；单独读一次 y 仪器的空载示数就分开了。"""
    exp = SPECS["ohm"]
    rng = np.random.default_rng(6)
    before, after = [], []
    for _ in range(20):
        ds = exp.simulate(rng, "ohm.y_offset", **_large(exp, "ohm.y_offset", 1))
        before.append(posterior(exp, ds).prob("ohm.y_offset"))
        ds2 = exp.extend(rng, ds, exp.menu_item("y_zero_check").plan)
        after.append(posterior(exp, ds2).prob("ohm.y_offset"))
    assert np.mean(after) > np.mean(before) + 0.3
    assert np.mean(after) > 0.8


def test_pendulum_length_to_bob_top_is_found_by_rechecking_origin():
    """摆长量到摆球顶端（少了半径）= x 起点偏差；重新核对摆长起点后 x_offset 的后验上升。"""
    exp = SPECS["pend"]
    rng = np.random.default_rng(7)
    before, after = [], []
    for _ in range(20):
        ds = exp.simulate(rng, "pend.x_offset", magnitude=0.02, linear={"slope": 2 * math.pi / math.sqrt(G)})
        before.append(posterior(exp, ds).prob("pend.x_offset"))
        ds2 = exp.extend(rng, ds, exp.menu_item("x_zero_check").plan)
        after.append(posterior(exp, ds2).prob("pend.x_offset"))
    assert np.mean(after) > np.mean(before) + 0.2


def test_posterior_is_fast():
    for exp in SPECS.values():
        ds = exp.simulate(np.random.default_rng(8), exp.causes[1].id)
        t0 = time.perf_counter()
        posterior(exp, ds)
        assert time.perf_counter() - t0 < 0.5


def test_spec_from_card_builds_other_experiment_path():
    card = json.loads((Path(__file__).resolve().parents[1] /
                       "lab_detective/knowledge/data/cards/lever.json").read_text(encoding="utf-8"))
    spec = spec_from_card(card, x_range=[5.0, 25.0], n_points=5,
                          slope_guess=19.6, resolutions={"x": 0.1, "y": 0.1}, sigma={"y": 0.04})
    exp = GenericExperiment(spec)
    assert exp.q == -1 and exp.xv["symbol"] == "L1" and exp.yv["symbol"] == "F1"
    assert exp.id == "other_lever"
    ds = exp.simulate(np.random.default_rng(9), exp.random_only_id, linear={"slope": 19.6})
    post = posterior(exp, ds)
    assert post.probs.sum() == pytest.approx(1.0)


# ── 程序化任务族 ─────────────────────────────────────────────────────────


def test_procedural_family_is_reproducible_and_varied():
    a = list(procedural_task_family(np.random.default_rng(10), 200))
    b = list(procedural_task_family(np.random.default_rng(10), 200))
    assert [(e.id, c) for e, c in a] == [(e.id, c) for e, c in b]
    assert [e.causes[1].description_zh for e, _ in a] == [e.causes[1].description_zh for e, _ in b]
    assert len({e.id for e, _ in a}) == 200
    assert all(c in e.cause_ids() for e, c in a)
    assert len({e.id.rsplit("_", 1)[0] for e, _ in a}) == len(law_keys())
    descs = {c.description_zh for e, _ in a for c in e.causes}
    assert len(descs) >= 80
    assert all(any("一" <= ch <= "鿿" for ch in c.description_zh) for e, _ in a for c in e.causes)


def test_procedural_tasks_are_diagnosable():
    rng = np.random.default_rng(11)
    wins = n = big_ok = big_n = 0
    for exp, cid in procedural_task_family(rng, 60):
        post = posterior(exp, exp.simulate(rng, exp.random_only_id))
        assert post.probs.sum() == pytest.approx(1.0)
        wins += post.top() == exp.random_only_id
        n += 1
        for c in exp.causes:
            if exp.suffix(c.id) in ("scatter", "drift"):
                sign = exp.fault_signs[c.id][0]
                big_ok += posterior(exp, exp.simulate(rng, c.id, **_large(exp, c.id, sign))).top() == c.id
                big_n += 1
    assert wins >= 0.6 * n
    assert big_ok >= 0.7 * big_n


def _ngrams(text: str, k: int) -> set[str]:
    t = "".join(text.split())
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def _word_ngrams(text: str, k: int) -> set[tuple[str, ...]]:
    w = text.lower().replace(",", " ").replace(".", " ").replace(";", " ").split()
    return {tuple(w[i:i + k]) for i in range(len(w) - k + 1)}


def test_every_card_builds_an_other_path_experiment_without_id_clash():
    """每张实验卡都能走"其他实验"通道；生成的原因 id 绝不和注册的手写孪生撞名。"""
    cards = sorted((Path(__file__).resolve().parents[1] / "lab_detective/knowledge/data/cards").glob("*.json"))
    assert cards
    registered = {c for e in EXPERIMENTS.values() for c in e.cause_ids()}
    for path in cards:
        card = json.loads(path.read_text(encoding="utf-8"))
        try:
            spec = spec_from_card(card, x_range=[1.0, 5.0])
            exp = GenericExperiment(spec)
        except ValueError:
            # 只允许"明确不支持"的情况：关系族不在表里、power 缺指数、卡里缺变量
            assert card["relation_family"] not in FAMILY_EXPONENT or card["relation_family"] == "power" \
                or not any(v["role"] == "measured" for v in card["variables"])
            continue
        assert not set(exp.cause_ids()) & registered, path.name
    for exp, _ in procedural_task_family(np.random.default_rng(13), 50):
        assert not set(exp.cause_ids()) & registered


def test_bad_spec_raises_value_error():
    with pytest.raises(ValueError):
        GenericExperiment({"relation_family": "power"})
    with pytest.raises(ValueError):
        GenericExperiment({"relation_family": "exponential"})


def test_no_wording_overlap_with_named_experiments():
    """程序化任务族用于训练，措辞不能照抄 12 个命名实验（尤其留出实验）的原因描述（中文 8 字、英文 6 词）。"""
    held = [c for e in EXPERIMENTS.values() for c in e.causes]
    assert len(EXPERIMENTS) >= 6
    zh_ref = set().union(*(_ngrams(c.description_zh + "|" + c.fix_zh, 8) for c in held))
    en_ref = set().union(*(_word_ngrams(c.description_en + " | " + c.fix_en, 6) for c in held))
    for exp, _ in procedural_task_family(np.random.default_rng(12), 200):
        for c in exp.causes:
            assert not (_ngrams(c.description_zh, 8) | _ngrams(c.fix_zh, 8)) & zh_ref, c.id
            assert not (_word_ngrams(c.description_en, 6) | _word_ngrams(c.fix_en, 6)) & en_ref, c.id
