"""孪生：预期读数表、"测量前先预测"、幽灵曲线，全部用胡克定律的解析式对照。"""
import json
import math

import numpy as np
import pytest

from lab_detective.engine import posterior
from lab_detective.labsim import get_experiment
from lab_detective.twin import expected_readings, ghost_curves, instantiate, noise_free_dataset, nominal_truth

EXP = get_experiment("hooke")
G = 9.8
SD_EXT = math.sqrt(0.05**2 + 0.1**2 / 12)


def test_predict_table_with_known_k():
    rows = instantiate(EXP, {"inv_k": 0.04}).predict_table("standard")
    assert [r["setting"] for r in rows] == [50, 100, 150, 200, 250]
    last = rows[-1]["readings"][0]
    assert last["mean"] == pytest.approx(100 * 0.04 * 0.25 * G)  # x = (100/k)·F = 9.8 cm
    assert last["sd_reading"] == pytest.approx(SD_EXT)
    assert last["sd_param"] == pytest.approx(0.0, abs=1e-12)
    assert last["lo"] == pytest.approx(last["mean"] - SD_EXT)
    json.dumps(rows, allow_nan=False, ensure_ascii=False)


def test_predict_table_with_k_unknown_propagates_prior():
    last = instantiate(EXP).predict_table()[-1]["readings"][0]
    F = 0.25 * G
    assert last["mean"] == pytest.approx(100 * 0.05 * F)
    assert last["sd_param"] == pytest.approx(100 * F * 0.0115, rel=1e-9)  # 1/k 先验 sd 线性传播
    assert last["sd"] == pytest.approx(math.hypot(100 * F * 0.0115, SD_EXT), rel=1e-9)


def test_unknown_param_rejected():
    with pytest.raises(KeyError):
        instantiate(EXP, {"spring_k": 25.0})


def test_expected_readings_before_measuring():
    r = expected_readings(EXP, "hooke.hanger_omitted", {"hanger_g": 23.0}, "weigh_hanger")[0]["readings"][0]
    assert r["mean"] == pytest.approx(23.0) and r["unit"] == "g"
    assert r["sd"] == pytest.approx(math.sqrt(0.1**2 + 0.1**2 / 12))
    r = expected_readings(EXP, "hooke.zero_offset", {"zero_offset": 0.5}, "zero_check")[0]["readings"][0]
    assert r["mean"] == pytest.approx(0.5)
    r = expected_readings(EXP, "hooke.zero_offset", None, "zero_check")[0]["readings"][0]
    assert r["mean"] == pytest.approx(0.0) and r["sd_param"] == pytest.approx(0.4)
    r = expected_readings(EXP, "hooke.random_only", None, "zero_check")[0]["readings"][0]
    assert r["mean"] == 0.0 and r["sd_param"] == 0.0


def test_expected_readings_average_over_magnitude_prior():
    """超出弹性限度：幅度没给时按先验 U(0.05, 0.5) 抽样；x = (100/k)F·(1 + mag·(F/F_ref)²)。"""
    r = expected_readings(EXP, "hooke.beyond_limit", {"inv_k": 0.04}, "heavy_load")[0]["readings"][0]
    F, Fref = 0.35 * G, 0.25 * G
    base = 100 * 0.04 * F
    assert r["mean"] == pytest.approx(base * (1 + 0.275 * (F / Fref) ** 2), abs=0.6)
    assert r["sd_param"] == pytest.approx(base * (F / Fref) ** 2 * 0.45 / math.sqrt(12), rel=0.12)


def test_noise_free_dataset_matches_observe_up_to_rounding():
    truth = nominal_truth(EXP, "hooke.hanger_omitted", {"hanger_g": 30.0, "inv_k": 0.04})
    plan = EXP.protocols["standard"].plan
    nf = noise_free_dataset(EXP, truth, plan)
    obs = EXP.observe(np.random.default_rng(0), truth, plan, noise_scale=1e-9)
    for a, b in zip(nf.observations, obs):
        assert abs(a.values[0] - b.values[0]) <= 0.05 + 1e-9


def test_ghost_curves_hanger_omitted():
    rng = np.random.default_rng(1)
    ds = EXP.simulate(rng, "hooke.hanger_omitted", linear={"inv_k": 0.04}, nuisance={"hanger_g": 40.0})
    post = posterior(EXP, ds)
    g = ghost_curves(EXP, ds, post, top_k=3, n_grid=201)
    json.dumps(g, allow_nan=False, ensure_ascii=False)
    assert len(g) == 3 and g[0]["prob"] >= g[1]["prob"] >= g[2]["prob"]
    top = g[0]
    assert top["cause_id"] in ("hooke.hanger_omitted", "hooke.zero_offset")  # 标准数据下两者混淆
    x, y = np.array(top["x"]), np.array(top["y"])
    assert x[0] == 0.0 and x[-1] == pytest.approx(0.25 * G)
    assert np.all(np.diff(y) > 0)  # 不取整：稠密网格上没有台阶
    # 截距 = (100/k)·m_h·g = 4 cm/N × 0.392 N = 1.57 cm
    assert y[0] == pytest.approx(4.0 * 0.040 * G, abs=0.12)
    lin = EXP.linearise(ds)
    assert np.max(np.abs(np.interp(lin.x, x, y) - lin.y)) < 0.25


def test_ghost_curve_bends_up_beyond_limit():
    rng = np.random.default_rng(2)
    ds = EXP.simulate(rng, "hooke.beyond_limit", magnitude=0.5)
    g = ghost_curves(EXP, ds, top_k=1)
    assert g[0]["cause_id"] == "hooke.beyond_limit"
    assert np.all(np.diff(np.array(g[0]["y"]), 2) > 0)  # 横轴等距时二阶差分 > 0 即向上弯
