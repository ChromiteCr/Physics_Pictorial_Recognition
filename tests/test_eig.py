"""EIG 顾问与模拟闭环的检查：非负、无信息测量为 0、两原因精确解、胡克定律的物理预期、计时。"""
import json
import math
import time

import numpy as np
import pytest
from scipy import stats

from lab_detective import eig
from lab_detective.engine import posterior
from lab_detective.labsim import get_experiment

EXP = get_experiment("hooke")
R, H, Z = "hooke.random_only", "hooke.hanger_omitted", "hooke.zero_offset"


def test_batch_evidence_matches_engine():
    rng = np.random.default_rng(5)
    dss = [EXP.simulate(rng, c) for c in EXP.cause_ids()]
    dss.append(EXP.extend(rng, EXP.simulate(rng, H), EXP.menu_item("weigh_hanger").plan))
    dss.append(EXP.extend(rng, EXP.simulate(rng, Z), EXP.menu_item("zero_check").plan))
    lz = eig.batch_log_evidence(EXP, dss)
    ref = np.array([posterior(EXP, d).log_evidence for d in dss])
    assert np.allclose(lz, ref, rtol=0, atol=1e-8)


def test_recommend_nonnegative_sorted_serialisable():
    for k, cause in enumerate(EXP.cause_ids()):
        ds = EXP.simulate(np.random.default_rng(10 + k), cause)
        scores = eig.recommend(EXP, ds, n_samples=64, rng=np.random.default_rng(0))
        assert [m.item_id for m in scores] != []
        assert {m.item_id for m in scores} == {m.id for m in EXP.menu}
        bits = [m.eig_bits for m in scores]
        assert all(b >= 0 for b in bits)
        assert bits == sorted(bits, reverse=True)
        for m in scores:
            assert m.eig_per_min == pytest.approx(m.eig_bits / m.minutes)
            assert set(m.predicted) == set(EXP.cause_ids())
            assert all(np.isfinite(v["mean"]) and v["sd"] > 0 for v in m.predicted.values())
            assert set(m.separates) <= set(EXP.cause_ids())
        json.dumps([m.to_dict() for m in scores])


def test_uninformative_item_has_zero_eig():
    """只在"只有随机误差"和"原长读偏"之间比：两者对挂钩质量的预测完全相同（都是先验），称挂钩不提供信息。"""
    ds = EXP.simulate(np.random.default_rng(3), Z)
    e = eig.expected_information_gain(EXP, ds, "weigh_hanger", n_samples=64, cause_ids=(R, Z))
    assert e == pytest.approx(0.0, abs=1e-9)
    scores = eig.recommend(EXP, ds, n_samples=32, cause_ids=(R, Z))
    wh = next(m for m in scores if m.item_id == "weigh_hanger")
    assert wh.separates == []
    assert wh.predicted[R]["mean"] == pytest.approx(wh.predicted[Z]["mean"])


def test_two_cause_eig_matches_exact_sum():
    """随机误差 vs 原长读偏，追加"重读原长"。两种原因下这个读数的预测都是单个高斯（再按 0.1 cm 取整），
    EIG 可以对取整后的所有可能读数精确求和；与蒙特卡洛估计比较。"""
    ds = EXP.simulate(np.random.default_rng(21), Z, linear={"zero_offset": 0.25, "inv_k": 0.04})
    cids = (R, Z)
    post = posterior(EXP, ds, cids)
    p = post.probs
    assert 0.1 < p[1] < 0.9, p  # 后验确实没定
    cp = post.per_cause[1]
    assert cp.grid.size == 1
    m, C = cp.fit.theta_mean[0, 1], cp.fit.theta_cov[0, 1, 1]
    sig, d = 0.05, 0.1
    var_eng = sig**2 + d**2 / 12
    mu = np.array([0.0, m])
    sim_sd = np.sqrt([sig**2, C + sig**2])  # 模拟：δ ~ 后验，再加读数噪声、取整
    eng_var = np.array([var_eng, C + var_eng])  # 引擎更新用的预测方差
    y = np.arange(-80, 81) * d
    logpost = np.log(p)[:, None] + stats.norm.logpdf(y[None, :], mu[:, None], np.sqrt(eng_var)[:, None])
    logpost -= np.logaddexp.reduce(logpost, axis=0)
    kl = np.sum(np.exp(logpost) * (logpost - np.log(p)[:, None]), axis=0)
    exact = 0.0
    for c in range(2):
        py = stats.norm.cdf((y + d / 2 - mu[c]) / sim_sd[c]) - stats.norm.cdf((y - d / 2 - mu[c]) / sim_sd[c])
        exact += p[c] * np.sum(py * kl)
    exact /= math.log(2)
    mc = eig.expected_information_gain(EXP, ds, "zero_check", n_samples=4000,
                                       rng=np.random.default_rng(1), cause_ids=cids)
    assert exact > 0.1
    assert mc == pytest.approx(exact, abs=0.015)


def test_weigh_hanger_beats_repeat_when_split():
    """斜率自由时，漏算挂钩（x 偏移）和原长读偏（y 偏移）在标准数据上分不开；称挂钩能分开，重复一次 150 g 不能。"""
    ds = EXP.simulate(np.random.default_rng(0), H, nuisance={"hanger_g": 10.0}, linear={"inv_k": 0.03})
    cids = (H, Z)
    p = posterior(EXP, ds, cids).prob(H)
    assert 0.2 < p < 0.8, p
    rng = np.random.default_rng(2)
    wh = eig.expected_information_gain(EXP, ds, "weigh_hanger", rng=rng, cause_ids=cids)
    rep = eig.expected_information_gain(EXP, ds, "repeat_150", rng=rng, cause_ids=cids)
    assert wh > rep + 0.1
    assert rep < 0.05
    # 全部原因参与时结论不变
    full = {m.item_id: m for m in eig.recommend(EXP, ds, rng=np.random.default_rng(3))}
    assert full["weigh_hanger"].eig_bits > full["repeat_150"].eig_bits
    assert {H, Z} <= set(full["weigh_hanger"].separates)


def test_predictions_follow_the_physics():
    """原长记大 δ = 0.5 cm：重读原长时"原记录 − 重读"应为 +0.5 cm；只有随机误差时为 0。
    漏算 30 g 挂钩：称挂钩的预测应在 30 g 附近，而原长读偏下只能给先验（5~50 g 的均匀分布，均值 27.5、sd 13）。"""
    ds = EXP.simulate(np.random.default_rng(7), Z, linear={"zero_offset": 0.5, "inv_k": 0.04})
    zc = next(m for m in eig.recommend(EXP, ds, n_samples=32) if m.item_id == "zero_check")
    assert zc.predicted[Z]["mean"] == pytest.approx(0.5, abs=0.15)
    assert zc.predicted[Z]["unit"] == "cm"
    assert zc.predicted[R]["mean"] == pytest.approx(0.0, abs=1e-12)
    ds = EXP.simulate(np.random.default_rng(8), H, nuisance={"hanger_g": 30.0}, linear={"inv_k": 0.04})
    wh = next(m for m in eig.recommend(EXP, ds, n_samples=32) if m.item_id == "weigh_hanger")
    ph = wh.predicted[H]
    assert ph["unit"] == "g" and abs(ph["mean"] - 30.0) < 3 * ph["sd"] and ph["sd"] < 5.0
    assert wh.predicted[Z]["mean"] == pytest.approx(27.5, abs=0.2)
    assert wh.predicted[Z]["sd"] == pytest.approx(45 / math.sqrt(12), rel=0.02)


def test_recommend_timing():
    worst = 0.0
    for cause in (R, H, Z):
        ds = EXP.simulate(np.random.default_rng(1), cause)
        t = time.perf_counter()
        eig.recommend(EXP, ds, n_samples=128)
        worst = max(worst, time.perf_counter() - t)
    assert worst < 1.5, worst


def test_protocol_parsing():
    std = EXP.protocols["standard"].plan
    assert eig.protocol_plan(EXP, "standard") == tuple(std)
    assert eig.protocol_plan(EXP, "standard+weigh_hanger") == tuple(std) + (("hanger_mass", 0.0),)
    assert eig.repeat_plan(EXP, std) == (("main", 150.0),)
    with pytest.raises(ValueError):
        eig.protocol_plan(EXP, "standard+eig")
    ds = eig.simulate_protocol(EXP, np.random.default_rng(0), H, "standard+eig", n_samples=32)
    assert len(ds.observations) == 6
    assert ds.meta["protocol"] == "standard+eig"


def test_closed_loop_eig_beats_repeat_on_confounded_pair():
    p_eig, p_rep = [], []
    for seed in range(6):
        for cause in (H, Z):
            a = eig.closed_loop(EXP, np.random.default_rng(seed), cause, strategy="eig", n_samples=32)
            b = eig.closed_loop(EXP, np.random.default_rng(seed), cause, strategy="repeat", n_samples=32)
            assert a["p_true"][0] == b["p_true"][0]  # 同一种子 → 同一份基础数据（配对）
            assert len(a["p_true"]) == 2 and len(a["chosen"]) == 1 and len(a["top1"]) == 2
            assert b["chosen"] == ["repeat_150"]
            p_eig.append(a["p_true"][1])
            p_rep.append(b["p_true"][1])
    assert np.mean(p_eig) > np.mean(p_rep)


@pytest.mark.parametrize("strategy", ["eig_per_min", "random", "fixed:zero_check", "best_fixed"])
def test_closed_loop_other_strategies(strategy):
    r = eig.closed_loop(EXP, np.random.default_rng(4), Z, strategy=strategy, rounds=2, n_samples=32)
    assert len(r["p_true"]) == 3 and len(r["chosen"]) == 2
    assert all(c in {m.id for m in EXP.menu} for c in r["chosen"])
    assert all(0.0 <= p <= 1.0 for p in r["p_true"])
    if strategy.startswith("fixed:"):
        assert r["chosen"] == ["zero_check", "zero_check"]
    json.dumps(r)


def test_best_fixed_item_is_deterministic():
    a = eig.best_fixed_item(EXP, "standard", np.random.default_rng(9), n_cases=30)
    b = eig.best_fixed_item(EXP, "standard", np.random.default_rng(9), n_cases=30)
    assert a == b and a in {m.id for m in EXP.menu}
    assert a != "repeat_150"
