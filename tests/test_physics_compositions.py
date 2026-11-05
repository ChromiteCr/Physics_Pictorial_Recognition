"""组件组合层的验证（2）：训练集里没有的新组合装置。

每个新组合：零故障（噪声 ×1e-6）时面板斜率给出它的物理常数；标准方案下面板看得出来的大故障，
20 次模拟里引擎至少 14 次（70%）把它排第一；后验计算 < 0.5 s；默认是留出装置（迁移测试用）。
"""

import time

import numpy as np
import pytest

from lab_detective.engine import posterior
from lab_detective.labsim.core import SIGNATURES
from lab_detective.labsim.registry import EXPERIMENTS
from lab_detective.panel import compute_panel
from lab_detective.physics import catalog

NEW = catalog.new_compositions()
EXPS = {e.id: e.experiment() for e in catalog.CATALOG.values()}


def test_at_least_five_new_compositions_held_out_and_not_registered():
    assert len(NEW) >= 5
    curated = set(EXPERIMENTS)
    for e in NEW:
        exp = EXPS[e.id]
        assert e.held_out and exp.held_out and exp.transfer in ("near", "far")
        assert exp.id not in curated and "." not in exp.id
    # 组合出来的实验不进注册表，训练/留出清单不变
    assert not set(EXPS) & curated


@pytest.mark.parametrize("entry", NEW, ids=[e.id for e in NEW])
def test_interface_conventions(entry):
    exp = EXPS[entry.id]
    assert exp.causes[0].id == f"{exp.id}.random_only"
    assert all(c.id.startswith(f"{exp.id}.") and c.id.count(".") == 1 for c in exp.causes)
    assert len(exp.causes) >= 4
    assert all(set(c.signatures) <= set(SIGNATURES) for c in exp.causes)
    assert set(exp.protocols) >= {"standard", "single"}
    assert len(exp.menu) >= 1
    for m in exp.menu:
        assert all(k in exp.obs_kinds for k, _ in m.plan)


@pytest.mark.parametrize("entry", NEW, ids=[e.id for e in NEW])
def test_zero_fault_recovers_physical_constant(entry):
    exp = EXPS[entry.id]
    lin = {p.name: p.mean for p in exp.linear}
    ds = exp.simulate(np.random.default_rng(0), exp.random_only_id, linear=lin, noise_scale=1e-6)
    panel = compute_panel(exp.linearise(ds))
    label, fn, true = entry.constant
    # 只剩读数取整：光电门 0.1 ms 取整在 5 ms 的挡光时间上约 1%，其余远小于 1%
    assert fn(panel.slope, panel.intercept) == pytest.approx(true, rel=0.01), label
    assert "none" in panel.signatures or set(panel.signatures) <= {"scatter_deficit", "drift"}


@pytest.mark.parametrize("entry", NEW, ids=[e.id for e in NEW])
def test_engine_identifies_large_fault(entry):
    exp = EXPS[entry.id]
    suffix, kw = entry.large_fault
    cid = f"{exp.id}.{suffix}"
    rng = np.random.default_rng(1)
    hits, worst = 0, 0.0
    for _ in range(20):
        ds = exp.simulate(rng, cid, **kw)
        t0 = time.perf_counter()
        post = posterior(exp, ds)
        worst = max(worst, time.perf_counter() - t0)
        assert post.probs.sum() == pytest.approx(1.0)
        hits += post.top() == cid
    assert hits >= 14, f"{cid}: {hits}/20"
    assert worst < 0.5


@pytest.mark.parametrize("entry", NEW, ids=[e.id for e in NEW])
def test_large_fault_moves_the_panel(entry):
    """所选的大故障在标准方案的面板上确实可见（这是它能被认出来的原因）。"""
    exp = EXPS[entry.id]
    suffix, kw = entry.large_fault
    ds = exp.simulate(np.random.default_rng(2), f"{exp.id}.{suffix}", noise_scale=1e-6, **kw)
    assert compute_panel(exp.linearise(ds)).signatures != ("none",)


def test_incline_newton2_exact_law():
    """斜面牛二：零故障时 a = [mg − (M_t − m)g sinθ]/M_t（槽码从车上取下挂到小盘，M_t 不变）。"""
    exp = EXPS["incline_newton2"]
    ds = exp.simulate(np.random.default_rng(3), exp.random_only_id, noise_scale=1e-6)
    s = np.sin(np.radians(5.0))
    for o in ds.observations:
        m = o.setting / 1000.0
        assert o.values[0] == pytest.approx((m * 9.8 - (1.0 - m) * 9.8 * s) / 1.0, abs=6e-4)


def test_system_energy_pulley_inertia_lowers_slope_by_mass_ratio():
    """滑轮等效质量 J：ΔE_k/ΔE_p = (M + m)/(M + m + J)。"""
    exp = EXPS["system_energy"]
    lin = {p.name: p.mean for p in exp.linear}
    ds = exp.simulate(np.random.default_rng(4), "system_energy.pulley_inertia", magnitude=30.0, linear=lin,
                      noise_scale=1e-6)
    slope = compute_panel(exp.linearise(ds)).slope
    assert slope == pytest.approx(0.55 / 0.58, rel=0.01)


def test_spring_oscillator_count10_exposes_miscount():
    """数错次数在 k 未知时只改斜率、标准方案看不出；只数 10 次再测一次能把它暴露出来。"""
    exp = EXPS["spring_oscillator"]
    rng = np.random.default_rng(5)
    before, after = [], []
    for _ in range(10):
        ds = exp.simulate(rng, "spring_oscillator.miscount", magnitude=1.0)
        before.append(posterior(exp, ds).prob("spring_oscillator.miscount"))
        ds2 = exp.extend(rng, ds, exp.menu_item("count10").plan)
        after.append(posterior(exp, ds2).prob("spring_oscillator.miscount"))
    assert np.mean(after) > np.mean(before) + 0.3


def test_photogate_free_fall_drag_curves_down():
    exp = EXPS["photogate_free_fall"]
    ds = exp.simulate(np.random.default_rng(6), "photogate_free_fall.air_drag", magnitude=0.12, noise_scale=1e-6)
    p = compute_panel(exp.linearise(ds))
    assert "curve_down" in p.signatures or "slope_low" in p.signatures


def test_incline_slide_recovers_friction_coefficient():
    """由 a 和记录的倾角求 μ = (g sinθ − a)/(g cosθ)。"""
    exp = EXPS["incline_slide"]
    mu = 0.25
    a = 9.8 * (np.sin(np.radians(30)) - mu * np.cos(np.radians(30)))
    ds = exp.simulate(np.random.default_rng(7), exp.random_only_id, linear={"motion.inv_sqrt_a": 1 / np.sqrt(a)},
                      noise_scale=1e-6)
    slope = compute_panel(exp.linearise(ds)).slope
    a_fit = 1.0 / slope**2
    mu_fit = (9.8 * np.sin(np.radians(30)) - a_fit) / (9.8 * np.cos(np.radians(30)))
    assert mu_fit == pytest.approx(mu, abs=0.005)
