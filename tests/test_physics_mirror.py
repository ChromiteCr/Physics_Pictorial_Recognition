"""组件组合层的验证（1）：镜像装置与精选实验逐项一致。

对每个镜像（含仪器、已知倾角等构造选项），对精选实验的每个原因（含只有随机误差）和每种观测：
随机抽 5 组干扰量、幅度和线性参数，比较读数均值 c + U·θ 和总标准差（相对误差 < 1e-6，实际 ~1e-15）；
并比较线性参数的先验、方案与菜单的 (种类, 控制量) 序列、同一份数据的线性化结果。
两边物理相同（单故障世界、同样的精确/近似约定），没有刻意的差别；若精选实验以后改了物理，这里会报出来。
"""

import numpy as np
import pytest

from lab_detective.labsim.experiments.energy import Energy
from lab_detective.labsim.experiments.friction import Friction
from lab_detective.labsim.experiments.hooke import Hooke
from lab_detective.labsim.experiments.newton2 import Newton2
from lab_detective.labsim.experiments.pendulum import Pendulum
from lab_detective.physics import catalog

CASES = [
    ("hooke_composed", Hooke, {}, {}),
    ("friction_composed", Friction, {}, {}),
    ("newton2_composed", Newton2, {}, {}),
    ("newton2_composed", Newton2, {"timer": "photogate"}, {"timer": "photogate"}),
    ("energy_composed", Energy, {}, {}),
    ("energy_composed", Energy, {"timer": "ticker"}, {"timer": "ticker"}),
    ("energy_composed", Energy, {"angle_deg": 20.0}, {"angle_deg": 20.0}),
    ("pendulum_composed", Pendulum, {}, {}),
    ("pendulum_composed", Pendulum, {"timer": "stopwatch"}, {"timer": "stopwatch"}),
]
IDS = [f"{c[0]}-{'-'.join(f'{k}={v}' for k, v in c[2].items()) or 'default'}" for c in CASES]
P = 5


def _pair(case):
    eid, cls, cur_kw, cmp_kw = case
    entry = catalog.get(eid)
    return entry, cls(**cur_kw), entry.experiment(**cmp_kw)


def _suffix(cid):
    return cid.split(".", 1)[1]


def _settings_by_kind(cur):
    """精选实验方案和菜单里出现过的 (种类 → 控制量)，主测量再加几个点。"""
    out: dict[str, set] = {}
    plans = [p.plan for p in cur.protocols.values()] + [m.plan for m in cur.menu]
    for plan in plans:
        for k, s in plan:
            out.setdefault(k, set()).add(float(s))
    main = sorted(out["main"])
    out["main"] |= {0.5 * (main[0] + main[1]), 1.2 * main[-1]}
    for k in cur.obs_kinds:
        out.setdefault(k, {0.0})
    return {k: np.array(sorted(v)) for k, v in out.items()}


def _mean(c, U, theta, P, n):
    m = np.broadcast_to(c, (P, n, 1))[..., 0]
    if theta.size:
        m = m + (np.broadcast_to(U, (P, n, 1, theta.size)) @ theta)[..., 0]
    return m


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_same_causes_kinds_and_flags(case):
    entry, cur, cmp = _pair(case)
    assert {_suffix(c) for c in cur.cause_ids()} == {_suffix(c) for c in cmp.cause_ids()}
    assert set(cur.obs_kinds) == set(cmp.obs_kinds)
    assert (cmp.held_out, cmp.transfer) == (cur.held_out, cur.transfer)
    assert cmp.id != cur.id
    for c in cur.causes:
        k = cmp.cause(f"{cmp.id}.{_suffix(c.id)}")
        assert (k.mechanism, k.magnitude) == (c.mechanism, c.magnitude), c.id
    for kind, ok in cur.obs_kinds.items():
        a, b = ok.readings[0], cmp.obs_kinds[kind].readings[0]
        assert (a.unit, a.resolution) == (b.unit, b.resolution), kind


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_predictions_match_for_every_shared_cause(case):
    entry, cur, cmp = _pair(case)
    pm = entry.param_map
    rng = np.random.default_rng(0)
    settings = _settings_by_kind(cur)
    for cause in cur.causes:
        kid = f"{cmp.id}.{_suffix(cause.id)}"
        lin_cur, lin_cmp = cur.linear_params(cause.id), cmp.linear_params(kid)
        # 线性参数：同顺序、同先验
        assert [pm.get(p.name, p.name) for p in lin_cmp] == [p.name for p in lin_cur]
        assert [(p.mean, p.sd) for p in lin_cmp] == [(p.mean, p.sd) for p in lin_cur]
        mag = np.array([cause.magnitude.draw(rng) for _ in range(P)])[:, None]
        cpar = {"mag": mag}
        for k, pr in cur.nuisance.items():
            cpar[k] = np.array([pr.draw(rng) for _ in range(P)])[:, None]
        theta = np.array([p.draw(rng) for p in lin_cur])
        if "mu_pull" in cpar and lin_cur and lin_cur[0].name == "mu":  # 摩擦力：分母里的 μ 与线性参数同值
            cpar["mu_pull"] = np.full((P, 1), theta[0])
        kpar = {"mag": mag}
        inv = {v: k for k, v in pm.items()}
        for k in cmp.nuisance:
            kpar[k] = cpar[pm.get(k, k)]
        for k, pr in cur.nuisance.items():  # 组合版没有的干扰量只能是已知量（点先验）
            assert inv.get(k, k) in cmp.nuisance or pr.kind == "point", k
        for kind, s in settings.items():
            order = np.arange(s.size, dtype=float)
            c1, U1 = cur.predict(kind, s, order, cpar, cause.id)
            c2, U2 = cmp.predict(kind, s, order, kpar, kid)
            m1, m2 = _mean(c1, U1, theta, P, s.size), _mean(c2, U2, theta, P, s.size)
            np.testing.assert_allclose(m2, m1, rtol=1e-6, atol=1e-12, err_msg=f"{cause.id} / {kind}")
            sd1 = np.broadcast_to(cur.total_sd(kind, cpar, cause.id), (P, 1, 1))
            sd2 = np.broadcast_to(cmp.total_sd(kind, kpar, kid), (P, 1, 1))
            np.testing.assert_allclose(sd2, sd1, rtol=1e-9, err_msg=f"σ {cause.id} / {kind}")


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_protocols_menus_and_linearisation_match(case):
    entry, cur, cmp = _pair(case)
    for pid, p in cur.protocols.items():
        assert tuple(cmp.protocols[pid].plan) == tuple(p.plan), pid
    assert {tuple(m.plan) for m in cmp.menu} == {tuple(m.plan) for m in cur.menu}
    rng = np.random.default_rng(1)
    for cause in (cur.random_only_id, cur.causes[-1].id):
        ds = cur.simulate(rng, cause)
        for m in cur.menu:
            ds = cur.extend(rng, ds, m.plan)
        a, b = cur.linearise(ds), cmp.linearise(ds)
        for f in ("x", "y", "sy", "order"):
            np.testing.assert_allclose(getattr(b, f), getattr(a, f), rtol=1e-9, err_msg=f)
        assert (b.x_label, b.y_label) == (a.x_label, a.y_label)
        if a.ideal_slope is None:
            assert b.ideal_slope is None
        else:
            assert b.ideal_slope == pytest.approx(a.ideal_slope, rel=1e-12)
        assert b.ideal_intercept == a.ideal_intercept


def test_mirror_simulation_identical_with_same_seed():
    """同一个种子、同样的真值：两边模拟出的读数逐个相同（取整后）。"""
    entry, cur, cmp = _pair(CASES[4])  # 机械能（光电门）
    t_cur = cur.draw_truth(np.random.default_rng(3), "energy.push")
    t_cmp = {"cause": "energy_composed.push", "magnitude": t_cur["magnitude"],
             "nuisance": {"incline.theta_deg": t_cur["nuisance"]["theta_deg"]},
             "linear": {"timer.w_true": t_cur["linear"]["w_true"]}}
    plan = cur.protocols["standard"].plan
    o1 = cur.observe(np.random.default_rng(4), t_cur, plan)
    o2 = cmp.observe(np.random.default_rng(4), t_cmp, plan)
    assert [o.values for o in o1] == [o.values for o in o2]
