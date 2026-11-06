"""错配高保真世界：形状、可复现、各效应确实生效。"""

import numpy as np

from lab_detective.eig import protocol_plan
from lab_detective.engine import posterior
from lab_detective.labsim import hifi
from lab_detective.labsim.registry import get_experiment


def _ds(eid, cfg, seed=0, **kw):
    exp = get_experiment(eid)
    plan = protocol_plan(exp, "standard")
    return exp, hifi.simulate(exp, np.random.default_rng(seed), kw.pop("cause", exp.random_only_id), plan, cfg,
                              protocol="standard", **kw)


def test_shapes_and_reproducible():
    exp, a = _ds("newton2", hifi.HIFI, seed=3)
    _, b = _ds("newton2", hifi.HIFI, seed=3)
    assert [o.values for o in a.observations] == [o.values for o in b.observations]
    assert len(a.observations) == len(protocol_plan(exp, "standard"))
    assert a.truth["world"] == "hifi" and a.truth["effects"]["bend_rel"] > 0
    post = posterior(exp, a)  # 引擎能吃这种数据
    assert np.isclose(post.probs.sum(), 1.0)


def test_slip_forced_and_bend_only_where_relevant():
    _, d = _ds("hooke", hifi.GAUSS, slip_force=True)
    assert len(d.truth["effects"]["slips"]) == 1
    assert d.truth["effects"]["bend_rel"] == 0.0  # 胡克定律没有拖曳
    _, e = _ds("newton2", hifi.HIFI)
    assert abs(e.truth["effects"]["bend_rel"] - (hifi.HIFI.drag_rel + hifi.HIFI.string_rel)) < 1e-12


def test_noise_shift_is_wider():
    exp = get_experiment("hooke")
    plan = protocol_plan(exp, "standard")
    sd = {}
    for name, cfg in (("g", hifi.GAUSS), ("n", hifi.NOISE_SHIFT)):
        res = []
        for s in range(40):
            d = hifi.simulate(exp, np.random.default_rng(s), exp.random_only_id, plan, cfg, protocol="standard")
            d0 = exp.simulate(np.random.default_rng(s), exp.random_only_id, plan)
            res.append(d.observations[-1].values[0] - d0.observations[-1].values[0])
        sd[name] = np.std(res)
    assert sd["n"] > sd["g"]


def test_two_causes_differs_from_one():
    exp = get_experiment("hooke")
    plan = protocol_plan(exp, "standard")
    one = hifi.simulate(exp, np.random.default_rng(1), "hooke.zero_offset", plan, hifi.GAUSS)
    two = hifi.simulate(exp, np.random.default_rng(1), "hooke.zero_offset", plan, hifi.GAUSS,
                        second_cause="hooke.hanger_omitted")
    assert two.truth["effects"]["second_cause"] == "hooke.hanger_omitted"
    assert [o.values for o in one.observations] != [o.values for o in two.observations]
