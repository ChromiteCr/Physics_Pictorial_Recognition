"""知识点图谱与实验卡：结构、无环、前置知识点存在、胡克卡与 labsim 原因库一致。

其他实验卡由并行作者维护，这里只断言知识点表本身和胡克卡没有问题，不要求全部卡片都已就绪。
"""
import json

import pytest

from lab_detective import knowledge as K
from lab_detective.knowledge.schema import EXPERIMENT_IDS, MECHANISM_TAGS, validate_card
from lab_detective.labsim import get_experiment
from lab_detective.labsim.core import MECHANISMS


def test_kp_count_and_ids():
    kps = K.all_kps()
    assert 40 <= len(kps) <= 70
    ids = [k["id"] for k in kps]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("kp.") for i in ids)
    mcs = [m["id"] for k in kps for m in k["misconceptions"]]
    assert len(mcs) == len(set(mcs)) and all(m.startswith("mc.") for m in mcs)


def test_kp_level_validation_clean():
    assert K._kp_problems() == []
    assert K._cause_map_problems() == []


def test_prerequisites_exist_and_graph_acyclic():
    g = K.graph()
    for kid, pre in g.items():
        assert kid not in pre
        for p in pre:
            assert p in g, (kid, p)
    # 用 Kahn 算法独立检查无环
    indeg = {k: len(v) for k, v in g.items()}
    dependents = {k: [] for k in g}
    for k, pre in g.items():
        for p in pre:
            dependents[p].append(k)
    ready = [k for k, d in indeg.items() if d == 0]
    seen = 0
    while ready:
        k = ready.pop()
        seen += 1
        for d in dependents[k]:
            indeg[d] -= 1
            if indeg[d] == 0:
                ready.append(d)
    assert seen == len(g)


def test_every_experiment_has_knowledge_points():
    covered = {e for k in K.all_kps() for e in k["experiments"]}
    assert set(EXPERIMENT_IDS) <= covered


def test_measurement_science_points_present():
    need = {
        "kp.systematic_error", "kp.random_error", "kp.significant_figures", "kp.uncertainty",
        "kp.scale_reading", "kp.graph_method", "kp.graph_slope_intercept", "kp.control_variables",
        "kp.multiple_measurements", "kp.zero_adjustment", "kp.parallax", "kp.reaction_time",
    }
    assert need <= set(K.graph())


def test_learning_path_is_topological():
    for kid in K.graph():
        path = K.learning_path(kid)
        assert kid not in path
        pos = {k: i for i, k in enumerate(path)}
        for k in path:
            for p in K.kp(k)["prerequisites"]:
                assert pos[p] < pos[k], (kid, p, k)
        # 直接前置知识点都在路径里
        assert set(K.kp(kid)["prerequisites"]) <= set(path)
    lp = K.learning_path("kp.pendulum_g")
    assert "kp.simple_pendulum" in lp and "kp.shm" in lp and "kp.hooke_law" in lp
    assert lp.index("kp.shm") < lp.index("kp.simple_pendulum")
    assert K.learning_path("kp.si_units") == []


def test_aliases_resolve():
    assert K.canonical("kp.hookes_law") == "kp.hooke_law"
    assert K.kp("kp.moment_arm")["id"] == "kp.lever"
    assert K.canonical("kp.no_such_point") is None
    with pytest.raises(KeyError):
        K.kp("kp.no_such_point")


def test_hooke_card_validates():
    card = K.load_card("hooke")
    assert validate_card(card) == []
    assert K.card_errors("hooke") == []
    assert not [p for p in K.validate_all() if "hooke" in p]


def test_hooke_card_causes_match_registry():
    card = K.load_card("hooke")
    exp = get_experiment(card["labsim_id"])
    assert set(card["common_causes"]) == set(exp.cause_ids())
    assert set(card["cause_kps"]) <= set(card["common_causes"])
    for ids in card["cause_kps"].values():
        assert all(K.canonical(i) == i for i in ids)


def test_cause_map_covers_all_mechanisms():
    m = json.loads((K.DATA / "cause_kp_map.json").read_text(encoding="utf-8"))
    assert set(m["mechanism"]) == set(MECHANISMS) == set(MECHANISM_TAGS)


def test_related_kps_for_hooke():
    base = K.related_kps("hooke")
    assert base[0] == "kp.hooke_law"
    lim = K.related_kps("hooke", "hooke.beyond_limit")
    assert "kp.elastic_limit" in lim and "kp.ideal_model" in lim
    # 漏算挂钩（x 偏移）→ 系统误差、截距的意义、重力
    han = K.related_kps("hooke", "hooke.hanger_omitted")
    assert {"kp.systematic_error", "kp.graph_slope_intercept", "kp.gravity_weight"} <= set(han)
    # 读数散布 → 偶然误差
    assert "kp.random_error" in K.related_kps("hooke", "hooke.reading_scatter")
    # 所有返回的都是正式 id，且不重复
    for ids in (base, lim, han):
        assert len(ids) == len(set(ids)) and all(K.canonical(i) == i for i in ids)


def test_related_kps_without_card_falls_back():
    # 没有卡片、也不在注册表里的实验：退回 experiments 字段，未知原因不报错
    ids = K.related_kps("pendulum", "pendulum.unknown_cause")
    assert "kp.simple_pendulum" in ids


def test_missing_card_is_keyerror_and_loaders_robust():
    with pytest.raises(KeyError):
        K.load_card("no_such_experiment")
    cards = K.all_cards()
    assert "hooke" in cards
    cards["hooke"]["id"] = "changed"  # 返回的是副本
    assert K.load_card("hooke")["id"] == "hooke"
    assert isinstance(K.validate_all(), list)
