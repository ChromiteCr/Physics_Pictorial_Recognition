"""讲解声明校验器：正确讲解要通过，故意写错的数字、单位、方向、原因、测量要被抓出来（阳性对照）。"""
import copy

import pytest

from lab_detective.tutor.verifier import extract_numbers, normalize, verify


def make_fs():
    return {
        "experiment_id": "hooke",
        "experiment_name": "探究弹簧弹力与形变量的关系（胡克定律）",
        "protocol": "standard",
        "data": {"n_points": 5, "table": [{"setting": 50, "values": [1.2]}]},
        "panel": {"slope": 4.0123456789, "intercept": 1.5234, "signatures": ["intercept_pos"]},
        "agreement": {"quantity": "k", "value": 24.1, "uncertainty": 0.6, "unit": "N/m", "z": 2.4,
                      "band": "tension"},
        "diagnosis": {
            "ranked": [
                {"cause_id": "hooke.hanger_omitted", "name": "漏算挂钩质量", "prob": 0.7123,
                 "evidence": "截距 +1.52 cm（z=6.2）"},
                {"cause_id": "hooke.zero_offset", "name": "原长读数有偏差", "prob": 0.2, "evidence": "截距 +1.52 cm"},
                {"cause_id": "hooke.random_only", "name": "只有随机误差", "prob": 0.0877, "evidence": ""},
            ],
            "not_in_list": False, "misfit_p": 0.4,
        },
        "next_measurement": {"id": "weigh_hanger", "label": "用天平称挂钩质量", "eig_bits": 0.83,
                             "predicted": [{"cause_id": "hooke.hanger_omitted", "value": 23.0, "unit": "g"},
                                           {"cause_id": "hooke.zero_offset", "value": 5.0, "unit": "g"}]},
        "fix": {"cause_id": "hooke.hanger_omitted", "text": "称出挂钩质量，拉力按总质量计算",
                "predicted_rerun": "截距回到 0 附近"},
        "knowledge_points": ["kp.hooke_law", "kp.systematic_error"],
        "numbers": [{"name": "intercept", "value": 1.52, "unit": "cm"},
                    {"name": "k_predicted", "value": 22.0, "unit": "N/m"}],
    }


CAT = {
    "causes": [
        {"id": "hooke.hanger_omitted", "name_zh": "漏算挂钩质量", "name_en": "hanger mass left out", "experiment_id": "hooke"},
        {"id": "hooke.zero_offset", "name_zh": "原长读数有偏差", "name_en": "unloaded length misread", "experiment_id": "hooke"},
        {"id": "hooke.beyond_limit", "name_zh": "超出弹性限度", "name_en": "beyond the elastic limit", "experiment_id": "hooke"},
        {"id": "freefall.air_drag", "name_zh": "空气阻力", "name_en": "air drag", "experiment_id": "freefall"},
    ],
    "menus": {"hooke": [{"id": "weigh_hanger", "label_zh": "用天平称挂钩质量", "label_en": "weigh the hanger"},
                        {"id": "zero_check", "label_zh": "取下钩码，重读原长", "label_en": "re-read the unloaded length"},
                        {"id": "heavy_load", "label_zh": "再挂一组更重的钩码（350 g）", "label_en": "add a heavier load"}]},
}

TAIL = "下一步：用天平称挂钩质量。"

GOOD = [
    "最可能的原因是「漏算挂钩质量」（可能性 71%），证据：截距 +1.52 cm（z=6.2）。" + TAIL,
    "主要原因是漏算了挂钩的质量：截距 1.52×10⁻² m，测得 k = (24.1 ± 0.6) N/m，比预测偏大。下一步建议用天平称一下挂钩的质量。",
    "最可能是漏算挂钩质量。截距为１５．２毫米，劲度系数偏大（z = 2.4）。" + TAIL,
    "最可能的原因是漏算挂钩质量，截距 1.52 厘米偏大。下一步：用天平称挂钩质量，若是漏算挂钩质量应称得约 23 g，若是原长读数有偏差约 5.0 g。",
    "最可能的原因是漏算挂钩质量（71.2%），截距 1.5e-2 m。用天平称挂钩质量能把它和原长读数有偏差分开（信息增益 0.83 bit）。",
    "5 组数据里，第 3 组看着正常。最可能的原因是漏算挂钩质量，截距 1.52 cm，斜率与 1/k 成正比。" + TAIL,
    "最可能的原因是漏算挂钩质量，不是空气阻力。截距 1.52 cm。" + TAIL,
    "测得 24.1 N/m，大于预测的 22.0 N/m。最可能的原因是漏算挂钩质量。" + TAIL,
    "如果是原长读数有偏差，截距也会偏小或偏大；但最可能的原因是漏算挂钩质量，截距 1.52 cm。" + TAIL,
    "The most likely cause is 漏算挂钩质量: the intercept is 1.52 cm (z = 6.2). Next: weigh the hanger, 用天平称挂钩质量.",
]


@pytest.mark.parametrize("text", GOOD)
def test_correct_texts_pass(text):
    r = verify(text, make_fs(), catalog=CAT)
    assert r["ok"], r["problems"]
    assert all(r["coverage"].values())


BAD = [
    ("最可能的原因是漏算挂钩质量，截距 3.1 cm，测得 k = 24.1 N/m。" + TAIL, "number_mismatch"),
    ("最可能的原因是漏算挂钩质量，截距 -1.52 cm，测得 k = 24.1 N/m。" + TAIL, "number_mismatch"),
    ("最可能的原因是漏算挂钩质量，截距 2 cm，测得 k = 24.1 N/m。" + TAIL, "number_mismatch"),
    ("最可能的原因是漏算挂钩质量（可能性 90%），截距 1.52 cm。" + TAIL, "number_mismatch"),
    ("最可能的原因是漏算挂钩质量，截距 1.52 m，测得 k = 24.1 N/m。" + TAIL, "unit_mismatch"),
    ("最可能的原因是漏算挂钩质量，截距 1.52 cm，测得 k = 24.1 N/cm。" + TAIL, "unit_mismatch"),
    ("最可能的原因是漏算挂钩质量，截距 1.52 cm，挂钩应为 23 kg。" + TAIL, "unit_mismatch"),
    ("最可能的原因是漏算挂钩质量，截距偏小，为 1.52 cm。" + TAIL, "direction"),
    ("最可能的原因是漏算挂钩质量，截距 1.52 cm，劲度系数偏小。" + TAIL, "direction"),
    ("最可能的原因是漏算挂钩质量，截距 1.52 cm，k 的测量值偏低。" + TAIL, "direction"),
    ("测得 24.1 N/m，小于预测的 22.0 N/m。最可能的原因是漏算挂钩质量。" + TAIL, "direction"),
    ("最可能的原因是漏算挂钩质量，截距 1.52 cm，也可能是空气阻力。" + TAIL, "unknown_cause"),
    ("最可能的原因是漏算挂钩质量，截距 1.52 cm；超出弹性限度也有可能。" + TAIL, "unknown_cause"),
    ("主要原因是弹簧生锈，截距 1.52 cm。" + TAIL, "unknown_cause"),
    ("主要原因是原长读数有偏差，截距 1.52 cm。" + TAIL, "wrong_top_cause"),
    ("最可能的原因是漏算挂钩质量，截距 1.52 cm。下一步建议取下钩码，重读原长。", "wrong_next_measurement"),
    ("截距 1.52 cm，数据有问题。" + TAIL, "missing_top_cause"),
    ("最可能的原因是漏算挂钩质量，图线不过原点。" + TAIL, "missing_number_evidence"),
    ("最可能的原因是漏算挂钩质量，截距 1.52 cm。", "missing_next_measurement"),
]


@pytest.mark.parametrize("text,code", BAD)
def test_planted_errors_are_caught(text, code):
    r = verify(text, make_fs(), catalog=CAT)
    assert not r["ok"]
    assert code in {p["code"] for p in r["problems"]}, r["problems"]


def test_correct_texts_have_no_false_direction_or_cause_flags():
    """正确讲解里不应出现方向或原因类问题（误杀的代价比放过更大）。"""
    for text in GOOD:
        codes = {p["code"] for p in verify(text, make_fs(), catalog=CAT)["problems"]}
        assert not codes & {"direction", "unknown_cause", "wrong_top_cause"}, (text, codes)


def test_negative_z_flips_direction():
    fs = make_fs()
    fs["agreement"]["z"] = -2.4
    ok = verify("最可能的原因是漏算挂钩质量，截距 1.52 cm，劲度系数偏小。" + TAIL, fs, catalog=CAT)
    bad = verify("最可能的原因是漏算挂钩质量，截距 1.52 cm，劲度系数偏大。" + TAIL, fs, catalog=CAT)
    assert ok["ok"], ok["problems"]
    assert "direction" in {p["code"] for p in bad["problems"]}


def test_negation_is_handled():
    r = verify("最可能的原因是漏算挂钩质量，截距 1.52 cm，k 并不偏小。" + TAIL, make_fs(), catalog=CAT)
    assert r["ok"], r["problems"]


def test_not_in_list_and_missing_next_measurement():
    fs = make_fs()
    fs["diagnosis"]["not_in_list"] = True
    fs["diagnosis"]["misfit_p"] = 0.003
    fs["next_measurement"] = None
    r = verify("数据和清单里的原因都对不上（失配 p = 0.003），可能有清单之外的原因；截距 1.52 cm。", fs, catalog=CAT)
    assert r["ok"], r["problems"]


def test_strict_mode_only_accepts_numbers_list():
    text = "最可能的原因是漏算挂钩质量，测得 k = 24.1 N/m。" + TAIL
    assert verify(text, make_fs(), catalog=CAT)["ok"]
    r = verify(text, make_fs(), catalog=CAT, strict=True)
    assert not r["ok"] and "number_mismatch" in {p["code"] for p in r["problems"]}


def test_number_extraction_details():
    nums = extract_numbers(normalize("(24.1±0.6) N/m，1.5×10⁻² m，3e-3 s，71%，第 3 组，L0 与 x²"))
    got = [(round(n.value, 6), n.unit, n.exempt) for n in nums]
    assert (24.1, "N/m", "") in got and (0.6, "N/m", "") in got
    assert (0.015, "m", "") in got and (0.003, "s", "") in got and (71.0, "%", "") in got
    assert (3.0, "", "ordinal") in got
    assert len(got) == 6  # L0 的 0 和 x² 的 2 不算数字


def test_deterministic():
    t = GOOD[1]
    assert verify(t, make_fs(), catalog=CAT) == verify(t, copy.deepcopy(make_fs()), catalog=CAT)


def test_causes_listed_in_next_measurement_are_known():
    fs = make_fs()
    fs["next_measurement"]["predicted"].append(
        {"cause_id": "hooke.beyond_limit", "name": "超出弹性限度", "value": 23.0, "unit": "g"})
    text = "最可能的原因是漏算挂钩质量，截距 1.52 cm。下一步：用天平称挂钩质量；若是超出弹性限度，预计 23 g。"
    assert verify(text, fs, catalog=CAT)["ok"]
    bad = verify("最可能的原因是超出弹性限度，截距 1.52 cm。" + TAIL, fs, catalog=CAT)
    assert "wrong_top_cause" in {p["code"] for p in bad["problems"]}


def facts_style_fs():
    """照 facts.py 的命名习惯手搭的事实单（带点号的名字、中文带括号的一致性量），不导入 facts.py。"""
    nums = [{"name": "n_points", "value": 5, "unit": ""},
            {"name": "intercept", "value": -0.012, "unit": "s²"},
            {"name": "intercept_z", "value": -0.4, "unit": ""},
            {"name": "slope_ratio", "value": 0.9362, "unit": ""},
            {"name": "slope_ratio_z", "value": -28.9, "unit": ""},
            {"name": "agreement.value", "value": 0.9362, "unit": ""},
            {"name": "agreement.z", "value": -28.9, "unit": ""},
            {"name": "rerun.slope_ratio", "value": 1.0004, "unit": ""},
            {"name": "rerun.intercept", "value": 0.0, "unit": "s²"},
            {"name": "estimate.inv_sqrt_g", "value": 0.31934, "unit": "s/√m"},
            {"name": "prob.pendulum.miscount", "value": 0.93, "unit": ""},
            {"name": "prob_pct.pendulum.miscount", "value": 93, "unit": "%"},
            {"name": "data[0].t10", "value": 12.71, "unit": "s"},
            {"name": "next.predicted.pendulum.miscount", "value": 19.63, "unit": "s"}]
    for i in range(40):
        nums.append({"name": f"data[{i + 1}].t10", "value": 13.0 + i / 10, "unit": "s"})
    return {
        "experiment_id": "pendulum", "experiment_name": "用单摆测量重力加速度",
        "agreement": {"quantity": "斜率比（实测/理论）", "value": 0.9362, "uncertainty": 0.0022, "unit": "",
                      "z": -28.9, "band": "inconsistent"},
        "panel": {"signatures": ["slope_low"]},
        "diagnosis": {"ranked": [{"cause_id": "pendulum.miscount", "name": "全振动次数数错了", "prob": 0.93,
                                  "evidence": "斜率比 0.9362（z=-28.9，显著）"},
                                 {"cause_id": "pendulum.conical", "name": "摆球做了圆锥摆运动", "prob": 0.07, "evidence": ""}],
                      "not_in_list": False, "misfit_p": 0.55},
        "next_measurement": {"id": "count10", "label": "在 100 cm 处只数 10 次全振动再计时一次", "eig_bits": 0.37,
                             "predicted": [{"cause_id": "pendulum.miscount", "name": "全振动次数数错了", "value": 19.63, "unit": "s"}]},
        "fix": {"cause_id": "pendulum.miscount", "text": "摆球第一次经过最低点时数 0 并开始计时",
                "predicted_rerun": "按修正方法重做，预计斜率比 1.0004（图线应通过原点）"},
        "knowledge_points": ["kp.pendulum"],
        "numbers": nums,
    }


@pytest.mark.parametrize("phrase,ok", [("斜率比偏小", True), ("斜率比偏大", False), ("测得的斜率比小于理论值", True),
                                       ("截距为负", True), ("截距偏大", False)])
def test_direction_with_facts_style_names(phrase, ok):
    fs = facts_style_fs()
    t = f"最可能的原因是全振动次数数错了，斜率比 0.9362（z = -28.9），{phrase}。下一步：在 100 cm 处只数 10 次全振动再计时一次。"
    r = verify(t, fs, catalog=CAT)
    assert r["ok"] == ok, r["problems"]
    if not ok:
        assert "direction" in {p["code"] for p in r["problems"]}
