"""组件组合层的验证（3）：看图建模——识别结果 → 候选装置 + 需要学生回答的问题。"""

import numpy as np

from lab_detective.engine import posterior
from lab_detective.physics.from_photo import candidates_from_photo
from lab_detective.vision.schema import RecognitionResult


def _ids(cands):
    return [c.entry_id for c in cands]


def _keys(cand):
    return {q.key for q in cand.open_questions}


def test_cart_track_string():
    cands = candidates_from_photo({"apparatus": ["cart", "track", "string"], "cues": ["细线连着小车"]})
    ids = _ids(cands)
    assert ids[0] == "newton2_composed"
    assert {"system_energy", "incline_newton2"} <= set(ids[:4])
    assert "cart.mass_g" in _keys(cands[0]) and "hanger.masses" in _keys(cands[0])
    # 候选装置能直接变成可诊断的实验孪生
    exp = cands[0].experiment()
    ds = exp.simulate(np.random.default_rng(0), exp.random_only_id)
    assert posterior(exp, ds).probs.sum() > 0.999


def test_incline_cue_prefers_incline_setups_and_photogate_option():
    cands = candidates_from_photo({"apparatus": ["小车", "导轨", "细线", "定滑轮", "钩码", "光电门"],
                                   "cues": ["轨道一端垫得很高，成了斜面"]})
    assert cands[0].entry_id == "incline_newton2"
    assert "incline.angle_deg" in _keys(cands[0]) and "gate.flag_mm" in _keys(cands[0])
    n2 = next(c for c in cands if c.entry_id == "newton2_composed")
    assert n2.options == {"timer": "photogate"}
    assert n2.setup.main_instrument.__class__.__name__ == "Photogate"


def test_spring_alone():
    cands = candidates_from_photo({"apparatus": ["spring"], "cues": []})
    assert _ids(cands)[:2] == ["hooke_composed", "spring_oscillator"]
    assert "hanger.masses" in _keys(cands[0])
    # 线索说在振动时，振子排第一，并且要问弹簧自身质量
    cands = candidates_from_photo({"apparatus": ["spring", "slotted_mass"], "cues": ["钩码上下振动", "手机计时"]})
    assert cands[0].entry_id == "spring_oscillator"
    assert "spring.mass_g" in _keys(cands[0])


def test_dynamometer_and_wooden_block():
    rec = RecognitionResult(letter="B", experiment_id="friction", open_name="滑动摩擦力",
                            cues=["测力计大致水平放置", "测力计挨着小车或木块"],
                            apparatus=["dynamometer", "wooden_block"], confidence=0.8)
    cands = candidates_from_photo(rec)
    top = cands[0]
    assert top.entry_id == "friction_composed" and top.mirror_of == "friction"
    assert "block.weight_n" in _keys(top)
    assert any("识别结果就是这个实验" in r for r in top.reasons)
    assert top.to_dict()["open_questions"][0]["unit"] == "N"


def test_pendulum_and_free_fall_and_nothing():
    c = candidates_from_photo({"apparatus": ["iron_block", "string", "iron_stand", "stopwatch"],
                               "cues": ["细线下方挂着小球"]})
    assert c[0].entry_id == "pendulum_composed" and c[0].options == {"timer": "stopwatch"}
    c = candidates_from_photo({"apparatus": ["ball", "photogate", "iron_stand"], "cues": ["电磁铁吸住小球，竖直下落"]})
    assert c[0].entry_id == "photogate_free_fall" and "ball.diameter_cm" in _keys(c[0])
    assert candidates_from_photo({"apparatus": ["lever"], "cues": []}) == []
