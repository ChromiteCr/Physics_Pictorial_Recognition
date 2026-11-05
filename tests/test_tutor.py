"""探究式导师：会话状态机、上下文组装、模板讲解（必须通过校验器）、脚本化导师的整条流程。

不导入 labsim 注册表和 facts.py（都在并行开发），实验信息和事实单用小夹具。
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from lab_detective.llm import ScriptedModel
from lab_detective.tutor import templates as tpl
from lab_detective.tutor.context import SYSTEM_PROMPT_ZH, build_context
from lab_detective.tutor.policy import ScriptedTutor, guarded_explanation
from lab_detective.tutor.session import MAX_HINT, MAX_TURNS, IllegalTransition, LabSession
from lab_detective.tutor.verifier import verify

CAT = {"causes": [], "menus": {}}
KP = {"kp.hooke_law": "胡克定律", "kp.systematic_error": "系统误差"}

CARD = {
    "id": "hooke", "name_zh": "探究弹簧弹力与形变量的关系", "grade": "高中",
    "goal": "探究弹簧弹力与伸长量的关系，测出劲度系数。",
    "formula": "F = k x",
    "variables": [{"symbol": "m", "name": "钩码质量", "unit": "g", "role": "control"},
                  {"symbol": "x", "name": "伸长量", "unit": "cm", "role": "measured"},
                  {"symbol": "F", "name": "拉力", "unit": "N", "role": "derived"}],
    "steps": ["竖直悬挂弹簧，读原长", "依次挂钩码，读长度", "算伸长量"],
    "data_table": {"columns": ["钩码质量 m/g", "伸长量 x/cm"]},
    "knowledge_points": ["kp.hooke_law", "kp.systematic_error"],
}


def make_fs():
    return {
        "experiment_id": "hooke", "experiment_name": "探究弹簧弹力与形变量的关系（胡克定律）",
        "protocol": "standard",
        "data": {"n_points": 5, "table": [{"setting": 50, "values": [1.2]}]},
        "panel": {"slope": 4.0123456789, "intercept": 1.5234567, "signatures": ["intercept_pos"]},
        "agreement": {"quantity": "k", "value": 24.1, "uncertainty": 0.6, "unit": "N/m", "z": 2.4, "band": "tension"},
        "diagnosis": {"ranked": [
            {"cause_id": "hooke.hanger_omitted", "name": "漏算挂钩质量", "prob": 0.7123, "evidence": "截距 +1.52 cm（z=6.2）"},
            {"cause_id": "hooke.zero_offset", "name": "原长读数有偏差", "prob": 0.2, "evidence": "截距 +1.52 cm"},
            {"cause_id": "hooke.random_only", "name": "只有随机误差", "prob": 0.0877, "evidence": ""}],
            "not_in_list": False, "misfit_p": 0.4},
        "next_measurement": {"id": "weigh_hanger", "label": "用天平称挂钩质量", "eig_bits": 0.83,
                             "predicted": [{"cause_id": "hooke.hanger_omitted", "value": 23.0, "unit": "g"},
                                           {"cause_id": "hooke.zero_offset", "value": 5.0, "unit": "g"}]},
        "fix": {"cause_id": "hooke.hanger_omitted", "text": "称出挂钩质量，拉力按（钩码 + 挂钩）的总质量计算",
                "predicted_rerun": "截距回到 0 附近"},
        "knowledge_points": ["kp.hooke_law", "kp.systematic_error"],
        "numbers": [{"name": "intercept", "value": 1.52, "unit": "cm"}],
    }


def fs_variants():
    base = make_fs()
    out = {"normal": base}
    v = copy.deepcopy(base)
    v["diagnosis"]["not_in_list"] = True
    v["diagnosis"]["misfit_p"] = 0.0031
    out["not_in_list"] = v
    v = copy.deepcopy(base)
    v["next_measurement"] = None
    out["no_next"] = v
    v = copy.deepcopy(base)
    v["diagnosis"]["ranked"] = [dict(base["diagnosis"]["ranked"][2], prob=0.93, evidence="截距 z=0.4"),
                                dict(base["diagnosis"]["ranked"][0], prob=0.05)]
    v["agreement"].update(z=0.4, band="consistent")
    v["fix"] = {"cause_id": "hooke.random_only", "text": "多测几组、取平均即可"}
    out["random_top"] = v
    v = copy.deepcopy(base)
    v["fix"] = {"text": "称出挂钩质量，" + "按照实验卡上的步骤重新计算每一组的拉力并重新作图，" * 6}
    v["knowledge_points"] = []
    out["long_fix"] = v
    return out


def fake_experiment():
    """只含脚本导师需要的属性的 hooke 替身。"""
    R = lambda key, label, unit, res: SimpleNamespace(key=key, label_zh=label, unit=unit, resolution=res)  # noqa: E731
    obs = {
        "main": SimpleNamespace(label_zh="挂钩码读伸长量", setting_label_zh="钩码质量", setting_unit="g",
                                readings=(R("extension", "伸长量", "cm", 0.1),)),
        "hanger_mass": SimpleNamespace(label_zh="用天平称挂钩质量", setting_label_zh="", setting_unit="",
                                       readings=(R("hanger_mass", "挂钩质量", "g", 0.1),)),
    }
    menu = {"weigh_hanger": SimpleNamespace(plan=(("hanger_mass", 0.0),))}
    return SimpleNamespace(
        name_zh="胡克定律", obs_kinds=obs,
        protocols={"standard": SimpleNamespace(plan=tuple(("main", m) for m in (50.0, 100.0, 150.0, 200.0, 250.0)))},
        menu_item=lambda i: menu[i])


# ── 会话 ────────────────────────────────────────────────────────────────


def test_full_phase_cycle_and_illegal_transitions():
    s = LabSession(experiment_id="hooke")
    for to in ("plan", "predict", "measure", "check", "measure", "check", "diagnose",
               "predict", "measure", "check", "diagnose", "explain", "reflect", "plan"):
        s.advance(to)
    assert s.phase == "plan" and s.eig_round == 0  # reflect → plan 重新开始，追加轮数清零
    with pytest.raises(IllegalTransition):
        s.advance("explain")
    with pytest.raises(IllegalTransition):
        LabSession().advance("measure")
    with pytest.raises(ValueError):
        LabSession(phase="nope")


def test_eig_rounds_are_capped():
    s = LabSession(phase="diagnose")
    for _ in range(3):
        s.advance("measure").advance("check").advance("diagnose")
    with pytest.raises(IllegalTransition):
        s.advance("measure")
    s.advance("explain")


def test_hint_ladder_caps_and_resets_on_phase_change():
    s = LabSession(phase="plan")
    levels = [s.request_hint() for _ in range(5)]
    assert levels == [1, 2, 3, 3, 3] and s.hints_used == MAX_HINT
    s.advance("predict")
    assert s.hint_level == 0 and s.hints_used == 3


def test_json_roundtrip_with_numpy_values():
    s = LabSession(experiment_id="hooke", card=CARD, fact_sheet=make_fs())
    s.advance("plan").advance("predict")
    s.record_prediction("main@250", np.float64(6.25), "cm")
    s.advance("measure")
    i = s.add_observation("main", np.float32(50.0), (np.float64(1.2),))
    s.record_check(i, {"predicted": np.float64(1.25), "z": np.float64(-0.4), "band": "consistent", "unit": "cm"})
    s.touch_kp("kp.hooke_law", "kp.hooke_law", "kp.systematic_error")
    s.add_turn("请测第 1 组" * 20, "1.2")
    s2 = LabSession.from_json(json.dumps(json.loads(s.to_json())))
    assert s2 == LabSession.from_dict(s.to_dict())
    assert s2.to_json() == s.to_json()
    assert s2.observations == [{"kind": "main", "setting": 50.0, "values": [1.2]}]
    assert s2.kp_touched == ["kp.hooke_law", "kp.systematic_error"]
    assert len(s2.turns[0]["tutor"]) <= 60
    with pytest.raises(IndexError):
        s.record_check(5, {})


def test_turns_are_capped():
    s = LabSession()
    for i in range(MAX_TURNS + 10):
        s.add_turn(f"t{i}", f"s{i}")
    assert len(s.turns) == MAX_TURNS and s.turns[-1]["tutor"] == f"t{MAX_TURNS + 9}"
    assert [t["student"] for t in s.last_turns(2)] == [f"s{MAX_TURNS + 8}", f"s{MAX_TURNS + 9}"]


# ── 上下文 ──────────────────────────────────────────────────────────────


def _session_in(phase):
    s = LabSession(experiment_id="hooke", card=CARD, fact_sheet=make_fs(), phase=phase)
    for i, m in enumerate((50.0, 100.0, 150.0, 200.0, 250.0)):
        s.add_observation("main", m, [round(0.4 * m / 50 + 1.5, 1)])
    s.record_prediction("main@250", 6.0, "cm")
    for i in range(5):
        s.add_turn(f"老师第{i}轮标记TURN{i}", f"学生第{i}轮")
    return s


@pytest.mark.parametrize("phase", ["identify", "plan", "predict", "measure", "check", "diagnose", "explain", "reflect"])
def test_context_is_deterministic_and_phase_scoped(phase):
    s = _session_in(phase)
    m1 = build_context(s, make_fs(), CARD, kp_names=KP)
    m2 = build_context(copy.deepcopy(s), make_fs(), CARD, kp_names=KP)
    assert m1 == m2
    assert [m["role"] for m in m1] == ["system", "user"] and m1[0]["content"] == SYSTEM_PROMPT_ZH
    user = m1[1]["content"]
    assert "TURN3" in user and "TURN4" in user and "TURN2" not in user  # 只放最近 2 轮
    assert "4.0123" not in user and "1.5234" not in user  # 不放面板原始浮点数
    assert sum(len(m["content"]) for m in m1) <= 3000
    if phase in ("diagnose", "explain"):
        assert "漏算挂钩质量" in user and "可引用数字" in user and "用天平称挂钩质量" in user
    if phase in ("plan", "measure", "identify"):
        assert "嫌疑人" not in user and "漏算挂钩质量" not in user


@pytest.mark.parametrize("budget", [3000, 1200, 800, 500])
def test_context_respects_budget(budget):
    s = _session_in("explain")
    msgs = build_context(s, make_fs(), CARD, budget_chars=budget, kp_names=KP, student_text="为什么" * 50)
    assert sum(len(m["content"]) for m in msgs) <= budget
    assert msgs[0]["content"] == SYSTEM_PROMPT_ZH
    assert "【任务】" in msgs[1]["content"]


# ── 模板 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(fs_variants()))
def test_template_explanation_passes_verifier(name):
    fs = fs_variants()[name]
    blocks = tpl.explanation_blocks(fs, KP)
    assert all(tpl.nchars(b) <= tpl.BLOCK_LIMIT for b in blocks.values()), {k: tpl.nchars(b) for k, b in blocks.items()}
    text = tpl.explanation_text(fs, KP)
    r = verify(text, fs, catalog=CAT)
    assert r["ok"], (text, r["problems"])


def test_template_content():
    b = tpl.explanation_blocks(make_fs(), KP)
    assert "「漏算挂钩质量」" in b["cause"] and "71%" in b["cause"]
    assert "1.52 cm" in b["evidence"] and "24.1 ± 0.6 N/m" in b["evidence"]
    assert "用天平称挂钩质量" in b["next"] and "23 g" in b["next"] and "5 g" in b["next"]
    assert b["kp"] == "关联知识点：胡克定律、系统误差。"
    assert "清单之外" in tpl.explanation_blocks(fs_variants()["not_in_list"], KP)["cause"]
    fs = make_fs()
    fs["next_measurement"]["predicted"].reverse()  # 事实单里的顺序不一定按排名
    nxt = tpl.block_next(fs)
    assert nxt.index("漏算挂钩质量") < nxt.index("原长读数有偏差")


def test_fmt_and_pct():
    assert [tpl.fmt(v) for v in (24.1, 1.52, 23.0, 0.7123456, 1234.0, -0.5, 0.0)] == \
        ["24.1", "1.52", "23", "0.7123", "1234", "-0.5", "0"]
    assert [tpl.pct(p) for p in (0.7123, 0.026, 0.997, 0.2)] == ["71", "2.6", "99.7", "20"]


# ── 脚本化导师 ──────────────────────────────────────────────────────────


def test_scripted_tutor_full_session():
    fs = make_fs()
    s = LabSession(experiment_id="hooke", card=CARD)
    tutor = ScriptedTutor(s, fs, CARD, experiment=fake_experiment(), kp_names=KP)

    m = tutor.next_message()
    assert m["phase"] == "identify" and m["expects"] == "choice" and "探究弹簧弹力" in m["text"]
    s.advance(tutor.suggest_next_phase())

    m = tutor.next_message()
    assert m["expects"] == "choice" and m["answer"] == "钩码质量 m" and "拉力 F" not in m["options"]
    s.request_hint()
    s.request_hint()
    s.request_hint()
    assert "钩码质量" in tutor.next_message()["text"].split("\n")[-1]  # 第 3 级提示接近答案
    s.advance(tutor.suggest_next_phase())

    m = tutor.next_message()
    assert m["phase"] == "predict" and m["expects"] == "prediction" and m["target"] == "main@250"
    assert "250 g" in m["text"] and m["unit"] == "cm"
    tutor.receive("6")
    assert s.prediction_for("main@250")["value"] == 6.0
    s.advance(tutor.suggest_next_phase())

    readings = [1.6, 3.6, 5.5, 7.6, 9.6]
    for i, x in enumerate(readings):
        assert s.phase == "measure"
        m = tutor.next_message()
        assert m["expects"] == "reading" and f"第 {i + 1} 组" in m["text"] and "记到 0.1 cm" in m["text"]
        tutor.receive(str(x))
        s.advance("check")
        band = "inconsistent" if i == 2 else "consistent"
        s.record_check(i, {"value": x, "predicted": x - 0.1, "uncertainty": 0.1, "unit": "cm", "z": 3.1 if i == 2 else 0.5,
                           "band": band})
        m = tutor.next_message()
        assert ("重读这一行" in m["options"]) == (band != "consistent")
        s.advance(tutor.suggest_next_phase())
    assert s.phase == "diagnose" and len(s.observations) == 5

    m = tutor.next_message()
    names = [r["name"] for r in sorted(fs["diagnosis"]["ranked"], key=lambda r: r["cause_id"])]
    assert m["expects"] == "choice" and m["options"] == names  # 按 id 排，不按概率，不泄露排名
    assert "漏算挂钩质量" not in m["text"]  # 学生选之前不揭示
    tutor.receive("原长读数有偏差")
    m = tutor.next_message()
    assert "引擎认为更可能的是「漏算挂钩质量」" in m["text"] and "用天平称挂钩质量" in m["text"]
    assert m["options"][0] == "做这个测量"

    tutor.start_next_measurement()
    s.advance(tutor.suggest_next_phase())
    m = tutor.next_message()
    assert s.phase == "predict" and m["expects"] == "choice"
    assert m["options"][:2] == ["约 5 g（原长读数有偏差）", "约 23 g（漏算挂钩质量）"]  # 按数值排
    tutor.receive(m["options"][1])
    s.advance("measure")
    m = tutor.next_message()
    assert m["expects"] == "reading" and m["target"] == {"kind": "hanger_mass", "setting": 0.0}
    assert "挂钩质量" in m["text"]
    tutor.receive("22.8")
    s.advance("check")
    assert tutor.suggest_next_phase() == "diagnose"
    s.advance("diagnose")
    s.pending_measurement = None
    assert tutor.suggest_next_phase() == "explain"
    s.advance("explain")

    m = tutor.next_message()
    assert m["text"] == tpl.explanation_text(fs, KP)
    assert verify(m["text"], fs, catalog=CAT)["ok"]
    tutor.receive("明白了，去写结论")
    assert s.kp_touched == ["kp.hooke_law", "kp.systematic_error"]
    s.advance("reflect")
    m = tutor.next_message()
    assert m["expects"] == "free" and "6 cm" in m["text"] and "9.6 cm" in m["text"]
    assert tutor.next_message() == m  # 确定性


def test_diagnose_bottom_out_hint_reveals_top():
    s = LabSession(experiment_id="hooke", card=CARD, phase="diagnose")
    tutor = ScriptedTutor(s, make_fs(), CARD, experiment=fake_experiment(), kp_names=KP)
    s.request_hint()
    assert "截距 +1.52 cm" not in tutor.next_message()["text"]
    s.request_hint()
    assert "截距 +1.52 cm" in tutor.next_message()["text"]
    s.request_hint()
    m = tutor.next_message()
    assert "最可能的原因是「漏算挂钩质量」" in m["text"]


GOOD_MODEL = "最可能的原因是漏算挂钩质量，截距 1.52 cm。下一步：用天平称挂钩质量。"
BAD_MODEL = "最可能的原因是空气阻力，截距 3.3 cm。"


@pytest.mark.parametrize("responses,source", [
    ([GOOD_MODEL], "model"),
    ([BAD_MODEL, GOOD_MODEL], "model_rewrite"),
    ([BAD_MODEL, BAD_MODEL], "template"),
    ([RuntimeError("服务器没开")], "template"),
])
def test_guarded_explanation(responses, source):
    fs = make_fs()
    s = LabSession(experiment_id="hooke", card=CARD, phase="explain")
    model = ScriptedModel(responses)
    out = guarded_explanation(model, s, fs, CARD, kp_names=KP, catalog=CAT)
    assert out["source"] == source and out["verify"]["ok"]
    if source == "template":
        assert out["text"] == tpl.explanation_text(fs, KP)
    if len(model.calls) == 2:
        assert "没有通过事实核对" in model.calls[1]["messages"][-1]["content"]
    assert guarded_explanation(None, s, fs, CARD, kp_names=KP, catalog=CAT)["source"] == "template"


def test_long_numbers_list_never_hits_hard_cut():
    """事实单 numbers 很长（facts.py 会登记 40+ 条）时，explain / diagnose 上下文也不应被硬截断。"""
    fs = make_fs()
    fs["numbers"] += [{"name": f"data[{i}].extension", "value": 1.0 + i / 7, "unit": "cm"} for i in range(60)]
    fs["numbers"] += [{"name": f"panel_extra_{i}", "value": 0.123456 * i, "unit": "cm/N"} for i in range(30)]
    for phase in ("diagnose", "explain"):
        s = _session_in(phase)
        user = build_context(s, fs, CARD, budget_chars=3000, kp_names=KP)[1]["content"]
        assert not user.endswith("…")
        line = next(x for x in user.split("\n") if x.startswith("【事实单】"))
        facts = json.loads(line[len("【事实单】"):])
        assert facts["嫌疑人"][0]["原因"] == "漏算挂钩质量"
        assert "data[" not in user
