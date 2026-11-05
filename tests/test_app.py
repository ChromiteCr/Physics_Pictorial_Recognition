"""Streamlit 界面（app.py + ui/）：AppTest 冒烟测试 + 界面背后纯逻辑的单元测试。

全部离线：socket 连接一律拦截；不加载 YOLO / torch（检测模块只在点"开始检测"时懒加载）；不调用任何模型。
"""
import json
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from lab_detective.labsim.core import Observation  # noqa: E402
from lab_detective.twin import instantiate  # noqa: E402
from ui import catalog, lab, llm  # noqa: E402

APP = str(ROOT / "app.py")
FORBIDDEN = ("explain_api", "zhipuai", "intent_classifier", "ultralytics", "torch", "detect")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """拦截所有 IP 连接（本机地址也拦），测试里出现联网就会失败或走离线兜底。"""
    real = socket.socket.connect

    def guarded(self, addr):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise OSError("测试中禁止联网")
        return real(self, addr)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("测试中禁止联网")))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _texts(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    for kind in ("info", "success", "warning", "error"):
        parts += [e.value for e in getattr(at, kind)]
    return "\n".join(parts)


def _pick(at, exp_id: str) -> None:
    at.selectbox(key="manual_pick").select(exp_id)
    at.button(key="pick_manual").click().run()
    assert not at.exception, at.exception
    assert at.session_state["exp_key"] == exp_id


# ── AppTest 冒烟测试 ─────────────────────────────────────────────────


def test_app_loads_offline_without_heavy_or_glm_modules():
    before = set(sys.modules)
    at = _app()
    assert len(at.tabs) == 5
    assert [t.label for t in at.tabs][0].startswith("①")
    new = set(sys.modules) - before
    assert not [m for m in FORBIDDEN if m in new], [m for m in FORBIDDEN if m in new]
    assert "explain_api" not in sys.modules and "intent_classifier" not in sys.modules
    assert at.toggle(key="model_mode").value is False  # 默认脚本模式


def test_hooke_scripted_flow_reaches_diagnosis_and_explanation():
    at = _app()
    _pick(at, "hooke")
    # 先预测再测量：孪生预测表在写下预测之前不显示
    assert "先写下你的预测" in _texts(at)
    at.number_input(key="pred_value").set_value(12.0)
    at.button(key="pred_save").click().run()
    assert not at.exception, at.exception
    assert at.session_state["lab"].student_predictions[0]["value"] == 12.0
    assert "孪生的预测表" in _texts(at)
    # 演示数据：注入"漏算挂钩质量"
    at.selectbox(key="sim_cause").select("hooke.hanger_omitted")
    at.button(key="simulate").click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["obs"]) == 5
    assert at.session_state["lab"].checks and at.session_state["lab"].checks[-1]["band"] is not None
    # 先猜，再揭晓嫌疑人看板
    at.radio(key="guess_cause@0").set_value("漏算挂钩质量")
    at.button(key="reveal").click().run()
    assert not at.exception, at.exception
    text = _texts(at)
    assert "嫌疑人看板" in text and "漏算挂钩质量" in text
    assert "和引擎的判断一致" in text
    assert "下一次测量" in text and "修正方法" in text
    fs = at.session_state["lab"].fact_sheet
    assert fs["diagnosis"]["ranked"][0]["cause_id"] == "hooke.hanger_omitted"
    assert "事实核对通过" in text  # ⑤ 模板讲解经过校验器
    assert "相关知识点" in text


def test_extra_measurement_and_rerun_round_trip():
    at = _app()
    _pick(at, "friction")
    at.button(key="pred_skip").click().run()
    at.selectbox(key="sim_cause").select("friction.zero_vertical")
    at.button(key="simulate").click().run()
    at.button(key="reveal").click().run()
    assert not at.exception, at.exception
    at.button(key="simulate_extra").click().run()
    at.button(key="add_extra").click().run()
    assert not at.exception, at.exception
    extras = at.session_state["extras"]
    assert len(extras) == 1 and extras[0]["obs"]
    at.button(key="reveal").click().run()
    at.button(key="simulate_rerun").click().run()
    assert not at.exception, at.exception
    assert "复测后最可能的原因" in _texts(at)


def test_other_experiment_composed_from_components():
    at = _app()
    at.selectbox(key="manual_pick").select(catalog.OTHER).run()
    [b for b in at.button if b.label == "组合出孪生"][0].click().run()
    assert not at.exception, at.exception
    key = at.session_state["exp_key"]
    assert catalog.is_other(key)
    exp = catalog.get_experiment(key)
    at.button(key="pred_skip").click().run()
    at.selectbox(key="sim_cause").select(exp.cause_ids()[0])
    at.button(key="simulate").click().run()
    at.button(key="reveal").click().run()
    assert not at.exception, at.exception
    assert "嫌疑人看板" in _texts(at)


def test_model_mode_without_server_falls_back_to_scripted():
    at = _app()
    at.toggle(key="model_mode").set_value(True).run()
    assert not at.exception, at.exception
    assert at.session_state["llm_probe"]["ok"] is False
    assert any("脚本模式" in w.value for w in at.warning)


def test_p1_detection_canvas_and_clarifying_question(tmp_path):
    """P1 的检测结果 → 器材表 → 识别实验 → 澄清问题（不跑 YOLO：直接放进检测结果）。"""
    from PIL import Image

    img = tmp_path / "setup.png"
    Image.new("RGB", (320, 200), (40, 60, 50)).save(img)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["init_image_path"] = str(img)
    at.session_state["annotated_image_rgb"] = np.zeros((200, 320, 3), dtype=np.uint8)
    at.session_state["components"] = [
        {"class_id": 0, "class_name": "cart", "confidence": 0.9, "bbox": [20.0, 20.0, 80.0, 60.0]},
        {"class_id": 1, "class_name": "track", "confidence": 0.8, "bbox": [0.0, 50.0, 320.0, 70.0]},
    ]
    at.run()
    assert not at.exception, at.exception
    at.button(key="recognize_btn").click().run()
    assert not at.exception, at.exception
    recog = at.session_state["recog"]
    assert recog["needs_question"] and recog["question"]["text"] == "你要测什么？"
    q = at.radio(key=f"clarify_{recog['question']['id']}")
    q.set_value("悬挂重物拉动小车时的加速度")
    at.button(key="clarify_btn").click().run()
    assert not at.exception, at.exception
    assert at.session_state["recog"]["experiment_id"] == "newton2"
    at.button(key="use_recognized").click().run()
    assert at.session_state["exp_key"] == "newton2"


# ── 纯逻辑 ───────────────────────────────────────────────────────────


def test_first_row_prediction_equals_pre_lab_twin_and_later_rows_narrow():
    exp = catalog.get_experiment("hooke")
    plan = lab.plan_for(exp, "standard")
    m0, s0 = lab.predictive(exp, [], "main", plan[0][1])[0]
    row = instantiate(exp).predict_rows(plan)[0]["readings"][0]
    assert m0 == pytest.approx(row["mean"], rel=1e-9)
    assert s0 == pytest.approx(row["sd"], rel=1e-6)
    ds = lab.simulate(exp, exp.random_only_id, "standard", 3)
    checks = lab.live_checks(exp, ds.observations)
    assert len(checks) == 5
    assert checks[-1]["uncertainty"] < checks[0]["uncertainty"] / 3  # 用数据校准后预测变窄
    assert {c["band"] for c in checks} <= {"consistent", "tension", "inconsistent"}


def test_live_check_flags_gross_misreading():
    exp = catalog.get_experiment("hooke")
    ds = lab.simulate(exp, exp.random_only_id, "standard", 5)
    obs = list(ds.observations)
    last = obs[-1]
    obs[-1] = Observation(last.kind, last.setting, (last.values[0] * 10,))  # 小数点错了一位
    assert lab.live_checks(exp, obs)[-1]["band"] == "inconsistent"


def test_table_round_trip_and_partial_rows():
    exp = catalog.get_experiment("average_speed")
    ds = lab.simulate(exp, exp.random_only_id, "standard", 1)
    df = lab.obs_to_df(exp, "main", ds.observations)
    back, partial = lab.df_to_observations(exp, "main", df)
    assert [o.values for o in back] == [o.values for o in ds.observations]
    assert partial == 0
    df.loc[0, "t2"] = np.nan
    back, partial = lab.df_to_observations(exp, "main", df)
    assert len(back) == len(ds.observations) - 1 and partial == 1
    tmpl = lab.template_df(exp, "main", [s for _, s in lab.plan_for(exp, "standard")])
    assert lab.df_to_observations(exp, "main", tmpl) == ([], 0)
    assert set(lab.column_labels(exp, "main")) == {"setting", "t1", "t2", "t3"}


def test_parse_settings_and_spec_validation():
    assert catalog.parse_settings("10, 20 30；40、50") == [10, 20, 30, 40, 50]
    with pytest.raises(ValueError):
        catalog.parse_settings("10, abc")
    x = {"name": "体积", "symbol": "V", "unit": "cm³", "tool": "量筒", "resolution": 1.0}
    y = {"name": "质量", "symbol": "m", "unit": "g", "tool": "电子天平", "resolution": 0.1}
    with pytest.raises(ValueError):
        catalog.make_spec("", x, y, "proportional", [10, 20])  # 少于 3 个取值
    with pytest.raises(ValueError):
        catalog.make_spec("", x, y, "inverse", [0, 10, 20])  # 反比要求 x > 0
    spec = catalog.make_spec("", x, y, "proportional", [10, 20, 30, 40])
    key = catalog.other_key(spec)
    assert catalog.get_experiment(key) is catalog.get_experiment(catalog.other_key(json.loads(json.dumps(spec))))
    exp = catalog.get_experiment(key)
    assert exp.cause_ids()[0].endswith(".random_only")


def test_calibrate_other_centres_slope_prior_on_data():
    x = {"name": "体积", "symbol": "V", "unit": "cm³", "tool": "量筒", "resolution": 1.0}
    y = {"name": "质量", "symbol": "m", "unit": "g", "tool": "电子天平", "resolution": 0.1}
    key = catalog.other_key(catalog.make_spec("", x, y, "proportional", [10, 20, 30, 40, 50]))
    obs = [Observation("main", v, (8.9 * v,)) for v in (10, 20, 30)]
    k2 = catalog.calibrate_other(key, obs)
    assert catalog.spec_of(k2)["slope_guess"] == pytest.approx(8.9)
    assert catalog.calibrate_other("hooke", obs) == "hooke"
    assert catalog.calibrate_other(key, obs[:1]) == key


def test_rerun_compare_verdicts():
    before = {"numbers": [{"name": "rerun.intercept", "value": 0.0}, {"name": "rerun.intercept_se", "value": 0.06}]}
    after = {"panel": {"intercept": 0.05, "intercept_se": 0.06},
             "diagnosis": {"ranked": [{"cause_id": "hooke.random_only", "name": "只有随机误差", "prob": 0.8}]}}
    r = lab.rerun_compare(before, after, "hooke.random_only")
    assert r["verdict"] == "fixed" and r["band"] == "consistent"
    after["diagnosis"]["ranked"][0]["prob"] = 0.3
    assert lab.rerun_compare(before, after, "hooke.random_only")["verdict"] == "likely"
    after["diagnosis"]["ranked"][0]["cause_id"] = "hooke.zero_offset"
    assert lab.rerun_compare(before, after, "hooke.random_only")["verdict"] == "not_fixed"


def test_design_helpers_for_every_curated_experiment():
    for item in catalog.curated():
        exp = catalog.get_experiment(item["id"])
        a = lab.range_advice(exp, "standard")
        assert a["lo"] <= a["hi"] and a["n"] >= 1
    rows = lab.separability(catalog.get_experiment("friction"), "standard", n=30)
    assert all(r["status"] in ("confused", "partial") for r in rows)


def test_probe_only_local_and_parses_models():
    assert llm.probe("https://api.example.com/v1")["ok"] is False

    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seen = {}

    def opener(url, timeout):
        seen["url"], seen["timeout"] = url, timeout
        return Resp(json.dumps({"data": [{"id": "qwen-4b-lab"}]}).encode())

    r = llm.probe("http://127.0.0.1:8080/v1/", opener=opener)
    assert r == {"ok": True, "models": ["qwen-4b-lab"], "error": ""}
    assert seen == {"url": "http://127.0.0.1:8080/v1/models", "timeout": 0.5}
    assert llm.probe("http://127.0.0.1:9/v1")["ok"] is False  # 被拦截的连接 → 不可用，不抛异常


def test_ui_text_has_no_fixed_experiment_count_and_no_online_fonts():
    files = [ROOT / "app.py", *sorted((ROOT / "ui").glob("*.py"))]
    for f in files:
        text = f.read_text(encoding="utf-8")
        for bad in ("12 个", "十二个", "12个实验", "fonts.googleapis", "explain_api", "zhipuai", "GLM"):
            assert bad not in text, f"{f.name} 含有 {bad!r}"


def test_streamlit_config_is_offline():
    cfg = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "gatherUsageStats = false" in cfg
