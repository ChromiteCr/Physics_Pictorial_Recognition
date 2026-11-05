"""vision：字母表、schema 往返、规则线索、Set-of-Mark、提示词、识别流水线（假模型）、数据集构建辅助。

全部离线，不加载 YOLO（检测框直接传入）。
"""

from __future__ import annotations

import json
import math

import pytest
from PIL import Image

from lab_detective.vision import build_dataset as bd
from lab_detective.vision import cues, letters, pipeline, prompt, schema, som
from lab_detective.vision.schema import RecognitionParseError, RecognitionResult

# ── 常用检测框 ─────────────────────────────────────────────────────────

CART_TRACK = [
    {"class_name": "cart", "bbox": [100, 100, 150, 130]},
    {"class_name": "track", "bbox": [0, 90, 400, 140]},
]
FRICTION = CART_TRACK + [
    {"class_name": "dynamometer", "bbox": [160, 105, 230, 120]},
    {"class_name": "string", "bbox": [148, 108, 162, 118]},
]
MOMENTUM = [
    {"class_name": "cart", "bbox": [50, 100, 100, 130]},
    {"class_name": "cart", "bbox": [300, 100, 350, 130]},
    {"class_name": "track", "bbox": [0, 90, 400, 140]},
]


class FakeModel:
    """假模型：记录调用，返回固定文本（或抛异常）。"""

    def __init__(self, text: str | None = None, exc: Exception | None = None):
        self.text, self.exc, self.calls = text, exc, []

    def chat(self, messages, json_schema=None, images=None):
        self.calls.append({"messages": messages, "json_schema": json_schema, "images": images})
        if self.exc:
            raise self.exc
        return {"text": self.text}


@pytest.fixture()
def photo(tmp_path):
    p = tmp_path / "photo.jpg"
    Image.new("RGB", (800, 400), (200, 200, 200)).save(p)
    return p


# ── 字母表 ─────────────────────────────────────────────────────────────


def test_letter_table_fixed():
    expect = ["hooke", "friction", "average_speed", "newton2", "momentum", "energy",
              "incline_friction", "free_fall", "atwood", "pendulum", "projectile", "lever"]
    assert letters.LETTERS == tuple("ABCDEFGHIJKL") + ("Z",)
    assert list(letters.EXPERIMENT_IDS) == expect
    assert len(set(letters.EXPERIMENT_IDS)) == 12
    for e in letters.TABLE:
        assert e.name_zh and e.short_zh
        assert letters.letter_for(e.experiment_id) == e.letter
        assert letters.experiment_for(e.letter) == e.experiment_id
    assert [e.held_out for e in letters.TABLE[:12]] == [False] * 6 + [True] * 6
    assert letters.experiment_for("Z") is None and letters.letter_for("nope") == "Z"
    assert set(letters.OLD_ID_MAP.values()) <= set(expect)


def test_letter_table_matches_labsim_registry():
    from lab_detective.labsim.registry import EXPERIMENTS

    assert letters.letter_for("hooke") == "A"
    # 已注册的实验必须都在字母表里（其他实验由并行开发者按这份 id 实现）
    for exp_id in EXPERIMENTS:
        if not exp_id.startswith("generic"):
            assert exp_id in letters.EXPERIMENT_IDS, exp_id


def test_normalize_letter():
    for raw, want in [("a", "A"), ("A.", "A"), ("(D)", "D"), ("Z", "Z"), ("B：摩擦", "B"),
                      ("hooke", None), ("M", None), ("", None), (3, None)]:
        assert letters.normalize_letter(raw) == want, raw


# ── schema ─────────────────────────────────────────────────────────────


def test_schema_round_trip():
    r = RecognitionResult(letter="J", experiment_id=None, open_name="单摆测重力加速度",
                          cues=["细线下挂小球"], apparatus=["string", "ball"], confidence=0.7,
                          source="model", question={"id": "q.ball", "options": []})
    assert r.experiment_id == "pendulum"  # 字母说了算
    d = json.loads(r.to_json())
    assert schema.validate(d) == []
    assert RecognitionResult.from_dict(d) == r
    assert schema.validate(r.model_part(), schema.MODEL_OUTPUT_SCHEMA) == []
    back = schema.parse_model_output(prompt.assistant_target(r))
    assert back.model_part() == r.model_part()


def test_schema_rejects_bad():
    with pytest.raises(ValueError):
        RecognitionResult(letter="Q", experiment_id=None, open_name="x")
    bad = {"letter": "A", "open_name": "x", "cues": [], "apparatus": ["unicorn"], "confidence": 2}
    assert len(schema.validate(bad, schema.MODEL_OUTPUT_SCHEMA)) == 2


def test_parse_tolerant():
    text = ('<think>先看看……</think>好的，结果如下：\n```json\n'
            '{"letter": "d", "open_name": "牛顿第二定律", "cues": "小车连细线，跨过滑轮", '
            '"apparatus": ["小车", "滑轮", "钩码", "魔法棒"], "confidence": "85%", '
            '"experiment_id": "energy"}\n```')
    r = schema.parse_model_output(text)
    assert r.letter == "D" and r.experiment_id == "newton2"
    assert r.cues == ["小车连细线", "跨过滑轮"]
    assert r.apparatus == ["cart", "pulley", "slotted_mass"]
    assert r.confidence == pytest.approx(0.85)
    assert any("魔法棒" in w for w in r.warnings) and any("不一致" in w for w in r.warnings)
    # 字母缺失时用 experiment_id 反推
    assert schema.parse_model_output('{"experiment_id": "lever", "open_name": ""}').letter == "L"
    for bad in ["没有 JSON", '{"letter": "Q"}', "```json\n[1, 2]\n```"]:
        with pytest.raises(RecognitionParseError):
            schema.parse_model_output(bad)


# ── 规则线索 ───────────────────────────────────────────────────────────


def test_rules_confident_cases():
    a = cues.analyze(FRICTION)
    assert a["confident"] and a["candidates"][0]["experiment_id"] == "friction"
    assert a["apparatus"] == ["cart", "track", "string", "dynamometer"]
    assert "测力计大致水平放置" in a["cues"] and "小车在轨道上" in a["cues"]
    m = cues.analyze(MOMENTUM)
    assert m["confident"] and m["candidates"][0]["letter"] == "E"
    assert 0.5 < m["confidence"] < 1.0


def test_rules_tie_and_exclusions():
    a = cues.analyze(CART_TRACK)
    assert not a["confident"] and sorted(a["tied"]) == ["C", "D", "F"]
    # ruler / spring 推理时不输出，规则输入也忽略
    a2 = cues.analyze(CART_TRACK + [{"class_name": "ruler", "bbox": [0, 0, 10, 10]},
                                    {"class_name": "spring", "bbox": [0, 0, 5, 5]}])
    assert a2["apparatus"] == ["cart", "track"] and a2["tied"] == a["tied"]
    assert cues.analyze([])["candidates"] == [] and cues.rule_confidence([]) == 0.0
    # 单摆弱规则：只进候选，永远不单独"有把握"
    pend = cues.analyze([{"class_name": "string", "bbox": [195, 0, 205, 200]},
                         {"class_name": "iron_block", "bbox": [185, 200, 215, 230]}])
    assert pend["tied"] == ["J"] and not pend["confident"]


# ── Set-of-Mark ────────────────────────────────────────────────────────


def test_som_draws_and_resizes(tmp_path):
    img = Image.new("RGB", (2000, 1000), (255, 255, 255))
    dets = [{"class_name": "cart", "bbox": [500, 400, 900, 700]},
            {"class_name": "track", "bbox": [100, 650, 1900, 800]}]
    out = tmp_path / "som.jpg"
    im, legend = som.draw_marks(img, dets, out_path=out)
    assert im.size == (768, 384) and out.exists()
    s = 768 / 2000
    assert legend[0]["bbox"] == pytest.approx([500 * s, 400 * s, 900 * s, 700 * s], abs=0.2)
    assert [e["id"] for e in legend] == [1, 2] and legend[0]["name_zh"] == "小车"
    # 框线是编号 1 的颜色，框内部仍是白色
    x2, ymid = round(900 * s) - 1, round(550 * s)
    assert im.getpixel((x2, ymid)) == som.PALETTE[0]
    assert im.getpixel((round(700 * s), ymid)) == (255, 255, 255)
    assert "1 = 小车（cart）" in som.legend_text(legend)
    small, _ = som.draw_marks(Image.new("RGB", (300, 200)), [])
    assert small.size == (300, 200)  # 不放大


def test_som_respects_exif_orientation(tmp_path):
    img = Image.new("RGB", (400, 200), (0, 0, 0))
    exif = Image.Exif()
    exif[274] = 6  # 顺时针转 90°
    p = tmp_path / "rot.jpg"
    img.save(p, exif=exif)
    im, _ = som.draw_marks(p, [])
    assert im.size == (200, 400)


# ── 提示词 ─────────────────────────────────────────────────────────────


def test_prompt_stable_and_complete(tmp_path):
    legend = [{"id": 1, "class_name": "cart", "name_zh": "小车", "bbox": [0, 0, 1, 1]}]
    m1 = prompt.build_messages("images/a.jpg", legend)
    assert m1 == prompt.build_messages("images/a.jpg", legend)
    assert m1[0]["role"] == "system" and m1[1]["content"][0] == {"type": "image", "image": "images/a.jpg"}
    for e in letters.TABLE:
        assert f"{e.letter} = {e.name_zh}" in prompt.SYSTEM_PROMPT
    assert "1 = 小车（cart）" in m1[1]["content"][1]["text"]
    assert len(prompt.prompt_hash()) == 12
    p = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8)).save(p)
    oa = prompt.to_openai_messages(prompt.build_messages(p, legend))
    assert oa[1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


# ── 澄清问题 ───────────────────────────────────────────────────────────


def test_question_cart_tie_picks_goal_question():
    q = pipeline.choose_question({"C": 1 / 3, "D": 1 / 3, "F": 1 / 3})
    assert q["id"] == "q.goal_cart"
    assert q["gain_bits"] == pytest.approx(math.log2(3), abs=1e-4)
    assert [o["letters"] for o in q["options"]] == [["C"], ["D"], ["F"], []]
    q2 = pipeline.choose_question({"A": 0.5, "B": 0.5})
    assert q2["id"] == "q.dynamometer"
    assert pipeline.choose_question({"A": 1.0}) is None


def test_question_gain_matches_definition():
    # incline 题：C、F 答"是"，D 答"否"，均匀先验下增益 = H(1/3,1/3,1/3) − 2/3·1
    gain = pipeline.expected_gain({"C": 1 / 3, "D": 1 / 3, "F": 1 / 3}, pipeline.QUESTION_BANK[3][2])
    assert gain == pytest.approx(math.log2(3) - 2 / 3, abs=1e-9)


def test_question_bank_consistency():
    covered = set()
    for qid, text, options in pipeline.QUESTION_BANK:
        assert qid.startswith("q.") and text
        for label, ls in options:
            assert label and set(ls) <= set(letters.LETTERS) - {"Z"}
            covered |= set(ls)
    assert covered == set("ABCDEFGHIJKL")


# ── 流水线 ─────────────────────────────────────────────────────────────


def test_pipeline_confident_rules_skip_model(photo):
    fm = FakeModel('{"letter": "A"}')
    r = pipeline.recognize(photo, fm, detections=FRICTION)
    assert r.letter == "B" and r.source == "rule" and not r.needs_question
    assert fm.calls == []
    assert schema.validate(r.to_dict()) == []


def test_pipeline_model_breaks_tie(photo, tmp_path):
    fm = FakeModel('```json\n{"letter": "D", "open_name": "牛顿第二定律", "cues": ["细线跨过滑轮挂钩码"], '
                   '"apparatus": ["cart", "track", "pulley"], "confidence": 0.9}\n```')
    r = pipeline.recognize(photo, fm, detections=CART_TRACK, work_dir=tmp_path)
    assert r.letter == "D" and r.experiment_id == "newton2" and r.source == "model"
    assert not r.needs_question and "pulley" in r.apparatus
    call = fm.calls[0]
    assert call["json_schema"] == schema.MODEL_OUTPUT_SCHEMA
    assert not call["images"] and (tmp_path / "som.jpg").exists()  # 图片只在消息里出现一次
    assert call["messages"][1]["content"][0] == {"type": "image", "image": str(tmp_path / "som.jpg")}
    assert "1 = 小车（cart）" in call["messages"][1]["content"][1]["text"]
    assert schema.validate(r.to_dict()) == []


def test_pipeline_low_confidence_asks_question_then_answer(photo, tmp_path):
    fm = FakeModel('{"letter": "C", "open_name": "平均速度", "cues": [], "apparatus": [], "confidence": 0.3}')
    r = pipeline.recognize(photo, fm, detections=CART_TRACK, work_dir=tmp_path)
    assert r.needs_question and r.question["id"] == "q.goal_cart" and r.source == "model"
    assert r.letter == "C"  # 模型稍微偏向 C，作为默认选中项
    labels = [o["label"] for o in r.question["options"]]
    r2 = pipeline.answer(r, labels.index("悬挂重物拉动小车时的加速度"))
    assert r2.letter == "D" and r2.source == "question" and not r2.needs_question
    r3 = pipeline.answer(r, len(labels) - 1)  # 都不是 → 其他实验
    assert r3.letter == "Z" and r3.experiment_id is None
    assert schema.validate(r2.to_dict()) == [] and schema.validate(r.to_dict()) == []


def test_pipeline_model_failure_falls_back(photo, tmp_path):
    for fm in (FakeModel(exc=RuntimeError("服务没启动")), FakeModel("我觉得是胡克定律")):
        r = pipeline.recognize(photo, fm, detections=CART_TRACK, work_dir=tmp_path)
        assert r.source == "rule_fallback" and r.needs_question
        assert r.letter in {"C", "D", "F"} and any("模型识别失败" in w for w in r.warnings)


def test_pipeline_no_model_no_detections_asks(photo):
    r = pipeline.recognize(photo, None, detections=[])
    assert r.needs_question and r.question is not None
    assert set(r.question["prior"]) == set(letters.LETTERS)


def test_manual_choice_always_wins(photo, tmp_path):
    fm = FakeModel('{"letter": "D", "open_name": "x", "cues": [], "apparatus": [], "confidence": 0.99}')
    r = pipeline.recognize(photo, fm, detections=FRICTION, manual="E")
    assert r.letter == "E" and r.source == "manual" and fm.calls == []
    auto = pipeline.recognize(photo, fm, detections=CART_TRACK, work_dir=tmp_path)
    m = pipeline.choose_manual(auto, "pendulum", open_name="单摆")
    assert m.letter == "J" and m.source == "manual" and m.open_name == "单摆" and m.confidence == 1.0
    assert pipeline.choose_manual(auto, "Z").experiment_id is None


# ── 数据集构建辅助（不加载 YOLO） ───────────────────────────────────────


def test_build_dataset_small(tmp_path):
    src = tmp_path / "data"
    (src / "self_captured" / "images").mkdir(parents=True)
    (src / "self_captured" / "labels").mkdir(parents=True)
    for split in ("train", "val"):
        (src / "dataset" / "images" / split).mkdir(parents=True)
        (src / "dataset" / "labels" / split).mkdir(parents=True)
    a = Image.new("RGB", (1600, 1200), (120, 10, 10))
    a.save(src / "self_captured" / "images" / "IMG_1.jpg")
    (src / "self_captured" / "labels" / "IMG_1.txt").write_text(
        "0 0.25 0.5 0.1 0.1\n1 0.5 0.55 0.9 0.2\n4 0.5 0.5 0.1 0.1\n")
    a.save(src / "self_captured" / "images" / "IMG_1_inv.jpg")
    (src / "dataset" / "images" / "val" / "self_captured_IMG_1.jpg").write_bytes(
        (src / "self_captured" / "images" / "IMG_1.jpg").read_bytes())
    (src / "dataset" / "images" / "train" / "self_captured_IMG_1_inv.jpg").write_bytes(b"x")

    summary = bd.build(src, tmp_path / "out", screen=None, log=lambda *_: None)
    assert summary["n_kept"] == 1 and summary["excluded"] == {"augmentation_inverted": 1}
    row = json.loads((tmp_path / "out" / "manifest.jsonl").read_text().splitlines()[0])
    assert row["id"] == "sc-IMG_1" and row["yolo_split"] == "val" and (row["width"], row["height"]) == (768, 576)
    assert row["gt_boxes"][0]["bbox"] == pytest.approx([153.6, 259.2, 230.4, 316.8])
    assert row["provisional_letter"] in {"C", "D", "F"} and row["needs_review"]
    assert {b["class_name"] for b in row["gt_boxes"]} == {"cart", "track", "ruler"}
    with pytest.raises(ValueError):
        bd._check_allowed(src / "video_frames" / "images" / "x.jpg")
