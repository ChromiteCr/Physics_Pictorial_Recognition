"""识别 bench 与审核图册：时间读取、时段切分、测试集生成、打分、规则基线、图册页面。

全部用合成图片和假检测器，不加载 YOLO，不读真实照片。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from datetime import datetime

import pytest
from PIL import Image

from lab_detective.bench import recognition_bench as rb
from lab_detective.bench import review_gallery as rg
from lab_detective.bench.freeze import freeze, verify
from lab_detective.training.format import Example, read_jsonl
from lab_detective.vision import cues
from lab_detective.vision.prompt import SYSTEM_PROMPT

BASE = datetime(2026, 7, 12, 10, 0, 0)


def _iso(sec: float) -> str:
    return datetime.fromtimestamp(BASE.timestamp() + sec).isoformat(timespec="seconds")


def _row(i, sec=None, letter="Z", boxes=None, yolo="train", **kw):
    r = {"id": f"sc-IMG_{i:04d}", "file": f"images/sc-IMG_{i:04d}.jpg", "orig_path": f"self_captured/images/IMG_{i:04d}.jpg",
         "source": "self_captured", "sha1": f"s{i}", "src_sha1": f"o{i}", "yolo_split": yolo,
         "gt_boxes": boxes or [], "provisional_letter": letter, "rule_candidates": [], "rule_tied": [],
         "rule_confident": False}
    if sec is not None:
        r["taken_at"] = _iso(sec)
    r.update(kw)
    return r


def _sessioned_rows(n_sessions=8, per=2, gap=3600):
    """n_sessions 个时段，每段 per 张（相隔 5 s），时段之间隔 gap 秒。"""
    rows = []
    for s in range(n_sessions):
        for j in range(per):
            rows.append(_row(s * per + j, s * gap + j * 5))
    return rows


def _write_images(root, rows, size=(160, 120)):
    (root / "images").mkdir(parents=True, exist_ok=True)
    for r in rows:
        Image.new("RGB", size, (120, 130, 140)).save(root / r["file"], quality=85)


DETS = [{"class_id": 0, "class_name": "cart", "confidence": 0.9, "bbox": [10, 10, 60, 50]},
        {"class_id": 1, "class_name": "track", "confidence": 0.8, "bbox": [0, 40, 150, 90]}]


# ── 拍摄时间 ───────────────────────────────────────────────────────────


def test_parse_exif_datetime_variants():
    want = datetime(2026, 7, 12, 11, 51, 32)
    assert rb.parse_exif_datetime("2026:07:12 11:51:32") == want
    assert rb.parse_exif_datetime(b"2026:07:12 11:51:32\x00") == want
    assert rb.parse_exif_datetime("2026-07-12 11:51:32") == want
    assert rb.parse_exif_datetime("2026-07-12T11:51:32") == want
    assert rb.parse_exif_datetime("garbage") is None
    assert rb.parse_exif_datetime(None) is None
    assert rb.parse_exif_datetime("0000:00:00 00:00:00") is None


def test_filename_time_patterns():
    assert rb.filename_time("IMG_9307.jpg") is None  # iPhone 编号没有时间
    assert rb.filename_time("IMG_20260712_115132.jpg") == datetime(2026, 7, 12, 11, 51, 32)
    assert rb.filename_time("photo 2026-07-12 11.51.32.png") == datetime(2026, 7, 12, 11, 51, 32)
    assert rb.filename_time("20261340_115132.jpg") is None  # 月份 13 不合法


def test_exif_time_reads_datetime_tag_and_prefers_original(tmp_path):
    p1 = tmp_path / "a.jpg"
    ex = Image.Exif()
    ex[306] = "2026:07:12 11:51:32"
    Image.new("RGB", (32, 32)).save(p1, exif=ex)
    assert rb.exif_time(p1) == datetime(2026, 7, 12, 11, 51, 32)

    p2 = tmp_path / "b.jpg"
    ex = Image.Exif()
    ex[306] = "2026:07:12 12:00:00"
    ex.get_ifd(0x8769)[36867] = "2026:07:12 11:00:00"  # DateTimeOriginal 优先
    Image.new("RGB", (32, 32)).save(p2, exif=ex)
    assert rb.exif_time(p2) == datetime(2026, 7, 12, 11, 0, 0)

    p3 = tmp_path / "c.jpg"
    Image.new("RGB", (32, 32)).save(p3)
    assert rb.exif_time(p3) is None
    assert rb.exif_time(tmp_path / "missing.jpg") is None


def test_photo_time_priority(tmp_path):
    (tmp_path / "self_captured/images").mkdir(parents=True)
    ex = Image.Exif()
    ex[306] = "2026:07:12 11:00:00"
    Image.new("RGB", (32, 32)).save(tmp_path / "self_captured/images/IMG_0001.jpg", exif=ex)
    row = _row(1)
    assert rb.photo_time(row, tmp_path) == (datetime(2026, 7, 12, 11, 0, 0), "exif_orig")
    row["taken_at"] = "2026-07-12T09:00:00"
    assert rb.photo_time(row, tmp_path) == (datetime(2026, 7, 12, 9, 0, 0), "manifest")
    named = {"id": "x", "orig_path": "IMG_20260101_010203.jpg"}
    assert rb.photo_time(named) == (datetime(2026, 1, 1, 1, 2, 3), "filename")
    assert rb.photo_time({"id": "IMG_5"}) == (None, "none")


# ── 时段切分 ───────────────────────────────────────────────────────────


def test_group_sessions_gap_is_inclusive_and_sorted_by_time_then_id():
    dated = [(100.0, "b"), (0.0, "a"), (1800.0, "c"), (3601.0, "d"), (100.0, "a2")]
    sess = rb.group_sessions(dated, 1800)
    assert sess == [["a", "a2", "b", "c"], ["d"]]  # 间隔恰好 1800 仍是同一时段；同时刻按 id 排
    assert len(rb.group_sessions(dated, 60)) == 4
    assert rb.group_sessions([], 60) == []


def test_undated_sessions_use_file_number_runs():
    out = rb.undated_sessions(["sc-IMG_0005", "sc-IMG_0001", "sc-IMG_0003", "sc-IMG_0100", "noNumber"])
    assert out == [["sc-IMG_0001", "sc-IMG_0003", "sc-IMG_0005"], ["sc-IMG_0100"], ["noNumber"]]


def test_session_split_keeps_sessions_whole_and_is_deterministic():
    rows = _sessioned_rows(8, 3)
    a = rb.session_split(rows, gap_s=1800)
    b = rb.session_split(list(reversed(rows)), gap_s=1800)
    assert a["photos"] == b["photos"]  # 与输入顺序无关
    assert a["n_sessions"] == 8 and a["n_test_sessions"] == 2
    assert a["session_fraction_test"] == 0.25 and a["photo_fraction_test"] == 0.25
    by_session = {}
    for v in a["photos"].values():
        by_session.setdefault(v["session"], set()).add(v["split"])
    assert all(len(s) == 1 for s in by_session.values())  # 没有时段横跨两边
    assert sum(next(iter(s)) == "test" for s in by_session.values()) == 2
    assert a["nearest_train_gap_s"]["min"] >= 1800
    assert a["params"]["gap_mode"] == "fixed"


def test_session_split_seed_changes_choice_and_adding_photo_keeps_assignments():
    rows = _sessioned_rows(8, 2)
    base = rb.session_split(rows, gap_s=1800, seed=0)
    others = [rb.session_split(rows, gap_s=1800, seed=s)["photos"] for s in range(1, 6)]
    assert any(o != base["photos"] for o in others)
    # 已有时段里多一张照片，所有旧照片的划分不变
    more = rows + [_row(99, 3 * 3600 + 10)]
    after = rb.session_split(more, gap_s=1800, seed=0)
    for pid, v in base["photos"].items():
        assert after["photos"][pid]["split"] == v["split"]


def test_session_split_refuses_single_session_and_lists_gap_table():
    rows = [_row(i, i * 7) for i in range(60)]  # 7 分钟内连拍，30 分钟间隔下只有 1 个时段
    with pytest.raises(rb.DegenerateSplit) as ei:
        rb.session_split(rows, gap_s=1800)
    msg = str(ei.value)
    assert "只有 1 个时段" in msg and "'1800': 1" in msg and "--gap-s" in msg


def test_session_split_auto_picks_largest_gap_with_enough_sessions():
    # 8 簇，簇内相隔 2 s、簇间隔 100 s：1800/600/180 下 1 个时段，60 s 下 8 个
    rows = []
    for c in range(8):
        for j in range(4):
            rows.append(_row(c * 4 + j, c * 100 + j * 2))
    info = rb.session_split(rows, gap_s="auto", auto_min_sessions=8)
    assert info["params"]["gap_mode"] == "auto" and info["params"]["gap_s"] == 60
    assert info["n_sessions"] == 8
    info2 = rb.session_split(rows, gap_s=60)
    assert info2["photos"] == info["photos"]


def test_session_split_reads_exif_from_source_dir_and_uses_times_cache(tmp_path):
    (tmp_path / "self_captured/images").mkdir(parents=True)
    rows = []
    for i in range(12):
        ex = Image.Exif()
        ex[306] = datetime.fromtimestamp(BASE.timestamp() + (i // 3) * 7200 + (i % 3) * 3).strftime("%Y:%m:%d %H:%M:%S")
        Image.new("RGB", (16, 16)).save(tmp_path / f"self_captured/images/IMG_{i:04d}.jpg", exif=ex)
        rows.append(_row(i))  # 没有 taken_at：只能读原图 EXIF
    info = rb.session_split(rows, src_root=tmp_path, gap_s=1800)
    assert info["n_sessions"] == 4 and info["n_undated"] == 0
    assert {v["time_source"] for v in info["photos"].values()} == {"exif_orig"}
    cache = {pid: v["taken_at"] for pid, v in info["photos"].items()}
    shutil.rmtree(tmp_path / "self_captured")  # 原图不在了，用缓存仍得到同样的划分
    again = rb.session_split(rows, src_root=tmp_path, gap_s=1800, times=cache)
    assert {k: v["split"] for k, v in again["photos"].items()} == {k: v["split"] for k, v in info["photos"].items()}
    assert {v["time_source"] for v in again["photos"].values()} == {"cache"}


def test_session_split_undated_photos_fall_back_to_number_runs_with_warning():
    rows = [_row(i) for i in list(range(1, 6)) + list(range(100, 105)) + list(range(200, 205)) + list(range(300, 305))]
    info = rb.session_split(rows, gap_s=1800)
    assert info["n_undated"] == 20 and info["n_sessions"] == 4
    assert any("没有拍摄时间" in w for w in info["warnings"])
    assert all(v["taken_at"] is None for v in info["photos"].values())


def test_split_helpers_and_overlap_check():
    info = rb.session_split(_sessioned_rows(8, 2), gap_s=1800)
    test, train = rb.split_ids(info, "test"), rb.split_ids(info, "train")
    assert len(test) == 4 and len(train) == 12 and not set(test) & set(train)
    assert rb.check_no_overlap(train, info)["clean"]
    bad = rb.check_no_overlap([*train, test[0]], info)
    assert not bad["clean"] and bad["overlap"] == [test[0]]


def test_split_cli_refuses_degenerate_and_writes_split(tmp_path, capsys):
    manifest = tmp_path / "manifest.jsonl"
    rows = [_row(i, (i // 5) * 100 + (i % 5) * 3) for i in range(40)]  # 8 簇，簇间隔 100 s
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = tmp_path / "bench"
    assert rb.main(["--manifest", str(manifest), "--out", str(out), "split", "--src", str(tmp_path)]) == 2
    assert not (out / "split.json").exists()
    assert "拒绝切分" in capsys.readouterr().err
    assert rb.main(["--manifest", str(manifest), "--out", str(out), "split", "--src", str(tmp_path), "--gap-s", "auto"]) == 0
    info = rb.load_split(out / "split.json")
    assert info["status"] == "draft" and info["n_sessions"] >= 4
    # 已有 split.json 不允许悄悄覆盖
    assert rb.main(["--manifest", str(manifest), "--out", str(out), "split", "--src", str(tmp_path), "--gap-s", "auto"]) == 2
    assert rb.main(["--manifest", str(manifest), "--out", str(out), "split", "--src", str(tmp_path), "--gap-s", "auto", "--force"]) == 0


def _parser_flags(parser):
    flags = set(parser._option_string_actions)
    for act in parser._actions:
        if isinstance(act, argparse._SubParsersAction):
            for sp in act.choices.values():
                flags |= set(sp._option_string_actions)
    return flags


def test_documented_usage_lines_match_the_real_cli():
    """文档里的每条用法：用到的选项都存在，--manifest/--out 写在子命令前面。"""
    parser = rb.build_parser()
    flags = _parser_flags(parser)
    subs = {"split", "confirm", "build", "score", "rules"}
    lines = [l for l in rb.__doc__.splitlines() if "python -m lab_detective.bench.recognition_bench" in l]
    assert len(lines) == len(subs)
    seen = set()
    for l in lines:
        assert set(re.findall(r"--[a-z][a-z-]*", l)) <= flags, l
        tokens = l.replace("[", " ").replace("]", " ").split()
        sub = next(t for t in tokens[3:] if t in subs)
        seen.add(sub)
        after = tokens[tokens.index(sub) + 1:]
        assert not {"--manifest", "--out"} & set(after), f"{sub}: 顶层选项必须写在子命令前面"
    assert seen == subs


def test_documented_score_invocation_runs(tmp_path):
    _, _, out, _, _ = _built(tmp_path)
    preds = tmp_path / "p.jsonl"
    exs = list(read_jsonl(out / "test.jsonl"))
    preds.write_text("".join(json.dumps({"id": e.id, "text": _text("C")}) + "\n" for e in exs), encoding="utf-8")
    assert rb.main(["--out", str(out), "score", "--pred", str(preds), "--report", str(tmp_path / "rep.json"),
                    "--arm", "demo"]) == 0
    rep = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))
    assert rep["arm"] == "demo" and rep["n_test"] == len(exs)
    with pytest.raises(SystemExit):  # 把 --out 写在子命令后面是用法错误
        rb.main(["score", "--pred", str(preds), "--out", str(tmp_path / "rep.json")])


def test_draft_split_is_flagged_until_student_confirms(tmp_path, capsys):
    rows, split, out, meta, _ = _built(tmp_path)
    assert split["status"] == "draft" and meta["split"]["status"] == "draft"
    assert any("草稿" in w for w in meta["warnings"])
    # 没有 split.json 时 confirm 报错；有了之后记下确认人，build 的警告消失
    assert rb.main(["--out", str(out), "confirm", "--who", "学生"]) == 2
    rb.write_split(split, out / "split.json")
    assert rb.main(["--out", str(out), "confirm", "--who", "学生"]) == 0
    confirmed = rb.load_split(out / "split.json")
    assert confirmed["status"] == "confirmed" and confirmed["confirmed_by"] == "学生" and confirmed["confirmed_at"]
    meta2 = rb.build_test_set(rows, confirmed, out, image_root=tmp_path / "photos")
    assert not any("草稿" in w for w in meta2["warnings"]) and meta2["split"]["status"] == "confirmed"


# ── review.json 与 gold ───────────────────────────────────────────────


def test_load_review_normalizes_and_rejects_bad_input(tmp_path):
    ids = ["a", "b"]
    assert rb.load_review({"a": "c", "b": "Z"}, ids) == {"a": "C", "b": "Z"}
    p = tmp_path / "review.json"
    p.write_text(json.dumps({"a": "D"}), encoding="utf-8")
    assert rb.load_review(p, ids) == {"a": "D"}
    with pytest.raises(ValueError, match="非法字母"):
        rb.load_review({"a": "7"}, ids)
    with pytest.raises(ValueError, match="不在 manifest"):
        rb.load_review({"zzz": "A"}, ids)
    with pytest.raises(ValueError):
        rb.load_review({"a": None}, ids)
    p.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="对象"):
        rb.load_review(p)


def test_gold_for_flags_provisional():
    r = _row(1, letter="C")
    assert rb.gold_for(r, None) == ("C", "provisional")
    assert rb.gold_for(r, {r["id"]: "D"}) == ("D", "reviewed")
    assert rb.gold_for(r, {"other": "D"}) == ("C", "provisional")


# ── 测试集生成 ─────────────────────────────────────────────────────────


def _built(tmp_path, review=None, rows=None):
    rows = rows or [_row(i, s * 3600 + j * 5, letter="C" if s % 2 else "Z",
                         boxes=[{"class_id": 0, "class_name": "cart", "bbox": [5, 5, 40, 40]}],
                         yolo="train" if i % 4 else "test")
                    for i, (s, j) in enumerate((s, j) for s in range(8) for j in range(2))]
    _write_images(tmp_path / "photos", rows)
    split = rb.session_split(rows, gap_s=1800)
    calls = []

    def detector(path):
        calls.append(path)
        return DETS

    out = tmp_path / "bench"
    meta = rb.build_test_set(rows, split, out, image_root=tmp_path / "photos", review=review, detector=detector,
                             detector_info={"weights": "fake"})
    return rows, split, out, meta, calls


def test_build_test_set_format_gold_and_prompt(tmp_path):
    rows, split, out, meta, calls = _built(tmp_path)
    exs = list(read_jsonl(out / "test.jsonl"))
    test_ids = rb.split_ids(split, "test")
    assert sorted(e.meta["photo_id"] for e in exs) == test_ids and len(calls) == len(test_ids) == meta["n_test"]
    e = exs[0]
    e.validate()
    assert e.task == "recognize" and e.source == "photo" and e.id == rb.example_id(e.meta["photo_id"])
    assert e.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}  # 与 vision/prompt.py 一致
    user = e.messages[1]["content"]
    assert user[0] == {"type": "image", "image": f"images/{e.meta['photo_id']}.jpg"}
    assert (out / user[0]["image"]).exists()
    assert "1 = 小车（cart）" in user[1]["text"] and "2 = 轨道（track）" in user[1]["text"]  # 图例来自检测框
    gold = json.loads(e.messages[-1]["content"])
    assert gold["letter"] == e.meta["gold_letter"] and e.meta["gold_fields"] == ["letter"]
    assert e.meta["gold_status"] == "provisional" and e.meta["provisional_letter"] == e.meta["gold_letter"]
    assert e.meta["detections"] == DETS and e.meta["yolo_seen"] == (e.meta["yolo_split"] in ("train", "val"))
    assert {"photo_id", "session", "prompt_hash", "yolo_split", "taken_at"} <= set(e.meta)
    # 没有任何学生核实：meta.json 里必须写明
    m = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert m["gold_status_counts"] == {"provisional": len(exs)} and m["n_reviewed_total"] == 0
    assert any("没有任何学生核实" in w for w in m["warnings"])
    assert all(p.name.endswith(".jpg") for p in (out / "images").iterdir())
    assert len(list((out / "images").iterdir())) == len(exs)  # 只有测试照片的图


def test_build_test_set_review_overrides_and_detection_cache_is_reused(tmp_path):
    rows, split, out, meta, calls = _built(tmp_path)
    test_ids = rb.split_ids(split, "test")
    review = {test_ids[0]: "G", test_ids[1]: "Z"}
    meta2 = rb.build_test_set(rows, split, out, image_root=tmp_path / "photos", review=review, detector=None,
                              review_sha256="abc")  # detector=None：只靠缓存
    assert len(calls) == len(test_ids)  # 没有重新检测
    exs = {e.meta["photo_id"]: e for e in read_jsonl(out / "test.jsonl")}
    assert exs[test_ids[0]].meta["gold_letter"] == "G" and exs[test_ids[0]].meta["gold_status"] == "reviewed"
    assert exs[test_ids[0]].meta["held_out_gold"] is True and exs[test_ids[0]].meta["experiment_id"] == "incline_friction"
    assert exs[test_ids[1]].meta["experiment_id"] is None
    assert exs[test_ids[-1]].meta["gold_status"] == "provisional"
    assert meta2["gold_status_counts"]["reviewed"] == 2 and meta2["review_sha256"] == "abc"
    assert any("没核实" in w for w in meta2["warnings"])


def test_build_test_set_requires_detector_or_cache(tmp_path):
    rows = _sessioned_rows(8, 2)
    _write_images(tmp_path / "photos", rows)
    split = rb.session_split(rows, gap_s=1800)
    with pytest.raises(ValueError, match="检测缓存"):
        rb.build_test_set(rows, split, tmp_path / "bench", image_root=tmp_path / "photos")


def test_build_output_can_be_frozen_and_verified(tmp_path):
    _, _, out, _, _ = _built(tmp_path)
    freeze(out, who="测试")
    assert verify(out)["ok"]
    (out / "test.jsonl").write_text("changed", encoding="utf-8")
    assert not verify(out)["ok"]


# ── 打分 ───────────────────────────────────────────────────────────────


def _ex(i, gold, status="reviewed", yolo="train", source="self_captured", apparatus=("cart", "track")):
    return Example(f"recbench-p{i}", "recognize", "photo",
                   [{"role": "user", "content": "q"}, {"role": "assistant", "content": "{}"}],
                   {"photo_id": f"p{i}", "gold_letter": gold, "gold_status": status, "yolo_split": yolo,
                    "source": source, "gold_apparatus_partial": list(apparatus)})


def _text(letter, conf=0.9, apparatus=("cart",)):
    return json.dumps({"letter": letter, "open_name": "x", "cues": [], "apparatus": list(apparatus),
                       "confidence": conf}, ensure_ascii=False)


GOLD = ["A", "A", "C", "C", "C", "E", "Z", "Z", "Z", "G"]


def _examples(**kw):
    return [_ex(i, g, **kw) for i, g in enumerate(GOLD)]


def test_score_top1_per_letter_confusion_and_missing_and_invalid():
    exs = _examples()
    preds = {
        "recbench-p0": {"text": _text("A")},
        "recbench-p1": {"text": _text("C")},  # 错
        "recbench-p2": {"text": "```json\n" + _text("C") + "\n```"},  # 围栏也能解析
        "recbench-p3": {"text": _text("C")},
        "recbench-p4": {"text": "完全不是 JSON"},  # 非法：算错
        "recbench-p5": {"text": _text("E")},
        "recbench-p6": {"text": _text("Z", 0.8)},
        "recbench-p7": {"text": _text("A")},  # 错
        # p8 缺失：算错、不丢弃
        "recbench-p9": {"text": _text("G")},
        "recbench-ghost": {"text": _text("A")},  # 多出来的 id 不计分
    }
    rep = score_ok(exs, preds)
    m = rep["reviewed"]
    assert m["n"] == 10 and m["top1"]["k"] == 6 and rep["n_missing"] == 1 and rep["extra_prediction_ids"] == ["recbench-ghost"]
    assert m["json_valid"]["k"] == 8  # p4 非法、p8 缺失
    assert m["top1"]["wilson95"][0] < 0.6 < m["top1"]["wilson95"][1]
    assert m["per_letter_recall"]["A"]["k"] == 1 and m["per_letter_recall"]["C"]["k"] == 2
    assert (m["per_letter_recall"]["Z"]["k"], m["per_letter_recall"]["Z"]["n"]) == (1, 3)
    assert m["confusion"]["C"] == {"C": 2, "∅": 1} and m["confusion"]["Z"]["A"] == 1 and m["confusion"]["Z"]["∅"] == 1
    assert m["closed_set_gold_not_Z"]["n"] == 7 and (m["held_out_G_L"]["k"], m["held_out_G_L"]["n"]) == (1, 1)
    assert m["other_Z_recall"]["k"] == 1 and m["other_Z_precision"]["k"] == 1 and m["other_Z_precision"]["n"] == 1
    assert rep["headline_subset"] == "reviewed" and rep["provisional"] == {"n": 0}
    json.dumps(rep, allow_nan=False)  # 报告必须是合法 JSON（没有 NaN）


def score_ok(exs, preds, **kw):
    rep = rb.score(exs, preds, **kw)
    json.dumps(rep, allow_nan=False)
    return rep


def test_score_invalid_policy_other_counts_invalid_as_z():
    exs = _examples()
    preds = {e.id: {"text": _text(e.meta["gold_letter"])} for e in exs}
    preds["recbench-p6"] = {"text": "垃圾"}  # gold Z 的照片输出非法
    preds["recbench-p2"] = {"text": "垃圾"}  # gold C 的照片输出非法
    wrong = rb.score(exs, preds)["reviewed"]["top1"]["k"]
    other = rb.score(exs, preds, policy=rb.ScoringPolicy("other"))["reviewed"]["top1"]["k"]
    assert wrong == 8 and other == 9  # 非法当作 Z：gold 恰好是 Z 的那张变对
    with pytest.raises(ValueError):
        rb.ScoringPolicy("bogus")


def test_score_matches_predictions_by_photo_id_and_rejects_duplicates(tmp_path):
    exs = _examples()
    rep = rb.score(exs, {f"p{i}": {"text": _text(g)} for i, g in enumerate(GOLD)})
    assert rep["reviewed"]["top1"]["k"] == 10
    p = tmp_path / "preds.jsonl"
    p.write_text(json.dumps({"id": "recbench-p0", "text": _text("A")}) + "\n" +
                 json.dumps({"id": "recbench-p0", "text": _text("C")}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        rb.read_predictions(p)


def test_score_separates_reviewed_from_provisional_and_flags_no_evidence():
    exs = _examples(status="provisional")
    preds = {e.id: {"text": _text(e.meta["gold_letter"])} for e in exs}
    rep = score_ok(exs, preds)
    assert rep["headline_subset"] is None and rep["reviewed"] == {"n": 0}
    assert rep["provisional"]["top1"]["k"] == 10
    assert any("不是证据" in c for c in rep["caveats"]) and any("按构造" in c for c in rep["caveats"])
    mixed = [_ex(0, "A"), _ex(1, "C", status="provisional")]
    rep2 = rb.score(mixed, {e.id: {"text": _text(e.meta["gold_letter"])} for e in mixed})
    assert rep2["reviewed"]["n"] == 1 and rep2["provisional"]["n"] == 1 and rep2["all"]["n"] == 2


def test_score_stratifies_by_source_and_yolo_split():
    exs = [_ex(0, "A", yolo="train"), _ex(1, "A", yolo="test", source="cc"), _ex(2, "C", yolo="test", source="cc")]
    preds = {"recbench-p0": {"text": _text("A")}, "recbench-p1": {"text": _text("C")}, "recbench-p2": {"text": _text("C")}}
    m = rb.score(exs, preds)["reviewed"]
    assert m["by_source"]["self_captured"]["k"] == 1
    assert (m["by_source"]["cc"]["k"], m["by_source"]["cc"]["n"]) == (1, 2)
    assert m["by_yolo_split"]["train"]["rate"] == 1.0 and m["by_yolo_split"]["test"]["rate"] == 0.5


def test_auroc_known_values():
    assert rb.auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert rb.auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert rb.auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5  # 全并列
    assert rb.auroc([0.9, 0.4, 0.6, 0.1], [True, True, False, False]) == pytest.approx(0.75)
    assert math.isnan(rb.auroc([0.1, 0.2], [True, True]))
    lo, hi = rb.auroc_ci([0.9, 0.8, 0.7, 0.2, 0.1, 0.3] * 5, [True, True, True, False, False, False] * 5)
    assert 0.9 <= lo <= hi <= 1.0
    assert all(math.isnan(v) for v in rb.auroc_ci([0.1, 0.2], [True, True]))


def test_auroc_other_from_confidence_and_from_logprobs():
    exs = [_ex(0, "Z"), _ex(1, "Z"), _ex(2, "C"), _ex(3, "C")]
    # 只有 confidence：预测 Z 取 conf，否则取 1−conf
    conf_preds = {"recbench-p0": {"text": _text("Z", 0.9)}, "recbench-p1": {"text": _text("C", 0.6)},
                  "recbench-p2": {"text": _text("C", 0.9)}, "recbench-p3": {"text": _text("Z", 0.3)}}
    a = rb.score(exs, conf_preds)["reviewed"]["auroc_other"]
    # P(Z)：p0=0.9 p1=0.4 p2=0.1 p3=0.3；正类 p0、p1 都高于负类 → AUROC 1
    assert a["auroc"] == 1.0 and a["n_pos"] == 2 and a["n_neg"] == 2
    # 有 logprobs 时用 logprobs（带空格的 token 也认），压倒 confidence
    lp_preds = {
        "recbench-p0": {"text": _text("A", 0.99), "letter_logprobs": {" Z": math.log(0.7), " A": math.log(0.3)}},
        "recbench-p1": {"text": _text("A", 0.99), "letter_logprobs": {"Z": math.log(0.2), "C": math.log(0.8)}},
        "recbench-p2": {"text": _text("C", 0.99), "letter_logprobs": {"Z": math.log(0.5), "C": math.log(0.5)}},
        "recbench-p3": {"text": _text("C", 0.99), "letter_logprobs": {"Z": math.log(0.1), "C": math.log(0.9)}},
    }
    a2 = rb.score(exs, lp_preds)["reviewed"]["auroc_other"]
    # P(Z)：0.7 0.2 0.5 0.1 → 正类(0.7,0.2) vs 负类(0.5,0.1)：4 对里赢 3 → 0.75
    assert a2["auroc"] == pytest.approx(0.75)
    p = rb.parse_prediction("x", {"text": _text("C"), "letter_probs": {"Z": 2.0, "C": 6.0}})
    assert p.p_other == pytest.approx(0.25)


def test_auroc_is_null_when_only_one_class_present():
    exs = [_ex(0, "C"), _ex(1, "A")]
    rep = score_ok(exs, {"recbench-p0": {"text": _text("C")}, "recbench-p1": {"text": _text("A")}})
    au = rep["reviewed"]["auroc_other"]
    assert au["auroc"] is None and au["n_pos"] == 0 and au["ci95"] == [None, None]


def test_apparatus_f1_uses_yolo_vocabulary_only():
    exs = [_ex(0, "C", apparatus=("cart", "track")), _ex(1, "C", apparatus=("cart",))]
    preds = {"recbench-p0": {"text": _text("C", apparatus=("cart", "pulley"))},  # pulley 不在 YOLO 词表，不算
             "recbench-p1": {"text": _text("C", apparatus=("cart", "string"))}}
    f = rb.score(exs, preds)["reviewed"]["apparatus_f1"]
    assert (f["tp"], f["fp"], f["fn"]) == (2, 1, 1)
    assert f["f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))


# ── 规则基线 ───────────────────────────────────────────────────────────

MOMENTUM = [{"class_name": "cart", "bbox": [50, 100, 100, 130]}, {"class_name": "cart", "bbox": [300, 100, 350, 130]},
            {"class_name": "track", "bbox": [0, 90, 400, 140]}]
CART_TRACK = MOMENTUM[1:]


def test_rule_probs_sum_to_one_and_no_candidates_means_z():
    assert rb.rule_probs([]) == {"Z": 1.0}
    pr = rb.rule_probs(cues.rule_candidates(MOMENTUM))
    assert sum(pr.values()) == pytest.approx(1.0, abs=1e-4)
    top = cues.rule_candidates(MOMENTUM)[0]
    assert pr[top["letter"]] > pr["Z"]
    assert pr[top["letter"]] == pytest.approx(cues.rule_confidence(cues.rule_candidates(MOMENTUM)), abs=1e-3)


def test_rules_prediction_confident_tied_and_none():
    p = rb.rules_prediction(MOMENTUM)
    assert p["letter"] == "E" and p["confident"] is True
    t = rb.rules_prediction(CART_TRACK)
    assert t["letter"] in "CDF" and t["confident"] is False and set(t["tied"]) == {"C", "D", "F"}
    z = rb.rules_prediction([])
    assert z["letter"] == "Z" and z["confidence"] == 1.0 and z["confident"] is False


def _rule_ex(i, boxes, dets, gold, status="reviewed"):
    e = _ex(i, gold, status=status)
    e.meta.update({"gt_boxes": boxes, "detections": dets})
    return e


def test_rules_baseline_is_circular_on_provisional_gold_and_reports_confident_subset():
    def provisional(boxes):  # 与 build_dataset.provisional_label 同一规则
        c = cues.analyze(boxes)["candidates"]
        return c[0]["letter"] if c else "Z"

    exs = [_rule_ex(i, b, b, provisional(b), "provisional") for i, b in enumerate((MOMENTUM, CART_TRACK, []))]
    # 暂定 gold = 规则对真值框的输出，所以规则 + 真值框按构造满分
    rep = rb.score(exs, rb.run_rules(exs, "gt"))
    assert rep["provisional"]["top1"]["k"] == 3 and rep["headline_subset"] is None
    assert rep["provisional"]["confident_only"]["coverage"]["k"] == 1  # 只有动量守恒这张规则有把握
    assert rep["provisional"]["confident_only"]["accuracy"]["rate"] == 1.0
    # Z 的 AUROC：没有候选的照片 P(Z)=1，高于其他
    assert rep["provisional"]["auroc_other"]["auroc"] == 1.0


def test_rules_on_detections_differs_when_detector_misses():
    exs = [_rule_ex(0, MOMENTUM, CART_TRACK, "E"), _rule_ex(1, MOMENTUM, MOMENTUM, "E")]  # 第 0 张检测漏了一辆车
    gt = rb.score(exs, rb.run_rules(exs, "gt"))["reviewed"]
    yolo = rb.score(exs, rb.run_rules(exs, "detections"))["reviewed"]
    assert gt["top1"]["k"] == 2 and yolo["top1"]["k"] == 1
    with pytest.raises(KeyError):
        rb.run_rules(exs, "bogus")


def test_rules_cli_runs_on_built_test_set(tmp_path, capsys):
    _, _, out, _, _ = _built(tmp_path)
    runs = tmp_path / "runs"
    assert rb.main(["--out", str(out), "rules", "--runs", str(runs)]) == 0
    for arm in ("rules_gt", "rules_yolo"):
        rep = json.loads((runs / arm / "report.json").read_text(encoding="utf-8"))
        assert rep["n_missing"] == 0 and rep["arm"] == arm
    assert "rules_yolo" in capsys.readouterr().out
    # score 命令：用规则预测文件打分
    assert rb.main(["--out", str(out), "score", "--pred", str(runs / "rules_gt" / "predictions.jsonl"),
                    "--report", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["n_test"] > 0


# ── 审核图册 ───────────────────────────────────────────────────────────


def _gallery_rows():
    rows = [
        _row(1, letter="E", boxes=[{"class_name": "cart", "bbox": [0, 0, 5, 5]}, {"class_name": "cart", "bbox": [9, 9, 20, 20]}],
             rule_confident=True, rule_tied=["E"], rule_candidates=[{"letter": "E", "score": 1.9}]),
        _row(2, letter="C", rule_tied=["C", "D", "F"],
             rule_candidates=[{"letter": "C", "score": 1.0}, {"letter": "D", "score": 1.0}, {"letter": "F", "score": 1.0}]),
        _row(3, letter="Z"),
    ]
    return rows


def _manifest(tmp_path, rows):
    _write_images(tmp_path, rows)
    p = tmp_path / "manifest.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return p


def _payload(html_text):
    m = re.search(r'<script type="application/json" id="payload">(.*?)</script>', html_text, re.S)
    return json.loads(m.group(1))


def test_hint_for_never_preselects_z_and_only_confident_is_single():
    rows = _gallery_rows()
    h1, h2, h3 = (rg.hint_for(r) for r in rows)
    assert h1["kind"] == "confident" and h1["letters"] == ["E"]
    assert h2["kind"] == "tied" and h2["letters"] == ["C", "D", "F"]
    assert h3["kind"] == "none" and h3["letters"] == [] and "不代表" in h3["text"]
    z_tied = rg.hint_for(_row(9, letter="Z", rule_tied=["Z", "C"], rule_candidates=[{"letter": "C", "score": 1}]))
    assert "Z" not in z_tied["letters"]
    assert rg.boxes_text(rows[0]) == "小车×2"


def test_gallery_embeds_every_photo_without_preselection(tmp_path):
    rows = _gallery_rows()
    out = tmp_path / "review.html"
    s = rg.build_gallery(_manifest(tmp_path, rows), out, split=None)
    text = out.read_text(encoding="utf-8")
    assert s["n_photos"] == 3 and s["hint_kinds"] == {"confident": 1, "tied": 1, "none": 1}
    pl = _payload(text)
    assert [p["id"] for p in pl["photos"]] == sorted(r["id"] for r in rows)
    assert [l["letter"] for l in pl["letters"]] == list("ABCDEFGHIJKLZ")
    assert all(p["thumb"].startswith("data:image/jpeg;base64,") and p["file"] == f"images/{p['id']}.jpg" for p in pl["photos"])
    # 没有任何预选：载荷里没有标签状态，页面脚本的初始标签是空对象
    assert not any(k in pl for k in ("labels", "selected", "review"))
    assert "var labels = {}" in text
    # 离线：没有任何外部资源
    assert not re.search(r'(src|href)\s*=\s*["\']https?://', text) and "cdn" not in text.lower()
    assert rg.gallery_id([r["id"] for r in rows]) == pl["gid"]


def test_gallery_escapes_script_terminators_and_html(tmp_path):
    evil = _row(1, letter="Z", boxes=[{"class_name": "</script><b>x", "bbox": [0, 0, 1, 1]}])
    evil["id"] = "sc-</script><img onerror=1>"
    evil["file"] = "images/ok.jpg"
    _write_images(tmp_path, [evil])
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(evil, ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "review.html"
    rg.build_gallery(manifest, out, split=None)
    text = out.read_text(encoding="utf-8")
    assert text.count("</script>") == 2  # 载荷块和脚本块各一个，数据里的 </script> 已被转义
    assert _payload(text)["photos"][0]["id"] == evil["id"]  # 转义后 JSON 解析回原样


def test_gallery_marks_test_photos_and_sorts_by_time(tmp_path):
    rows = [_row(1, 500), _row(2, 100), _row(3, 300)]
    manifest = _manifest(tmp_path, rows)
    split = {"photos": {rows[0]["id"]: {"split": "test", "session": 2, "taken_at": _iso(500)},
                        rows[1]["id"]: {"split": "train", "session": 1, "taken_at": _iso(100)},
                        rows[2]["id"]: {"split": "train", "session": 1, "taken_at": _iso(300)}}}
    sp = tmp_path / "split.json"
    sp.write_text(json.dumps(split), encoding="utf-8")
    out = tmp_path / "r.html"
    s = rg.build_gallery(manifest, out, split=sp)
    pl = _payload(out.read_text(encoding="utf-8"))
    assert s["n_test"] == 1
    assert [p["id"] for p in pl["photos"]] == [rows[1]["id"], rows[2]["id"], rows[0]["id"]]  # 按拍摄时间
    assert [p["test"] for p in pl["photos"]] == [False, False, True]


def test_gallery_missing_image_is_an_error(tmp_path):
    rows = _gallery_rows()
    manifest = _manifest(tmp_path, rows)
    (tmp_path / rows[1]["file"]).unlink()
    with pytest.raises(FileNotFoundError):
        rg.build_gallery(manifest, tmp_path / "r.html", split=None)


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 来跑页面里的纯逻辑")
def test_gallery_logic_block_in_node(tmp_path):
    rows = _gallery_rows()
    out = tmp_path / "review.html"
    rg.build_gallery(_manifest(tmp_path, rows), out, split=None)
    text = out.read_text(encoding="utf-8")
    logic = text.split("/*LOGIC-BEGIN*/")[1].split("/*LOGIC-END*/")[0]
    harness = r"""
const L = new Function(require('fs').readFileSync(0, 'utf8') +
  '; return {isDone, visibleIds, nextTodo, buildExport, mergeImport};')();
const photos = [{id:'a',test:true},{id:'b',test:false},{id:'c',test:true},{id:'d',test:false}];
const labels = {c:'C', a:'Z', ghost:'A'};
const out = {};
out.all = L.visibleIds(photos, labels, 'all', '');
out.todo = L.visibleIds(photos, labels, 'todo', '');
out.done = L.visibleIds(photos, labels, 'done', '');
out.test = L.visibleIds(photos, labels, 'test', '');
out.testTodo = L.visibleIds(photos, labels, 'test_todo', '');
out.letter = L.visibleIds(photos, labels, 'all', 'C');
out.next1 = L.nextTodo(['a','b','c','d'], labels, 'a');
out.nextWrap = L.nextTodo(['a','b','c','d'], {b:'B',c:'C',d:'D'}, 'c');
out.nextNone = L.nextTodo(['a','b'], {a:'A',b:'B'}, 'a');
out.export = JSON.stringify(L.buildExport(labels, ['a','b','c','d']));
out.exportEmpty = JSON.stringify(L.buildExport({}, ['a']));
out.merge = L.mergeImport({a:'E', zzz:'A', b:'Q', c:'z'}, ['a','b','c'], ['A','E','Z']);
out.mergeBadRoot = L.mergeImport([1,2], ['a'], ['A']);
console.log(JSON.stringify(out));
"""
    res = subprocess.run(["node", "-e", harness], input=logic, capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    o = json.loads(res.stdout)
    assert o["all"] == ["a", "b", "c", "d"] and o["todo"] == ["b", "d"] and o["done"] == ["a", "c"]
    assert o["test"] == ["a", "c"] and o["testTodo"] == [] and o["letter"] == ["c"]
    assert o["next1"] == "b" and o["nextWrap"] == "a" and o["nextNone"] is None
    # 导出只含照片集里存在的 id，键排序；没点过的不出现
    assert o["export"] == '{"a":"Z","c":"C"}' and o["exportEmpty"] == "{}"
    assert o["merge"] == {"labels": {"a": "E"}, "unknown": ["zzz"], "bad": ["b", "c"]}  # 小写 z 不是合法字母
    assert o["mergeBadRoot"]["bad"] and o["mergeBadRoot"]["labels"] == {}


def test_dev_seen_photos_never_in_test():
    from lab_detective.bench import recognition_bench as rb
    rows = [{"id": f"self_captured_IMG_{9300 + i}"} for i in range(40)]
    times = {r["id"]: f"2026:07:12 {10 + i // 4:02d}:{(i % 4) * 5:02d}:00" for i, r in enumerate(rows)}
    sp = rb.session_split(rows, times=times, gap_s=1800)
    for pid, v in sp["photos"].items():
        if rb._dev_seen(pid):
            assert v["split"] == "train", pid


def _with_dev_seen(rows, picks):
    """把第 idx 行的 id 换成开发时看过的照片（IMG_9307 等），让它所在的时段被固定划进训练侧。"""
    out = [dict(r) for r in rows]
    for idx, stem in picks.items():
        out[idx]["id"] = f"sc-{stem}"
    return out


def test_dev_seen_sessions_are_recorded_and_shrink_the_test_set_with_a_warning():
    base = _sessioned_rows(8, 2)  # 第 i 行属于时段 i // 2
    # 时段 0–4 被占用，还剩 3 个可选，目标 k=2 仍能满足
    rows = _with_dev_seen(base, {0: "IMG_9307", 2: "IMG_9314", 4: "IMG_9316", 6: "IMG_9341", 8: "IMG_9359"})
    info = rb.session_split(rows, gap_s=1800)
    forced = info["dev_seen_forced_train"]
    assert forced["sessions"] == [1, 2, 3, 4, 5] and "sc-IMG_9307" in forced["photos"]
    assert info["n_test_sessions"] == 2
    assert any("开发时看过" in w for w in info["warnings"])
    assert all(info["photos"][pid]["split"] == "train" for pid in forced["photos"])
    # 再占用时段 5、6：只剩 1 个可选时段、目标 2 个，测试集缩到 1 个并警告
    rows2 = _with_dev_seen(rows, {10: "IMG_9361", 12: "IMG_9363"})
    info2 = rb.session_split(rows2, gap_s=1800)
    assert info2["n_test_sessions"] == 1 and any("少于目标" in w for w in info2["warnings"])
    assert info2["n_test_photos"] == 2
    # 全部时段都被占用：拒绝，不悄悄给空测试集
    rows3 = _with_dev_seen(rows2, {14: "IMG_9449"})
    with pytest.raises(rb.DegenerateSplit, match="开发时看过"):
        rb.session_split(rows3, gap_s=1800)


def test_rebuild_after_resplit_removes_stale_images_and_detections(tmp_path):
    rows, split, out, _, _ = _built(tmp_path)
    old_ids = set(rb.split_ids(split, "test"))
    other = rb.session_split(rows, gap_s=1800, seed=3)
    new_ids = set(rb.split_ids(other, "test"))
    assert new_ids != old_ids
    rb.build_test_set(rows, other, out, image_root=tmp_path / "photos", detector=lambda p: DETS)
    assert {p.stem for p in (out / "images").iterdir()} == new_ids
    assert {d["id"] for d in rb.read_jsonl_dicts(out / "detections.jsonl")} == new_ids
    assert {e.meta["photo_id"] for e in read_jsonl(out / "test.jsonl")} == new_ids

