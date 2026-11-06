"""第二批训练样本：诊断样本的 meta.causes、引导（guide）、识别（recognize）、蒸馏任务、实验卡。"""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from lab_detective.training import tasks_diagnose, tasks_guide
from lab_detective.training.splits import HELD_OUT_EXPERIMENTS, WITHHELD_CAUSES
from lab_detective.labsim.registry import get_experiment

warnings.simplefilter("ignore")


# ── 诊断样本带 meta.causes（服务器打乱标签对照要用）────────────────────

def test_diagnose_meta_causes_present():
    for i in range(3):
        ex = tasks_diagnose.build_example(i, "train", n_samples=16)
        causes = ex.meta["causes"]
        assert causes and all(set(c) == {"id", "name"} for c in causes)
        assert {c["id"] for c in causes} == set(ex.meta["id_map"])
        exp_id = ex.meta["id_map"][causes[0]["id"]].split(".")[0]
        assert all(c["name"] for c in causes), exp_id


# ── 引导 ──────────────────────────────────────────────────────────────

def _held_out_names() -> list[str]:
    names = []
    for eid in HELD_OUT_EXPERIMENTS:
        names.append(get_experiment(eid).name_zh)
    for eid, cid in WITHHELD_CAUSES.items():
        names.append(get_experiment(eid).cause(cid).name_zh)
    return names


def test_guide_examples_deterministic_and_clean():
    banned = _held_out_names()
    cats = set()
    for i in range(30):
        ex = tasks_guide.build_example(i, "train")
        if ex is None:
            continue
        again = tasks_guide.build_example(i, "train")
        assert ex.to_dict() == again.to_dict()
        assert ex.task == "guide" and ex.messages[-1]["role"] == "assistant"
        blob = json.dumps(ex.to_dict(), ensure_ascii=False)
        for n in banned:
            assert n not in blob, (i, n)
        text = ex.messages[-1]["content"]
        assert tasks_guide.n_questions(text) <= 1
        cats.add(ex.meta["category"])
        if ex.meta["category"] == "diagnose_ask":
            assert not any(c["name"] in text for c in ex.meta["causes"])
        # 上下文只有 system + user 两条（不放完整对话）
        assert [m["role"] for m in ex.messages] == ["system", "user", "assistant"]
    assert len(cats) >= 5


def test_guide_val_differs_from_train():
    a = tasks_guide.build_example(0, "train")
    b = tasks_guide.build_example(0, "val")
    assert a is None or b is None or a.meta["seed"] != b.meta["seed"]


def test_one_question_and_problems():
    t = "先检查一下：仪器静止了吗？视线和刻度平齐了吗？"
    assert tasks_guide.n_questions(tasks_guide.one_question(t)) == 1
    assert tasks_guide.guide_problems("你觉得是「漏算挂钩质量」吗？", "diagnose_ask", None, ["漏算挂钩质量"], False)
    assert not tasks_guide.guide_problems("你觉得最可能的原因是哪一个？", "diagnose_ask", None, ["漏算挂钩质量"], False)
    assert tasks_guide.guide_problems("对吗？为什么？", "plan", None, [], False)


# ── 识别 ──────────────────────────────────────────────────────────────

def _synthetic_rec(tmp_path, n=12):
    from PIL import Image

    root = tmp_path / "rec"
    (root / "images").mkdir(parents=True)
    rows, photos = [], {}
    for i in range(n):
        pid = f"sc-IMG_{i:04d}"
        Image.new("RGB", (320, 240), (40 + i * 10, 90, 120)).save(root / "images" / f"{pid}.jpg")
        rows.append({"id": pid, "file": f"images/{pid}.jpg", "sha1": "x", "gt_boxes": [
            {"class_id": 0, "class_name": "wooden_block", "bbox": [100, 100, 180, 150]},
            {"class_id": 1, "class_name": "dynamometer", "bbox": [200, 110, 300, 130]}],
            "provisional_letter": "E" if i % 2 == 0 else "Z", "rule_confident": i % 2 == 0,
            "rule_confidence": 0.9 if i % 2 == 0 else 0.0})
        photos[pid] = {"session": i // 3 + 1, "split": "test" if i < 3 else "train", "taken_at": None,
                       "time_source": "none"}
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"status": "draft", "params": {}, "photos": photos}), encoding="utf-8")
    return root, split


def test_recognize_rule_confident_train_side_only(tmp_path):
    from lab_detective.training import tasks_recognize as R

    root, split = _synthetic_rec(tmp_path)
    tr, va, rep = R.build_examples(root / "manifest.jsonl", split, root / "review.json", root)
    ids = {e.meta["photo_id"] for e in tr + va}
    assert ids and not ids & {"sc-IMG_0000", "sc-IMG_0001", "sc-IMG_0002"}  # 测试侧照片不进训练
    assert all(e.meta["letter"] == "E" and e.meta["label_status"] == "rule_confident" for e in tr + va)
    assert rep["test_side_skipped"] == 3 and rep["label_source"] == "rule_confident"
    assert any("只有一个类别" in w for w in rep["warnings"])
    assert len(tr + va) == 3 * rep["eligible"]
    for e in tr + va:
        (img,) = e.images()
        assert img.startswith("images/aug/") and (root / img).exists()
        out = json.loads(e.messages[-1]["content"])
        assert out["letter"] == "E" and 0.5 <= out["confidence"] <= 0.95
    # 同一张照片的 3 个变体在同一边；验证集按整张照片
    by_photo = {}
    for e in tr + va:
        by_photo.setdefault(e.meta["photo_id"], set()).add(e.meta["split"])
    assert all(len(v) == 1 for v in by_photo.values())
    # 确定性
    tr2, va2, _ = R.build_examples(root / "manifest.jsonl", split, root / "review.json", root)
    assert [e.to_dict() for e in tr] == [e.to_dict() for e in tr2]


def test_recognize_review_overrides_and_missing_split(tmp_path):
    from lab_detective.training import tasks_recognize as R

    root, split = _synthetic_rec(tmp_path)
    (root / "review.json").write_text(json.dumps({"sc-IMG_0005": "c", "sc-IMG_0001": "E"}), encoding="utf-8")
    tr, va, rep = R.build_examples(root / "manifest.jsonl", split, root / "review.json", root)
    assert {e.meta["photo_id"] for e in tr + va} == {"sc-IMG_0005"}  # 0001 在测试侧
    assert all(e.meta["letter"] == "C" and e.meta["label_status"] == "reviewed" for e in tr + va)
    with pytest.raises(R.NoSplit):
        R.build_examples(root / "manifest.jsonl", tmp_path / "nope.json", None, root)


def test_augment_boxes_shift_inside():
    from PIL import Image
    from lab_detective.training import tasks_recognize as R

    im = Image.new("RGB", (400, 300))
    boxes = [{"class_name": "cart", "bbox": [0, 0, 100, 100]}, {"class_name": "track", "bbox": [50, 200, 390, 290]}]
    out, nb, p = R.augment(im, boxes, np.random.default_rng(0))
    W, H = out.size
    assert W >= 400 * (1 - R.MAX_CROP) - 1 and H >= 300 * (1 - R.MAX_CROP) - 1
    assert 0.9 <= p["brightness"] <= 1.1
    for b in nb:
        x1, y1, x2, y2 = b["bbox"]
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H


# ── 蒸馏任务 ──────────────────────────────────────────────────────────

def _first_diag_job(start=1_000_000):
    from lab_detective.training import distill_jobs as D

    for i in range(start, start + 40):
        j = D.diagnose_job(i)
        if j is not None:
            return i, j
    raise AssertionError("找不到清单变体")


def test_distill_diagnose_prompt_matches_training_and_filters():
    from lab_detective.training import distill_jobs as D

    i, job = _first_diag_job()
    ex = tasks_diagnose.build_example(i, "train")
    assert job.messages == ex.messages[:-1]  # 教师看到的就是训练提示
    assert job.meta["causes"] == ex.meta["causes"]
    ref = D.diagnose_reference(i)
    assert ref["top"] == ex.meta["top_cause"]
    ok, info = job.check(ex.messages[-1]["content"])  # 引擎自己的目标应当保留
    assert ok == ex.meta["explanation_verified"], info
    obj = json.loads(ex.messages[-1]["content"])
    assert not job.check("这不是 JSON")[0]
    if len(obj["ranked"]) > 1 and len(ref["tied"]) == 1:  # 把第一名换成别的原因 → 拒绝
        a, b = obj["ranked"][0]["cause_id"], obj["ranked"][-1]["cause_id"]
        obj["ranked"][0]["cause_id"], obj["ranked"][-1]["cause_id"] = b, a
        ok2, info2 = job.check(json.dumps(obj, ensure_ascii=False))
        assert not ok2 and info2["reason"] == "top_cause"


def test_distill_guide_check_and_job_from_id():
    from lab_detective.training import distill_jobs as D

    jobs = D.build_jobs("guide", 3, D.START["hy4"])
    assert len(jobs) == 3 and all(j.task == "guide" for j in jobs)
    j = jobs[0]
    assert D.job_from_id(j.id).messages == j.messages
    assert not j.check("第一个问题？第二个问题？")[0]
    assert not j.check("请记下 12345.6 这个数。")[0]  # 提示里没有的数字


def test_distill_cap_hy4_roundtrip_and_no_key(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from lab_detective.training import distill, distill_jobs as D

    root = tmp_path / "distill"
    pack = root / "hy4_pack"
    info = D.export_pack({"diagnose": 2, "guide": 2}, pack)
    assert info["n_prompts"] == 4 and (pack / "README.md").exists()
    rows = [json.loads(x) for x in (pack / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    # 假教师：诊断题答引擎自己的目标（应保留），引导题答两个问题（应拒绝）
    resp = []
    for r in rows:
        task, _, idx = r["id"].rpartition("-")
        if task == "diagnose":
            text = tasks_diagnose.build_example(int(idx), "train").messages[-1]["content"]
        else:
            text = "第一个问题？第二个问题？"
        resp.append({"id": r["id"], "text": text})
    resp.append({"id": "nonsense-1", "text": "x"})
    (pack / "responses.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in resp),
                                          encoding="utf-8")
    monkeypatch.setattr(D, "MAX_DISTILLED", 1)  # 上限 1：第二条合格的也不留
    st = D.import_pack(pack, root)
    assert st["read"] == 5 and st["unknown_id"] == 1 and st["kept"] <= 1
    assert D.total_kept(root) == st["kept"]
    st2 = D.import_pack(pack, root)  # 重复导入不重复计
    assert st2["duplicate"] == 4 and D.total_kept(root) == st["kept"]
    out = D.to_train(root, tmp_path / "train")
    if st["kept"]:
        (f,) = out
        ex = json.loads(Path(f).read_text(encoding="utf-8").splitlines()[0])
        assert ex["source"] == "distill-hy4" and ex["task"] == "diagnose"
    # 没有 key：礼貌拒绝，不抛异常
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(distill, "KEY_FILE", tmp_path / "none.env")
    a = SimpleNamespace(task="diagnose", n=1, workers=1, offset=0, model="m", no_think=False, seed=0)
    assert D.cmd_deepseek(a, root=root) == 2
    assert "DEEPSEEK_API_KEY" in capsys.readouterr().err


def test_distill_deepseek_with_fake_teacher(tmp_path):
    from types import SimpleNamespace

    from lab_detective.training import distill_jobs as D

    def teacher(messages, max_tokens=2048):
        return "{}", {}

    a = SimpleNamespace(task="guide", n=2, workers=1, offset=0, model="m", no_think=False,
                        seed=D.BASE_SEED)
    assert D.cmd_deepseek(a, teacher=teacher, root=tmp_path) == 0
    raw = tmp_path / "deepseek_guide" / "raw_results.jsonl"
    assert len(raw.read_text(encoding="utf-8").splitlines()) == 2


# ── 实验卡 ────────────────────────────────────────────────────────────

def test_card_examples_valid_and_deterministic():
    from lab_detective.training import tasks_card as C

    fams = set()
    for i in range(20):
        ex = C.build_example(i, "train")
        assert ex.to_dict() == C.build_example(i, "train").to_dict()
        out = json.loads(ex.messages[-1]["content"])
        assert C.validate_card_output(out) == []
        assert out["relation_family"] == ex.meta["relation_family"]
        assert not any(ch in "0123456789" for s in out["steps"] for ch in s.replace("6 组", ""))  # 步骤不写设定值
        fams.add(ex.meta["relation_family"])
    assert len(fams) >= 2
    assert C.validate_card_output('{"name_zh": "x"}')


def test_student_facing_banned_terms():
    from lab_detective.training.distill_jobs import banned_terms
    assert banned_terms("原因照抄事实单里的名称") == ["事实单", "照抄"]
    assert banned_terms("最可能是摩擦没有平衡，截距 -0.3 m/s²，下一步称小车质量。") == []
