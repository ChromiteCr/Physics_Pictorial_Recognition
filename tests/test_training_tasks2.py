"""第二批训练样本：诊断样本的 meta.causes、引导（guide）、识别（recognize）、蒸馏任务、实验卡。"""

from __future__ import annotations

import json
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
