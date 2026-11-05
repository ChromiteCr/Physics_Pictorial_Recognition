"""merge_and_eval 的纯 Python 部分：预测行格式（bench 打分器要 text 键）、示例拼接。"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
import merge_and_eval as M  # noqa: E402


class _Args:
    arm, backend, model, adapter, shots = "S1", "vllm", "merged/s1", None, 0


def _item(rid="recbench-x", task="recognize", gold='{"letter": "C"}'):
    rec = {"id": rid, "task": task, "source": "photo", "meta": {"experiment_id": "hooke", "split": "test"},
           "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": gold}]}
    return {"rec": rec, "bench_file": "bench_recognition_test.jsonl", "gold": gold, "prompt": "p", "images": []}


def test_row_has_text_for_bench_scorer():
    out = '{"letter": "C", "open_name": "平均速度", "cues": [], "apparatus": [], "confidence": 0.8}'
    row = M.make_row(_item(), _Args, out, 10, 20, "stop", latency=0.5)
    assert row["text"] == out == row["prediction"] and row["latency_s"] == 0.5
    assert row["json_ok"] is True and row["has_think"] is False
    json.dumps(row, ensure_ascii=False)
    rb = pytest.importorskip("lab_detective.bench.recognition_bench")
    pred = rb.parse_prediction(row["id"], row)
    assert pred.letter == "C" and pred.error is None  # 打分器能读出字母
    err = M.make_row(_item(), _Args, "", None, 0, "error", "boom")
    assert err["text"] == "" and rb.parse_prediction(err["id"], err).letter is None  # 出错按无效计


def test_think_marker_detected():
    row = M.make_row(_item(), _Args, "<think>\n想一想</think>{}", 1, 1, "stop")
    assert row["has_think"] is True


def test_shots_are_text_only_same_task_and_before_question():
    def r(rid, task, img=False):
        content = ([{"type": "image", "image": "images/a.jpg"}] if img else []) + [{"type": "text", "text": rid}]
        return {"id": rid, "task": task, "messages": [{"role": "system", "content": "S"},
                                                      {"role": "user", "content": content},
                                                      {"role": "assistant", "content": "A-" + rid}]}
    pool = [r(f"t{i}", "diagnose") for i in range(5)] + [r("img", "diagnose", True), r("k", "knowledge")]
    target = r("bench-1", "diagnose")
    shots = M.pick_shots(target, pool, 3, seed=0)
    assert len(shots) == 3 and all(s["task"] == "diagnose" and s["id"] != "img" for s in shots)
    assert shots == M.pick_shots(target, list(reversed(pool)), 3, seed=0)  # 与池子顺序无关
    msgs = M.with_shots(target["messages"][:-1], shots)
    assert msgs[0]["role"] == "system" and sum(m["role"] == "system" for m in msgs) == 1
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant"] * 3 + ["user"]
    assert msgs[-1] == target["messages"][1]
