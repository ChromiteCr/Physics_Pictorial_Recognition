import numpy as np
import pytest

from lab_detective.bench import stats
from lab_detective.bench.freeze import freeze, verify
from lab_detective.training.format import Example, leakage_report, read_jsonl, write_jsonl


def _ex(i, text="题目", ans="{}"):
    return Example(f"x{i}", "diagnose", "engine",
                   [{"role": "system", "content": "s"}, {"role": "user", "content": text},
                    {"role": "assistant", "content": ans}], {"split": "train"})


def test_jsonl_roundtrip_and_validation(tmp_path):
    p = tmp_path / "a.jsonl"
    assert write_jsonl(p, [_ex(1), _ex(2, "另一题")]) == 2
    got = list(read_jsonl(p))
    assert [e.id for e in got] == ["x1", "x2"]
    with pytest.raises(AssertionError):
        Example("y", "bogus", "engine", [{"role": "assistant", "content": "x"}]).validate()


def test_leakage_detects_id_and_content():
    rep = leakage_report([_ex(1, "A"), _ex(2, "B")], [_ex(1, "C"), _ex(9, "B")])
    assert rep["id_overlap"] == ["x1"]
    assert rep["content_overlap"] == [("x9", "x2")]
    assert not rep["clean"]


def test_freeze_detects_change(tmp_path):
    (tmp_path / "t.jsonl").write_text("1\n")
    freeze(tmp_path, "测试者")
    assert verify(tmp_path)["ok"]
    (tmp_path / "t.jsonl").write_text("2\n")
    assert not verify(tmp_path)["ok"]


def test_stats_basics():
    lo, hi = stats.wilson(18, 24)
    assert 0.55 < lo < 0.56 and 0.88 < hi < 0.89
    r = stats.mcnemar_exact([1, 1, 1, 0, 1], [0, 0, 1, 0, 1])
    assert r["a_only"] == 2 and r["b_only"] == 0 and r["p_value"] == pytest.approx(0.5)
    assert stats.sign_test([1, 2, 3, 4, 5, 6])["p_value"] == pytest.approx(2 / 64)
    assert stats.sign_test([1, 1, 1, 1, 1, 1, 1, 1, -1])["p_value"] == pytest.approx(20 / 512)
    assert stats.holm([0.01, 0.04, 0.03]) == [True, False, False]
    e = stats.ece([0.9, 0.9, 0.1], [1, 1, 0])
    assert e["ece"] == pytest.approx(0.1, abs=1e-9)
    assert stats.chance_product_of_marginals(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.5)
