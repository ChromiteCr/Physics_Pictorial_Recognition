"""物理 bench：与训练同一套目标构造（逐字对照）、小规模生成、打分（引擎参照 / 故意答错 / 规则）、配对比较、K1/K3。"""

import json
from argparse import Namespace

import numpy as np
import pytest

from lab_detective.bench import physics_bench as pb
from lab_detective.bench import run_models, score
from lab_detective.training import prompts
from lab_detective.training import tasks_design as tdz
from lab_detective.training import tasks_diagnose as td
from lab_detective.training.format import read_jsonl
from lab_detective.training.splits import example_rng, example_seed

SPLITS = ("in_distribution", "held_out", "swapped_table", "oracle_in_prompt", "unmodelled", "clean", "design")


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    s = pb.build(SPLITS, out, small=True, train_dir=out / "no_train", log=lambda *_: None)
    return out, s


def test_target_matches_training_generator():
    """训练集 0 号样本是"给清单"变体：bench 的目标构造必须逐字给出同样的提示和答案。"""
    rng = example_rng("diagnose", "train", 0)
    case = td.simulate_case(rng)
    assert rng.random() >= td.P_PROPOSE + td.P_OMIT_TRUE
    r = pb.diagnose_target(case, case.allowed, rng)
    ex = td.build_example(0, "train")
    assert r["messages"] == ex.messages[:-1]
    assert json.dumps(r["target"], ensure_ascii=False) == ex.messages[-1]["content"]
    best = max(r["eig"], key=r["eig"].get)
    if r["target"]["next_measurement"]:
        assert r["target"]["next_measurement"]["id"] == best  # 同一个种子，EIG 与事实单完全一致


def test_design_matches_training_generator():
    ex = tdz.build_example(0, "train")
    mine, _, _ = pb._design_one(0, 0, example_rng("design", "train", 0))
    assert mine.messages == ex.messages


def test_seed_namespace_disjoint():
    train = {example_seed("diagnose", sp, i) for sp in ("train", "val") for i in range(300)}
    bench = {pb.split_seed(s, i) for s in pb.SPLITS for i in range(300)}
    assert not train & bench
    assert len(bench) == 300 * len(pb.SPLITS)


def test_build_small(bench):
    out, s = bench
    held = pb.held_out_ids()
    assert any(h.startswith("physics:") for h in held) and "pendulum" in held
    assert s["splits"]["held_out"]["n"] == len(held)
    for sp in SPLITS:
        exs = list(read_jsonl(out / f"bench_{sp}.jsonl"))
        assert exs and all(e.messages[-1]["role"] == "assistant" for e in exs)
        for e in exs:
            assert not prompts.validate_output(e.meta["schema"], e.messages[-1]["content"])
    orc = list(read_jsonl(out / "bench_oracle_in_prompt.jsonl"))
    assert all(pb.ORACLE_HEAD in e.messages[-2]["content"] for e in orc)
    gold = pb.read_gold(out / "bench_unmodelled.gold.jsonl")
    assert all(g["out_of_list"] and g["subsplit"] in ("detectable", "degenerate") for g in gold.values())
    gold = pb.read_gold(out / "bench_in_distribution.gold.jsonl")
    for g in gold.values():
        assert g["truth_disp"] in g["eq_class"]
        assert sorted(c for grp in g["eq_classes"] for c in grp) == sorted(g["id_map"])


def _preds(out, model, splits=SPLITS, transform=None):
    path = out / f"{model}.jsonl"
    a = Namespace(model=model if model in ("engine", "rule") else "engine", splits=list(splits),
                  bench_dir=str(out), out=str(path), base_url="", provider="mlx", api_key_env=None,
                  thinking="off", max_tokens=64, temperature=0.0, timeout=1.0, limit=None, resume=False,
                  log_calls=False)
    run_models.run(a)
    if transform:
        rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                r["text"] = transform(r["text"])
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return score.load_predictions(path)


def _wrong(text):
    """故意答错：原因排名倒过来（最不可能的拿最高概率）；设计题点数多写 1。"""
    o = json.loads(text)
    if "ranked" in o:
        rk = o["ranked"]
        ids = [r["cause_id"] for r in reversed(rk)]
        o["ranked"] = [dict(r, cause_id=c) for r, c in zip(rk, ids)]  # 概率顺序不变，原因倒过来
        o["explanation"] = "随便猜的。"
    else:
        o["recommend"]["n_points"] += 1
    return json.dumps(o, ensure_ascii=False)


def test_engine_reference_and_wrong_predictor(bench):
    out, _ = bench
    gold = score.load_gold(out)
    eng = score.score_rows(gold, _preds(out, "engine"))
    rep = score.report(eng, gold, "engine")
    for sp in ("in_distribution", "held_out", "oracle_in_prompt", "clean"):
        a = rep["splits"][sp]["all"]
        assert a["json_valid"] == 1.0 and a["gold_agree"] == 1.0
        assert a["eig_regret"] < 1e-9
        assert a["eq_top1"] >= a["top1"]
    assert rep["k1_oracle_gold_agree"] == 1.0
    assert rep["splits"]["design"]["all"]["n_points_ok"] == 1.0
    assert rep["k3_swapped_table"]["n"] == pb.SMALL
    bad = score.score_rows(gold, _preds(out, "bad", transform=_wrong))
    rb = score.report(bad, gold, "bad")
    ea, ba = rep["overall_diagnose"], rb["overall_diagnose"]
    assert ba["gold_agree"] < ea["gold_agree"]
    assert ba["log_score"] > ea["log_score"]
    assert rb["splits"]["design"]["all"]["n_points_ok"] == 0.0
    cmp = score.compare_all(eng, bad)
    mc = cmp["in_distribution"]["eq_top1"]["mcnemar"]
    assert mc["a_only"] >= mc["b_only"] and 0.0 <= mc["p_value"] <= 1.0
    md = score.to_markdown(rep, cmp, "bad")
    assert "配对比较" in md and "in_distribution" in md


def test_rule_and_missing_predictions(bench):
    out, _ = bench
    gold = score.load_gold(out, ["in_distribution", "clean"])
    rows = score.score_rows(gold, _preds(out, "rule", ("in_distribution", "clean")))
    assert all(r["json_valid"] for r in rows)
    empty = score.score_rows(gold, {})
    assert all(not r["json_valid"] and r["missing"] and r["eig_regret"] in (0.0, 1.0) for r in empty)


def test_parse_tolerant_and_mcnemar_wiring():
    assert score.parse('```json\n{"a": 1}\n```') == {"a": 1}
    assert score.parse("前面有字 {\"a\": 2} 后面也有") == {"a": 2}
    assert score.parse("不是 JSON") is None
    a = [{"id": str(i), "family": f"f{i % 3}", "eq_top1": i % 4 != 0} for i in range(40)]
    b = [{"id": str(i), "family": f"f{i % 3}", "eq_top1": i % 2 == 0} for i in range(40)]
    c = score.compare(a, b)
    assert c["n"] == 40 and c["mcnemar"]["discordant"] == c["mcnemar"]["a_only"] + c["mcnemar"]["b_only"]
    assert c["mcnemar"]["a_only"] == 20 and c["mcnemar"]["b_only"] == 10


def test_strip_reasoning_transport():
    body = {"choices": [{"message": {"content": "{}", "reasoning_content": "想了很久"}}]}
    t, st = run_models._strip_reasoning(lambda *a: (200, json.dumps(body).encode()))
    status, raw = t("u", {}, b"", 1.0)
    assert status == 200 and "reasoning_content" not in raw.decode() and st["reasoning_chars"] == 4
