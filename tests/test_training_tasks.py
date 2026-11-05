"""训练样本生成器：schema、划分与泄漏、提示里不含特征标签和引擎概率、讲解过校验器、确定性。"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from lab_detective import knowledge
from lab_detective.labsim.core import SIGNATURES
from lab_detective.labsim.registry import EXPERIMENTS, get_experiment
from lab_detective.training import generate as gen
from lab_detective.training import prompts, splits
from lab_detective.training import tasks_diagnose as td
from lab_detective.training import tasks_explain as te
from lab_detective.training import tasks_knowledge as tk
from lab_detective.training.format import read_jsonl
from lab_detective.tutor.context import SYSTEM_PROMPT_ZH
from lab_detective.tutor.templates import BAND_ZH, SIGNATURE_ZH, nchars
from lab_detective.tutor.verifier import verify

warnings.simplefilter("ignore")

COUNTS = {"diagnose": 30, "design": 6, "explain": 14, "knowledge": 30}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen")
    rep = gen.generate(COUNTS, out, workers=1, chunk=8, quiet=True)
    ex = {t: {s: list(read_jsonl(out / s / f"{t}.jsonl")) for s in ("train", "val")} for t in COUNTS}
    return {"out": out, "report": rep, "ex": ex}


@pytest.fixture(scope="module")
def named_diag():
    """只用训练实验（不含程序化任务族）的诊断样本，覆盖全部 6 个训练实验。"""
    return [td.build_example(i, "train", generic_fraction=0.0) for i in range(40)]


def _all(data, split="train"):
    return [e for t in COUNTS for e in data["ex"][t][split]]


def _text(ex) -> str:
    return ex.prompt_text() + "\n" + ex.messages[-1]["content"]


# ── 划分 ──────────────────────────────────────────────────────────────


def test_splits_follow_registry_and_withheld_is_sane():
    assert splits.check_withheld() == []
    assert set(splits.TRAIN_EXPERIMENTS) == {k for k, e in EXPERIMENTS.items() if not e.held_out}
    assert set(splits.HELD_OUT_EXPERIMENTS) == {k for k, e in EXPERIMENTS.items() if e.held_out}
    assert not set(splits.TRAIN_EXPERIMENTS) & set(splits.HELD_OUT_EXPERIMENTS)
    for eid, cid in splits.WITHHELD_CAUSES.items():
        exp = get_experiment(eid)
        assert cid not in splits.allowed_causes(exp)
        assert len(splits.allowed_causes(exp)) == len(exp.causes) - 1
    with pytest.raises(ValueError):
        splits.allowed_causes(get_experiment(splits.HELD_OUT_EXPERIMENTS[0]))


def test_seeds_are_deterministic_and_train_val_disjoint():
    assert splits.example_seed("diagnose", "train", 7) == splits.example_seed("diagnose", "train", 7)
    tr = {splits.example_seed("diagnose", "train", i) for i in range(3000)}
    va = {splits.example_seed("diagnose", "val", i) for i in range(3000)}
    assert len(tr) == 3000 and not tr & va
    assert splits.example_seed("diagnose", "train", 0) != splits.example_seed("design", "train", 0)
    with pytest.raises(ValueError):
        splits.example_seed("diagnose", "bench", 0)
    bench = {splits.bench_seed("diagnose", i) for i in range(3000)}
    assert not bench & tr and not bench & va
    assert splits.split_counts(100) == (95, 5)
    assert splits.split_counts(20000) == (19000, 1000)
    assert splits.split_counts(1) == (1, 0)


def test_protocol_and_cause_mix():
    rng = np.random.default_rng(0)
    exp = get_experiment("hooke")
    protos = Counter()
    for _ in range(4000):
        p = td.sample_protocol(rng, exp)
        protos["extra" if "+" in p else p] += 1
    assert abs(protos["single"] / 4000 - 0.15) < 0.03
    assert abs(protos["standard"] / 4000 - 0.60) < 0.03
    assert abs(protos["extra"] / 4000 - 0.25) < 0.03
    allowed = splits.allowed_causes(exp)
    causes = Counter(td.sample_cause(rng, exp, allowed) for _ in range(4000))
    assert abs(causes[exp.random_only_id] / 4000 - 0.25) < 0.03
    assert splits.WITHHELD_CAUSES["hooke"] not in causes


# ── schema ────────────────────────────────────────────────────────────


def test_counts_and_val_fraction(data):
    rep = data["report"]["tasks"]
    for t in ("diagnose", "design", "explain"):
        n_tr, n_va = splits.split_counts(COUNTS[t])
        assert rep[t]["val"] == n_va and rep[t]["train"] + rep[t]["dropped"] <= n_tr
        assert rep[t]["train_val_overlap_removed"] == 0
    assert rep["knowledge"]["train"] > 0
    assert (data["out"] / "generate_report.json").exists()


def test_all_examples_valid_against_schemas(data, named_diag):
    for t in COUNTS:
        for split in ("train", "val"):
            for ex in data["ex"][t][split]:
                ex.validate()
                assert ex.task == t and ex.meta["split"] == split
                assert ex.messages[0] == {"role": "system", "content": SYSTEM_PROMPT_ZH}
                if t == "explain":
                    continue
                variant = ex.meta.get("schema", t)
                assert prompts.validate_output(variant, ex.messages[-1]["content"]) == [], ex.id
    for ex in named_diag:
        assert prompts.validate_output(ex.meta["schema"], ex.messages[-1]["content"]) == [], ex.id


def test_diagnose_target_matches_engine_conventions(named_diag):
    variants = Counter(ex.meta["variant"] for ex in named_diag)
    assert variants["list"] > 0 and variants["propose"] > 0
    for ex in named_diag:
        tgt = json.loads(ex.messages[-1]["content"])
        probs = [r["prob"] for r in tgt["ranked"]]
        assert all(abs(p * 20 - round(p * 20)) < 1e-9 for p in probs)  # 0.05 的倍数
        assert probs == sorted(probs, reverse=True)
        assert nchars(tgt["explanation"]) <= prompts.EXPLAIN_LIMIT
        if ex.meta["variant"] == "propose":
            assert "【候选原因】" not in ex.prompt_text()
            assert len(tgt["ranked"]) <= prompts.MAX_PROPOSE
            names = {c["name"] for c in ex.meta["causes"]}
            assert {r["name"] for r in tgt["ranked"]} <= names
        else:
            ids = set(ex.meta["id_map"])
            assert {r["cause_id"] for r in tgt["ranked"]} == ids  # 全部候选都列出
            assert ex.meta["id_map"][tgt["ranked"][0]["cause_id"]] == ex.meta["top_cause"]
            if tgt["next_measurement"]:
                assert tgt["next_measurement"]["id"] in ex.meta["menu_map"]
                top2 = [r["cause_id"] for r in tgt["ranked"][:2]]
                assert [p["cause_id"] for p in tgt["next_measurement"]["predicted"]] == top2
            if ex.meta["next_no_separation"]:  # 引擎说没有测量能分开 → 目标写 null，讲解也不推荐测量
                assert tgt["next_measurement"] is None and "下一步：" not in tgt["explanation"]
            if ex.meta["variant"] == "omit_true":
                assert ex.meta["truth_cause"] not in ex.meta["id_map"].values()
    # 讲解绝大多数能直接通过校验器
    assert sum(ex.meta["explanation_verified"] for ex in named_diag) >= 0.85 * len(named_diag)


# ── 泄漏：留出实验、扣下的原因 ─────────────────────────────────────────


def _forbidden_strings() -> list[str]:
    out = []
    for eid in splits.HELD_OUT_EXPERIMENTS:
        out.append(EXPERIMENTS[eid].name_zh)
        try:
            out.append(knowledge.load_card(eid)["name_zh"])
        except KeyError:
            pass
    for eid, cid in splits.WITHHELD_CAUSES.items():
        c = get_experiment(eid).cause(cid)
        out += [cid, c.name_zh, c.description_zh, c.name_en]
    return [s for s in dict.fromkeys(out) if s]


def test_no_held_out_experiment_or_withheld_cause_in_train(data, named_diag):
    forbidden = _forbidden_strings()
    withheld = splits.withheld_ids()
    exs = _all(data) + _all(data, "val") + named_diag
    for ex in exs:
        assert ex.meta["experiment_id"] not in splits.HELD_OUT_EXPERIMENTS
        for k in ("truth_cause", "top_cause"):
            assert ex.meta.get(k) not in withheld
        assert not set((ex.meta.get("id_map") or {}).values()) & withheld
        text = _text(ex)
        hits = [s for s in forbidden if s in text]
        assert not hits, (ex.id, hits)
    # 训练实验都出现过
    assert {ex.meta["experiment_id"] for ex in named_diag} == set(splits.TRAIN_EXPERIMENTS)


def test_knowledge_pool_excludes_held_out_topics():
    pool = tk.question_pool()
    keys = {it["key"] for it in pool}
    for cid in splits.withheld_ids():
        assert not any(k.startswith(f"cause|{cid}|") for k in keys)
    allowed = tk.train_kps()
    held_only = {k["id"] for k in knowledge.all_kps()
                 if k["experiments"] and all(e in splits.HELD_OUT_EXPERIMENTS for e in k["experiments"])}
    assert "kp.pendulum_g" in held_only and "kp.pendulum_g" not in allowed
    for it in pool:
        assert it["target"]["knowledge_points"][0]["id"] in allowed or it["kind"] == "cause"
        assert not any(n in it["question"] for n in tk.held_out_names())
    tr = {it["key"].rsplit("|", 1)[0] for it in pool if tk.split_of(it) == "train"}
    va = {it["key"].rsplit("|", 1)[0] for it in pool if tk.split_of(it) == "val"}
    assert va and not tr & va


# ── 提示里不含特征标签和引擎概率 ───────────────────────────────────────


def _strip_fixed(text: str) -> str:
    for b in prompts.fixed_blocks():
        text = text.replace(b, "")
    return text


def test_prompts_have_no_signature_tags_or_engine_outputs(data, named_diag):
    exs = data["ex"]["diagnose"]["train"] + data["ex"]["design"]["train"] + named_diag
    phrases = list(SIGNATURE_ZH.values()) + list(BAND_ZH.values())
    words = ("概率", "后验", "信息增益", "嫌疑", "consistent", "tension", "inconsistent", "eig")
    for ex in exs:
        p = _strip_fixed(ex.prompt_text())
        tags = [s for s in SIGNATURES if s in p]
        assert not tags, (ex.id, tags)
        assert not [s for s in phrases if s in p], ex.id
        assert not [w for w in words if w in p], ex.id
        exp_id = ex.meta.get("instance_id") or ex.meta["experiment_id"]
        if exp_id in EXPERIMENTS:
            assert not [c for c in EXPERIMENTS[exp_id].cause_ids() if c in p], ex.id
            assert not [m.id for m in EXPERIMENTS[exp_id].menu if f" {m.id}" in p or f"{m.id}：" in p], ex.id
        tgt = json.loads(ex.messages[-1]["content"])
        if ex.task == "diagnose" and len(tgt.get("evidence", "")) > 8:
            assert tgt["evidence"] not in p
    # 固定文字确实不含样本信息：两条不同样本的固定块相同
    a, b = named_diag[0].prompt_text(), named_diag[1].prompt_text()
    assert prompts.SIGNATURE_BLOCK in a and prompts.SIGNATURE_BLOCK in b


def test_build_diagnose_prompt_takes_no_engine_output():
    import inspect

    for fn in (prompts.build_diagnose_messages, prompts.build_design_messages):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"post", "posterior", "fact_sheet", "fs", "probs", "signatures"}


# ── 讲解过校验器 ──────────────────────────────────────────────────────


def test_explanation_targets_pass_verifier(data):
    exs = data["ex"]["explain"]["train"] + data["ex"]["explain"]["val"]
    assert exs
    by_id = {ex.id: ex for ex in exs}
    for split, n in (("train", splits.split_counts(COUNTS["explain"])[0]), ("val", splits.split_counts(COUNTS["explain"])[1])):
        for i in range(n):
            case, fs, rng, _ = te.explain_case(i, split)
            ex = by_id.get(f"expl-{case.family}-{split}-{i:06d}")
            if ex is None:
                continue
            text = ex.messages[-1]["content"]
            v = verify(text, fs)
            assert v["ok"], (ex.id, v["problems"])
            assert nchars(text) <= te.LIMIT
            assert ex.meta["fact_sheet_sha"] == te._sha(fs)


def test_explain_prompt_is_inference_context(data):
    from lab_detective.tutor.context import PHASE_TASKS_ZH

    for ex in data["ex"]["explain"]["train"][:5]:
        assert PHASE_TASKS_ZH["explain"] in ex.messages[1]["content"]
        assert len(ex.messages) == 3


# ── 打乱标签对照能用 ──────────────────────────────────────────────────


def test_label_shuffle_control_changes_diagnose_targets(named_diag):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
    import common as C

    recs = [ex.to_dict() for ex in named_diag]
    new, st = C.apply_label_shuffle(recs, seed=1)
    assert st["by_how"]["meta"] == len(recs)
    assert st["frac_changed"] >= 0.9
    assert st["frac_top_changed"] >= 0.4
    for r in new:
        json.loads(r["messages"][-1]["content"])  # 置换后仍是合法 JSON


# ── 确定性 ────────────────────────────────────────────────────────────


def test_examples_are_deterministic():
    a = td.build_example(3, "train").to_dict()
    b = td.build_example(3, "train").to_dict()
    assert a == b
    assert td.build_example(4, "train").to_dict() != a
    assert tk.build_example(2, "train").to_dict() == tk.build_example(2, "train").to_dict()


def test_worker_count_does_not_change_output(tmp_path):
    counts = {"diagnose": 8, "explain": 4, "knowledge": 6}
    gen.generate(counts, tmp_path / "w1", workers=1, chunk=3, quiet=True)
    gen.generate(counts, tmp_path / "w2", workers=2, chunk=3, quiet=True)
    for t in counts:
        for split in ("train", "val"):
            f1 = (tmp_path / "w1" / split / f"{t}.jsonl").read_bytes()
            f2 = (tmp_path / "w2" / split / f"{t}.jsonl").read_bytes()
            assert f1 == f2, (t, split)
