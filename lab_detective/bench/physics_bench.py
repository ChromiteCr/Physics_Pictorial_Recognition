"""物理 bench 生成：各测试划分 → results/bench/physics/bench_<划分>.jsonl（训练样本格式，assistant = 标准答案 JSON）
+ 真值旁注 bench_<划分>.gold.jsonl + 事实单 bench_<划分>.facts.jsonl（校验讲解用）。

提示和目标与训练完全同一套构造：prompts.diagnose_prompt / design_prompt，目标按 tasks_diagnose / tasks_design 的做法
（同样的随机数消耗顺序；tests/test_physics_bench.py 有逐字对照测试）。bench 只用"给清单"的诊断变体。
种子：splits.bench_seed(任务, 划分偏移 + 序号)，与训练 / 验证的种子命名空间不重叠。

划分（默认条数见 SIZES，--small 每个划分只做 SMALL 条）：
  in_distribution  与训练同分布（训练实验 + 程序化任务族，去掉扣下的原因）
  held_out         全部留出的 labsim 实验 + 物理组件层里留出的组合实验，每个实验 HELD_OUT_EACH 条
  noise_shift      噪声放大 2 倍并改成 Student-t（labsim/hifi.py NOISE_SHIFT）
  withheld_cause   真实原因 = 训练时扣下的原因（候选清单里有它）
  hifi             错配的高保真世界（hifi.HIFI：t 噪声、计时拖尾、拖曳、线质量、抄错）
  clean            只有随机误差（误报率）
  unmodelled       两个原因同时作用 / 一个读数抄错 ×10；真实原因不在清单里；按引擎失配判据分 detectable / degenerate
  swapped_table    K3 换表对照：保留样本 A 的真实原因，提示里的数据表换成同一实验同一方案的另一份数据 B
  oracle_in_prompt K1：把标准答案直接放进提示末尾
  design           实验前设计（tasks_design 的目标）
（引导 guide 没有生成器，这里不做。）

真值旁注里的等价类：图谱（atlas_table，每个原因 ATLAS_N 份）里两两判别正确率 < 0.75（与 tasks_design 的 confused
同一条线）的原因对合并（并查集），得分时"判成同一等价类里的原因"算对。
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import warnings
import zlib
from pathlib import Path

import numpy as np

from ..atlas import atlas_table
from ..eig import protocol_plan, recommend
from ..engine import posterior
from ..labsim import hifi
from ..labsim.core import Dataset, Experiment
from ..labsim.registry import EXPERIMENTS, get_experiment
from ..paths import RESULTS
from ..training import prompts
from ..training import tasks_design as tdz
from ..training import tasks_diagnose as td
from ..training.format import Example, leakage_report, read_jsonl, write_jsonl
from ..training.splits import (HELD_OUT_EXPERIMENTS, TRAIN_EXPERIMENTS, WITHHELD_CAUSES, allowed_causes,
                               bench_seed)

OUT = RESULTS / "bench" / "physics"
SPLITS = ("in_distribution", "held_out", "noise_shift", "withheld_cause", "hifi", "clean", "unmodelled",
          "swapped_table", "oracle_in_prompt", "design")
SIZES = {"in_distribution": 600, "noise_shift": 300, "withheld_cause": 300, "hifi": 300, "clean": 150,
         "unmodelled": 150, "swapped_table": 600, "oracle_in_prompt": 100, "design": 200}
HELD_OUT_EACH = 100
SMALL = 2
SPLIT_OFFSET = {s: (i + 1) * 1_000_000 for i, s in enumerate(SPLITS)}
ATLAS_N = 32
CONFUSED_BELOW = tdz.CONFUSED_BELOW
EIG_SAMPLES = td.EIG_SAMPLES
ORACLE_HEAD = "【参考答案】（本题的正确输出，照抄即可）"

_EQ_CACHE: dict[tuple, list[list[str]]] = {}


def split_seed(split: str, index: int) -> int:
    task = "design" if split == "design" else "diagnose"
    return bench_seed(task, SPLIT_OFFSET[split] + int(index))


# ── 实验列表 ──────────────────────────────────────────────────────────


def held_out_ids() -> list[str]:
    """留出实验：labsim 注册表的 held_out + 组件层目录里 held_out 的组合（id 前缀 "physics:"）。"""
    from ..physics import catalog

    ids = list(HELD_OUT_EXPERIMENTS)
    ids += [f"physics:{e.id}" for e in catalog.CATALOG.values() if e.held_out]
    return ids


def load_experiment(eid: str) -> Experiment:
    if eid.startswith("physics:"):
        from ..physics import catalog

        return catalog.experiment(eid.split(":", 1)[1])
    return get_experiment(eid)


# ── 等价类 ────────────────────────────────────────────────────────────


def equivalence_classes(exp: Experiment, protocol: str, cand, generic: bool = False) -> list[list[str]]:
    """候选原因按"这个方案下分不开"合并成等价类（真实 id）。程序化实例不缓存。"""
    key = (exp.id, protocol, tuple(cand))
    if not generic and key in _EQ_CACHE:
        return _EQ_CACHE[key]
    rng = np.random.default_rng(zlib.crc32(f"{exp.id}|{protocol}|{','.join(cand)}|eq".encode()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = atlas_table(exp, [protocol], ATLAS_N, rng, cause_ids=list(cand))
    parent = {c: c for c in cand}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for r in rows:
        if r["acc"] < CONFUSED_BELOW:
            parent[find(r["cause_a"])] = find(r["cause_b"])
    groups: dict[str, list[str]] = {}
    for c in cand:
        groups.setdefault(find(c), []).append(c)
    out = list(groups.values())
    if not generic:
        _EQ_CACHE[key] = out
    return out


# ── 诊断目标（与 tasks_diagnose.build_example 的 list 变体逐行相同）────────────────


def _verify_sheet(fs: dict, probs: dict[str, float], top2: list[str]) -> dict:
    """与 tasks_diagnose.short_explanation 校验时用的事实单相同（概率改成取整值，下一次测量只留前两名）。"""
    f2 = copy.deepcopy(fs)
    for r in (f2.get("diagnosis") or {}).get("ranked") or []:
        if r.get("cause_id") in probs:
            r["prob"] = probs[r["cause_id"]]
    nm = f2.get("next_measurement")
    if nm:
        nm.pop("eig_bits", None)
        order = {c: i for i, c in enumerate(top2)}
        nm["predicted"] = sorted((p for p in nm.get("predicted") or [] if p.get("cause_id") in order),
                                 key=lambda p: order[p["cause_id"]])
    return f2


def diagnose_target(case: td.Case, cand, rng: np.random.Generator, n_samples: int = EIG_SAMPLES) -> dict:
    """case + 候选 → 提示、目标 JSON、事实单、引擎后验、各追加测量的 EIG。rng 的消耗顺序与训练生成器相同。"""
    exp = case.exp
    cand = list(cand)
    eig_int = int(rng.integers(0, 2**62))
    fs = td.fact_sheet(case, cand, np.random.default_rng(eig_int), n_samples)
    post = posterior(exp, case.ds, cand)
    order = sorted(range(len(cand)), key=lambda j: (-float(post.probs[j]), j))
    ranked_ids = [cand[j] for j in order]
    probs = {cand[j]: td.round_prob(post.probs[j]) for j in range(len(cand))}
    top2 = ranked_ids[:2]
    messages, disp, menu_disp = prompts.diagnose_prompt(exp, case.ds, cand, rng, propose=False)
    sigs = list((fs.get("panel") or {}).get("signatures") or ["none"])
    dg = fs.get("diagnosis") or {}
    evidence = td.clean_evidence((dg.get("ranked") or [{}])[0].get("evidence", ""))
    nm = fs.get("next_measurement")
    no_separation = bool(nm) and not nm.get("separates")
    if no_separation:
        nm = None
    fs_t = dict(fs, next_measurement=nm)
    pred_by = {p["cause_id"]: p for p in (nm or {}).get("predicted") or []}
    explanation, verified = td.short_explanation(fs_t, probs, top2)
    target = {
        "signatures": sigs, "evidence": evidence,
        "ranked": [{"cause_id": disp[c], "prob": probs[c], "expect": list(exp.cause(c).signatures)}
                   for c in ranked_ids],
        "not_in_list": bool(dg.get("not_in_list")),
        "next_measurement": None if not nm else {
            "id": menu_disp[nm["id"]],
            "predicted": [{"cause_id": disp[c], "value": pred_by[c]["value"], "unit": pred_by[c]["unit"]}
                          for c in top2 if c in pred_by]},
        "agreement": (fs.get("agreement") or {}).get("band"),
        "explanation": explanation,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = recommend(exp, case.ds, n_samples, np.random.default_rng(eig_int), cause_ids=tuple(cand))
    eig = {menu_disp[s.item_id]: float(s.eig_bits) for s in scores}
    return {"messages": messages, "target": target, "disp": disp, "menu_disp": menu_disp,
            "fs": _verify_sheet(fs_t, probs, top2), "probs_raw": {c: float(post.probs[j]) for j, c in enumerate(cand)},
            "ranked_ids": ranked_ids, "eig": eig, "verified": verified, "no_separation": no_separation,
            "not_in_list": bool(dg.get("not_in_list")), "signatures": sigs}


def _diag_record(split: str, index: int, seed: int, case: td.Case, cand, rng, *, truth: str,
                 out_of_list: bool = False, subsplit: str | None = None, extra: dict | None = None,
                 oracle: bool = False) -> tuple[Example, dict, dict]:
    exp = case.exp
    r = diagnose_target(case, cand, rng)
    disp, menu_disp = r["disp"], r["menu_disp"]
    target = r["target"]
    errs = prompts.validate_output("diagnose", target)
    if errs:
        raise ValueError(f"{split}#{index} 标准答案不合 schema：{errs}")
    messages = r["messages"]
    text = json.dumps(target, ensure_ascii=False)
    if oracle:
        messages = copy.deepcopy(messages)
        messages[-1]["content"] += "\n\n" + ORACLE_HEAD + "\n" + text
    protocol = case.protocol
    eq = equivalence_classes(exp, protocol, cand, case.generic)
    truth_disp = None if out_of_list else disp[truth]
    eq_disp = [[disp[c] for c in g] for g in eq]
    truth_class = next((g for g in eq_disp if truth_disp in g), []) if truth_disp else []
    if subsplit is None and out_of_list:
        subsplit = "detectable" if r["not_in_list"] else "degenerate"
    ex_id = f"bench-{split}-{case.family}-{index:05d}"
    meta = {"experiment_id": case.family, "split": f"bench:{split}", "seed": seed, "schema": "diagnose",
            "variant": "list", "generic": case.generic, "protocol": protocol,
            "causes": [{"id": d, "name": exp.cause(c).name_zh} for c, d in disp.items()],
            "id_map": {d: c for c, d in disp.items()}, "menu_map": {d: m for m, d in menu_disp.items()},
            "prompt_version": prompts.PROMPT_VERSION}
    if case.generic:
        meta["instance_id"] = exp.id
    ex = Example(ex_id, "diagnose", "engine", messages + [{"role": "assistant", "content": text}], meta)
    ex.validate()
    world = (case.ds.truth or {}).get("world", "labsim")
    gold = {"id": ex_id, "split": split, "schema": "diagnose", "family": case.family, "experiment": exp.id,
            "protocol": protocol, "world": world, "subsplit": subsplit,
            "truth_cause": truth, "truth_disp": truth_disp, "out_of_list": out_of_list,
            "eq_class": truth_class, "eq_classes": eq_disp,
            "engine_probs": {disp[c]: p for c, p in r["probs_raw"].items()},
            "engine_top": disp[r["ranked_ids"][0]], "engine_not_in_list": r["not_in_list"],
            "eig": r["eig"], "gold_next": (target["next_measurement"] or {}).get("id"),
            "expect": {disp[c]: list(exp.cause(c).signatures) for c in cand},
            "signatures": r["signatures"], "explanation_verified": r["verified"],
            "id_map": meta["id_map"], "menu_map": meta["menu_map"]}
    if extra:
        gold.update(extra)
    return ex, gold, {"id": ex_id, "fact_sheet": r["fs"]}


# ── 各划分的一条 ──────────────────────────────────────────────────────


def _case(exp, family, generic, cid, protocol, ds, allowed) -> td.Case:
    ds.meta["protocol"] = protocol
    return td.Case(exp, family, generic, cid, protocol, ds, tuple(allowed))


def _curated_train(rng) -> Experiment:
    return get_experiment(TRAIN_EXPERIMENTS[int(rng.integers(len(TRAIN_EXPERIMENTS)))])


def build_one(split: str, index: int, held: list[str] | None = None):
    seed = split_seed(split, index)
    rng = np.random.default_rng(seed)
    if split == "design":
        return _design_one(index, seed, rng)
    if split in ("in_distribution", "oracle_in_prompt"):
        case = td.simulate_case(rng)
        rng.random()  # 与训练生成器的变体抽签对齐
        return _diag_record(split, index, seed, case, case.allowed, rng, truth=case.true_cause,
                            oracle=split == "oracle_in_prompt")
    if split == "held_out":
        held = held or held_out_ids()
        eid = held[index % len(held)]
        exp = load_experiment(eid)
        cid = td.sample_cause(rng, exp, exp.cause_ids())
        protocol = td.sample_protocol(rng, exp)
        ds = exp.simulate(rng, cid, protocol_plan(exp, protocol))
        case = _case(exp, eid.replace("physics:", ""), False, cid, protocol, ds, exp.cause_ids())
        rng.random()
        return _diag_record(split, index, seed, case, case.allowed, rng, truth=cid)
    if split in ("noise_shift", "hifi"):
        if split == "noise_shift":
            exp, family, generic = td.sample_experiment(rng)
        else:
            exp = _curated_train(rng)
            family, generic = exp.id, False
        allowed = allowed_causes(exp)
        cid = td.sample_cause(rng, exp, allowed)
        protocol = td.sample_protocol(rng, exp)
        cfg = hifi.NOISE_SHIFT if split == "noise_shift" else hifi.HIFI
        ds = hifi.simulate(exp, rng, cid, protocol_plan(exp, protocol), cfg, world=split, protocol=protocol)
        case = _case(exp, family, generic, cid, protocol, ds, allowed)
        rng.random()
        return _diag_record(split, index, seed, case, allowed, rng, truth=cid)
    if split == "withheld_cause":
        exp = get_experiment(TRAIN_EXPERIMENTS[index % len(TRAIN_EXPERIMENTS)])
        cid = WITHHELD_CAUSES[exp.id]
        protocol = td.sample_protocol(rng, exp)
        ds = exp.simulate(rng, cid, protocol_plan(exp, protocol))
        case = _case(exp, exp.id, False, cid, protocol, ds, exp.cause_ids())
        rng.random()
        return _diag_record(split, index, seed, case, exp.cause_ids(), rng, truth=cid)
    if split == "clean":
        exp = _curated_train(rng)
        cid = exp.random_only_id
        ds = exp.simulate(rng, cid, protocol_plan(exp, "standard"))
        case = _case(exp, exp.id, False, cid, "standard", ds, allowed_causes(exp))
        rng.random()
        return _diag_record(split, index, seed, case, case.allowed, rng, truth=cid)
    if split == "unmodelled":
        exp = _curated_train(rng)
        allowed = allowed_causes(exp)
        protocol = "standard"
        plan = protocol_plan(exp, protocol)
        if index % 2 == 0:  # 两个原因同时作用
            others = [c for c in allowed if c != exp.random_only_id]
            a, b = (others[int(j)] for j in rng.choice(len(others), 2, replace=False))
            ds = hifi.simulate(exp, rng, a, plan, hifi.GAUSS, second_cause=b, world="unmodelled", protocol=protocol)
            truth, kind = f"{a}+{b}", "two_causes"
        else:  # 只有随机误差，但有一个读数抄错 ×10 / ÷10
            ds = hifi.simulate(exp, rng, exp.random_only_id, plan, hifi.GAUSS, world="unmodelled",
                               protocol=protocol, slip_force=True)
            truth, kind = f"{exp.random_only_id}+slip", "slip"
        case = _case(exp, exp.id, False, truth, protocol, ds, allowed)
        rng.random()
        return _diag_record(split, index, seed, case, allowed, rng, truth=truth, out_of_list=True,
                            extra={"unmodelled_kind": kind})
    if split == "swapped_table":
        exp = _curated_train(rng)
        allowed = allowed_causes(exp)
        protocol = td.sample_protocol(rng, exp)
        cid_a = td.sample_cause(rng, exp, allowed)
        cid_b = td.sample_cause(rng, exp, allowed)
        exp.simulate(rng, cid_a, protocol_plan(exp, protocol))  # A 的数据（不显示）
        ds_b = exp.simulate(rng, cid_b, protocol_plan(exp, protocol))
        case = _case(exp, exp.id, False, cid_a, protocol, ds_b, allowed)
        rng.random()
        return _diag_record(split, index, seed, case, allowed, rng, truth=cid_a,
                            extra={"shown_cause": cid_b})
    raise ValueError(f"未知划分 {split}")


def _design_one(index: int, seed: int, rng: np.random.Generator):
    """与 tasks_design.build_example 相同的构造，只是种子来自 bench 命名空间。"""
    exp, family, generic = td.sample_experiment(rng, tdz.DESIGN_GENERIC_FRACTION)
    protocol = "single" if rng.random() < tdz.P_SINGLE and "single" in exp.protocols else "standard"
    causes = allowed_causes(exp)
    rows_exp = tdz.expected_rows(exp, protocol, rng)
    b = tdz._budget(exp, generic, rng)
    arows = tdz._atlas(exp, protocol, causes, generic, rng)
    messages, disp, menu_disp = prompts.design_prompt(exp, protocol, rows_exp, tdz.budget_line(b), causes, rng)
    rec = tdz.recommend(exp, protocol, b)
    dom = tdz.dominant_source(b)
    unres = tdz.unresolved(arows, disp, menu_disp)
    names = {d: exp.cause(c).name_zh for c, d in disp.items()}
    labels = {d: exp.menu_item(m).label_zh for m, d in menu_disp.items()}
    target = {
        "expected": [{"setting": r["setting"], "value": r["value"], "sd": r["sd"], "unit": r["unit"]} for r in rows_exp],
        "dominant_source": dom, "recommend": rec, "unresolved": unres,
        "explanation": tdz.design_explanation(exp, protocol, rec, dom, unres, names, labels),
    }
    errs = prompts.validate_output("design", target)
    if errs:
        raise ValueError(f"design#{index} 标准答案不合 schema：{errs}")
    ex_id = f"bench-design-{family}-{index:05d}"
    meta = {"experiment_id": family, "split": "bench:design", "seed": seed, "schema": "design", "generic": generic,
            "protocol": protocol, "causes": [{"id": d, "name": names[d]} for d in disp.values()],
            "id_map": {d: c for c, d in disp.items()}, "menu_map": {d: m for m, d in menu_disp.items()},
            "prompt_version": prompts.PROMPT_VERSION}
    ex = Example(ex_id, "design", "engine",
                 messages + [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}], meta)
    ex.validate()
    gold = {"id": ex_id, "split": "design", "schema": "design", "family": family, "experiment": exp.id,
            "protocol": protocol, "target": target}
    return ex, gold, None


# ── 整体生成 ──────────────────────────────────────────────────────────


def split_size(split: str, small: bool, sizes: dict | None = None) -> int:
    if small:
        return len(held_out_ids()) if split == "held_out" else SMALL
    sizes = dict(SIZES, **(sizes or {}))
    if split == "held_out":
        return sizes.get("held_out_each", HELD_OUT_EACH) * len(held_out_ids())
    return int(sizes[split])


def _job(args):
    split, index, held = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_one(split, index, held)


def build(splits=SPLITS, out: str | Path = OUT, small: bool = False, sizes: dict | None = None,
          workers: int = 1, train_dir: str | Path | None = None, log=print) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    held = held_out_ids()
    summary: dict = {"splits": {}, "small": small, "prompt_version": prompts.PROMPT_VERSION,
                     "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    all_examples: list[Example] = []
    pool = None
    if workers > 1:
        import multiprocessing as mp

        pool = mp.get_context("spawn").Pool(min(int(workers), 6))
    try:
        for split in splits:
            n = split_size(split, small, sizes)
            t0 = time.perf_counter()
            jobs = [(split, i, held) for i in range(n)]
            rows = list(pool.imap(_job, jobs, chunksize=8) if pool else map(_job, jobs))
            exs = [r[0] for r in rows]
            golds = [r[1] for r in rows]
            facts = [r[2] for r in rows if r[2] is not None]
            write_jsonl(out / f"bench_{split}.jsonl", exs)
            _write(out / f"bench_{split}.gold.jsonl", golds)
            if facts:
                _write(out / f"bench_{split}.facts.jsonl", facts)
            dt = time.perf_counter() - t0
            info = {"n": len(exs), "seconds": round(dt, 1)}
            if split == "unmodelled":
                info["subsplits"] = {k: sum(g.get("subsplit") == k for g in golds) for k in ("detectable", "degenerate")}
            summary["splits"][split] = info
            all_examples.extend(exs)
            log(f"[bench] {split}: {len(exs)} 条，{dt:.1f} s")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    train_dir = Path(train_dir) if train_dir else RESULTS / "train"
    if train_dir.is_dir():
        train = [e for f in sorted(train_dir.glob("*.jsonl")) for e in read_jsonl(f)]
        rep = leakage_report(train, all_examples)
        summary["leakage"] = {k: rep[k] for k in ("n_train", "n_test", "clean")} | {
            "id_overlap": rep["id_overlap"][:20], "content_overlap": rep["content_overlap"][:20]}
        log(f"[bench] 泄漏检查：{'干净' if rep['clean'] else '有重叠！'}（训练 {rep['n_train']} 条）")
    else:
        summary["leakage"] = {"skipped": f"{train_dir} 不存在"}
    (out / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _write(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_gold(path: str | Path) -> dict[str, dict]:
    out = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                g = json.loads(line)
                out[g["id"]] = g
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="生成物理 bench（冻结前的草稿；冻结由学生做）")
    ap.add_argument("--splits", nargs="*", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--small", action="store_true", help=f"每个划分只做 {SMALL} 条（留出实验每个 1 条），测试用")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--train-dir", default=None, help="泄漏检查用的训练集目录（默认 results/train）")
    for s in SIZES:
        ap.add_argument(f"--n-{s.replace('_', '-')}", type=int, default=None)
    ap.add_argument("--n-held-out-each", type=int, default=None)
    a = ap.parse_args(argv)
    sizes = {s: getattr(a, f"n_{s}") for s in SIZES if getattr(a, f"n_{s}") is not None}
    if a.n_held_out_each is not None:
        sizes["held_out_each"] = a.n_held_out_each
    s = build(a.splits, a.out, a.small, sizes, a.workers, a.train_dir)
    print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
