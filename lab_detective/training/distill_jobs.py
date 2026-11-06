"""蒸馏任务：给教师（DeepSeek / Hy4）出题、按物理引擎过滤、导出 / 导入，全部放在 results/distill/ 下。

教师盲做：题目就是学生训练时看到的提示（prompts.py / tutor.context），教师看不到引擎答案。只保留：
- 诊断（diagnose，只用"给清单"变体）：JSON 合法（prompts.validate_output）；教师排第一的原因在引擎的等价类里
  （后验与引擎第一名之比 > 0.5 的原因都算并列）；下一次测量是 EIG 前两名之一（引擎认为没有测量能分开时，
  写 null 或任一项都行）；explanation 通过声明校验器（对照取整到 0.05 的事实单，与引擎目标同一做法）。
- 引导（guide）：最多一个问题（要学生回答的阶段恰好一个）；学生选嫌疑原因之前不说原因名称；
  诊断揭示 / 讲解通过校验器；其他阶段教师写的数字（≥ 10 的）必须在提示里出现过。
所有教师、所有任务合计最多 distill.MAX_DISTILLED = 5,248 条（读 results/distill/**/raw_results.jsonl 里已保留的条数）。

题号 = "<任务>-<序号>"，序号决定一切（split 一律 train、基础种子同训练集）：DeepSeek 从 1,000,000 起、Hy4 从 2,000,000 起，
与引擎样本（从 0 起）和彼此都不重叠；导入时只凭题号就能重建题目和检查。

    python -m lab_detective.training.distill_jobs deepseek --task diagnose --n 2000 [--workers 4]
    python -m lab_detective.training.distill_jobs hy4-export --task diagnose --n 300 [--task guide --n 300] [--out results/distill/hy4_pack]
    python -m lab_detective.training.distill_jobs hy4-import [--pack results/distill/hy4_pack]
    python -m lab_detective.training.distill_jobs to-train      # 保留的输出 → results/train/distill_<教师>.jsonl
    python -m lab_detective.training.distill_jobs status
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

from . import distill, prompts
from .distill import MAX_DISTILLED, DistillJob
from .format import Example, write_jsonl
from .splits import BASE_SEED, example_rng

REPO = Path(__file__).resolve().parents[2]
DISTILL_DIR = REPO / "results" / "distill"
PACK_DIR = DISTILL_DIR / "hy4_pack"
START = {"deepseek": 1_000_000, "hy4": 2_000_000}
TASKS = ("diagnose", "guide")
TIE_RATIO = 0.5
EIG_SAMPLES = 64
ASK_CATEGORIES = ("identify", "plan", "predict", "diagnose_ask", "predict_next")
# 引导蒸馏只出这些阶段的题（预测 / 测量阶段的上下文里没有下一行的设定值，教师只能编）
GUIDE_DISTILL_CATEGORIES = ("plan", "check", "diagnose_ask", "diagnose_reveal", "predict_next", "explain", "reflect")
MAX_GUIDE_CHARS = 360


# ── 诊断：复现 build_example 的随机数顺序，只拼提示（不算事实单）─────────────


def _diag_case(index: int, base_seed: int = BASE_SEED):
    """→ (case, variant, cand, eig_seed, rng)；与 tasks_diagnose.build_example 的随机数顺序完全一致。"""
    from .tasks_diagnose import GENERIC_FRACTION, P_OMIT_TRUE, P_PROPOSE, simulate_case

    rng = example_rng("diagnose", "train", index, base=base_seed)
    case = simulate_case(rng, GENERIC_FRACTION)
    exp = case.exp
    u = rng.random()
    if u < P_PROPOSE:
        variant = "propose"
    elif u < P_PROPOSE + P_OMIT_TRUE and case.true_cause != exp.random_only_id and len(case.allowed) >= 4:
        variant = "omit_true"
    else:
        variant = "list"
    cand = [c for c in case.allowed if not (variant == "omit_true" and c == case.true_cause)]
    eig_seed = int(rng.integers(0, 2**62))
    return case, variant, cand, eig_seed, rng


def diagnose_prompt(index: int, base_seed: int = BASE_SEED) -> dict | None:
    """"给清单"变体的提示和编号表；其他变体返回 None。"""
    case, variant, cand, eig_seed, rng = _diag_case(index, base_seed)
    if variant != "list":
        return None
    messages, disp, menu_disp = prompts.diagnose_prompt(case.exp, case.ds, cand, rng, propose=False)
    return {"messages": messages, "id_map": {d: c for c, d in disp.items()},
            "menu_map": {d: m for m, d in menu_disp.items()}, "experiment_id": case.family,
            "truth_cause": case.true_cause,
            "causes": [{"id": d, "name": case.exp.cause(c).name_zh} for c, d in disp.items()]}


def diagnose_reference(index: int, base_seed: int = BASE_SEED, n_samples: int = EIG_SAMPLES) -> dict:
    """引擎参照：后验、等价类、EIG 前两名、校验用事实单（概率取整到 0.05，与引擎目标的讲解同一做法）。"""
    from ..eig import recommend
    from ..engine import posterior
    from .tasks_diagnose import fact_sheet, round_prob

    case, variant, cand, eig_seed, _ = _diag_case(index, base_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = fact_sheet(case, cand, np.random.default_rng(eig_seed), n_samples)
        post = posterior(case.exp, case.ds, cand)
        scores = list(recommend(case.exp, case.ds, n_samples, np.random.default_rng(eig_seed), cause_ids=cand) or [])
    probs = {c: float(post.probs[j]) for j, c in enumerate(cand)}
    top = max(cand, key=lambda c: (probs[c], -cand.index(c)))
    tied = sorted(c for c in cand if probs[top] > 0 and probs[c] / probs[top] > TIE_RATIO)
    nm = fs.get("next_measurement")
    no_sep = not nm or not nm.get("separates")
    top2 = [s.item_id for s in scores[:2]]
    if nm and nm.get("id") and nm["id"] not in top2:
        top2.append(nm["id"])
    ranked_ids = sorted(cand, key=lambda c: (-probs[c], cand.index(c)))
    f2 = copy.deepcopy(fs)
    rp = {c: round_prob(p) for c, p in probs.items()}
    for r in (f2.get("diagnosis") or {}).get("ranked") or []:
        if r.get("cause_id") in rp:
            r["prob"] = rp[r["cause_id"]]
    if no_sep:  # 引擎目标此时 next_measurement 写 null、讲解说"分不开 / 不需要"，校验也按 null
        f2["next_measurement"] = None
    if f2.get("next_measurement"):
        f2["next_measurement"].pop("eig_bits", None)
        keep = set(ranked_ids[:2])
        f2["next_measurement"]["predicted"] = [p for p in f2["next_measurement"].get("predicted") or []
                                               if p.get("cause_id") in keep]
    return {"top": top, "tied": tied, "probs": probs, "top2_items": top2, "no_separation": no_sep,
            "fs": fs, "fs_rounded": f2, "variant": variant}


def check_diagnose(text: str, ref: dict, id_map: dict, menu_map: dict) -> tuple[bool, dict]:
    from ..tutor.verifier import verify
    from ..vision.schema import RecognitionParseError, extract_json

    try:
        obj = extract_json(text)
    except RecognitionParseError as e:
        return False, {"reason": "json", "detail": str(e)}
    errs = prompts.validate_output("diagnose", obj)
    if errs:
        return False, {"reason": "schema", "detail": errs[:3]}
    ranked = obj["ranked"]
    best = max(range(len(ranked)), key=lambda j: (float(ranked[j]["prob"]), -j))
    top = id_map.get(ranked[best]["cause_id"])
    if top not in ref["tied"]:
        return False, {"reason": "top_cause", "teacher": top, "engine_tied": ref["tied"]}
    nm = obj.get("next_measurement")
    if nm is None:
        if not ref["no_separation"]:
            return False, {"reason": "next_null", "engine_top2": ref["top2_items"]}
    else:
        item = menu_map.get(nm.get("id"))
        if item is None or (not ref["no_separation"] and item not in ref["top2_items"]):
            return False, {"reason": "next_item", "teacher": item, "engine_top2": ref["top2_items"]}
    ex = obj["explanation"]
    if not (verify(ex, ref["fs_rounded"])["ok"] or verify(ex, ref["fs"])["ok"]):
        v = verify(ex, ref["fs_rounded"])
        return False, {"reason": "verifier", "detail": [str(p) for p in v.get("problems", [])[:3]]}
    return True, {"reason": "ok", "teacher_top": top}


def diagnose_job(index: int, base_seed: int = BASE_SEED) -> DistillJob | None:
    p = diagnose_prompt(index, base_seed)
    if p is None:
        return None
    cache: dict = {}

    def check(text: str) -> tuple[bool, dict]:
        if "ref" not in cache:  # 参照懒算：导出提示词包时不需要
            cache["ref"] = diagnose_reference(index, base_seed)
        return check_diagnose(text, cache["ref"], p["id_map"], p["menu_map"])

    meta = {"experiment_id": p["experiment_id"], "index": index, "base_seed": base_seed, "variant": "list",
            "id_map": p["id_map"], "menu_map": p["menu_map"], "truth_cause": p["truth_cause"],
            "causes": p["causes"], "prompt_version": prompts.PROMPT_VERSION}
    return DistillJob(f"diagnose-{index}", "diagnose", p["messages"], check, meta)


# ── 引导 ──────────────────────────────────────────────────────────────

def check_guide(text: str, category: str, fs: dict | None, cause_names, revealed: bool, prompt_text: str) -> tuple[bool, dict]:
    from .tasks_guide import guide_problems, n_questions

    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    probs = guide_problems(t, category, fs, cause_names, revealed, prompt_text)
    if category in ASK_CATEGORIES and n_questions(t) != 1:
        probs.append("要学生回答的阶段必须恰好问一个问题")
    if len(t) > MAX_GUIDE_CHARS:
        probs.append(f"太长（{len(t)} 字）")
    return (not probs), {"reason": "ok" if not probs else "guide", "detail": probs[:3]}


def guide_job(index: int, base_seed: int = BASE_SEED) -> DistillJob | None:
    from .tasks_guide import guide_seed, pick_turn

    got = pick_turn(guide_seed("train", index, base_seed))
    if got is None:
        return None
    sim, turn, _ = got
    if turn.category not in GUIDE_DISTILL_CATEGORIES:
        return None
    prompt_text = "\n".join(m["content"] for m in turn.messages if isinstance(m["content"], str))

    def check(text: str) -> tuple[bool, dict]:
        return check_guide(text, turn.category, turn.fs, sim.names, turn.revealed, prompt_text)

    meta = {"experiment_id": sim.eid, "index": index, "base_seed": base_seed, "category": turn.category,
            "causes": [{"id": c, "name": n} for c, n in zip(sim.allowed, sim.names)]}
    return DistillJob(f"guide-{index}", "guide", turn.messages, check, meta)


BUILDERS = {"diagnose": diagnose_job, "guide": guide_job}


def build_jobs(task: str, n: int, start: int, base_seed: int = BASE_SEED, max_scan: int | None = None) -> list[DistillJob]:
    """从序号 start 起依次出题，跳过不合格的（如非清单变体），直到 n 道。"""
    out, i = [], start
    stop = start + (max_scan or 4 * n + 50)
    while len(out) < n and i < stop:
        j = BUILDERS[task](i, base_seed)
        if j is not None:
            out.append(j)
        i += 1
    return out


def job_from_id(job_id: str, base_seed: int = BASE_SEED) -> DistillJob | None:
    task, _, idx = job_id.rpartition("-")
    if task not in BUILDERS or not idx.isdigit():
        return None
    return BUILDERS[task](int(idx), base_seed)


# ── 封顶计数 ──────────────────────────────────────────────────────────


def kept_counts(root: str | Path = DISTILL_DIR, exclude: str | Path | None = None) -> dict[str, int]:
    """results/distill/**/raw_results.jsonl 里已保留的条数（按文件）。exclude：不计的那个文件。"""
    root = Path(root)
    out = {}
    ex = Path(exclude).resolve() if exclude else None
    for f in sorted(root.rglob("raw_results.jsonl")) if root.exists() else []:
        if ex is not None and f.resolve() == ex:
            continue
        n = 0
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("kept"):
                n += 1
        out[str(f.relative_to(root))] = n
    return out


def total_kept(root: str | Path = DISTILL_DIR, exclude: str | Path | None = None) -> int:
    return sum(kept_counts(root, exclude).values())


# ── 子命令 ────────────────────────────────────────────────────────────

NO_KEY_MSG = ("没有找到 DeepSeek 的 API key，蒸馏没有开始。\n"
              f"请在 {distill.KEY_FILE} 里写一行 DEEPSEEK_API_KEY=你的key（这个文件不进仓库），然后重新运行。")


def cmd_deepseek(a, teacher=None, root: Path = DISTILL_DIR) -> int:
    if teacher is None:
        if not distill.load_key():
            print(NO_KEY_MSG, file=sys.stderr)
            return 2
        out_dir = root / f"deepseek_{a.task}"
        teacher = distill.DeepSeekTeacher(model=a.model, thinking=not a.no_think,
                                          log_path=out_dir / "api_log.jsonl")
    out_dir = root / f"deepseek_{a.task}"
    other = total_kept(root, exclude=out_dir / "raw_results.jsonl")
    if other >= MAX_DISTILLED:
        print(f"已保留 {other} 条，达到上限 {MAX_DISTILLED}，不再蒸馏。")
        return 0
    jobs = build_jobs(a.task, a.n, START["deepseek"] + a.offset, a.seed)
    stats = distill.run(jobs, teacher, getattr(teacher, "name", "deepseek"), out_dir, max_kept=MAX_DISTILLED,
                        workers=a.workers, already_kept=other)
    print(json.dumps({"task": a.task, "jobs": len(jobs), **stats}, ensure_ascii=False))
    return 0


def export_pack(spec: dict[str, int], folder: Path = PACK_DIR, base_seed: int = BASE_SEED) -> dict:
    """spec: {任务: 题数} → folder/prompts.jsonl + README.md + pack.json。"""
    jobs = []
    info = {"base_seed": base_seed, "tasks": {}, "prompt_version": prompts.PROMPT_VERSION}
    for task, n in spec.items():
        js = build_jobs(task, n, START["hy4"], base_seed)
        jobs += js
        info["tasks"][task] = {"n": len(js), "first_id": js[0].id if js else None, "last_id": js[-1].id if js else None}
    n = distill.export_prompt_pack(jobs, folder)
    info["n_prompts"] = n
    (folder / "pack.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def import_pack(folder: Path = PACK_DIR, root: Path = DISTILL_DIR, base_seed: int | None = None) -> dict:
    """读 folder/responses.jsonl，按题号重建题目和检查，过滤后写 root/hy4/raw_results.jsonl（遵守总上限）。"""
    folder = Path(folder)
    resp = folder / "responses.jsonl"
    if not resp.exists():
        raise FileNotFoundError(f"没有找到 {resp}：把 Hy4 的回答按 README 写成 responses.jsonl 放进这个文件夹")
    if base_seed is None:
        pj = folder / "pack.json"
        base_seed = json.loads(pj.read_text(encoding="utf-8"))["base_seed"] if pj.exists() else BASE_SEED
    out_dir = root / "hy4"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw_results.jsonl"
    done = set()
    if raw.exists():
        done = {json.loads(line)["job_id"] for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()}
    kept_total = total_kept(root)
    stats = {"read": 0, "kept": 0, "rejected": 0, "unknown_id": 0, "duplicate": 0, "over_cap": 0}
    with raw.open("a", encoding="utf-8") as f:
        for line in resp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            stats["read"] += 1
            if r.get("id") in done:
                stats["duplicate"] += 1
                continue
            job = job_from_id(str(r.get("id", "")), base_seed)
            if job is None:
                stats["unknown_id"] += 1
                continue
            ok, info = job.check(str(r.get("text", "")))
            if ok and kept_total >= MAX_DISTILLED:
                ok, info = False, {"reason": "over_cap"}
                stats["over_cap"] += 1
            stats["kept" if ok else "rejected"] += 1
            kept_total += int(ok)
            done.add(job.id)
            f.write(json.dumps({"job_id": job.id, "task": job.task, "teacher": "hy4", "text": r.get("text", ""),
                                "kept": ok, "info": info, "usage": {}, "meta": job.meta}, ensure_ascii=False) + "\n")
    stats["kept_total_all_teachers"] = kept_total
    return stats


def to_train(root: Path = DISTILL_DIR, out: Path = REPO / "results" / "train", base_seed: int = BASE_SEED) -> dict:
    """保留的教师输出 → 训练样本（提示原样，答案 = 教师最终输出），按教师写 distill_<教师>.jsonl，总数不超过上限。"""
    rep, left = {}, MAX_DISTILLED
    for raw in sorted(Path(root).rglob("raw_results.jsonl")):
        teacher = "hy4" if raw.parent.name.startswith("hy4") else "deepseek"
        exs = []
        for line in raw.read_text(encoding="utf-8").splitlines():
            if not line.strip() or left <= 0:
                continue
            r = json.loads(line)
            if not r.get("kept"):
                continue
            job = job_from_id(r["job_id"], (r.get("meta") or {}).get("base_seed", base_seed))
            if job is None:
                continue
            text = re.sub(r"<think>.*?</think>", "", r["text"], flags=re.S).strip()
            meta = dict(job.meta, teacher=r.get("teacher"), split="train")
            exs.append(Example(f"distill-{teacher}-{job.id}", job.task, f"distill-{teacher}",
                               list(job.messages) + [{"role": "assistant", "content": text}], meta))
            left -= 1
        if exs:
            path = Path(out) / f"distill_{raw.parent.name}.jsonl"
            rep[str(path)] = write_jsonl(path, exs)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="蒸馏任务：出题、DeepSeek 运行、Hy4 导出 / 导入（总上限 5,248）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("deepseek")
    d.add_argument("--task", choices=TASKS, required=True)
    d.add_argument("--n", type=int, required=True)
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--offset", type=int, default=0, help="题号偏移（续跑别的题时用）")
    d.add_argument("--model", default="deepseek-flash")
    d.add_argument("--no-think", action="store_true")
    d.add_argument("--seed", type=int, default=BASE_SEED)
    e = sub.add_parser("hy4-export")
    e.add_argument("--task", action="append", choices=TASKS, required=True)
    e.add_argument("--n", action="append", type=int, required=True)
    e.add_argument("--out", default=str(PACK_DIR))
    e.add_argument("--seed", type=int, default=BASE_SEED)
    i = sub.add_parser("hy4-import")
    i.add_argument("--pack", default=str(PACK_DIR))
    sub.add_parser("to-train")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "deepseek":
        return cmd_deepseek(a)
    if a.cmd == "hy4-export":
        if len(a.task) != len(a.n):
            ap.error("--task 和 --n 要成对给出")
        info = export_pack(dict(zip(a.task, a.n)), Path(a.out), a.seed)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    if a.cmd == "hy4-import":
        print(json.dumps(import_pack(Path(a.pack)), ensure_ascii=False, indent=2))
        return 0
    if a.cmd == "to-train":
        print(json.dumps(to_train(), ensure_ascii=False, indent=2))
        return 0
    print(json.dumps({"kept": kept_counts(), "total": total_kept(), "cap": MAX_DISTILLED}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
