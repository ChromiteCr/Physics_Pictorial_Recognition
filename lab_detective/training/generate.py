"""训练样本生成器（命令行）：诊断、设计、讲解、知识问答 → results/{train,val}/<任务>.jsonl。

用法：
    python -m lab_detective.training.generate --n-diagnose 2000 --n-design 400 --n-explain 800 \
        --n-knowledge 500 --workers 6 --out results

- 每个任务的 --n 是训练 + 验证的总数；验证取 5%（独立种子命名空间，见 splits.py）。
- 每条样本只由 (基础种子, 任务, 划分, 序号) 决定，和进程数、分块无关：--workers 1 和 6 写出的文件逐字节相同。
- 多进程（spawn，≤ 6 个），按块分发序号，主进程按序号顺序写文件。
- 讲解样本的讲解过不了校验器、知识题库用完时，这条样本丢弃并计数（不补）。
- 写完后按输入内容哈希检查：训练集内部重复、训练与验证重叠（重叠的训练样本删掉）。
- 打印并写出 <out>/generate_report.json：条数、丢弃数、每个任务的吞吐量（条/秒，墙钟）和单进程耗时。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

from .format import Example, content_hash, write_jsonl
from .splits import (BASE_SEED, HELD_OUT_EXPERIMENTS, TRAIN_EXPERIMENTS, VAL_FRACTION, WITHHELD_CAUSES,
                     split_counts)

TASKS = ("diagnose", "design", "explain", "knowledge", "guide", "card", "recognize")
MAX_WORKERS = 6
REPO = Path(__file__).resolve().parents[2]


def _builder(task: str):
    if task == "diagnose":
        from .tasks_diagnose import build_example
    elif task == "design":
        from .tasks_design import build_example
    elif task == "explain":
        from .tasks_explain import build_example
    elif task == "knowledge":
        from .tasks_knowledge import build_example
    elif task == "guide":
        from .tasks_guide import build_example
    elif task == "card":
        from .tasks_card import build_example
    else:
        raise ValueError(task)
    return build_example


def _init_worker() -> None:
    warnings.simplefilter("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def run_chunk(job: tuple) -> tuple[list[dict | None], float]:
    """工作进程：生成 [start, stop) 这些序号的样本 → (样本字典列表（丢弃的为 None）, 用时秒)。"""
    task, split, start, stop, base_seed, opts = job
    t0 = time.perf_counter()
    build = _builder(task)
    out: list[dict | None] = []
    for i in range(start, stop):
        ex = build(i, split, base_seed=base_seed, **opts)
        out.append(ex.to_dict() if ex is not None else None)
    return out, time.perf_counter() - t0


def _chunks(task: str, split: str, n: int, chunk: int, base_seed: int, opts: dict) -> list[tuple]:
    return [(task, split, a, min(a + chunk, n), base_seed, opts) for a in range(0, n, chunk)]


def generate_task(task: str, split: str, n: int, workers: int, chunk: int, base_seed: int,
                  opts: dict | None = None, pool=None) -> tuple[list[Example], dict]:
    """生成一个 (任务, 划分) 的 n 条样本（按序号顺序）。返回 (样本, 统计)。"""
    opts = dict(opts or {})
    jobs = _chunks(task, split, n, chunk, base_seed, opts)
    t0 = time.perf_counter()
    results = pool.imap(run_chunk, jobs) if pool is not None else map(run_chunk, jobs)
    examples: list[Example] = []
    dropped, cpu = 0, 0.0
    for rows, dt in results:
        cpu += dt
        for d in rows:
            if d is None:
                dropped += 1
            else:
                examples.append(Example.from_dict(d))
    wall = time.perf_counter() - t0
    stats = {"task": task, "split": split, "requested": n, "written": len(examples), "dropped": dropped,
             "wall_s": round(wall, 3), "cpu_s": round(cpu, 3),
             "per_sec_wall": round(len(examples) / wall, 2) if wall > 0 else None,
             "per_sec_one_worker": round(len(examples) / cpu, 2) if cpu > 0 else None}
    return examples, stats


def _dedupe(train: list[Example], val: list[Example]) -> tuple[list[Example], dict]:
    val_h = {content_hash(e) for e in val}
    seen: set[str] = set()
    keep, dup_internal, dup_val = [], 0, 0
    for e in train:
        h = content_hash(e)
        if h in val_h:
            dup_val += 1
            continue
        if h in seen:
            dup_internal += 1  # 训练集内部重复保留（同一道题的不同序号），只统计
        seen.add(h)
        keep.append(e)
    return keep, {"train_val_overlap_removed": dup_val, "train_internal_duplicates": dup_internal}


def generate(counts: dict[str, int], out: str | Path, workers: int = 1, chunk: int = 25,
             base_seed: int = BASE_SEED, opts: dict | None = None, quiet: bool = False) -> dict:
    """counts: {任务: 总条数}。写出 <out>/train/<任务>.jsonl 和 <out>/val/<任务>.jsonl，返回报告。"""
    out = Path(out)
    workers = max(1, min(int(workers), MAX_WORKERS))
    opts = opts or {}
    report = {"base_seed": base_seed, "workers": workers, "chunk": chunk, "val_fraction": VAL_FRACTION,
              "train_experiments": list(TRAIN_EXPERIMENTS), "held_out_experiments": list(HELD_OUT_EXPERIMENTS),
              "withheld_causes": dict(WITHHELD_CAUSES), "tasks": {}}
    from .prompts import PROMPT_VERSION

    report["prompt_version"] = PROMPT_VERSION
    pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker) if workers > 1 else None
    try:
        for task in TASKS:
            n = int(counts.get(task, 0) or 0)
            if n <= 0:
                continue
            if task == "recognize":  # 照片样本：主进程一次生成全部合格照片（n 截断训练条数），不走序号分块
                from .tasks_recognize import generate_recognize

                report["tasks"][task] = generate_recognize(out, n, quiet=quiet)
                continue
            n_train, n_val = split_counts(n)
            if task == "knowledge":
                from .tasks_knowledge import pool_size

                n_train = min(n_train, pool_size("train", base_seed))
                n_val = min(n_val, pool_size("val", base_seed))
            t_opts = dict(opts.get(task, {}))
            val, s_val = generate_task(task, "val", n_val, workers, chunk, base_seed, t_opts, pool)
            train, s_train = generate_task(task, "train", n_train, workers, chunk, base_seed, t_opts, pool)
            train, dd = _dedupe(train, val)
            n_tr = write_jsonl(out / "train" / f"{task}.jsonl", train)
            n_va = write_jsonl(out / "val" / f"{task}.jsonl", val)
            wall = s_train["wall_s"] + s_val["wall_s"]
            cpu = s_train["cpu_s"] + s_val["cpu_s"]
            made = s_train["written"] + s_val["written"]
            rep = {"train": n_tr, "val": n_va, "dropped": s_train["dropped"] + s_val["dropped"], **dd,
                   "wall_s": round(wall, 2), "examples_per_sec": round(made / wall, 2) if wall > 0 else None,
                   "examples_per_sec_one_worker": round(made / cpu, 2) if cpu > 0 else None,
                   "options": t_opts}
            report["tasks"][task] = rep
            if not quiet:
                print(f"{task:9s} train {n_tr:6d}  val {n_va:5d}  丢弃 {rep['dropped']:4d}  "
                      f"{rep['examples_per_sec']} 条/秒（单进程 {rep['examples_per_sec_one_worker']} 条/秒）",
                      flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    out.mkdir(parents=True, exist_ok=True)
    (out / "generate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="生成训练 / 验证样本（引擎标注，不调用任何模型）")
    ap.add_argument("--n-diagnose", type=int, default=0)
    ap.add_argument("--n-design", type=int, default=0)
    ap.add_argument("--n-explain", type=int, default=0)
    ap.add_argument("--n-knowledge", type=int, default=0)
    ap.add_argument("--n-guide", type=int, default=0)
    ap.add_argument("--n-card", type=int, default=0)
    ap.add_argument("--n-recognize", type=int, default=0, help="识别样本训练条数上限（合格照片 × 3 个变体，大于总数时取全部）")
    ap.add_argument("--workers", type=int, default=1, help=f"进程数（最多 {MAX_WORKERS}）")
    ap.add_argument("--chunk", type=int, default=25, help="每块样本数")
    ap.add_argument("--seed", type=int, default=BASE_SEED, help="基础种子（bench 必须用别的命名空间）")
    ap.add_argument("--out", default=str(REPO / "results"))
    ap.add_argument("--generic-frac", type=float, default=None, help="诊断 / 讲解里程序化任务族的比例（默认 0.5）")
    ap.add_argument("--design-generic-frac", type=float, default=None, help="设计任务里程序化任务族的比例（默认 0.7）")
    ap.add_argument("--eig-samples", type=int, default=None, help="EIG 蒙特卡洛样本数（默认 64）")
    a = ap.parse_args(argv)
    if a.workers > MAX_WORKERS:
        print(f"--workers 最多 {MAX_WORKERS}，按 {MAX_WORKERS} 运行", file=sys.stderr)
    opts: dict[str, dict] = {"diagnose": {}, "design": {}, "explain": {}, "knowledge": {}}
    if a.generic_frac is not None:
        opts["diagnose"]["generic_fraction"] = a.generic_frac
        opts["explain"]["generic_fraction"] = a.generic_frac
    if a.design_generic_frac is not None:
        opts["design"]["generic_fraction"] = a.design_generic_frac
    if a.eig_samples is not None:
        opts["diagnose"]["n_samples"] = a.eig_samples
        opts["explain"]["n_samples"] = a.eig_samples
    counts = {"diagnose": a.n_diagnose, "design": a.n_design, "explain": a.n_explain, "knowledge": a.n_knowledge,
              "guide": a.n_guide, "card": a.n_card, "recognize": a.n_recognize}
    rep = generate(counts, a.out, a.workers, a.chunk, a.seed, opts)
    print(json.dumps(rep["tasks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
