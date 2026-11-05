"""真实故障注入实验的预先规定分析（只能由学生本人启动）。

顺序（任何一步不通过就停止）：
1. 数据已冻结且没被改过（FIELD_MANIFEST.json），冻结时用的就是这张公开表；
2. 拆封：密封表的哈希等于公开表里的承诺值，公开表也没被改过；用种子重算的随机分配与两张表一致；
3. 每次运行、每个引擎：只用基础方案、基础方案 + 每一项追加测量（各自单独接在基础方案后面，不累积）、
   基础方案 + 全部追加测量，分别算后验：真实原因的后验概率 p_true、最可能原因是否就是真实原因、熵、失配 p 值；
   并用冻结的基础数据复算信息增益推荐，核对与当时的回执一致；
4. 汇总（规则取自公开表里的 analysis_spec，建表时已受哈希保护）：
   - 各阶段 top-1 正确率及 Wilson 区间、p_true 均值；
   - 主线 B：配对 Wilcoxon 符号秩检验，p_true(基础 + 信息增益推荐) − p_true(基础 + 最佳固定测量)，
     合并课内实验（单摆单独报告），两项测量相同时差为 0，按预先规定处理并报告个数；次要对比同法描述；
   - 涉及图谱最难分开的那对条件的运行单独报告；
   - 图谱预测效度：只用标准方案时，在图谱给出的每一对原因里两两判别的真实正确率与图谱区间对照；
   - 不设故障时的误报、"不在清单"的失配次数、没做的运行和跳过的槽位如实列出。
只输出数字，不写解读；解读由学生本人写。

命令（必须在终端里由学生本人运行，会要求输入确认语）：
    python -m lab_detective.fieldkit.analysis --public <公开表> --sealed <密封表> --data <数据文件夹> --out <输出文件夹>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from ..bench.stats import paired_bootstrap, wilson
from ..engine import posterior
from . import ingest
from .catalog import FIELDKIT_VERSION, SLOT_LABEL
from .live import eig_seed, recommend_item
from .schedule import regenerate_check, verify_commitment
from .sheet import read_sheet

CONFIRM_PHRASE = "我本人启动分析"
LN2 = math.log(2.0)
STAGES = ("base", "eig", "best_fixed", "repeat", "twin_eig", "all")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def metrics(engine, ds, truth: str, misfit_alpha: float) -> dict:
    post = posterior(engine, ds)
    top = post.top()
    return {"p_true": post.prob(truth), "top": top, "top_prob": float(post.probs.max()),
            "correct": bool(top == truth), "entropy_bits": post.entropy() / LN2,
            "misfit_p": float(post.misfit_p), "misfit": bool(post.misfit_p < misfit_alpha)}


def pair_correct(engine, ds, truth: str, other: str) -> bool:
    """只在这一对原因里、等先验比较证据（与图谱的两两判别同一规则；完全相等时算一半，这里记为不对）。"""
    post = posterior(engine, ds, cause_ids=(truth, other))
    return bool(post.log_evidence[0] > post.log_evidence[1])


def _wilcoxon(d: list[float], zero_method: str, alternative: str) -> dict:
    d = [float(x) for x in d]
    n_zero = sum(1 for x in d if x == 0.0)
    out = {"n_pairs": len(d), "n_zero": n_zero, "mean_diff": float(np.mean(d)) if d else None,
           "median_diff": float(np.median(d)) if d else None, "statistic": None, "p_value": None, "note": ""}
    nz = [x for x in d if x != 0.0]
    if len(nz) == 0:
        out["note"] = "所有差都为 0（或没有配对），检验无法进行"
        return out
    try:
        r = stats.wilcoxon(d, zero_method=zero_method, alternative=alternative)
        out["statistic"], out["p_value"] = float(r.statistic), float(r.pvalue)
    except ValueError as e:
        out["note"] = f"检验无法进行：{e}"
    if len(d) >= 2:
        out["bootstrap_mean_diff"] = paired_bootstrap(d, [0.0] * len(d), n_boot=5000, seed=0)
    return out


def _acc(rows: list[bool], conf: float) -> dict:
    k, n = int(sum(rows)), len(rows)
    lo, hi = wilson(k, n, conf) if n else (None, None)
    return {"k": k, "n": n, "acc": k / n if n else None, "lo": lo, "hi": hi}


def _verify_all(public: dict, sealed: dict, data_dir: Path) -> list[str]:
    errs = []
    v = ingest.verify(data_dir)
    if not v["ok"]:
        errs.append(f"数据没有冻结或冻结后被改过：{v}")
    elif v.get("commitment") != public.get("commitment"):
        errs.append("冻结数据时用的公开表与现在给的公开表不是同一张")
    vc = verify_commitment(public, sealed)
    errs += vc["errors"]
    if sealed.get("fieldkit_version") == FIELDKIT_VERSION:
        errs += regenerate_check(public, sealed)["errors"]
    return errs


def analyse_run(sheet, sheet_path: Path, public: dict, key: dict, spec: dict) -> dict:
    datasets = ingest.run_datasets(sheet, public)
    run = next(r for r in public["runs"] if r["code"] == key["code"])
    out = {"run_id": key["run_id"], "code": key["code"], "experiment": key["experiment"], "session": run["session"],
           "condition": key["condition"], "setter": key["setter"], "blind": key["blind"], "slots": run["slots"],
           "engines": {}}
    info = public["experiments"][key["experiment"]]
    rp = Path(sheet_path).with_suffix(".eig.json")
    receipt = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else None
    for name, res in datasets.items():
        if name not in key["truth"]:
            continue
        eng = res["engine"]
        truth = key["truth"][name]["cause_id"]
        m = {"truth": truth, "equivalent": key["truth"][name]["equivalent"],
             "base": metrics(eng, res["base"], truth, spec["misfit_alpha"]),
             "single": {s: metrics(eng, ds, truth, spec["misfit_alpha"]) for s, ds in res["single"].items()},
             "all": metrics(eng, res["all"], truth, spec["misfit_alpha"]), "skipped": res["skipped"]}
        checks = {}
        for slot in ("eig", "twin_eig"):
            if slot not in run["slots"] or (slot == "eig") != (name == info["primary"]):
                continue
            allowed = tuple(info["allowed_items"] if slot == "eig" else info.get("twin_allowed_items", ()))
            rec = recommend_item(eng, res["base"], allowed, int(spec["eig_n_samples"]), eig_seed(key["code"], slot))
            got = (receipt or {}).get("slots", {}).get(slot, {}).get("chosen")
            checks[slot] = {"recomputed": rec["chosen"], "receipt": got, "reproduced": rec["chosen"] == got}
        m["eig_check"] = checks
        out["engines"][name] = m
    return out


def summarise(runs: list[dict], public: dict, spec: dict, missing: list[str]) -> dict:
    conf = float(spec["ci_level"])
    exps = public["design"]["experiments"]
    summary: dict = {"accuracy": {}, "p_true_mean": {}, "contrasts": {}, "confounded_pair": {},
                     "atlas_validity": {}, "clean_false_alarm": {}, "misfit": {}, "eig_reproduced": {},
                     "missing_runs": missing, "skipped": []}
    for r in runs:
        for name, m in r["engines"].items():
            for s in m["skipped"]:
                summary["skipped"].append({"run_id": r["run_id"], "engine": name, "slot": s})

    def stage_val(m, stage, field):
        if stage == "base":
            return m["base"][field]
        if stage == "all":
            return m["all"][field]
        return m["single"][stage][field] if stage in m["single"] else None

    for e in exps:
        info = public["experiments"][e]
        for name in info["engine_ids"]:
            rs = [r for r in runs if r["experiment"] == e and name in r["engines"]]
            if not rs:
                continue
            k = f"{e}/{name}"
            summary["accuracy"][k], summary["p_true_mean"][k] = {}, {}
            for st in STAGES:
                vals = [stage_val(r["engines"][name], st, "correct") for r in rs]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                summary["accuracy"][k][st] = _acc(vals, conf)
                pts = [stage_val(r["engines"][name], st, "p_true") for r in rs]
                summary["p_true_mean"][k][st] = float(np.mean([p for p in pts if p is not None]))
            clean = [r for r in rs if r["condition"] == "clean"]
            summary["clean_false_alarm"][k] = _acc([not r["engines"][name]["base"]["correct"] for r in clean], conf)
            summary["misfit"][k] = {st: int(sum(1 for r in rs if (stage_val(r["engines"][name], st, "misfit") or False)))
                                    for st in ("base", "all")}
            summary["eig_reproduced"][k] = [
                {"run_id": r["run_id"], "slot": s, **c} for r in rs for s, c in r["engines"][name]["eig_check"].items()]

    def diffs(rs, name, a, b):
        out = []
        for r in rs:
            m = r["engines"].get(name)
            if m and a in m["single"] and b in m["single"]:
                out.append(m["single"][a]["p_true"] - m["single"][b]["p_true"])
        return out

    contrasts = [tuple(spec["primary_contrast"])] + [tuple(c) for c in spec["secondary_contrasts"]]
    zm, alt = spec["wilcoxon_zero_method"], spec["wilcoxon_alternative"]
    pool = [e for e in exps if public["experiments"][e]["mainline_b"]] if spec["pool"] == "mainline_b" else list(exps)
    for i, (a, b) in enumerate(contrasts):
        label = f"{a} − {b}"
        block = {"role": "primary" if i == 0 else "secondary"}
        prs = [r for r in runs if r["experiment"] in pool]
        block["pooled"] = dict(_wilcoxon(
            [d for e in pool for d in diffs([r for r in prs if r["experiment"] == e],
                                            public["experiments"][e]["primary"], a, b)], zm, alt),
            experiments=pool)
        block["by_experiment"] = {}
        for e in exps:
            name = public["experiments"][e]["primary"]
            block["by_experiment"][e] = _wilcoxon(diffs([r for r in runs if r["experiment"] == e], name, a, b), zm, alt)
        summary["contrasts"][label] = block

    for e in exps:
        info = public["experiments"][e]
        pairs = info.get("atlas_pairs") or []
        name = info["primary"]
        rs = [r for r in runs if r["experiment"] == e and name in r["engines"]]
        if pairs:
            p0 = pairs[0]
            sub = [r for r in rs if r["condition"] in (p0["condition_a"], p0["condition_b"])]
            a, b = spec["primary_contrast"]
            summary["confounded_pair"][e] = {"pair": [p0["condition_a"], p0["condition_b"]], "status": p0["status"],
                                             "contrast": _wilcoxon(diffs(sub, name, a, b), zm, alt)}
    return summary


def atlas_validity_real(runs_data: list[tuple], public: dict, conf: float) -> dict:
    """runs_data: [(运行结果, 主引擎, 基础数据集)]；对每一对图谱条件算真实的两两判别正确率。"""
    out = {}
    for e in public["design"]["experiments"]:
        info = public["experiments"][e]
        rows = []
        for p in info.get("atlas_pairs") or []:
            if p["status"] == "same_class":
                continue
            flags = []
            for r, eng, base in runs_data:
                if r["experiment"] != e or r["condition"] not in (p["condition_a"], p["condition_b"]):
                    continue
                truth, other = ((p["cause_a"], p["cause_b"]) if r["condition"] == p["condition_a"]
                                else (p["cause_b"], p["cause_a"]))
                flags.append(pair_correct(eng, base, truth, other))
            real = _acc(flags, conf)
            inside = None if real["acc"] is None else bool(p["lo"] <= real["acc"] <= p["hi"])
            covers = None if real["acc"] is None else bool(real["lo"] <= p["acc"] <= real["hi"])
            rows.append({"condition_a": p["condition_a"], "condition_b": p["condition_b"], "status": p["status"],
                         "atlas": {"acc": p["acc"], "lo": p["lo"], "hi": p["hi"]}, "real": real,
                         "real_point_in_atlas_interval": inside, "atlas_point_in_real_interval": covers})
        out[e] = rows
    return out


def run_analysis(public: dict, sealed: dict, data_dir: str | Path, out_dir: str | Path, confirm: str) -> dict:
    if confirm != CONFIRM_PHRASE:
        raise PermissionError("正式分析只能由学生本人启动（确认语不对）")
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    errs = _verify_all(public, sealed, data_dir)
    if errs:
        raise ValueError("拆封前检查没通过：\n" + "\n".join(errs))
    spec = public["analysis_spec"]
    key_by_code = {k["code"]: k for k in sealed["key"]}
    sheets = {}
    for p in ingest.find_sheets(data_dir):
        sh = read_sheet(p)
        sheets[sh.meta["code"]] = (sh, p)
    runs, runs_data = [], []
    for run in public["runs"]:
        if run["code"] not in sheets:
            continue
        sh, path = sheets[run["code"]]
        key = key_by_code[run["code"]]
        r = analyse_run(sh, path, public, key, spec)
        runs.append(r)
        info = public["experiments"][run["experiment"]]
        ds = ingest.run_datasets(sh, public).get(info["primary"])
        if ds is not None:
            runs_data.append((r, ds["engine"], ds["base"]))
    missing = [r["run_id"] for r in public["runs"] if r["code"] not in sheets]
    summary = summarise(runs, public, spec, missing)
    summary["atlas_validity"] = atlas_validity_real(runs_data, public, float(spec["ci_level"]))
    result = _clean({"analysis_spec": spec, "commitment": public["commitment"], "runs": runs, "summary": summary})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "analysis_tables.md").write_text(tables_markdown(result, public), encoding="utf-8")
    return result


def _yn(v) -> str:
    return "—" if v is None else ("是" if v else "否")


def _f(v, nd=3) -> str:
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def tables_markdown(result: dict, public: dict) -> str:
    s = result["summary"]
    L = ["# 真实实验分析结果（只有数字，解读由你本人写）", "", f"承诺值：`{result['commitment']}`", ""]
    L += ["## 各阶段 top-1 正确率（Wilson 区间）与真实原因的平均后验", "",
          "| 实验/引擎 | 阶段 | 对/总 | 正确率 | 区间 | 平均 p_true |", "|---|---|---|---|---|---|"]
    for k, by in s["accuracy"].items():
        for st, a in by.items():
            lab = "只用基础方案" if st == "base" else ("全部追加测量" if st == "all" else "基础 + " + SLOT_LABEL.get(st, st))
            L.append(f"| {k} | {lab} | {a['k']}/{a['n']} | {_f(a['acc'])} | [{_f(a['lo'])}, {_f(a['hi'])}] | "
                     f"{_f(s['p_true_mean'][k].get(st))} |")
    L += ["", "## 配对 Wilcoxon（p_true 之差）", "",
          "| 对比 | 范围 | 配对数 | 差为 0 | 平均差 | 中位差 | 统计量 | p 值 | 备注 |", "|---|---|---|---|---|---|---|---|---|"]
    for label, block in s["contrasts"].items():
        rows = [("合并：" + "、".join(block["pooled"]["experiments"]), block["pooled"])]
        rows += [(e, w) for e, w in block["by_experiment"].items()]
        for scope, w in rows:
            L.append(f"| {label}（{'主' if block['role'] == 'primary' else '次'}） | {scope} | {w['n_pairs']} | "
                     f"{w['n_zero']} | {_f(w['mean_diff'])} | {_f(w['median_diff'])} | {_f(w['statistic'])} | "
                     f"{_f(w['p_value'], 4)} | {w['note']} |")
    L += ["", "## 图谱最难分开的那一对条件（单独报告）", "",
          "| 实验 | 条件对 | 图谱状态 | 配对数 | 平均差 | p 值 |", "|---|---|---|---|---|---|"]
    for e, c in s["confounded_pair"].items():
        w = c["contrast"]
        L.append(f"| {e} | {' / '.join(c['pair'])} | {c['status']} | {w['n_pairs']} | {_f(w['mean_diff'])} | "
                 f"{_f(w['p_value'], 4)} |")
    L += ["", "## 图谱预测效度（只用标准方案，两两判别）", "",
          "| 实验 | 条件对 | 图谱状态 | 图谱正确率 [区间] | 真实 对/总 | 真实正确率 [区间] | 真实点估计在图谱区间内 | "
          "图谱点估计在真实区间内 |", "|---|---|---|---|---|---|---|---|"]
    for e, rows in s["atlas_validity"].items():
        for r in rows:
            a, re_ = r["atlas"], r["real"]
            L.append(f"| {e} | {r['condition_a']} / {r['condition_b']} | {r['status']} | {_f(a['acc'])} "
                     f"[{_f(a['lo'])}, {_f(a['hi'])}] | {re_['k']}/{re_['n']} | {_f(re_['acc'])} "
                     f"[{_f(re_['lo'])}, {_f(re_['hi'])}] | {_yn(r['real_point_in_atlas_interval'])} | "
                     f"{_yn(r['atlas_point_in_real_interval'])} |")
    L += ["", "## 其他", ""]
    for k, a in s["clean_false_alarm"].items():
        mf = s["misfit"][k]
        L.append(f"- {k}：不设故障时只用基础方案判错 {a['k']}/{a['n']}；失配（p < 规定值）次数："
                 f"只用基础方案 {mf['base']}，全部追加测量 {mf['all']}")
    bad = [x for v in s["eig_reproduced"].values() for x in v if not x["reproduced"]]
    L.append(f"- 信息增益推荐复算不一致：{len(bad)} 次" + (f"（{', '.join(x['run_id'] for x in bad)}）" if bad else ""))
    L.append(f"- 没有做的运行：{', '.join(s['missing_runs']) or '无'}")
    L.append(f"- 跳过的追加测量：{len(s['skipped'])} 项")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="真实实验的预先规定分析（学生本人启动）")
    ap.add_argument("--public", required=True)
    ap.add_argument("--sealed", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not sys.stdin.isatty():
        sys.exit("正式分析必须在终端里由学生本人运行（不接受管道输入）。")
    typed = input(f"请输入“{CONFIRM_PHRASE}”后回车：").strip()
    public = json.loads(Path(a.public).read_text(encoding="utf-8"))
    sealed = json.loads(Path(a.sealed).read_text(encoding="utf-8"))
    res = run_analysis(public, sealed, a.data, a.out, typed)
    print(f"完成：{len(res['runs'])} 次运行。结果在 {Path(a.out) / 'analysis_results.json'} 和 analysis_tables.md。")


if __name__ == "__main__":
    main()
