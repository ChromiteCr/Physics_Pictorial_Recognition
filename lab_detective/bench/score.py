"""物理 bench 打分：读模型预测（JSONL：{"id", "text", "model", ...}）和真值旁注，按划分 / 实验 / 方案汇总，
两份预测之间做配对比较，写 markdown + JSON 报告。

诊断（schema = diagnose）逐条指标：
  json_valid      能解析且通过 prompts.validate_output（含讲解字数）
  top1            排第一的 cause_id = 真实原因（真实原因不在清单里的样本不计入）
  eq_top1         排第一的原因落在真实原因的等价类里（图谱判为分不开的原因合并，见 physics_bench）
  gold_agree      排第一的原因 = 引擎的第一名（K1 / 引擎参照用）
  log_score       −ln p(真实原因)（概率按 ranked 归一化，缺的记 0，下限 1e-6）
  brier           多类 Brier：Σ(p − y)²
  ece             置信度 = 第一名的概率，正确 = top1；ece_eq：置信度 = 第一名所在等价类的概率和，正确 = eq_top1
  nil_auroc       "不在清单里"判别的 AUROC（正类 = 真实原因不在清单里；打分 = not_in_list + 0.5·(1 − 第一名概率)）
  eig_regret      (最大 EIG − 所选测量的 EIG) / 最大 EIG；最大 EIG < eig.NO_SEPARATION_BITS、或引擎与模型都写 null
                  时记 0；写 null 而引擎有推荐、或编号不存在记 1
  verifier_pass   explanation 通过 tutor/verifier.verify（对照引擎事实单）
设计（schema = design）：json_valid、点数一致、主要来源类型一致、分不开原因对的 F1。
K3（swapped_table）：真实 id 空间里的准确率与机会水平（按实验分层的两个边际分布乘积，同时给出不分层的值）。
配对比较（bench/stats.py）：eq_top1 与 top1 的精确 McNemar、跨实验符号检验、按实验成簇的配对 bootstrap。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ..eig import NO_SEPARATION_BITS
from ..llm.client import parse_json
from ..training import prompts
from . import stats
from .physics_bench import OUT

EPS = 1e-6


# ── 读取 ──────────────────────────────────────────────────────────────


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_gold(bench_dir: str | Path = OUT, splits: Iterable[str] | None = None) -> dict[str, dict]:
    """真值旁注（含事实单）→ {样本 id: gold}。"""
    bench_dir = Path(bench_dir)
    out: dict[str, dict] = {}
    for p in sorted(bench_dir.glob("bench_*.gold.jsonl")):
        split = p.name[len("bench_"):-len(".gold.jsonl")]
        if splits is not None and split not in splits:
            continue
        facts = {r["id"]: r["fact_sheet"] for r in _jsonl(bench_dir / f"bench_{split}.facts.jsonl")}
        for g in _jsonl(p):
            g = dict(g, fact_sheet=facts.get(g["id"]))
            out[g["id"]] = g
    return out


def load_predictions(path: str | Path) -> dict[str, dict]:
    return {r["id"]: r for r in _jsonl(Path(path))}


def parse(text) -> dict | None:
    if isinstance(text, dict):
        return text
    try:
        obj = parse_json(text or "")
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


# ── 逐条打分 ──────────────────────────────────────────────────────────


def _probs(obj: dict, cands: list[str]) -> tuple[dict[str, float], str | None]:
    """ranked → (候选上的归一化概率, 第一名)。第一名 = ranked 里第一个合法的 cause_id。"""
    raw = {c: 0.0 for c in cands}
    top = None
    for r in obj.get("ranked") or []:
        if not isinstance(r, dict):
            continue
        cid = r.get("cause_id")
        if cid not in raw:
            continue
        if top is None:
            top = cid
        try:
            p = float(r.get("prob", 0.0))
        except (TypeError, ValueError):
            p = 0.0
        if math.isfinite(p) and p > 0:
            raw[cid] = max(raw[cid], p)
    s = sum(raw.values())
    if s <= 0:
        probs = {c: (1.0 if c == top else 0.0) if top else 1.0 / len(cands) for c in cands}
    else:
        probs = {c: v / s for c, v in raw.items()}
    return probs, top


def score_diagnose(gold: dict, text) -> dict:
    obj = parse(text)
    cands = list(gold["id_map"])
    row = {"id": gold["id"], "split": gold["split"], "family": gold["family"], "protocol": gold["protocol"],
           "out_of_list": bool(gold["out_of_list"]), "subsplit": gold.get("subsplit"), "parsed": obj is not None}
    row["json_valid"] = obj is not None and not prompts.validate_output("diagnose", obj)
    if obj is None:
        obj = {}
    probs, top = _probs(obj, cands)
    truth = gold.get("truth_disp")
    eq = set(gold.get("eq_class") or [])
    conf = probs.get(top, 0.0) if top else 0.0
    top_class = next((g for g in gold.get("eq_classes") or [] if top in g), [top] if top else [])
    row.update(top=top, top_real=gold["id_map"].get(top) if top else None, conf=conf,
               conf_eq=sum(probs.get(c, 0.0) for c in top_class))
    if not gold["out_of_list"]:
        p_true = probs.get(truth, 0.0)
        row.update(top1=top == truth, eq_top1=top in eq, log_score=-math.log(max(p_true, EPS)),
                   brier=sum((probs[c] - (1.0 if c == truth else 0.0)) ** 2 for c in cands))
    row["gold_agree"] = top == gold.get("engine_top")
    nil = obj.get("not_in_list") is True
    row["nil"] = nil
    row["nil_score"] = (1.0 if nil else 0.0) + 0.5 * (1.0 - conf)
    eig = gold.get("eig") or {}
    best = max(eig.values(), default=0.0)
    nm = obj.get("next_measurement")
    chosen = nm.get("id") if isinstance(nm, dict) else None
    if best < NO_SEPARATION_BITS or (chosen is None and nm is None and gold.get("gold_next") is None):
        regret = 0.0  # 没有测量能分开剩下的原因：写 null 或任选一项都不算后悔
    elif chosen in eig:
        regret = max(0.0, (best - eig[chosen]) / best)
    else:
        regret = 1.0
    row["eig_regret"] = regret
    fs = gold.get("fact_sheet")
    expl = obj.get("explanation")
    if fs is not None and isinstance(expl, str) and expl.strip():
        from ..tutor.verifier import verify

        try:
            row["verifier_pass"] = bool(verify(expl, fs)["ok"])
        except Exception:  # noqa: BLE001  校验器自身出错按不通过计
            row["verifier_pass"] = False
    else:
        row["verifier_pass"] = False
    return row


def score_design(gold: dict, text) -> dict:
    obj = parse(text)
    tgt = gold["target"]
    row = {"id": gold["id"], "split": gold["split"], "family": gold["family"], "protocol": gold["protocol"],
           "parsed": obj is not None, "json_valid": obj is not None and not prompts.validate_output("design", obj)}
    obj = obj or {}
    rec = obj.get("recommend") if isinstance(obj.get("recommend"), dict) else {}
    dom = obj.get("dominant_source") if isinstance(obj.get("dominant_source"), dict) else {}
    row["n_points_ok"] = rec.get("n_points") == tgt["recommend"]["n_points"]
    row["source_type_ok"] = dom.get("type") == tgt["dominant_source"]["type"]

    def pairs(lst):
        out = set()
        for u in lst or []:
            if isinstance(u, dict) and isinstance(u.get("pair"), list) and len(u["pair"]) == 2:
                out.add(frozenset(map(str, u["pair"])))
        return out

    g, p = pairs(tgt["unresolved"]), pairs(obj.get("unresolved"))
    if not g and not p:
        f1 = 1.0
    else:
        tp = len(g & p)
        f1 = 2 * tp / (len(g) + len(p)) if (len(g) + len(p)) else 1.0
    row["unresolved_f1"] = f1
    return row


def score_rows(gold: dict[str, dict], preds: dict[str, dict]) -> list[dict]:
    """每条 bench 样本一行；没有预测的样本按空输出打分（算作错误和非法 JSON）。"""
    rows = []
    for gid, g in gold.items():
        text = (preds.get(gid) or {}).get("text", "")
        r = score_design(g, text) if g["schema"] == "design" else score_diagnose(g, text)
        r["missing"] = gid not in preds
        rows.append(r)
    return rows


# ── 汇总 ──────────────────────────────────────────────────────────────


def _mean(rows, key) -> float | None:
    v = [float(r[key]) for r in rows if key in r and r[key] is not None]
    return float(np.mean(v)) if v else None


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    out: dict = {"n": len(rows), "json_valid": _mean(rows, "json_valid"), "missing": sum(r["missing"] for r in rows)}
    if "n_points_ok" in rows[0]:
        out.update(n_points_ok=_mean(rows, "n_points_ok"), source_type_ok=_mean(rows, "source_type_ok"),
                   unresolved_f1=_mean(rows, "unresolved_f1"))
        return out
    inl = [r for r in rows if not r["out_of_list"]]
    out.update(n_in_list=len(inl), top1=_mean(inl, "top1"), eq_top1=_mean(inl, "eq_top1"),
               gold_agree=_mean(rows, "gold_agree"), log_score=_mean(inl, "log_score"), brier=_mean(inl, "brier"),
               eig_regret=_mean(rows, "eig_regret"), verifier_pass=_mean(rows, "verifier_pass"),
               nil_rate=_mean(rows, "nil"))
    if inl:
        out["top1_ci95"] = list(stats.wilson(sum(bool(r["top1"]) for r in inl), len(inl)))
        out["eq_top1_ci95"] = list(stats.wilson(sum(bool(r["eq_top1"]) for r in inl), len(inl)))
        out["ece"] = stats.ece([r["conf"] for r in inl], [bool(r["top1"]) for r in inl])["ece"]
        out["ece_eq"] = stats.ece([r["conf_eq"] for r in inl], [bool(r["eq_top1"]) for r in inl])["ece"]
    out["nil_auroc"] = nil_auroc(rows)
    out["nil_auroc_detectable"] = nil_auroc([r for r in rows if r.get("subsplit") != "degenerate"])
    return out


def nil_auroc(rows: list[dict]) -> float | None:
    """正类 = 真实原因不在清单里。nil_auroc_detectable 去掉"退化"样本（引擎失配判据也看不出来的，任何模型都抓不到）；
    可检测 / 退化的划分来自引擎自己的失配判据，所以引擎在可检测子集上的 AUROC≈1 是构造使然，不是引擎有效的证据。"""
    rows = [r for r in rows if "nil_score" in r]
    y = [r["out_of_list"] for r in rows]
    if not rows or all(y) or not any(y):
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, [r["nil_score"] for r in rows]))


def k3_chance(rows: list[dict], gold: dict[str, dict]) -> dict:
    """换表对照：真实 id 空间里的准确率和机会水平（两个边际分布的乘积）。"""
    rs = [r for r in rows if r["split"] == "swapped_table" and not r["out_of_list"]]
    if not rs:
        return {}
    truth = [gold[r["id"]]["truth_cause"] for r in rs]
    pred = [r["top_real"] or "(无)" for r in rs]
    acc = float(np.mean([t == p for t, p in zip(truth, pred)]))
    # 换表只在同一实验内换，机会水平按实验分层：每个实验内两个边际分布的乘积，再按样本数加权
    fam: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rs):
        fam[r["family"]].append(i)
    chance = sum(len(ix) / len(rs) * stats.chance_product_of_marginals([truth[i] for i in ix], [pred[i] for i in ix])
                 for ix in fam.values())
    return {"n": len(rs), "acc_real": acc, "chance": float(chance),
            "chance_pooled": stats.chance_product_of_marginals(truth, pred),
            "acc_ci95": list(stats.wilson(sum(t == p for t, p in zip(truth, pred)), len(rs)))}


def report(rows: list[dict], gold: dict[str, dict], model: str = "") -> dict:
    by_split: dict[str, list] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    rep: dict = {"model": model, "splits": {}, "overall_diagnose": {}}
    for s, rs in sorted(by_split.items()):
        fam: dict[str, list] = defaultdict(list)
        prot: dict[str, list] = defaultdict(list)
        sub: dict[str, list] = defaultdict(list)
        for r in rs:
            fam[r["family"]].append(r)
            prot[str(r["protocol"]).split("+")[0] + ("+追加" if "+" in str(r["protocol"]) else "")].append(r)
            if r.get("subsplit"):
                sub[r["subsplit"]].append(r)
        rep["splits"][s] = {"all": summarize(rs), "by_experiment": {k: summarize(v) for k, v in sorted(fam.items())},
                            "by_protocol": {k: summarize(v) for k, v in sorted(prot.items())}}
        if sub:
            rep["splits"][s]["by_subsplit"] = {k: summarize(v) for k, v in sorted(sub.items())}
    diag = [r for r in rows if "nil_score" in r and r["split"] not in ("swapped_table", "oracle_in_prompt")]
    rep["overall_diagnose"] = summarize(diag) if diag else {}
    rep["k3_swapped_table"] = k3_chance(rows, gold)
    if "oracle_in_prompt" in by_split:
        rep["k1_oracle_gold_agree"] = _mean(by_split["oracle_in_prompt"], "gold_agree")
    return rep


# ── 配对比较 ──────────────────────────────────────────────────────────


def compare(rows_a: list[dict], rows_b: list[dict], key: str = "eq_top1", n_boot: int = 5000, seed: int = 0) -> dict:
    """同一批样本上 A 对 B：精确 McNemar、跨实验符号检验、按实验成簇的配对 bootstrap。"""
    b_by = {r["id"]: r for r in rows_b}
    pairs = [(a, b_by[a["id"]]) for a in rows_a if a["id"] in b_by and key in a and key in b_by[a["id"]]]
    if not pairs:
        return {"n": 0}
    ya = [bool(a[key]) if not isinstance(a[key], float) else a[key] for a, _ in pairs]
    yb = [bool(b[key]) if not isinstance(b[key], float) else b[key] for _, b in pairs]
    out: dict = {"n": len(pairs), "key": key}
    if all(isinstance(v, bool) for v in ya + yb):
        out["mcnemar"] = stats.mcnemar_exact(ya, yb)
    fam: dict[str, list] = defaultdict(list)
    for (a, _), va, vb in zip(pairs, ya, yb):
        fam[a["family"]].append(float(va) - float(vb))
    out["sign_test_experiments"] = stats.sign_test([float(np.mean(v)) for v in fam.values()])
    out["paired_bootstrap"] = stats.paired_bootstrap([float(v) for v in ya], [float(v) for v in yb], n_boot=n_boot,
                                                     seed=seed, clusters=[a["family"] for a, _ in pairs])
    return out


def compare_all(rows_a: list[dict], rows_b: list[dict]) -> dict:
    out = {}
    for s in sorted({r["split"] for r in rows_a}):
        ra = [r for r in rows_a if r["split"] == s]
        rb = [r for r in rows_b if r["split"] == s]
        keys = ("json_valid", "n_points_ok") if s == "design" else ("eq_top1", "top1", "json_valid", "log_score")
        out[s] = {k: compare(ra, rb, k) for k in keys}
    return out


# ── 报告 ──────────────────────────────────────────────────────────────


def _f(v, d=3) -> str:
    return "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def to_markdown(rep: dict, cmp: dict | None = None, name_b: str = "") -> str:
    lines = [f"# 物理 bench 报告：{rep.get('model') or '(未命名)'}", ""]
    lines += ["| 划分 | n | JSON 合法 | top-1 | 等价类 top-1 | 引擎一致 | log score | Brier | ECE | ECE(等价类) | EIG 后悔 | 校验通过 | 不在清单 AUROC |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, d in rep["splits"].items():
        a = d["all"]
        lines.append(f"| {s} | {a['n']} | {_f(a.get('json_valid'))} | {_f(a.get('top1'))} | {_f(a.get('eq_top1'))} | "
                     f"{_f(a.get('gold_agree'))} | {_f(a.get('log_score'))} | {_f(a.get('brier'))} | {_f(a.get('ece'))} | {_f(a.get('ece_eq'))} | "
                     f"{_f(a.get('eig_regret'))} | {_f(a.get('verifier_pass'))} | {_f(a.get('nil_auroc'))} |")
        if s == "design":
            lines.append(f"|  ↳ 设计 | | 点数 {_f(a.get('n_points_ok'))} | 来源 {_f(a.get('source_type_ok'))} | "
                         f"分不开对 F1 {_f(a.get('unresolved_f1'))} | | | | | | | | |")
    o = rep.get("overall_diagnose") or {}
    if o:
        lines += ["", f"诊断合计（不含 K1/K3）：n = {o['n']}，等价类 top-1 = {_f(o.get('eq_top1'))}，"
                      f"不在清单 AUROC = {_f(o.get('nil_auroc'))}（只算可检测的：{_f(o.get('nil_auroc_detectable'))}）"]
    for s, d in rep["splits"].items():
        for k, v in (d.get("by_subsplit") or {}).items():
            lines.append(f"{s} / {k}：n = {v['n']}，判为不在清单的比例 {_f(v.get('nil_rate'))}")
    k3 = rep.get("k3_swapped_table") or {}
    if k3:
        lines.append(f"K3 换表对照：准确率 {_f(k3['acc_real'])}（95% 区间 {_f(k3['acc_ci95'][0])}–{_f(k3['acc_ci95'][1])}），"
                     f"机会水平 {_f(k3['chance'])}，n = {k3['n']}")
    if rep.get("k1_oracle_gold_agree") is not None:
        lines.append(f"K1 答案进提示：与标准答案一致率 {_f(rep['k1_oracle_gold_agree'])}")
    if cmp:
        lines += ["", f"## 配对比较：{rep.get('model')} 对 {name_b}", "",
                  "| 划分 | 指标 | n | 差值 | McNemar p | 跨实验符号检验 p | bootstrap 95% 区间 |", "|---|---|---|---|---|---|---|"]
        for s, d in cmp.items():
            for k, c in d.items():
                if not c.get("n"):
                    continue
                mc = c.get("mcnemar") or {}
                bs = c["paired_bootstrap"]
                lines.append(f"| {s} | {k} | {c['n']} | {_f(bs['diff'])} | {_f(mc.get('p_value'), 4)} | "
                             f"{_f(c['sign_test_experiments']['p_value'], 4)} | {_f(bs['ci95'][0])} – {_f(bs['ci95'][1])} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="物理 bench 打分与配对比较")
    ap.add_argument("pred", help="模型预测 JSONL（run_models.py 的输出）")
    ap.add_argument("--against", default=None, help="另一份预测，做配对比较")
    ap.add_argument("--bench-dir", default=str(OUT))
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="报告前缀（默认与预测文件同名 .report.json / .report.md）")
    a = ap.parse_args(argv)
    gold = load_gold(a.bench_dir, a.splits)
    preds = load_predictions(a.pred)
    ids = set(preds)
    gold = {k: v for k, v in gold.items() if a.splits or k in ids} or gold
    model = next((p.get("model", "") for p in preds.values()), "")
    rows = score_rows(gold, preds)
    rep = report(rows, gold, model)
    cmp = None
    name_b = ""
    if a.against:
        pb = load_predictions(a.against)
        name_b = next((p.get("model", "") for p in pb.values()), a.against)
        cmp = compare_all(rows, score_rows(gold, pb))
        rep["compare"] = {"against": name_b, "results": cmp}
    prefix = Path(a.out) if a.out else Path(a.pred).with_suffix("")
    Path(str(prefix) + ".report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str),
                                                  encoding="utf-8")
    Path(str(prefix) + ".report.md").write_text(to_markdown(rep, cmp, name_b), encoding="utf-8")
    print(to_markdown(rep, cmp, name_b))


if __name__ == "__main__":
    main()
