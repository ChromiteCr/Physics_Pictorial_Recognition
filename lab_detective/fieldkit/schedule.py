"""预先提交的随机表：真实故障注入实验的运行顺序、条件分配、追加测量顺序，加上密封卡和记录表模板。

设计：
- 单元 = 实验 × 条件（不设故障 + 3 种故障）× 重复；第 r 次重复放在第 (r mod 实验课次数) + 1 次实验课，
  每次课里实验的先后、实验内条件的先后都随机（每次课每个单元一次，课次之间平衡）。
- 每次运行都做全部追加测量（信息增益推荐、最佳固定测量、一次简单重复），顺序随机，
  同一实验内每个槽位出现在每个位置的次数最多差 1（循环拉丁方）。
- 运行代码从种子随机流里抽，与条件无关；卡片上只有代码和动作，没有条件名。
- 提交：公开表（给学生）+ 密封表（给成年人，含种子、盐、条件和真值）。
  承诺值 commitment = sha256(密封表的规范 JSON)，写在公开表里；密封表里又记着公开表核心内容的 sha256。
  种子要保密（有种子就能重算分配）。易变的东西（时间、代码版本）不进哈希。
- 图谱检查：每个实验在所选条件的真值原因之间算两两判别正确率（标准方案，独立的固定种子），
  如实记下状态（confused / partial / separable；真值相同记 same_class），不把 partial 说成混淆。

命令（生成后不要打开 sealed/ 文件夹，直接交给成年人打印）：
    python -m lab_detective.fieldkit.schedule build --out <文件夹>
    python -m lab_detective.fieldkit.schedule verify --public <公开表> --sealed <密封表>
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import secrets
import zlib
from pathlib import Path

import numpy as np

from .catalog import (
    DEFAULT_EXPERIMENTS,
    DEFAULT_SLOTS,
    EXCLUDED,
    FIELD_EXPERIMENTS,
    FIELDKIT_VERSION,
    OPTIONAL_SLOTS,
    SLOT_LABEL,
    TWIN_PREFIX,
    FieldExperiment,
    allowed_items,
    best_fixed,
    build_engines,
    fill_magnitude,
    howto,
    pendulum_spec,
    resolve_truth,
)
from .sheet import PENDING, Row, Sheet, fmt_num, fmt_scalar, write_sheet

CODE_ALPHABET = "ACDEFHJKLMNPQRTUVWXY3479"  # 去掉容易看错的 0/O、1/I、2/Z、5/S、8/B、6/G

# 预先规定的分析（写进公开表、受哈希保护）。数值都是占位默认值，建表前由学生逐项确认（见 docs/student/02）。
ANALYSIS_SPEC_DEFAULT = {
    "version": 1,
    "primary_metric": "p_true",  # 真实原因（或其等价原因）的后验概率
    "followup_mode": "single",  # 每项追加测量单独接在基础方案后面算后验（不累积）
    "primary_contrast": ["eig", "best_fixed"],
    "secondary_contrasts": [["eig", "repeat"], ["best_fixed", "repeat"]],
    "pool": "mainline_b",  # 合并检验只用课内实验；单摆单独报告
    "wilcoxon_zero_method": "wilcox",  # 两项测量相同时差为 0：丢弃并报告个数
    "wilcoxon_alternative": "two-sided",
    "alpha": 0.05,
    "ci_level": 0.95,
    "misfit_alpha": 0.01,  # "不在清单"判据（只描述），与产品事实单 facts.py 的判据一致
    "eig_n_samples": 256,
    "missing_rule": "exclude_and_list",
}
ATLAS_CHECK_DEFAULT = {"n_per_cause": 200, "confused_below": 0.75, "separable_at": 0.9}


# ── 哈希 ─────────────────────────────────────────────────────────────────


def canonical(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_obj(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def public_core(public: dict) -> dict:
    return {k: v for k, v in public.items() if k not in ("commitment", "unhashed")}


def verify_commitment(public: dict, sealed: dict) -> dict:
    errors = []
    sealed_body = {k: v for k, v in sealed.items() if k != "unhashed"}
    if sha256_obj(sealed_body) != public.get("commitment"):
        errors.append("密封表的哈希与公开表里的承诺值不一致（密封表被改过，或不是同一次生成的）")
    if sealed.get("public_sha256") != sha256_obj(public_core(public)):
        errors.append("公开表的内容与密封表里记下的哈希不一致（公开表被改过）")
    return {"ok": not errors, "errors": errors}


# ── 随机分配（只依赖种子和设计参数，不依赖物理代码） ─────────────────────────────


def _codes(rng: np.random.Generator, n: int) -> list[str]:
    seen, out = set(), []
    while len(out) < n:
        c = "".join(CODE_ALPHABET[int(i)] for i in rng.integers(0, len(CODE_ALPHABET), 4))
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _slot_orders(rng: np.random.Generator, slots: tuple[str, ...], n: int) -> list[tuple[str, ...]]:
    """循环拉丁方：每 k 次运行里每个槽位在每个位置各出现一次；整体最多差 1。"""
    k = len(slots)
    base = [slots[int(i)] for i in rng.permutation(k)]
    rows = [tuple(base[(i + j) % k] for j in range(k)) for i in range(k)]
    out: list[tuple[str, ...]] = []
    while len(out) < n:
        out.extend(rows[int(i)] for i in rng.permutation(k))
    return out[:n]


def assign(seed: int, experiments: tuple[str, ...], repeats: int, sessions: int,
           slots: dict[str, tuple[str, ...]]) -> dict:
    """随机分配：运行顺序、代码、条件、幅度、追加测量顺序、盐。"""
    rng = np.random.default_rng(seed)
    salt = "".join(f"{int(x):016x}" for x in rng.integers(0, 2**62, 2))
    # 幅度：每个（实验，条件）的各次重复轮流用打乱后的幅度选项
    mags: dict[tuple[str, str], list] = {}
    for e in experiments:
        for rec in FIELD_EXPERIMENTS[e].recipes:
            opts = list(range(len(rec.magnitudes)))
            seq: list = []
            while len(seq) < repeats and opts:
                seq.extend(opts[int(i)] for i in rng.permutation(len(opts)))
            mags[(e, rec.key)] = seq[:repeats] if opts else [None] * repeats
    runs = []
    for s in range(1, sessions + 1):
        reps = [r for r in range(repeats) if r % sessions == s - 1]
        if not reps:
            continue
        for ei in rng.permutation(len(experiments)):
            e = experiments[int(ei)]
            cells = [(rec.key, r) for r in reps for rec in FIELD_EXPERIMENTS[e].recipes]
            for ci in rng.permutation(len(cells)):
                key, r = cells[int(ci)]
                runs.append({"session": s, "experiment": e, "condition": key, "repeat": r,
                             "magnitude_index": mags[(e, key)][r]})
    codes = _codes(rng, len(runs))
    orders = {e: iter(_slot_orders(rng, slots[e], sum(1 for x in runs if x["experiment"] == e)))
              for e in experiments}
    for i, run in enumerate(runs):
        run["run_id"] = f"R{i + 1:02d}"
        run["code"] = codes[i]
        run["slots"] = list(next(orders[run["experiment"]]))
    return {"salt": salt, "runs": runs}


# ── 每个实验的静态信息（依赖物理代码） ────────────────────────────────────


def _plan(plan) -> list:
    return [[str(k), float(s)] for k, s in plan]


def _item_for_plan(engine, plan) -> str | None:
    for m in engine.menu:
        if [[k, float(s)] for k, s in m.plan] == _plan(plan):
            return m.id
    return None


def _atlas_pairs(fexp: FieldExperiment, engine, truth_by_cond: dict, params: dict) -> list[dict]:
    from ..atlas import pairwise_accuracy

    rows = []
    conds = [r.key for r in fexp.recipes]
    for a, b in itertools.combinations(conds, 2):
        ta, tb = truth_by_cond[a], truth_by_cond[b]
        row = {"condition_a": a, "condition_b": b, "cause_a": ta, "cause_b": tb}
        if ta == tb:
            row.update(acc=None, lo=None, hi=None, n_total=0, status="same_class")
        else:
            seed = zlib.crc32(f"fieldkit|atlas|{engine.id}|{ta}|{tb}".encode())
            acc, lo, hi = pairwise_accuracy(engine, "standard", ta, tb, int(params["n_per_cause"]),
                                            np.random.default_rng(seed))
            status = ("confused" if hi < params["confused_below"]
                      else "separable" if lo >= params["separable_at"] else "partial")
            row.update(acc=float(acc), lo=float(lo), hi=float(hi), n_total=2 * int(params["n_per_cause"]),
                       status=status)
        rows.append(row)
    rank = {"same_class": 0, "confused": 1, "partial": 2, "separable": 3}
    rows.sort(key=lambda r: (rank[r["status"]], r["acc"] if r["acc"] is not None else -1.0))
    return rows


def experiment_info(key: str, config: dict, slots: tuple[str, ...], atlas: dict | None) -> tuple[dict, dict]:
    """→ (公开表里这个实验的信息, 各条件的真值 {条件: {引擎: (原因 id, 是否等价)}})。"""
    fexp = FIELD_EXPERIMENTS[key]
    for rec in fexp.recipes:
        for eng, suffixes in rec.truth.items():
            for s in suffixes:
                if f"{key}.{s}" in EXCLUDED:
                    raise ValueError(f"{key}.{s} 会损坏器材，不能在真实器材上做：{EXCLUDED[key + '.' + s]}")
    spec = pendulum_spec(config) if key == "pendulum" else None
    engines = build_engines(key, config, spec)
    prim = engines[fexp.primary]
    allowed = allowed_items(fexp, prim, config)
    base = prim.protocols["standard"].plan
    from ..eig import repeat_plan

    rep = repeat_plan(prim, base)
    info = {
        "name_zh": fexp.name_zh, "primary": fexp.primary, "engine_ids": {k: e.id for k, e in engines.items()},
        "mainline_b": fexp.mainline_b, "config": config, "spec": spec,
        "base_plan": _plan(base), "base_howto": fexp.base_howto, "setting_note": fexp.setting_note,
        "slots": list(slots), "allowed_items": list(allowed),
        "best_fixed": best_fixed(prim, allowed),
        "repeat": {"item": _item_for_plan(prim, rep) or "repeat", "plan": _plan(rep)},
        "howto": {m: howto(fexp, prim, m) for m in allowed},
        "units": {k: [r.unit for r in ok.readings] for k, ok in prim.obs_kinds.items()},
    }
    if "twin_eig" in slots:
        tw = engines["twin"]
        t_allowed = allowed_items(fexp, tw, config)
        info["twin_allowed_items"] = list(t_allowed)
        info["twin_howto"] = {m: howto(fexp, tw, m) for m in t_allowed}
        info["twin_units"] = {k: [r.unit for r in ok.readings] for k, ok in tw.obs_kinds.items()}
    truths = {}
    for rec in fexp.recipes:
        truths[rec.key] = {eng: resolve_truth(engines[eng], suf) for eng, suf in rec.truth.items() if eng in engines}
    if atlas is not None:
        truth_prim = {c: t[fexp.primary][0] for c, t in truths.items()}
        info["atlas_pairs"] = _atlas_pairs(fexp, prim, truth_prim, atlas)
    return info, truths


# ── 建表 ─────────────────────────────────────────────────────────────────


def build_schedule(experiments=DEFAULT_EXPERIMENTS, repeats: int = 2, sessions: int = 2, seed: int | None = None,
                   config: dict | None = None, extra_slots: dict | None = None, analysis: dict | None = None,
                   atlas: dict | bool | None = True) -> tuple[dict, dict]:
    """→ (公开表, 密封表)。seed 不给时用系统随机数（请让它只留在密封表里）。"""
    experiments = tuple(experiments)
    for e in experiments:
        if e not in FIELD_EXPERIMENTS:
            raise ValueError(f"不认识的实验 {e!r}（可选：{', '.join(FIELD_EXPERIMENTS)}）")
    if len(set(experiments)) != len(experiments):
        raise ValueError("实验重复")
    if repeats < 1 or sessions < 1:
        raise ValueError("repeats、sessions 至少为 1")
    seed = int(secrets.randbits(62)) if seed is None else int(seed)
    extra_slots = dict(extra_slots or {})
    slots = {}
    for e in experiments:
        ex = tuple(extra_slots.get(e, ()))
        bad = [s for s in ex if s not in OPTIONAL_SLOTS.get(e, ())]
        if bad:
            raise ValueError(f"{e} 不能加槽位 {bad}")
        slots[e] = DEFAULT_SLOTS + ex
    cfg = {e: dict(FIELD_EXPERIMENTS[e].config, **(config or {}).get(e, {})) for e in experiments}
    spec = dict(ANALYSIS_SPEC_DEFAULT, **(analysis or {}))
    atlas_params = None if atlas is False or atlas is None else dict(
        ATLAS_CHECK_DEFAULT, **(atlas if isinstance(atlas, dict) else {}))

    infos, truths, warnings = {}, {}, []
    for e in experiments:
        infos[e], truths[e] = experiment_info(e, cfg[e], slots[e], atlas_params)
        pairs = infos[e].get("atlas_pairs")
        if pairs is not None and not any(p["status"] in ("confused", "same_class") for p in pairs):
            p = pairs[0]
            warnings.append(f"{infos[e]['name_zh']}：所选条件里没有图谱判为混淆的一对；最接近的是 "
                            f"{p['condition_a']} / {p['condition_b']}（{p['status']}，正确率 {p['acc']:.2f}）。")

    asg = assign(seed, experiments, repeats, sessions, slots)
    pub_runs, key = [], []
    for run in asg["runs"]:
        e = run["experiment"]
        fexp = FIELD_EXPERIMENTS[e]
        rec = next(r for r in fexp.recipes if r.key == run["condition"])
        mag = rec.magnitudes[run["magnitude_index"]] if run["magnitude_index"] is not None else None
        mtext = fill_magnitude(mag.text, cfg[e]) if mag else ""
        steps = [fill_magnitude(s.replace("{mag}", mtext), cfg[e]) for s in rec.steps]
        student = fill_magnitude(rec.student_text.replace("{mag}", mtext), cfg[e]) if rec.setter == "student" else ""
        pub_runs.append({k: run[k] for k in ("run_id", "code", "session", "experiment", "slots")})
        key.append({
            "run_id": run["run_id"], "code": run["code"], "experiment": e, "condition": rec.key,
            "repeat": run["repeat"], "setter": rec.setter, "blind": rec.blind,
            "magnitude": None if mag is None else {"value": mag.value, "text": mtext},
            "truth": {eng: {"cause_id": t[0], "equivalent": t[1]} for eng, t in truths[e][rec.key].items()},
            "card": {"steps": steps, "student_text": student, "safety": rec.safety},
        })
    public = {
        "fieldkit_version": FIELDKIT_VERSION,
        "design": {"experiments": list(experiments), "repeats": repeats, "sessions": sessions,
                   "n_runs": len(pub_runs), "conditions_per_experiment": {e: len(FIELD_EXPERIMENTS[e].recipes)
                                                                           for e in experiments}},
        "analysis_spec": spec,
        "atlas_check": atlas_params,
        "experiments": infos,
        "runs": pub_runs,
        "warnings": warnings,
    }
    sealed = {
        "fieldkit_version": FIELDKIT_VERSION,
        "seed": seed,
        "salt": asg["salt"],
        "build_args": {"experiments": list(experiments), "repeats": repeats, "sessions": sessions,
                       "slots": {e: list(s) for e, s in slots.items()}},
        "public_sha256": sha256_obj(public),
        "key": key,
    }
    public["commitment"] = sha256_obj(sealed)
    return public, sealed


def regenerate_check(public: dict, sealed: dict) -> dict:
    """用密封表里的种子重算随机分配，与两张表逐项比对（不依赖物理代码）。"""
    a = sealed["build_args"]
    asg = assign(int(sealed["seed"]), tuple(a["experiments"]), int(a["repeats"]), int(a["sessions"]),
                 {e: tuple(s) for e, s in a["slots"].items()})
    errors = []
    if asg["salt"] != sealed.get("salt"):
        errors.append("盐不一致")
    if len(asg["runs"]) != len(public["runs"]) or len(asg["runs"]) != len(sealed["key"]):
        errors.append("运行次数不一致")
    for r, p, k in zip(asg["runs"], public["runs"], sealed["key"]):
        if (r["run_id"], r["code"], r["experiment"], r["session"], r["slots"]) != \
                (p["run_id"], p["code"], p["experiment"], p["session"], p["slots"]):
            errors.append(f"{p['run_id']}：公开表与重算结果不一致")
        if (r["code"], r["condition"]) != (k["code"], k["condition"]):
            errors.append(f"{k['run_id']}：密封表与重算结果不一致")
    return {"ok": not errors, "errors": errors}


# ── 记录表 ───────────────────────────────────────────────────────────────


def _unit(info: dict, kind: str) -> str:
    if kind.startswith(TWIN_PREFIX):
        u = info.get("twin_units", {}).get(kind[len(TWIN_PREFIX):], [""])
    else:
        u = info["units"].get(kind, [""])
    return u[0] if u else ""


def _menu_plan(info: dict, engines_cache: dict, e: str, item: str) -> list:
    if e not in engines_cache:
        engines_cache[e] = build_engines(e, info["config"], info.get("spec"))
    prim = engines_cache[e][info["primary"]]
    return _plan(prim.menu_item(item).plan)


def static_followups(info: dict, e: str, slots: list[str], engines_cache: dict) -> list[tuple[list[str], str, list]]:
    """按槽位顺序列出（槽位们, 菜单项, plan）；同一项测量合并成一行组，信息增益类槽位先放待定。"""
    groups: list[tuple[list[str], str, list]] = []
    for s in slots:
        if s in ("eig", "twin_eig"):
            groups.append(([s], PENDING, []))
            continue
        if s == "best_fixed":
            item, plan = info["best_fixed"], _menu_plan(info, engines_cache, e, info["best_fixed"])
        else:
            item, plan = info["repeat"]["item"], info["repeat"]["plan"]
        for g in groups:
            if g[1] != PENDING and g[2] == plan:
                g[0].append(s)
                break
        else:
            groups.append(([s], item, plan))
    return groups


def run_sheet(public: dict, run: dict | None, experiment: str, engines_cache: dict | None = None) -> Sheet:
    """一次运行的记录表（run=None 时是该实验的通用模板）。"""
    engines_cache = {} if engines_cache is None else engines_cache
    info = public["experiments"][experiment]
    slots = run["slots"] if run else info["slots"]
    sh = Sheet()
    sh.meta["fieldkit_version"] = str(public["fieldkit_version"])
    sh.meta["run_id"] = run["run_id"] if run else ""
    sh.meta["code"] = run["code"] if run else ""
    sh.meta["experiment"] = experiment
    sh.meta["session"] = str(run["session"]) if run else ""
    sh.meta["followup_order"] = " > ".join(slots)
    for k, v in info["config"].items():
        sh.meta[f"config.{k}"] = fmt_scalar(v)
    sh.notes = [
        f"# 实验侦探 真实实验记录表：{info['name_zh']}（一次运行一个文件，只记器材读数，不写任何人的信息）",
        f"# 基础方案：{info['base_howto']}",
        f"# {info['setting_note']}；value 列填读数；做不了的追加测量在 value 填 skip。",
        "# 追加测量按 followup_order 的顺序做；stage 为 eig 的那一行做完基础方案后运行："
        "python -m lab_detective.fieldkit.live <本文件> --public <公开表>，它会把要做的测量填进来。",
    ]
    step = 1
    for kind, s in info["base_plan"]:
        sh.rows.append(Row(step, "base", "standard", kind, fmt_num(s), "", _unit(info, kind)))
        step += 1
    for stages, item, plan in static_followups(info, experiment, slots, engines_cache):
        if item == PENDING:
            sh.rows.append(Row(step, "+".join(stages), PENDING, "", "", "", ""))
            step += 1
            continue
        for kind, s in plan:
            sh.rows.append(Row(step, "+".join(stages), item, kind, fmt_num(s), "", _unit(info, kind)))
            step += 1
    return sh


def run_list_markdown(public: dict) -> str:
    lines = ["# 运行顺序（公开）", "",
             f"承诺值（密封表的 sha256）：`{public['commitment']}`", "",
             "| 运行 | 代码 | 实验课 | 实验 | 追加测量顺序 |", "|---|---|---|---|---|"]
    for r in public["runs"]:
        name = public["experiments"][r["experiment"]]["name_zh"]
        order = " → ".join(SLOT_LABEL.get(s, s) for s in r["slots"])
        lines.append(f"| {r['run_id']} | {r['code']} | 第 {r['session']} 次 | {name} | {order} |")
    if public.get("warnings"):
        lines += ["", "## 提示", ""] + [f"- {w}" for w in public["warnings"]]
    return "\n".join(lines) + "\n"


def write_outputs(public: dict, sealed: dict, out: str | Path) -> dict:
    from .cards import render_cards

    out = Path(out)
    pub, sea = out / "public", out / "sealed"
    for d in (pub / "runs", pub / "templates", sea):
        d.mkdir(parents=True, exist_ok=True)
    (pub / "schedule_public.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    (sea / "schedule_sealed.json").write_text(json.dumps(sealed, ensure_ascii=False, indent=2), encoding="utf-8")
    (sea / "sealed_cards.html").write_text(render_cards(public, sealed), encoding="utf-8")
    (pub / "run_list.md").write_text(run_list_markdown(public), encoding="utf-8")
    cache: dict = {}
    for e in public["design"]["experiments"]:
        write_sheet(run_sheet(public, None, e, cache), pub / "templates" / f"{e}.csv")
    for r in public["runs"]:
        write_sheet(run_sheet(public, r, r["experiment"], cache), pub / "runs" / f"{r['run_id']}_{r['code']}.csv")
    return {"public": str(pub / "schedule_public.json"), "sealed": str(sea / "schedule_sealed.json"),
            "public_sha256": hashlib.sha256((pub / "schedule_public.json").read_bytes()).hexdigest(),
            "commitment": public["commitment"]}


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_config(items: list[str]) -> dict:
    """--config friction.scale_division=0.2 newton2.cart_mass_g=480"""
    from .sheet import parse_scalar

    out: dict = {}
    for it in items or []:
        k, v = it.split("=", 1)
        e, p = k.split(".", 1)
        out.setdefault(e, {})[p] = parse_scalar(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="真实故障注入实验：随机表与密封卡")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    b.add_argument("--repeats", type=int, default=2)
    b.add_argument("--sessions", type=int, default=2)
    b.add_argument("--seed", type=int, default=None, help="不给就用系统随机数（推荐）")
    b.add_argument("--config", nargs="*", default=[], help="实验.参数=值，例如 newton2.cart_mass_g=480")
    b.add_argument("--twin-eig", action="store_true", help="单摆加做留出孪生的信息增益推荐")
    v = sub.add_parser("verify")
    v.add_argument("--public", required=True)
    v.add_argument("--sealed", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        exps = tuple(x.strip() for x in a.experiments.split(",") if x.strip())
        extra = {"pendulum": ("twin_eig",)} if a.twin_eig and "pendulum" in exps else {}
        public, sealed = build_schedule(exps, a.repeats, a.sessions, a.seed, _parse_config(a.config), extra)
        res = write_outputs(public, sealed, a.out)
        print(f"共 {public['design']['n_runs']} 次运行。")
        print(f"公开表：{res['public']}")
        print(f"  公开表文件 sha256：{res['public_sha256']}")
        print(f"  承诺值（密封表 sha256）：{res['commitment']}")
        print("把这两个哈希抄进研究计划（附日期）。sealed/ 文件夹不要打开，整个交给成年人打印、保管。")
        for w in public["warnings"]:
            print("提示：" + w)
    else:
        pub, sea = load_json(a.public), load_json(a.sealed)
        r1, r2 = verify_commitment(pub, sea), regenerate_check(pub, sea)
        print(json.dumps({"commitment": r1, "regenerate": r2}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
