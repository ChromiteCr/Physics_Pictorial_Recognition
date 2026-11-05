"""实验中途用：做完基础方案后，算出"信息增益推荐"的那项追加测量，填进记录表，并留一张回执。

- 只读基础方案的读数（追加测量还没有任何读数时才允许运行），记下基础读数的指纹；
- 用与产品相同的 eig.recommend，固定种子（由运行代码决定）和样本数（公开表里预先规定），结果可复算；
- 危险的菜单项（例如真实弹簧上挂 350 g）不选；回执只记选中的项目、基础读数指纹、种子标记和样本数
  （不记排名和各项的信息增益，免得透露引擎眼下偏向哪个原因；分析时会复算）；
- 推荐的项目和最佳固定测量或简单重复相同时只做一次，槽位合并标在同一组行上；
- 屏幕上只显示要做的动作，不显示后验和任何原因，学生保持不知道引擎的猜测。

命令：python -m lab_detective.fieldkit.live <记录表.csv> --public <公开表.json>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import zlib
from pathlib import Path

import numpy as np

from .catalog import FIELD_EXPERIMENTS, SLOT_LABEL, TWIN_PREFIX, build_engines, howto
from .sheet import PENDING, SKIP, Row, Sheet, base_hash, fmt_num, read_sheet, write_sheet


def eig_seed(code: str, slot: str) -> int:
    return zlib.crc32(f"fieldkit|{code}|{slot}".encode())


def engine_config(public: dict, sheet: Sheet) -> dict:
    """公开表里的配置，被记录表表头里学生实际填的配置覆盖。"""
    info = public["experiments"][sheet.meta["experiment"]]
    cfg = dict(info["config"])
    cfg.update({k: v for k, v in sheet.config.items() if v != ""})
    return cfg


def sheet_engines(public: dict, sheet: Sheet) -> dict:
    e = sheet.meta["experiment"]
    return build_engines(e, engine_config(public, sheet), public["experiments"][e].get("spec"))


def base_dataset(engine, sheet: Sheet):
    from ..labsim.core import Dataset, Observation

    obs = []
    for r in sheet.base_rows():
        obs.append(Observation(r.kind, float(r.setting), (float(r.value),)))
    return Dataset(engine.id, obs, None, {"stage": "base"})


def _check_base(sheet: Sheet) -> None:
    for r in sheet.base_rows():
        try:
            v = float(r.value)
        except ValueError as e:
            raise ValueError(f"基础方案第 {r.step} 行还没有读数（或不是数字）：{r.value!r}") from e
        if not math.isfinite(v):
            raise ValueError(f"基础方案第 {r.step} 行读数不是有限数")
    for r in sheet.followup_rows():
        if r.value.strip():
            raise ValueError("追加测量已经有读数了：推荐必须在做任何追加测量之前算出，这次运行请如实记录并告诉开发者")


def recommend_item(engine, ds, allowed: tuple[str, ...], n_samples: int, seed: int) -> dict:
    from ..eig import recommend

    scores = recommend(engine, ds, n_samples=n_samples, rng=np.random.default_rng(seed))
    ranking = [{"item": s.item_id, "eig_bits": round(float(s.eig_bits), 6)} for s in scores]
    chosen = next((s.item_id for s in scores if s.item_id in allowed), None)
    return {"chosen": chosen, "unrestricted_top": scores[0].item_id, "ranking": ranking}


def _fill(sheet: Sheet, slot: str, item: str, plan, units: dict, prefix_twin: set[str]) -> None:
    """把待定的槽位换成具体测量；与已有组相同就合并槽位。"""
    plan = [[str(k), float(s)] for k, s in plan]
    kinds = [(TWIN_PREFIX + k if k in prefix_twin else k) for k, _ in plan]
    groups: dict[str, list[Row]] = {}
    for r in sheet.followup_rows():
        if r.item != PENDING:
            groups.setdefault(r.stage, []).append(r)
    for stage, rows in groups.items():
        if [[r.kind, float(r.setting)] for r in rows] == [[k, s] for k, (_, s) in zip(kinds, plan)]:
            for r in rows:
                r.stage = stage + "+" + slot
            sheet.rows = [r for r in sheet.rows if not (r.item == PENDING and r.stage == slot)]
            return
    out, step = [], 1
    for r in sheet.rows:
        if r.item == PENDING and r.stage == slot:
            for k, (_, s) in zip(kinds, plan):
                u = units.get(k, [""])
                out.append(Row(0, slot, item, k, fmt_num(s), "", u[0] if u else ""))
        else:
            out.append(r)
    for r in out:
        r.step = step
        step += 1
    sheet.rows = out


def run_live(sheet_path: str | Path, public: dict, now: str | None = None) -> dict:
    sheet_path = Path(sheet_path)
    sheet = read_sheet(sheet_path)
    e = sheet.meta.get("experiment", "")
    run = next((r for r in public["runs"] if r["code"] == sheet.meta.get("code")), None)
    if run is None or run["experiment"] != e:
        raise ValueError("记录表的代码或实验与公开表对不上")
    _check_base(sheet)
    info = public["experiments"][e]
    engines = sheet_engines(public, sheet)
    n_samples = int(public["analysis_spec"]["eig_n_samples"])
    bh = base_hash(sheet)
    receipt_path = sheet_path.with_suffix(".eig.json")
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text(encoding="utf-8"))
        if old["base_sha256"] != bh:
            raise ValueError("基础读数在推荐之后被改过；请不要改，告诉开发者")
        return old
    receipt = {"code": run["code"], "run_id": run["run_id"], "experiment": e, "base_sha256": bh,
               "n_samples": n_samples, "slots": {},
               "created_at": now or _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    fexp = FIELD_EXPERIMENTS[e]
    for slot in run["slots"]:
        if slot not in ("eig", "twin_eig"):
            continue
        eng = engines[info["primary"]] if slot == "eig" else engines["twin"]
        allowed = tuple(info["allowed_items"] if slot == "eig" else info["twin_allowed_items"])
        ds = base_dataset(eng, sheet)
        rec = recommend_item(eng, ds, allowed, n_samples, eig_seed(run["code"], slot))
        if rec["chosen"] is None:
            raise ValueError("没有可做的菜单项")
        plan = eng.menu_item(rec["chosen"]).plan
        prim_kinds = set(engines[info["primary"]].obs_kinds)
        twin_only = set() if slot == "eig" else {k for k, _ in plan if k not in prim_kinds}
        units = dict(info["units"])
        units.update({TWIN_PREFIX + k: v for k, v in info.get("twin_units", {}).items()})
        _fill(sheet, slot, rec["chosen"], plan, units, twin_only)
        receipt["slots"][slot] = {"chosen": rec["chosen"], "seed_tag": f"fieldkit|{run['code']}|{slot}",
                                  "howto": howto(fexp, eng, rec["chosen"])}
    write_sheet(sheet, sheet_path)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def todo_lines(public: dict, sheet: Sheet) -> list[str]:
    """按顺序列出要做的追加测量（只有动作）。"""
    e = sheet.meta["experiment"]
    info = public["experiments"][e]
    fexp = FIELD_EXPERIMENTS[e]
    lines, seen = [], []
    for r in sheet.followup_rows():
        if (r.stage, r.item) in seen:
            continue
        seen.append((r.stage, r.item))
        text = info["howto"].get(r.item) or info.get("twin_howto", {}).get(r.item) or fexp.howto.get(r.item, "")
        if r.item == "repeat":
            text = "按表里这几行的控制量重测一次（和基础方案同样的做法）。"
        label = " + ".join(SLOT_LABEL.get(s, s) for s in r.stages)
        lines.append(f"{len(lines) + 1}. [{label}] {text}")
    lines.append(f"做不了的项目在 value 填 {SKIP}。")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="做完基础方案后：算出信息增益推荐的追加测量")
    ap.add_argument("sheet")
    ap.add_argument("--public", required=True)
    a = ap.parse_args()
    public = json.loads(Path(a.public).read_text(encoding="utf-8"))
    run_live(a.sheet, public)
    print("已填好记录表，回执存在同名 .eig.json 文件里（不要改它）。接下来按顺序做：")
    for line in todo_lines(public, read_sheet(a.sheet)):
        print(line)


if __name__ == "__main__":
    main()
