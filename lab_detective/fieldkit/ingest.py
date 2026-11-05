"""读入、检查、冻结真实实验的记录表，转成引擎用的数据集（Dataset）。

检查（错误会阻止冻结，提示只是提醒）：
- 列和表头键必须是模板里的（不允许多出任何列，例如姓名）；
- 代码、实验、追加测量顺序与公开表一致；基础方案每一行都有有限的数字读数；
- 观测种类属于该实验的引擎，单位没被改；控制量偏离计划太多、读数不是仪器分辨率的整数倍只提示；
- 有信息增益推荐的运行必须有回执，回执里的基础读数指纹与现在的基础读数一致（推荐之后没改过基础数据），
  推荐的项目就是表里填的项目。
冻结：在数据文件夹写 FIELD_MANIFEST.json（每个记录表和回执的 sha256、公开表的 sha256），之后任何改动都能被发现。
清单里不记录任何人的信息。

命令：
    python -m lab_detective.fieldkit.ingest check  <数据文件夹> --public <公开表>
    python -m lab_detective.fieldkit.ingest freeze <数据文件夹> --public <公开表>
    python -m lab_detective.fieldkit.ingest verify <数据文件夹>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..labsim.core import Dataset, Observation
from .catalog import TWIN_PREFIX
from .live import sheet_engines
from .sheet import PENDING, SKIP, Sheet, SheetError, base_hash, read_sheet

MANIFEST = "FIELD_MANIFEST.json"
SHARED_KINDS = ("main",)  # 主引擎和留出孪生含义相同的观测种类（单摆：摆长 cm → 30 次全振动总时间 s）


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Checked:
    path: Path
    sheet: Sheet | None
    run: dict | None
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    receipt: dict | None = None


def _num(s: str) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _kind_info(engines: dict, primary: str, kind: str):
    """→ (引擎名, 原始种类) 或 None。twin. 前缀的只给留出孪生。"""
    if kind.startswith(TWIN_PREFIX):
        k = kind[len(TWIN_PREFIX):]
        return ("twin", k) if "twin" in engines and k in engines["twin"].obs_kinds else None
    return (primary, kind) if kind in engines[primary].obs_kinds else None


def check_sheet(path: str | Path, public: dict) -> Checked:
    path = Path(path)
    try:
        sheet = read_sheet(path)
    except SheetError as e:
        return Checked(path, None, None, [str(e)])
    c = Checked(path, sheet, None)
    code, e = sheet.meta.get("code", ""), sheet.meta.get("experiment", "")
    run = next((r for r in public["runs"] if r["code"] == code), None)
    if run is None:
        c.errors.append(f"代码 {code!r} 不在公开表里")
        return c
    c.run = run
    if e != run["experiment"]:
        c.errors.append(f"实验 {e!r} 与公开表（{run['experiment']}）不一致")
        return c
    if sheet.meta.get("run_id") != run["run_id"]:
        c.errors.append("运行编号与公开表不一致")
    if sheet.meta.get("followup_order", "").replace(" ", "") != ">".join(run["slots"]):
        c.errors.append("追加测量顺序（followup_order）被改过")
    info = public["experiments"][e]
    unknown_cfg = set(sheet.config) - set(info["config"])
    if unknown_cfg:
        c.errors.append(f"表头里有公开表没有的配置项：{sorted(unknown_cfg)}")
    try:
        engines = sheet_engines(public, sheet)
    except Exception as ex:  # 配置值不合法
        c.errors.append(f"仪器配置不合法：{ex}")
        return c

    base = sheet.base_rows()
    plan = info["base_plan"]
    if len(base) != len(plan):
        c.errors.append(f"基础方案应有 {len(plan)} 行，现在是 {len(base)} 行")
    for r, (k, s) in zip(base, plan):
        if r.kind != k:
            c.errors.append(f"第 {r.step} 行：基础方案的观测种类应为 {k}")
        v, st = _num(r.value), _num(r.setting)
        if v is None:
            c.errors.append(f"第 {r.step} 行：基础方案缺读数或不是数字（{r.value!r}）")
        if st is None:
            c.errors.append(f"第 {r.step} 行：控制量不是数字（{r.setting!r}）")
        elif s and abs(st - s) > 0.15 * abs(s):
            c.warnings.append(f"第 {r.step} 行：控制量 {st:g} 与计划 {s:g} 相差较多（照实记录即可）")

    stages_seen: list[str] = []
    for r in sheet.followup_rows():
        for s in r.stages:
            if s not in run["slots"]:
                c.errors.append(f"第 {r.step} 行：槽位 {s!r} 不在本次的追加测量里")
            if s not in stages_seen:
                stages_seen.append(s)
        if r.item == PENDING:
            c.errors.append(f"第 {r.step} 行：信息增益推荐还没填（做完基础方案后要运行 live）")
            continue
        ki = _kind_info(engines, info["primary"], r.kind)
        if ki is None:
            c.errors.append(f"第 {r.step} 行：观测种类 {r.kind!r} 不属于这个实验")
            continue
        if r.value.strip().lower() != SKIP and _num(r.value) is None:
            c.errors.append(f"第 {r.step} 行：读数缺失或不是数字（{r.value!r}；做不了请填 {SKIP}）")
        if _num(r.setting) is None:
            c.errors.append(f"第 {r.step} 行：控制量不是数字（{r.setting!r}）")
    missing = [s for s in run["slots"] if s not in stages_seen]
    if missing:
        c.errors.append(f"缺少追加测量槽位：{missing}")

    for r in sheet.rows:
        ki = _kind_info(engines, info["primary"], r.kind) if r.kind else None
        if ki is None:
            continue
        eng = engines[ki[0]]
        reading = eng.obs_kinds[ki[1]].readings[0]
        if r.unit != reading.unit:
            c.errors.append(f"第 {r.step} 行：单位应为 {reading.unit!r}，现在是 {r.unit!r}")
        v = _num(r.value)
        if v is not None and reading.resolution > 0:
            q = v / reading.resolution
            if abs(q - round(q)) > 1e-6 * max(1.0, abs(q)):
                c.warnings.append(f"第 {r.step} 行：读数 {r.value} 不是分辨率 {reading.resolution:g} 的整数倍")

    if any(s in ("eig", "twin_eig") for s in run["slots"]):
        rp = path.with_suffix(".eig.json")
        if not rp.exists():
            c.errors.append("缺少信息增益推荐的回执（.eig.json）")
        else:
            rc = json.loads(rp.read_text(encoding="utf-8"))
            c.receipt = rc
            if rc.get("base_sha256") != base_hash(sheet):
                c.errors.append("基础读数在推荐之后被改过（与回执里的指纹不一致）")
            for slot, rec in rc.get("slots", {}).items():
                items = {r.item for r in sheet.followup_rows() if slot in r.stages}
                if items != {rec["chosen"]}:
                    c.errors.append(f"槽位 {slot} 填的项目 {sorted(items)} 与回执里的推荐 {rec['chosen']} 不一致")
    return c


def find_sheets(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*.csv") if p.is_file())


def check_folder(folder: str | Path, public: dict) -> dict:
    folder = Path(folder)
    checked = [check_sheet(p, public) for p in find_sheets(folder)]
    by_code: dict[str, list] = {}
    for c in checked:
        if c.run:
            by_code.setdefault(c.run["code"], []).append(c)
    errors = {str(c.path.name): c.errors for c in checked if c.errors}
    for code, cs in by_code.items():
        if len(cs) > 1:
            errors.setdefault("（全部）", []).append(f"代码 {code} 有 {len(cs)} 个记录表")
    done = set(by_code)
    missing = [r["run_id"] for r in public["runs"] if r["code"] not in done]
    return {"checked": checked, "errors": errors,
            "warnings": {str(c.path.name): c.warnings for c in checked if c.warnings},
            "missing_runs": missing}


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def freeze(folder: str | Path, public: dict, public_path: str | Path | None = None) -> dict:
    """全部检查通过才写清单；有错误时抛出 ValueError（附错误列表）。"""
    folder = Path(folder)
    res = check_folder(folder, public)
    if res["errors"]:
        raise ValueError("记录表有错误，不能冻结：\n" + json.dumps(res["errors"], ensure_ascii=False, indent=2))
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name != MANIFEST
                   and (p.suffix == ".csv" or p.name.endswith(".eig.json")))
    manifest = {
        "kind": "fieldkit-data",
        "public_sha256": hashlib.sha256(Path(public_path).read_bytes()).hexdigest() if public_path else None,
        "commitment": public["commitment"],
        "files": [{"file": str(p.relative_to(folder)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
                  for p in files],
        "missing_runs": res["missing_runs"],
        "unhashed": {"frozen_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                     "code_commit": _git_commit()},
    }
    (folder / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def verify(folder: str | Path) -> dict:
    folder = Path(folder)
    mp = folder / MANIFEST
    if not mp.exists():
        return {"ok": False, "changed_or_missing": [], "unlisted": [], "error": "还没有冻结（没有清单）"}
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    bad = [e["file"] for e in manifest["files"]
           if not (folder / e["file"]).exists() or sha256_file(folder / e["file"]) != e["sha256"]]
    listed = {e["file"] for e in manifest["files"]}
    current = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p.name != MANIFEST
               and (p.suffix == ".csv" or p.name.endswith(".eig.json"))}
    extra = sorted(current - listed)
    return {"ok": not bad and not extra, "changed_or_missing": bad, "unlisted": extra,
            "commitment": manifest.get("commitment"), "public_sha256": manifest.get("public_sha256")}


# ── 转成数据集 ─────────────────────────────────────────────────────────────


def _obs(rows) -> list[Observation]:
    return [Observation(k, float(r.setting), (float(r.value),)) for r, k in rows]


def run_datasets(sheet: Sheet, public: dict) -> dict:
    """→ {引擎名: {"engine", "base", "single": {槽位: Dataset}, "all", "skipped": [槽位]}}。

    single：基础方案 + 这一项追加测量（多个槽位共用一项测量时，各槽位拿到同一份数据）；
    all：基础方案 + 全部追加测量（按实际顺序）。某一项填了 skip 时，该槽位没有数据，记在 skipped。
    观测种类不属于某个引擎的行在那个引擎里略过（例如单摆的 twin. 行只给留出孪生）。
    """
    e = sheet.meta["experiment"]
    info = public["experiments"][e]
    engines = sheet_engines(public, sheet)
    run = next(r for r in public["runs"] if r["code"] == sheet.meta["code"])
    out = {}
    for name, eng in engines.items():
        def own(r, name=name, eng=eng):
            if r.kind.startswith(TWIN_PREFIX):
                k = r.kind[len(TWIN_PREFIX):]
                return k if name == "twin" and k in eng.obs_kinds else None
            if name == info["primary"] or r.kind in SHARED_KINDS:
                return r.kind if r.kind in eng.obs_kinds else None
            return None

        base = [(r, own(r)) for r in sheet.base_rows()]
        if any(k is None for _, k in base):
            continue  # 这个引擎读不了基础方案
        meta = {"code": run["code"], "run_id": run["run_id"], "engine": name}
        res = {"engine": eng, "base": Dataset(eng.id, _obs(base), None, dict(meta, stage="base")),
               "single": {}, "skipped": []}
        all_rows = list(base)
        for slot in run["slots"]:
            rows = [r for r in sheet.followup_rows() if slot in r.stages]
            if any(r.value.strip().lower() == SKIP for r in rows):
                res["skipped"].append(slot)
                continue
            mine = [(r, own(r)) for r in rows if own(r) is not None]
            if not mine:
                continue  # 这一项测量这个引擎用不上
            res["single"][slot] = Dataset(eng.id, _obs(base + mine), None, dict(meta, stage=f"base+{slot}"))
        for r in sheet.followup_rows():
            k = own(r)
            if k is not None and r.value.strip().lower() != SKIP:
                all_rows.append((r, k))
        res["all"] = Dataset(eng.id, _obs(all_rows), None, dict(meta, stage="all"))
        out[name] = res
    return out


def datasets_json(folder: str | Path, public: dict) -> list[dict]:
    """全部运行的数据集（不含真值），可直接 json.dumps。"""
    rows = []
    for p in find_sheets(Path(folder)):
        sh = read_sheet(p)
        for name, res in run_datasets(sh, public).items():
            item = {"code": sh.meta["code"], "run_id": sh.meta["run_id"], "experiment": sh.meta["experiment"],
                    "engine": name, "base": res["base"].to_dict(), "all": res["all"].to_dict(),
                    "single": {k: v.to_dict() for k, v in res["single"].items()}, "skipped": res["skipped"]}
            rows.append(item)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="真实实验记录表：检查、冻结、校验")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "freeze"):
        s = sub.add_parser(name)
        s.add_argument("folder")
        s.add_argument("--public", required=True)
    v = sub.add_parser("verify")
    v.add_argument("folder")
    a = ap.parse_args()
    if a.cmd == "verify":
        print(json.dumps(verify(a.folder), ensure_ascii=False, indent=2))
        return
    public = json.loads(Path(a.public).read_text(encoding="utf-8"))
    if a.cmd == "check":
        res = check_folder(a.folder, public)
        print(json.dumps({"errors": res["errors"], "warnings": res["warnings"],
                          "missing_runs": res["missing_runs"]}, ensure_ascii=False, indent=2))
    else:
        m = freeze(a.folder, public, a.public)
        print(f"已冻结 {len(m['files'])} 个文件；把 {Path(a.folder) / MANIFEST} 的 sha256 抄进实验记录本（附日期）。")
        if m["missing_runs"]:
            print(f"没有做的运行：{', '.join(m['missing_runs'])}（分析时如实列出）")


if __name__ == "__main__":
    main()
