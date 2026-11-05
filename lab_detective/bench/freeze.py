"""冻结测试集：给每个文件算 SHA-256，写清单；之后任何改动都能被发现。

正式冻结必须由学生本人运行（ISEF：正式数据由学生启动），命令：
    python -m lab_detective.bench.freeze freeze <目录> --who <姓名>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
from pathlib import Path

from ..training.format import sha256_file

MANIFEST = "FROZEN_MANIFEST.json"


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def freeze(folder: str | Path, who: str, note: str = "") -> dict:
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name != MANIFEST)
    entries = [{"file": str(p.relative_to(folder)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
               for p in files]
    manifest = {
        "frozen_by": who,
        "frozen_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "code_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "note": note,
        "files": entries,
    }
    (folder / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def verify(folder: str | Path) -> dict:
    folder = Path(folder)
    manifest = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    bad = []
    for e in manifest["files"]:
        p = folder / e["file"]
        if not p.exists() or sha256_file(p) != e["sha256"]:
            bad.append(e["file"])
    current = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p.name != MANIFEST}
    extra = sorted(current - {e["file"] for e in manifest["files"]})
    return {"ok": not bad and not extra, "changed_or_missing": bad, "unlisted": extra,
            "frozen_by": manifest["frozen_by"], "frozen_at": manifest["frozen_at"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="冻结 / 校验测试集")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("folder")
    f.add_argument("--who", required=True)
    f.add_argument("--note", default="")
    v = sub.add_parser("verify")
    v.add_argument("folder")
    a = ap.parse_args()
    if a.cmd == "freeze":
        m = freeze(a.folder, a.who, a.note)
        print(f"已冻结 {len(m['files'])} 个文件；请把 {Path(a.folder) / MANIFEST} 的哈希抄进研究计划（附日期）。")
    else:
        print(json.dumps(verify(a.folder), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
