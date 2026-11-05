"""打包给 5090 的数据：训练/验证样本、用到的图片、冻结的评测集、服务器脚本、清单与哈希。

上行带宽只有约 100 KB/s，所以：只打包被样本引用的图片（已缩放到长边 ≤ 768 px），
JSONL 用 gzip，最后报告总大小和预计上传时间。

用法：
    python -m lab_detective.training.package --train results/train/*.jsonl --val results/val/*.jsonl \
        --bench results/bench --images-root results --out results/package
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

from .format import read_jsonl, sha256_file

REPO = Path(__file__).resolve().parents[2]
UPLOAD_KBPS = 100.0


def _copy_jsonl_gz(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fi, gzip.open(dst, "wb", compresslevel=9) as fo:
        shutil.copyfileobj(fi, fo)


def build_package(train: list[Path], val: list[Path], bench_dir: Path | None, images_root: Path,
                  out_root: Path, tag: str | None = None) -> dict:
    tag = tag or _dt.date.today().strftime("%Y%m%d")
    pkg = out_root / f"package_{tag}"
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "data").mkdir(parents=True)
    counts: Counter = Counter()
    images: set[str] = set()
    for split, files in (("train", train), ("val", val)):
        for f in files:
            for ex in read_jsonl(f):
                counts[(split, ex.task, ex.source)] += 1
                images.update(ex.images())
            _copy_jsonl_gz(f, pkg / "data" / split / (f.name + ".gz"))
    missing = []
    for rel in sorted(images):
        src = images_root / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = pkg / "data" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if bench_dir and bench_dir.exists():
        shutil.copytree(bench_dir, pkg / "bench")
    if (REPO / "server").exists():
        shutil.copytree(REPO / "server", pkg / "server", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO / "lab_detective" / "training" / "format.py", pkg / "server" / "ppr_format.py") \
        if (pkg / "server").exists() else None
    files = sorted(p for p in pkg.rglob("*") if p.is_file())
    manifest = {
        "tag": tag,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "counts": [{"split": s, "task": t, "source": src, "n": n} for (s, t, src), n in sorted(counts.items())],
        "n_images": len(images) - len(missing),
        "missing_images": missing,
        "files": [{"file": str(p.relative_to(pkg)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
                  for p in files],
    }
    (pkg / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tar_path = out_root / f"package_{tag}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(pkg, arcname=pkg.name)
    size = tar_path.stat().st_size
    report = {"package_dir": str(pkg), "tarball": str(tar_path), "bytes": size,
              "upload_minutes_at_100KBps": round(size / 1024 / UPLOAD_KBPS / 60, 1),
              "sha256": sha256_file(tar_path), "counts": manifest["counts"],
              "n_images": manifest["n_images"], "missing_images": len(missing)}
    (out_root / f"package_{tag}.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="打包给 5090 的训练数据")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--bench", default=None)
    ap.add_argument("--images-root", default=str(REPO / "results"))
    ap.add_argument("--out", default=str(REPO / "results"))
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    rep = build_package([Path(p) for p in a.train], [Path(p) for p in a.val],
                        Path(a.bench) if a.bench else None, Path(a.images_root), Path(a.out), a.tag)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
