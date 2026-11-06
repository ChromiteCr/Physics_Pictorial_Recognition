"""打包给 5090 的数据：训练/验证样本、用到的图片、冻结的评测集、服务器脚本、清单与哈希。

上行带宽只有约 100 KB/s，所以：只打包被样本引用的图片（已缩放到长边 ≤ 768 px），
整个包 tar.gz 压缩，最后报告总大小和预计上传时间。布局与 server/README.md 一致。

用法：
    python -m lab_detective.training.package --train results/train/*.jsonl --val results/val/*.jsonl \
        --bench results/bench/physics/*.jsonl results/bench/recognition/*.jsonl \
        --images-root results/recognition --out results
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

from .format import read_jsonl, sha256_file

REPO = Path(__file__).resolve().parents[2]
UPLOAD_KBPS = 100.0


def _merge_jsonl(files: list[Path], dst: Path, counts: Counter, split: str, images: dict[str, Path],
                 warnings: list[str], images_root: Path, local_first: bool = False) -> int:
    """images: 包内相对路径 → 源文件。local_first=True（bench）时先找 bench 文件旁边的 images/（识别 bench 的带框图），
    找不到再按 images_root 解析。同一路径指向两个内容不同的源文件时报错。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dst.open("w", encoding="utf-8") as fo:
        for f in files:
            for ex in read_jsonl(f):
                ex.validate()
                if ex.task == "diagnose" and "causes" not in ex.meta and len(warnings) < 5:
                    warnings.append(f"{ex.id}: diagnose 样本缺 meta.causes（服务器打乱标签对照要用）")
                counts[(split, ex.task, ex.source)] += 1
                for rel in ex.images():
                    src = f.parent / rel if local_first and (f.parent / rel).exists() else images_root / rel
                    prev = images.get(rel)
                    if prev is not None and prev != src and prev.exists() and src.exists() \
                            and sha256_file(prev) != sha256_file(src):
                        raise ValueError(f"图片路径冲突：{rel} 同时指向 {prev} 和 {src}（内容不同）")
                    images.setdefault(rel, src)
                fo.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
                n += 1
    return n


def build_package(train: list[Path], val: list[Path], bench_files: list[Path], images_root: Path,
                  out_root: Path, tag: str | None = None) -> dict:
    """布局与 server/README.md 一致：<pkg>/train.jsonl、val.jsonl、bench/bench_*.jsonl、images/、server/。
    样本里的图片路径（images/...）相对 images_root（bench 先找 bench 文件旁边），打包后相对数据包根目录。"""
    tag = tag or _dt.date.today().strftime("%Y%m%d")
    pkg = out_root / f"package_{tag}"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    counts: Counter = Counter()
    images: dict[str, Path] = {}
    warnings: list[str] = []
    _merge_jsonl(train, pkg / "train.jsonl", counts, "train", images, warnings, images_root)
    if val:
        _merge_jsonl(val, pkg / "val.jsonl", counts, "val", images, warnings, images_root)
    for bf in bench_files:
        name = bf.name if bf.name.startswith("bench_") else f"bench_{bf.parent.name}_{bf.name}"
        _merge_jsonl([bf], pkg / "bench" / name, counts, "bench", images, warnings, images_root, local_first=True)
    missing = []
    for rel in sorted(images):
        src = images[rel]
        if not src.exists():
            missing.append(rel)
            continue
        dst = pkg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if (REPO / "server").exists():
        shutil.copytree(REPO / "server", pkg / "server", ignore=shutil.ignore_patterns("__pycache__"))
    files = sorted(p for p in pkg.rglob("*") if p.is_file())
    manifest = {
        "tag": tag,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "counts": [{"split": s, "task": t, "source": src, "n": n} for (s, t, src), n in sorted(counts.items())],
        "n_images": len(images) - len(missing),
        "missing_images": missing,
        "warnings": warnings,
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
              "n_images": manifest["n_images"], "missing_images": len(missing), "warnings": warnings}
    (out_root / f"package_{tag}.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="打包给 5090 的训练数据")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--bench", nargs="*", default=[], help="bench JSONL 文件（冻结后的）")
    ap.add_argument("--images-root", default=str(REPO / "results" / "recognition"))
    ap.add_argument("--out", default=str(REPO / "results"))
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    rep = build_package([Path(p) for p in a.train], [Path(p) for p in a.val],
                        [Path(p) for p in a.bench], Path(a.images_root), Path(a.out), a.tag)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
