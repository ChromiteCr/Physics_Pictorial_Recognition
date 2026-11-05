"""从旧项目（只读）构建识别图片集：只用学生本人拍的照片。

    python -m lab_detective.vision.build_dataset [--src 旧 data 目录] [--out results/recognition]

来源规则（ISEF 设计 §16.5）：
- 只用 self_captured/images 和 dataset/images/*/self_captured_*（两者逐字节相同，按 sha1 去重，
  dataset 里的那份只用来查它进过 YOLO 的哪个划分）；
- 不碰 video_frames、openclaw（来源未确认）；
- *_inv 是颜色反相的增广图，不是照片，排除并记录；
- 用 COCO yolov8n（class 0 = person，conf ≥ 0.4）筛人像，含人像的不输出、记进 excluded.jsonl。
输出：images/<id>.jpg（先按 EXIF 转正，长边 ≤ 768，JPEG q85，不带 EXIF 元数据）+ manifest.jsonl + summary.json。
标签只是规则给的"暂定"值，必须由学生本人逐张核实。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from . import cues as _cues
from .letters import OTHER_LETTER, experiment_for
from .som import JPEG_QUALITY, MAX_SIDE, resize_long_side

REPO = Path(__file__).resolve().parents[2]
SRC_DEFAULT = Path("/Users/billgao/Documents/Coding/SCPKU/Physics-Pictorial-Recognition/Proj/data")
OUT_DEFAULT = REPO / "results" / "recognition"
PERSON_WEIGHTS = REPO / "assets" / "weights" / "yolov8n.pt"
PERSON_CONF = 0.4
FORBIDDEN_PARTS = ("video_frames", "openclaw")  # 来源未确认，永不读取
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check_allowed(path: Path) -> None:
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        raise ValueError(f"来源未确认的目录不能用：{path}")


def collect_sources(src: Path) -> tuple[list[dict], list[dict]]:
    """列出候选照片（按 sha1 去重）和被排除的文件。返回 (sources, excluded)。"""
    src = Path(src)
    sources: dict[str, dict] = {}
    excluded: list[dict] = []

    def add(img: Path, label: Path | None, source: str) -> str:
        _check_allowed(img)
        if img.stem.endswith("_inv"):
            excluded.append({"orig_path": str(img.relative_to(src)), "reason": "augmentation_inverted",
                             "parent": img.stem[: -len("_inv")].removeprefix("self_captured_")})
            return ""
        digest = sha1_file(img)
        if digest not in sources:
            stem = img.stem.removeprefix("self_captured_")
            sources[digest] = {"id": f"sc-{stem}", "orig_path": str(img.relative_to(src)), "source": source,
                               "src_sha1": digest, "label_path": label if label and label.exists() else None,
                               "yolo_split": None}
        return digest

    sc = src / "self_captured"
    for img in sorted((sc / "images").glob("*")):
        if img.suffix.lower() in IMAGE_EXTS:
            add(img, sc / "labels" / f"{img.stem}.txt", "self_captured")

    for split in ("train", "val", "test"):
        for img in sorted((src / "dataset" / "images" / split).glob("self_captured_*")):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            digest = add(img, src / "dataset" / "labels" / split / f"{img.stem}.txt", "dataset_self_captured")
            if digest:
                sources[digest]["yolo_split"] = split
    # 增广图在两处各出现一次，排除记录按文件名去重
    seen, uniq = set(), []
    for e in excluded:
        key = e["parent"]
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    return sorted(sources.values(), key=lambda s: s["id"]), uniq


def read_yolo_labels(path: Path | None, width: int, height: int) -> list[dict]:
    """YOLO 标注（cls cx cy w h，归一化，EXIF 转正后的坐标系）→ 像素 xyxy。"""
    if path is None or not Path(path).exists():
        return []
    boxes = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:])
        boxes.append({
            "class_id": cls,
            "class_name": _cues.YOLO_CLASSES.get(cls, "unknown"),
            "bbox": [round((cx - w / 2) * width, 1), round((cy - h / 2) * height, 1),
                     round((cx + w / 2) * width, 1), round((cy + h / 2) * height, 1)],
        })
    return boxes


def provisional_label(gt_boxes: list[dict]) -> dict:
    """用规则表给真值框打暂定实验标签（先去掉推理时不输出的类别，与 detect.py 一致）。"""
    a = _cues.analyze(gt_boxes)
    if a["candidates"]:
        top = a["candidates"][0]
        letter, exp_id = top["letter"], top["experiment_id"]
    else:
        letter, exp_id = OTHER_LETTER, experiment_for(OTHER_LETTER)
    return {
        "provisional_letter": letter,
        "provisional_experiment_id": exp_id,
        "rule_candidates": [{"letter": c["letter"], "score": c["score"]} for c in a["candidates"]],
        "rule_tied": a["tied"],
        "rule_confident": a["confident"],
        "rule_confidence": a["confidence"],
        "needs_review": not a["confident"],
        "label_status": "provisional",
    }


def load_photo(path: Path) -> Image.Image:
    """读图 → 先用 JPEG draft 降采样省内存 → EXIF 转正 → RGB → 长边 ≤ 768。"""
    im = Image.open(path)
    im.draft("RGB", (2 * MAX_SIDE, 2 * MAX_SIDE))
    im = ImageOps.exif_transpose(im).convert("RGB")
    return resize_long_side(im, MAX_SIDE)[0]


class PersonScreen:
    """COCO yolov8n 人像筛查（只加载一次，CPU）。"""

    def __init__(self, weights: Path = PERSON_WEIGHTS, conf: float = PERSON_CONF):
        os.environ.setdefault("YOLO_OFFLINE", "true")  # 关掉 ultralytics 的联网检查与统计上报
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(4)
        self.model = YOLO(str(weights))
        self.conf = conf

    def max_person_conf(self, im: Image.Image) -> float:
        r = self.model.predict(im, device="cpu", verbose=False, classes=[0], conf=0.1, imgsz=640)[0]
        confs = [float(c) for c, k in zip(r.boxes.conf.tolist(), r.boxes.cls.tolist()) if int(k) == 0]
        return max(confs, default=0.0)


def build(src: Path, out: Path, screen=None, limit: int | None = None, log=print) -> dict:
    """构建图片集。screen=None 时不做人像筛查（仅测试用；正式运行必须筛查）。"""
    src, out = Path(src), Path(out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    sources, excluded = collect_sources(src)
    if limit is not None:
        sources = sources[:limit]
    rows = []
    t0 = time.time()
    for i, s in enumerate(sources, 1):
        im = load_photo(src / s["orig_path"])
        W, H = im.size
        pconf = screen.max_person_conf(im) if screen is not None else None
        if pconf is not None and pconf >= PERSON_CONF:
            excluded.append({"id": s["id"], "orig_path": s["orig_path"], "src_sha1": s["src_sha1"],
                             "reason": "person", "person_max_conf": round(pconf, 3)})
            im.close()
            continue
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_QUALITY)
        data = buf.getvalue()
        (img_dir / f"{s['id']}.jpg").write_bytes(data)
        gt = read_yolo_labels(s["label_path"], W, H)
        rows.append({
            "id": s["id"],
            "file": f"images/{s['id']}.jpg",
            "source": s["source"],
            "orig_path": s["orig_path"],
            "sha1": hashlib.sha1(data).hexdigest(),
            "src_sha1": s["src_sha1"],
            "width": W,
            "height": H,
            "own_photo": True,
            "yolo_split": s["yolo_split"],
            "gt_boxes": gt,
            **provisional_label(gt),
            "person_screened": screen is not None,
            "person_max_conf": None if pconf is None else round(pconf, 3),
        })
        if i % 25 == 0:
            log(f"[{i}/{len(sources)}] 已处理，用时 {time.time() - t0:.0f} s")
        im.close()

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "excluded.jsonl", "w", encoding="utf-8") as f:
        for e in excluded:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    summary = {
        "src": str(src),
        "n_sources": len(sources),
        "n_kept": len(rows),
        "excluded": dict(Counter(e["reason"] for e in excluded)),
        "person_screened": screen is not None,
        "person_conf_threshold": PERSON_CONF,
        "provisional_letters": dict(sorted(Counter(r["provisional_letter"] for r in rows).items())),
        "needs_review": sum(r["needs_review"] for r in rows),
        "no_gt_boxes": sum(not r["gt_boxes"] for r in rows),
        "yolo_split": dict(Counter(str(r["yolo_split"]) for r in rows)),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="构建识别图片集（只用学生本人拍的照片）")
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT, help="旧项目 data 目录（只读）")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--person-weights", type=Path, default=PERSON_WEIGHTS)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 张（调试用）")
    args = ap.parse_args(argv)
    screen = PersonScreen(args.person_weights)
    summary = build(args.src, args.out, screen=screen, limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
