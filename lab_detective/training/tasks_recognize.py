"""识别（recognize）任务的训练样本：学生本人的照片 → 带编号框的 Set-of-Mark 图 + 识别提示 → RecognitionResult JSON。

照片来源与切分
- 切分沿用识别 bench（lab_detective/bench/recognition_bench.py）的按拍摄时段切分：只读
  results/bench/recognition/split.json，只用其中明确划在 train 一侧的照片；测试一侧的照片永远不进训练。
  split.json 不存在时拒绝生成（先运行 bench 的 split 子命令）。
- 标签：results/recognition/review.json（学生审核导出的 {id: 字母}）存在时，只用审核过的照片；
  不存在时只用规则有把握（rule_confident）的照片，并在报告里写明这只是暂定标签。
- 每张照片 3 个变体：原图 + 2 个轻度增强（裁掉 ≤ 8%、亮度 ±10%），先在原图上增强、再平移裁剪框、最后画编号框。
  三张图都写到 results/recognition/images/aug/（不改原图），样本里写 images/aug/<文件>，相对 results/recognition/，
  打包脚本按这个根目录复制图片。
- 编号框用标注框（gt_boxes）画，和推理时用 YOLO 检测框画的是同一种图（vision.som.draw_marks），不需要跑 YOLO。
- 目标：recognition_bench.gold_target（与 bench 的 gold 同一构造：字母 + 由框推出的构型线索和器材）；
  confidence 改为 0.9（审核过）或规则置信度（夹在 0.5–0.95），不教模型永远给 1.0。
- 验证集：按照片 id 的哈希取约 5%（至少 1 张），同一张照片的 3 个变体在同一边。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from ..bench.recognition_bench import gold_target, load_review, load_split, split_ids
from ..vision.letters import entry
from ..vision.prompt import PROMPT_VERSION, build_messages, prompt_hash
from ..vision.som import draw_marks
from .format import Example, sha256_file, write_jsonl

REPO = Path(__file__).resolve().parents[2]
REC_ROOT = REPO / "results" / "recognition"
MANIFEST = REC_ROOT / "manifest.jsonl"
REVIEW = REC_ROOT / "review.json"
SPLIT = REPO / "results" / "bench" / "recognition" / "split.json"
AUG_DIR = "images/aug"
VARIANTS = ("som", "a1", "a2")
MAX_CROP = 0.08
MAX_BRIGHT = 0.10
VAL_FRACTION = 0.05
TASK_CODE = 6


class NoSplit(RuntimeError):
    pass


def _read_manifest(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def eligible_photos(rows: list[dict], split: dict, review: dict[str, str] | None) -> tuple[list[dict], dict]:
    """→ ([{row, letter, status, confidence}], 计数)。只取 split 里明确在 train 一侧的照片。"""
    train_ids = set(split_ids(split, "train"))
    test_ids = set(split_ids(split, "test"))
    out, stats = [], Counter()
    for r in rows:
        if r["id"] in test_ids:
            stats["test_side_skipped"] += 1
            continue
        if r["id"] not in train_ids:
            stats["not_in_split_skipped"] += 1
            continue
        if review is not None:
            if r["id"] not in review:
                stats["not_reviewed_skipped"] += 1
                continue
            out.append({"row": r, "letter": review[r["id"]], "status": "reviewed", "confidence": 0.9})
        elif r.get("rule_confident"):
            conf = float(min(max(float(r.get("rule_confidence") or 0.0), 0.5), 0.95))
            out.append({"row": r, "letter": r["provisional_letter"], "status": "rule_confident", "confidence": conf})
        else:
            stats["not_confident_skipped"] += 1
    stats["eligible"] = len(out)
    return out, dict(stats)


def val_photo_ids(ids: list[str], frac: float = VAL_FRACTION) -> set[str]:
    """按 sha1(id) 排序取前 round(frac·n) 张（n ≥ 10 时至少 1 张）。与随机数无关，新增照片不打乱已有归属。"""
    if not ids:
        return set()
    k = int(np.floor(len(ids) * frac + 0.5))
    if len(ids) >= 10:
        k = max(k, 1)
    ranked = sorted(ids, key=lambda i: hashlib.sha1(f"rec-val|{i}".encode()).hexdigest())
    return set(ranked[:k])


def _rng(photo_id: str, variant: str) -> np.random.Generator:
    h = int(hashlib.sha1(f"{photo_id}|{variant}".encode()).hexdigest()[:15], 16)
    return np.random.default_rng([TASK_CODE, h])


def augment(im: Image.Image, boxes: list[dict], rng: np.random.Generator) -> tuple[Image.Image, list[dict], dict]:
    """轻度增强：每边裁掉总计 ≤ 8% 的宽 / 高（位置随机）、亮度 ×(1 ± ≤10%)。框平移并裁到图内，剩不到 30% 的框丢掉。"""
    W, H = im.size
    cw = int(W * rng.uniform(0.02, MAX_CROP))
    ch = int(H * rng.uniform(0.02, MAX_CROP))
    x0, y0 = int(rng.integers(0, cw + 1)), int(rng.integers(0, ch + 1))
    x1, y1 = W - (cw - x0), H - (ch - y0)
    out = im.crop((x0, y0, x1, y1))
    factor = float(1.0 + rng.uniform(-MAX_BRIGHT, MAX_BRIGHT))
    out = ImageEnhance.Brightness(out).enhance(factor)
    nb = []
    for b in boxes:
        bx1, by1, bx2, by2 = (float(v) for v in b["bbox"])
        nx1, ny1 = max(bx1 - x0, 0.0), max(by1 - y0, 0.0)
        nx2, ny2 = min(bx2 - x0, float(x1 - x0)), min(by2 - y0, float(y1 - y0))
        area0 = max(bx2 - bx1, 0) * max(by2 - by1, 0)
        area1 = max(nx2 - nx1, 0) * max(ny2 - ny1, 0)
        if area0 > 0 and area1 / area0 >= 0.3:
            nb.append({**b, "bbox": [round(nx1, 1), round(ny1, 1), round(nx2, 1), round(ny2, 1)]})
    return out, nb, {"crop": [x0, y0, x1, y1], "brightness": round(factor, 3)}


def render_variant(row: dict, image_root: Path, variant: str, write: bool = True) -> tuple[str, list[dict], list[dict], dict]:
    """→ (相对路径 images/aug/<id>_<变体>.jpg, 图例, 变换后的框, 增强参数)。"""
    src = Path(image_root) / row["file"]
    boxes = list(row.get("gt_boxes") or [])
    with Image.open(src) as im0:
        im = im0.convert("RGB")
    params: dict = {}
    if variant != "som":
        im, boxes, params = augment(im, boxes, _rng(row["id"], variant))
    rel = f"{AUG_DIR}/{row['id']}_{variant}.jpg"
    out_path = Path(image_root) / rel
    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    _, legend = draw_marks(im, boxes, out_path=out_path if write else None)
    return rel, legend, boxes, params


def _target(letter: str, boxes: list[dict], confidence: float) -> str:
    d = json.loads(gold_target(letter, boxes))
    d["confidence"] = round(float(confidence), 3)
    return json.dumps(d, ensure_ascii=False)


def build_examples(manifest: str | Path = MANIFEST, split_path: str | Path = SPLIT,
                   review_path: str | Path | None = REVIEW, image_root: str | Path = REC_ROOT,
                   write_images: bool = True) -> tuple[list[Example], list[Example], dict]:
    """→ (训练样本, 验证样本, 报告)。确定性：同样的输入得到同样的样本和图片。"""
    split_path = Path(split_path)
    if not split_path.exists():
        raise NoSplit(f"没有找到 {split_path}：先运行 python -m lab_detective.bench.recognition_bench split")
    split = load_split(split_path)
    rows = _read_manifest(Path(manifest))
    review = None
    if review_path is not None and Path(review_path).exists():
        review = load_review(review_path, [r["id"] for r in rows])
    photos, stats = eligible_photos(rows, split, review)
    val_ids = val_photo_ids([p["row"]["id"] for p in photos])
    train, val = [], []
    split_sha = sha256_file(split_path)
    for p in photos:
        r = p["row"]
        sp = split["photos"][r["id"]]
        for v in VARIANTS:
            rel, legend, boxes, params = render_variant(r, Path(image_root), v, write_images)
            msgs = build_messages(rel, legend) + [{"role": "assistant", "content": _target(p["letter"], boxes, p["confidence"])}]
            which = "val" if r["id"] in val_ids else "train"
            meta = {"experiment_id": entry(p["letter"]).experiment_id, "split": which, "seed": None,
                    "photo_id": r["id"], "variant": v, "augment": params, "letter": p["letter"],
                    "label_status": p["status"], "session": sp.get("session"),
                    "recognition_split_status": split.get("status"), "recognition_split_sha256": split_sha,
                    "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash(), "image_sha1": r.get("sha1")}
            ex = Example(f"rec-{r['id']}-{v}", "recognize", "photo", msgs, meta)
            ex.validate()
            (val if which == "val" else train).append(ex)
    letters = Counter(p["letter"] for p in photos)
    warnings = []
    if review is None:
        warnings.append("没有 review.json：只用了规则有把握的照片，标签是暂定的；学生审核导出 review.json 后请重新生成")
    if len(letters) <= 1:
        warnings.append(f"只有一个类别 {dict(letters)}：这份识别数据会教模型永远回答同一个字母，审核更多照片前不宜单独训练")
    if split.get("status") != "confirmed":
        warnings.append("识别切分还是草稿（split.json status=draft）：间隔或种子改了就必须重新生成识别样本")
    report = {**stats, "photos_train": len(photos) - len(val_ids), "photos_val": len(val_ids),
              "train": len(train), "val": len(val), "letters": dict(letters),
              "label_source": "review.json" if review is not None else "rule_confident",
              "split_status": split.get("status"), "split_params": split.get("params"), "warnings": warnings}
    return train, val, report


def pool_size(split: str = "train", **kw) -> int:
    tr, va, _ = build_examples(write_images=False, **kw)
    return len(tr if split == "train" else va)


def generate_recognize(out: str | Path, n: int | None = None, quiet: bool = False, **kw) -> dict:
    """写 <out>/train/recognize.jsonl、<out>/val/recognize.jsonl（generate.py 调用）。n 截断训练条数（None = 全部）。"""
    train, val, rep = build_examples(**kw)
    if n is not None and n > 0:
        train = train[:n]
    out = Path(out)
    rep["train"] = write_jsonl(out / "train" / "recognize.jsonl", train)
    rep["val"] = write_jsonl(out / "val" / "recognize.jsonl", val)
    if not quiet:
        print(f"recognize train {rep['train']:6d}  val {rep['val']:5d}  照片 {rep['eligible']}（{rep['label_source']}）",
              flush=True)
        for w in rep["warnings"]:
            print("  警告：" + w, flush=True)
    return rep
