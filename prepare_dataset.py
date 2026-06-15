"""
将 self_captured/ 和 openclaw/ 中的标注数据合并到 dataset/ 供 YOLOv8 训练使用。

用法：
    python prepare_dataset.py [--val-ratio 0.15] [--test-ratio 0.05]

数据放置规则：
  - data/self_captured/images/  放自己拍摄的图片（.jpg/.png）
  - data/self_captured/labels/  放对应的 YOLO 格式标注（.txt，与图片同名）
  - data/openclaw/images/       放 OpenClaw 下载的图片
  - data/openclaw/labels/       放对应标注

运行后生成：
  - data/dataset/images/{train,val,test}/
  - data/dataset/labels/{train,val,test}/
"""

import shutil
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SOURCES = [DATA_DIR / "self_captured", DATA_DIR / "openclaw"]
DATASET_DIR = DATA_DIR / "dataset"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    """收集图片-标注配对，只返回两者都存在的配对。"""
    pairs = []
    img_dir = source_dir / "images"
    lbl_dir = source_dir / "labels"
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            print(f"  [跳过] 缺少标注: {img_path.name}")
    return pairs


def split_and_copy(pairs: list, val_ratio: float, test_ratio: float):
    random.shuffle(pairs)
    n = len(pairs)
    n_test = max(1, int(n * test_ratio)) if n > 5 else 0
    n_val = max(1, int(n * val_ratio)) if n > 3 else 0

    splits = {
        "test": pairs[:n_test],
        "val": pairs[n_test : n_test + n_val],
        "train": pairs[n_test + n_val :],
    }

    for split, split_pairs in splits.items():
        img_out = DATASET_DIR / "images" / split
        lbl_out = DATASET_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in split_pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

    return {k: len(v) for k, v in splits.items()}


def main(val_ratio: float = 0.15, test_ratio: float = 0.05, seed: int = 42):
    random.seed(seed)

    # 清空旧数据集
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = DATASET_DIR / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    all_pairs = []
    for source in SOURCES:
        if not source.exists():
            continue
        pairs = collect_pairs(source)
        print(f"  {source.name}: {len(pairs)} 张有效图片")
        all_pairs.extend(pairs)

    if not all_pairs:
        print("没有找到任何标注数据，请先将图片和标注放入 self_captured/ 或 openclaw/ 目录。")
        return

    counts = split_and_copy(all_pairs, val_ratio, test_ratio)
    total = sum(counts.values())
    print(f"\n数据集已生成（共 {total} 张）:")
    for split, n in counts.items():
        print(f"  {split:5s}: {n} 张")
    print(f"\n数据集路径: {DATASET_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.val_ratio, args.test_ratio, args.seed)
