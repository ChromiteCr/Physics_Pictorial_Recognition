"""
将 self_captured/ 和 openclaw/ 中的标注数据合并到 dataset/ 供 YOLOv8 训练使用。

用法：
    python prepare_dataset.py [--val-ratio 0.15] [--test-ratio 0.05]
    python prepare_dataset.py --exclude openclaw   # 打包训练集但不含 openclaw 来源

数据放置规则：
  - data/self_captured/images/  放自己拍摄的图片（.jpg/.png）
  - data/self_captured/labels/  放对应的 YOLO 格式标注（.txt，与图片同名）
  - data/openclaw/images/       放 OpenClaw 下载的图片
  - data/openclaw/labels/       放对应标注
  - data/video_frames/images/   放 extract_video_frames.py 抽出的视频帧
  - data/video_frames/labels/   放对应标注（autolabel_glm.py 生成 + 人工校对）

运行后生成：
  - data/dataset/images/{train,val,test}/
  - data/dataset/labels/{train,val,test}/
"""

import shutil
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SOURCES = [DATA_DIR / "self_captured", DATA_DIR / "openclaw", DATA_DIR / "video_frames"]
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
    """
    pairs 中每项为 (img_path, lbl_path, source_tag)。
    输出文件名前缀来源标签（如 self_captured_img_0001.jpg），
    避免不同来源目录下同名文件（如两边都有 img_0001.jpg）互相覆盖。
    """
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
        for img_path, lbl_path, source_tag in split_pairs:
            dest_stem = f"{source_tag}_{img_path.stem}"
            shutil.copy2(img_path, img_out / f"{dest_stem}{img_path.suffix}")
            shutil.copy2(lbl_path, lbl_out / f"{dest_stem}.txt")

    return {k: len(v) for k, v in splits.items()}


def main(val_ratio: float = 0.15, test_ratio: float = 0.05, seed: int = 42,
         exclude: list[str] = ()):
    random.seed(seed)

    sources = [s for s in SOURCES if s.name not in exclude]
    if exclude:
        print(f"排除来源: {', '.join(exclude)}")

    # 清空旧数据集
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = DATASET_DIR / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    all_pairs = []
    source_counts = {}
    for source in sources:
        if not source.exists():
            continue
        pairs = collect_pairs(source)
        source_counts[source.name] = len(pairs)
        print(f"  {source.name}: {len(pairs)} 张有效图片")
        all_pairs.extend((img, lbl, source.name) for img, lbl in pairs)

    if not all_pairs:
        print("没有找到任何标注数据，请先将图片和标注放入 self_captured/ 或 openclaw/ 目录。")
        return

    openclaw_ratio = source_counts.get("openclaw", 0) / len(all_pairs)
    if openclaw_ratio > 0.25:
        print(f"  [提醒] openclaw 网络图片占比 {openclaw_ratio:.0%}，建议控制在 25% 以内，"
              f"以自拍数据为主保证真实场景一致性")

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
    parser.add_argument("--exclude", nargs="*", default=[],
                         help="排除的来源目录名（如 --exclude openclaw），默认全部来源都用")
    args = parser.parse_args()
    main(args.val_ratio, args.test_ratio, args.seed, exclude=args.exclude)
