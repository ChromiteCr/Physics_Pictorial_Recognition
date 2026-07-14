"""
从 data/video/ 下的视频里抽帧，供后续人工标注/GLM 辅助标注使用。

用法：
    python extract_video_frames.py --stride 10
    python extract_video_frames.py --stride 25 --videos 38815401569-1-192.mp4

抽出的帧存到 data/video_frames/images/，文件名格式 {视频名}_f{帧号:06d}.jpg，
带视频名前缀避免跟 self_captured/openclaw 撞名，也方便回溯某帧来自哪段视频第几帧。
已存在的输出文件会跳过，可以中断后重跑，或换更小的 stride 补抽。

每次跑完会在 data/video_frames/frame_hints.json 里补上新抽出帧的空白条目
（已有条目不覆盖），供人工填写"这张图里包含哪些类别"，给 autolabel_glm.py 用。
"""

import argparse
import json
from pathlib import Path

import cv2

PROJ = Path(__file__).parent
VIDEO_DIR = PROJ / "data" / "video"
OUT_DIR = PROJ / "data" / "video_frames" / "images"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def extract_video(video_path: Path, out_dir: Path, stride: int) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [跳过] 无法打开: {video_path.name}")
        return 0, 0

    stem = video_path.stem
    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            out_path = out_dir / f"{stem}_f{frame_idx:06d}.jpg"
            if not out_path.exists():
                cv2.imwrite(str(out_path), frame)
                saved += 1
        frame_idx += 1
    cap.release()
    return frame_idx, saved


def update_frame_hints(out_dir: Path):
    """给新抽出的帧在 frame_hints.json 里补空白条目（不覆盖已有条目）。"""
    hints_path = out_dir.parent / "frame_hints.json"
    existing = {}
    if hints_path.exists():
        existing = json.loads(hints_path.read_text())

    frame_names = sorted(p.name for p in out_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    added = 0
    for name in frame_names:
        if name not in existing:
            existing[name] = []
            added += 1

    hints_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n")
    print(f"\n物体提示清单: {hints_path}（新增 {added} 条空白条目，共 {len(existing)} 条）")
    print("可以编辑这个文件，为每张图填上它包含的类别（cart/track/string/ruler/dynamometer），"
          "供 autolabel_glm.py --hints-file 使用；留空的图片会走全类别检测。")


def main():
    parser = argparse.ArgumentParser(description="从视频抽帧")
    parser.add_argument("--stride", type=int, default=10,
                         help="每 N 帧取 1 帧（默认 10，也可传 25 等）")
    parser.add_argument("--videos", nargs="*", default=None,
                         help="只处理这些文件名（默认处理 --video-dir 下全部视频）")
    parser.add_argument("--video-dir", default=str(VIDEO_DIR), help="视频所在目录")
    parser.add_argument("--out", default=str(OUT_DIR), help="抽出的帧存放目录")
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride 必须是正整数")

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.videos:
        video_paths = [video_dir / name for name in args.videos]
        missing = [p for p in video_paths if not p.exists()]
        if missing:
            parser.error(f"找不到这些视频: {[p.name for p in missing]}")
    else:
        video_paths = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)

    if not video_paths:
        print(f"{video_dir} 下没有找到视频文件")
        return

    print(f"待处理 {len(video_paths)} 个视频，stride={args.stride}，输出到 {out_dir}")
    total_frames = total_saved = 0
    for video_path in video_paths:
        n_frames, n_saved = extract_video(video_path, out_dir, args.stride)
        total_frames += n_frames
        total_saved += n_saved
        print(f"  {video_path.name}: 共 {n_frames} 帧，新抽出 {n_saved} 张（已存在的已跳过）")

    print(f"\n完成。共 {total_frames} 帧，新抽出 {total_saved} 张图片到 {out_dir}")
    update_frame_hints(out_dir)


if __name__ == "__main__":
    main()
