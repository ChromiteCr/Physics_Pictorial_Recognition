"""
Part 1：识别实验初状态（力学版）。

核心函数 detect_components() 被 app.py（Streamlit 网页）直接调用，也可以用下面的
命令行方式单独跑，方便调试/离线核对检测效果：

    python detect.py <初状态照片> [--video 实验视频] [--weights ...] [--conf 0.25]

输出：
    <名称>_detected.jpg    标好框的检测结果图
    <名称>_components.json 器材列表（Part 1 的输出，供人工核对/后续接入 Part 2）

[2026-07-11] 架构调整：detect.py 不再包含物理量计算和动画生成——那些现在由用户
在 Part 2 手动输入实测数据后，走 physics.compute_experiment() + animate.py 完成。
detect.py 只负责"识别器材"这一件事，外加可选的视频轨迹提取（仅供动画画参考路径，
不做任何标定）。旧的 --mode momentum/energy_conservation/... 命令行入口已移除。
"""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from physics import track_marker

PROJ_DIR = Path(__file__).parent
DEFAULT_WEIGHTS = PROJ_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

CLASS_NAMES = {
    0: "cart",
    1: "track",
    2: "spring",
    3: "string",
    4: "ruler",
    5: "dynamometer",
}

CLASS_COLORS = {
    0: (255, 128, 0),
    1: (128, 128, 128),
    2: (0, 120, 255),
    3: (200, 0, 200),
    4: (0, 200, 200),
    5: (0, 0, 255),
}

# [2026-07-11] ruler 类仍参与训练（dataset.yaml 里保留），但检测结果里主动过滤掉
# 它：首轮训练混淆矩阵显示 track 真实类 0% 被正确识别、57% 被误判成 ruler，框经常
# 横跨大半张图，画出来/报出来都是噪声。calibrate() 因此拿不到 ruler，但 Part 2
# 已经不靠它做标定了（改成用户手动输入真实物理量），不影响新架构。
EXCLUDED_FROM_DETECT = {"ruler"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def detect_components(
    image_path: str | Path,
    weights_path: str | Path | None = None,
    conf: float = 0.25,
) -> dict:
    """
    Part 1 核心入口：对一张初状态照片跑 YOLO，返回器材列表 + 标好框的图。

    Returns:
        {
          "detections": [{"class_id", "class_name", "confidence", "bbox"}, ...],
          "annotated_image": np.ndarray (BGR)，供网页 st.image() 直接展示,
          "image_shape": (h, w),
        }
    """
    weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
    if not weights_path.exists():
        raise FileNotFoundError(
            f"找不到权重文件: {weights_path}\n请先运行 train.py 完成训练。"
        )

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    model = YOLO(str(weights_path))
    results = model(img, conf=conf)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES.get(cls_id, "unknown")
        if class_name in EXCLUDED_FROM_DETECT:
            continue
        detections.append({
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
        })

    annotated = img.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = CLASS_COLORS.get(det["class_id"], (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{det['class_name']} {det['confidence']:.2f}",
                    (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return {
        "detections": detections,
        "annotated_image": annotated,
        "image_shape": img.shape[:2],
    }


def load_video_frames(source: str | Path) -> tuple[list, float]:
    """从视频文件或帧序列目录加载帧，返回 (frames, fps)。"""
    source = Path(source)
    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        frames = [cv2.imread(str(p)) for p in paths]
        return frames, 30.0  # 帧序列目录默认假设 30fps

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def extract_reference_track(
    video_path: str | Path, color: str = "red", fps_override: float | None = None
) -> dict:
    """
    Part 1 的可选视频处理：只为 Part 2 的动画提供一条参考运动路径，
    不做标定、不参与任何物理量计算（这些现在都靠用户手动输入）。

    Returns:
        {"track": list[TrackPoint], "fps": float, "last_frame": np.ndarray | None}
    """
    frames, fps = load_video_frames(video_path)
    if fps_override:
        fps = fps_override
    track = track_marker(frames, fps, color=color)
    return {"track": track, "fps": fps, "last_frame": frames[-1] if frames else None}


def main():
    parser = argparse.ArgumentParser(description="Part 1：识别实验初状态（器材检测）")
    parser.add_argument("image", help="初状态照片路径")
    parser.add_argument("--video", default=None, help="可选：实验过程视频，仅用于提取参考运动路径")
    parser.add_argument("--marker-color", default="red", choices=["red", "green", "blue"])
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-dir", default="output")
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    result = detect_components(image_path, args.weights, args.conf)

    detected_out = out_dir / f"{stem}_detected.jpg"
    cv2.imwrite(str(detected_out), result["annotated_image"])
    print(f"检测结果: {detected_out}")

    components_out = out_dir / f"{stem}_components.json"
    components_out.write_text(
        json.dumps(result["detections"], ensure_ascii=False, indent=2)
    )
    print(f"器材列表: {components_out}")

    print("\n--- 检测到的器材 ---")
    for det in result["detections"]:
        print(f"  {det['class_name']:15s} conf={det['confidence']:.2f}  bbox={det['bbox']}")

    if args.video:
        track_result = extract_reference_track(args.video, color=args.marker_color)
        n_points = len(track_result["track"])
        print(f"\n参考轨迹提取完成：{n_points} 个轨迹点（仅供 Part 2 动画参考，不用于计算）")


if __name__ == "__main__":
    main()
