"""
端到端推理入口（力学版）

用法：
    python detect.py <视频或帧序列目录> --mode momentum \
        --mass1 200 --mass2 200 --collision-t 1.2 \
        [--marker-color red] [--marker-color2 green] [--weights ...] [--conf 0.25]

支持的 --mode：
    momentum            动量定理/守恒（需 --mass1 --mass2 --collision-t，双标记颜色）
    energy_conservation 机械能守恒（需 --mass）
    circular_motion     圆周运动（需 --center-x --center-y，可选 --mass）
    vibration           振动分析（高级，无需额外参数）

输出：
    <名称>_detected.jpg   首帧检测结果（标定用）
    <名称>_physics.json   物理量提取报告
    <名称>_overlay.jpg    轨迹+物理量叠加图
    <名称>_animation.gif  动画
"""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from physics import extract_physics, track_marker
from animate import draw_motion_overlay, generate_motion_animation

PROJ_DIR = Path(__file__).parent
DEFAULT_WEIGHTS = PROJ_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

CLASS_NAMES = {
    0: "cart",
    1: "track",
    2: "pendulum_bob",
    3: "spring",
    4: "string",
    5: "ruler",
}

CLASS_COLORS = {
    0: (255, 128, 0),
    1: (128, 128, 128),
    2: (0, 200, 80),
    3: (0, 120, 255),
    4: (200, 0, 200),
    5: (0, 200, 200),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_frames(source: Path) -> tuple[list, float]:
    """从视频文件或帧序列目录加载帧，返回 (frames, fps)。"""
    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        frames = [cv2.imread(str(p)) for p in paths]
        fps = 30.0  # 帧序列目录默认假设 30fps，可用 --fps 覆盖
        return frames, fps

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


def run(args):
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"找不到输入: {source}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"找不到权重文件: {weights_path}\n请先运行 train.py 完成训练。"
        )

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem if source.is_file() else source.name

    frames, fps = load_frames(source)
    if args.fps:
        fps = args.fps
    if not frames:
        raise RuntimeError("未读取到任何帧")

    # ── 1. 首帧检测（标定 + 场景识别） ──────────────────────────
    model = YOLO(str(weights_path))
    results = model(frames[0], conf=args.conf)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(cls_id, "unknown"),
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist(),
        })

    first_frame_vis = frames[0].copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = CLASS_COLORS.get(det["class_id"], (255, 255, 255))
        cv2.rectangle(first_frame_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(first_frame_vis, f"{det['class_name']} {det['confidence']:.2f}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    detected_out = out_dir / f"{stem}_detected.jpg"
    cv2.imwrite(str(detected_out), first_frame_vis)
    print(f"首帧检测: {detected_out}")

    # ── 2. 轨迹提取（经典 CV 标记跟踪） ──────────────────────────
    tracks = {"track1": track_marker(frames, fps, color=args.marker_color)}
    if args.mode == "momentum":
        tracks["track2"] = track_marker(frames, fps, color=args.marker_color2)

    # ── 3. 物理量提取 ────────────────────────────────────────────
    kwargs = {}
    if args.mode == "momentum":
        kwargs.update(mass1_g=args.mass1, mass2_g=args.mass2, collision_t=args.collision_t)
    elif args.mode == "energy_conservation":
        kwargs.update(mass_g=args.mass)
    elif args.mode == "circular_motion":
        kwargs.update(center_xy=(args.center_x, args.center_y), mass_g=args.mass)
    if args.ruler_length:
        kwargs["ruler_real_length_cm"] = args.ruler_length

    physics_data = extract_physics(args.mode, detections, tracks, fps, **kwargs)
    physics_out = out_dir / f"{stem}_physics.json"
    physics_out.write_text(json.dumps(physics_data, ensure_ascii=False, indent=2))
    print(f"物理报告: {physics_out}")

    # ── 4. 可视化叠加 + 动画 ─────────────────────────────────────
    overlay_out = out_dir / f"{stem}_overlay.jpg"
    draw_motion_overlay(frames[-1], tracks, physics_data, str(overlay_out))
    print(f"轨迹叠加: {overlay_out}")

    if "error" not in physics_data:
        anim_out = out_dir / f"{stem}_animation.gif"
        generate_motion_animation(physics_data, tracks, str(anim_out))
        print(f"动画: {anim_out}")

    print("\n--- 物理量摘要 ---")
    print(json.dumps(physics_data, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="力学探究实验识别与可视化")
    parser.add_argument("source", help="输入视频路径或帧序列图片目录")
    parser.add_argument("--mode", required=True,
                         choices=["momentum", "energy_conservation", "circular_motion", "vibration"])
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-dir", default="output")
    parser.add_argument("--fps", type=float, default=None, help="帧序列目录需手动指定")
    parser.add_argument("--marker-color", default="red", choices=["red", "green", "blue"])
    parser.add_argument("--marker-color2", default="green", choices=["red", "green", "blue"],
                         help="momentum 模式下第二个小车的标记颜色")
    parser.add_argument("--ruler-length", type=float, default=None, help="标定尺真实长度(cm)，默认20")
    # momentum
    parser.add_argument("--mass1", type=float, help="cart1 质量(g)")
    parser.add_argument("--mass2", type=float, help="cart2 质量(g)")
    parser.add_argument("--collision-t", type=float, help="碰撞发生时刻(s)")
    # energy_conservation / circular_motion
    parser.add_argument("--mass", type=float, help="物体质量(g)")
    # circular_motion
    parser.add_argument("--center-x", type=float, help="圆心像素x坐标")
    parser.add_argument("--center-y", type=float, help="圆心像素y坐标")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
