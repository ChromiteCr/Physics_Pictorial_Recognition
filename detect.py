"""
端到端推理入口

用法：
    python detect.py <图片路径> [--weights runs/detect/train/weights/best.pt] [--conf 0.25]

输出：
    <图片名>_detected.jpg    带 bounding box 的检测结果
    <图片名>_physics.json    物理量提取报告
    <图片名>_animation.gif   力图动画
"""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from physics import extract_physics
from animate import draw_force_overlay, generate_spring_animation

PROJ_DIR = Path(__file__).parent
DEFAULT_WEIGHTS = PROJ_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

CLASS_NAMES = {
    0: "wooden_block",
    1: "iron_block",
    2: "spring",
    3: "string",
    4: "dynamometer",
}

CLASS_COLORS = {
    0: (255, 128, 0),    # 橙色 - 木块
    1: (128, 128, 128),  # 灰色 - 铁块
    2: (0, 200, 80),     # 绿色 - 弹簧
    3: (200, 0, 200),    # 紫色 - 细绳
    4: (0, 120, 255),    # 蓝色 - 弹簧测力计
}


def run(image_path: str, weights: str, conf: float, save_dir: str):
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"找不到图片: {img_path}")

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"找不到权重文件: {weights_path}\n请先运行 train.py 完成训练。"
        )

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    # ── 1. 目标检测 ──────────────────────────────────────────────
    model = YOLO(str(weights_path))
    results = model(str(img_path), conf=conf)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(cls_id, "unknown"),
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
        })

    # 保存检测结果图
    img = cv2.imread(str(img_path))
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = CLASS_COLORS.get(det["class_id"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    detected_out = out_dir / f"{stem}_detected.jpg"
    cv2.imwrite(str(detected_out), img)
    print(f"检测结果: {detected_out}")

    # ── 2. 物理量提取 ────────────────────────────────────────────
    physics_data = extract_physics(detections, img.shape)
    physics_out = out_dir / f"{stem}_physics.json"
    physics_out.write_text(json.dumps(physics_data, ensure_ascii=False, indent=2))
    print(f"物理报告: {physics_out}")

    # ── 3. 动画生成 ──────────────────────────────────────────────
    overlay_out = out_dir / f"{stem}_force_overlay.jpg"
    draw_force_overlay(img_path, physics_data, str(overlay_out))
    print(f"力图叠加: {overlay_out}")

    if physics_data.get("spring"):
        anim_out = out_dir / f"{stem}_animation.gif"
        generate_spring_animation(physics_data["spring"], str(anim_out))
        print(f"弹簧动画: {anim_out}")

    print("\n--- 物理量摘要 ---")
    print(json.dumps(physics_data, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="物理实验图片识别与可视化")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--save-dir", default="output", help="输出目录")
    args = parser.parse_args()

    run(args.image, args.weights, args.conf, args.save_dir)


if __name__ == "__main__":
    main()
