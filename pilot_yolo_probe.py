"""
预实验：用现成的 COCO 预训练 YOLO 权重，直接识别一张力学实验照片，
观察通用 COCO 80 类能否覆盖砝码槽木块/弹簧测力计/导轨/细线/尺子等专用器材。

预期：COCO 没有这些专用类别，大概率漏检或错认成泛化类别（如 book/bottle），
用来佐证「需要用 autolabel.py + 自建 6 类数据集微调 YOLO」这一决策，
本身不是最终识别方案。

用法：
    python pilot_yolo_probe.py [图片路径，默认 testi.png] [--model yolov8n.pt] [--conf 0.15]

输出：
    output/<图片名>_pilot_detected.jpg   带 bbox 的检测结果图
    控制台打印每个检测框的 COCO 类别、置信度、坐标，并与期望的力学器材类别对照
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJ_DIR = Path(__file__).parent
DEFAULT_IMAGE = PROJ_DIR / "testi.png"

# 本项目最终需要识别的力学器材（对照组，来自 data/dataset.yaml 的 6 类；
# pendulum_bob 已删除，ruler 仅训练不参与 detect.py 输出，此处仍列出供参考）
EXPECTED_ELEMENTS = ["cart/小车", "track/导轨", "spring/裸弹簧(已停用)",
                     "string/细线", "ruler/标定尺(仅训练)", "dynamometer/弹簧测力计"]


def run(image_path: Path, model_name: str, conf: float, save_dir: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片: {image_path}")

    save_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)  # 首次运行会自动下载 COCO 预训练权重（需联网）

    results = model(str(image_path), conf=conf)[0]
    img = cv2.imread(str(image_path))

    print(f"模型: {model_name}（COCO 80类预训练，未针对本项目器材微调）")
    print(f"图片: {image_path.name}  尺寸: {img.shape[1]}x{img.shape[0]}")
    print(f"检测到 {len(results.boxes)} 个目标：\n")

    detected_names = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detected_names.append(name)
        print(f"  - {name:15s} conf={confidence:.2f}  bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, f"{name} {confidence:.2f}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out_path = save_dir / f"{image_path.stem}_pilot_detected.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"\n标注图已保存: {out_path}")

    print("\n--- 与本项目所需的力学器材类别对照 ---")
    print(f"期望识别: {', '.join(EXPECTED_ELEMENTS)}")
    if detected_names:
        print(f"COCO 实际识别到: {', '.join(sorted(set(detected_names)))}")
    else:
        print("COCO 实际识别到: 无（0 个目标通过置信度阈值）")
    print("结论: COCO 预训练类别本身不包含弹簧测力计/导轨/砝码槽木块等专用器材，"
          "预计只能识别到少量泛化类别或完全漏检——"
          "这正是需要用 autolabel.py 预标注 + 自建 6 类数据集微调 YOLO 的原因。")


def main():
    parser = argparse.ArgumentParser(description="预实验：用COCO预训练YOLO识别力学实验照片")
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE), help="图片路径")
    parser.add_argument("--model", default="yolov8n.pt", help="预训练权重（yolov8n/yolov8s等）")
    parser.add_argument("--conf", type=float, default=0.15, help="置信度阈值（预实验调低以观察弱检测）")
    parser.add_argument("--save-dir", default="output", help="输出目录")
    args = parser.parse_args()

    run(Path(args.image), args.model, args.conf, PROJ_DIR / args.save_dir)


if __name__ == "__main__":
    main()
