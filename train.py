"""
YOLOv8 训练脚本

使用前：
  1. 将图片放入 data/self_captured/images/ 和 data/openclaw/images/
  2. 将 YOLO 格式标注放入对应的 labels/ 目录
  3. 运行 python prepare_dataset.py 合并数据集
  4. 运行 python train.py

输出：
  runs/detect/train/weights/best.pt   最优权重
  runs/detect/train/weights/last.pt   最后一轮权重
"""

from pathlib import Path
from ultralytics import YOLO

PROJ_DIR = Path(__file__).parent
DATA_YAML = PROJ_DIR / "data" / "dataset.yaml"
RUNS_DIR = PROJ_DIR / "runs"

# 训练配置
MODEL = "yolov8s.pt"   # yolov8s（小）；精度不足时换 yolov8m.pt
EPOCHS = 100
IMG_SIZE = 640
BATCH = 16
WORKERS = 4
DEVICE = 0             # GPU 编号；无 GPU 则设为 "cpu"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"找不到数据集配置: {DATA_YAML}")

    # 检查训练集是否有图片
    train_img_dir = PROJ_DIR / "data" / "dataset" / "images" / "train"
    imgs = list(train_img_dir.glob("*.*")) if train_img_dir.exists() else []
    if not imgs:
        print("训练集为空，请先运行 prepare_dataset.py")
        return

    print(f"开始训练，训练集共 {len(imgs)} 张图片")
    model = YOLO(MODEL)
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project=str(RUNS_DIR / "detect"),
        name="train",
        exist_ok=False,
        pretrained=True,
        augment=True,
        # 数据增强参数
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
    )

    best = RUNS_DIR / "detect" / "train" / "weights" / "best.pt"
    print(f"\n训练完成，最优权重: {best}")
    print(f"mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
