"""
检测器训练脚本（力学版）

支持两种检测器，二选一：
  - YOLO（默认，yolov8s）：小数据集收敛快、稳，首选
  - RT-DETR（transformer 检测器）：满足「用 ViT/transformer」要求的安全替代，
    已在 COCO 预训练，可直接微调；显存占用更高，RTX 5090 32GB 足够

硬件提示（重要）：
  租赁的 RTX 5090 是 Blackwell 架构（sm_120），需要 CUDA 12.8+ 的 PyTorch。
  本机 anaconda 自带的 torch 大概率不支持 5090，请在租赁机上重建环境：
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install ultralytics

使用前：
  1. 将图片放入 data/self_captured/images/ 和 data/openclaw/images/
     （自拍为主，网络图片只作少量补充，详见 dataset.yaml 注释）
  2. 将 YOLO 格式标注放入对应的 labels/ 目录
     （可先用 autolabel.py 借 Grounding DINO 自动预标注）
  3. 运行 python prepare_dataset.py 合并数据集
  4. 运行 python train.py

输出：
  runs/detect/train/weights/best.pt   最优权重
  runs/detect/train/weights/last.pt   最后一轮权重
"""

from pathlib import Path

PROJ_DIR = Path(__file__).parent
DATA_YAML = PROJ_DIR / "data" / "dataset.yaml"
RUNS_DIR = PROJ_DIR / "runs"

# ── 训练配置 ──────────────────────────────────────────────────────
USE_RTDETR = False     # False=YOLO；True=RT-DETR（transformer，显存更高）
MODEL = "yolov8s.pt"   # yolov8s（小）；精度不足时换 yolov8m.pt / yolo11s.pt
RTDETR_MODEL = "rtdetr-l.pt"

EPOCHS = 100
IMG_SIZE = 640
BATCH = 32             # RTX 5090 32GB 下 yolov8s@640 可用 32~64；OOM 则调小
WORKERS = 8
DEVICE = 0             # 租赁 GPU 编号；本机无 GPU 冒烟测试时设为 "cpu"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"找不到数据集配置: {DATA_YAML}")

    # 检查训练集是否有图片
    train_img_dir = PROJ_DIR / "data" / "dataset" / "images" / "train"
    imgs = list(train_img_dir.glob("*.*")) if train_img_dir.exists() else []
    if not imgs:
        print("训练集为空，请先运行 prepare_dataset.py")
        return

    print(f"开始训练（{'RT-DETR' if USE_RTDETR else 'YOLO'}），训练集共 {len(imgs)} 张图片")

    if USE_RTDETR:
        from ultralytics import RTDETR
        model = RTDETR(RTDETR_MODEL)
    else:
        from ultralytics import YOLO
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
        # 数据增强：运动轨迹由经典 CV 单独提取，与 YOLO 检测的静态器件无关，
        # 故色彩/几何增强可放心开启，提升小数据集泛化
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
    map50 = results.results_dict.get("metrics/mAP50(B)")
    if map50 is not None:
        print(f"mAP@0.5: {map50:.4f}")


if __name__ == "__main__":
    main()
