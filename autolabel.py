"""
零样本自动预标注（提精度的关键杠杆）

用 Grounding DINO（文本提示检测）给原始图片自动打框，导出 YOLO 格式标注，
人工只需快速校对，而不是从零拉框 —— 把几十张扩成几百张带标注图。

依赖：transformers
硬件：RTX 5090 32GB 可全精度跑，无需量化。

用法：
    python autolabel.py --images data/self_captured/images \
                        --out    data/self_captured/labels
然后人工在 Roboflow / labelImg 里校对，再 prepare_dataset.py 合并。
"""

import argparse
from pathlib import Path

# 文本提示 → YOLO 类别 id（与 data/dataset.yaml 对齐）
#
# 实测教训（2026-07-10，用 testi.png 验证）：单词提示词(cart/track/ball/spring/
# string/ruler)几乎测不出东西——最高置信度仅 0.13，远低于任何合理阈值。换成
# 描述性短语后同一张图立刻测出 4/6 类：
#   "wooden block with holes" 0.73 / "thin string" 0.55 /
#   "metal cylindrical weight" 0.59 / "spring scale" ~0.42
# track/ruler 仍互相混淆（真正的尺子被认成 "metal track"，导轨被认成 "ruler"），
# cart/ball 未在该图中出现，尚未验证，等拍到小车/摆球照片后需要重新测试调整。
#
# 2026-07-10 第二轮：批量看过 images/ 下 41 张图后发现"弹簧测力计"整机（外壳+
# 刻度+挂钩）跟"裸弹簧"（振动实验用的线圈本体）视觉差异很大，硬塞进同一类会
# 互相污染，新增 dynamometer(6) 类专门表示测力计整机。
#
# 2026-07-10 第三轮：spring(3) 提示词("coiled metal spring")实测几乎全是误检——
# 圆孔（砝码木块/尺子的装订孔）、卷尺内芯、螺丝螺纹都会被误认成裸弹簧线圈，
# 76 个自动框里全部删除。已决定该类暂不参与本轮训练标注，故从提示词里去掉，
# 避免继续产生噪声框；class id 3 仍保留在 dataset.yaml 里（占位，不删编号），
# 只是不再主动标注。
# cart(0) 提示词一直测不出来（自动检测 0 命中率），已在 6 张确认含小车的图上
# 改为人工手动拉框，不再依赖 Grounding DINO。
#
# 2026-07-11 第四轮：删除 pendulum_bob 类（首轮训练验证集/训练集都是 0 实例，
# 混淆矩阵整行整列空白），原提示词 "ball" 一并去掉；后续 string/ruler/
# dynamometer 的 id 相应减 1，与新版 dataset.yaml（6 类）对齐。
PROMPT_TO_CLASS = {
    "cart": 0,
    "aluminum track rail": 1,
    "thin string": 3,
    "measuring ruler with markings": 4,
    "spring scale dynamometer": 5,
}
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_grounding_dino(device: str = "cuda"):
    """加载 HuggingFace Grounding DINO（首次会自动下载权重）。"""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return processor, model


def to_yolo_line(cls_id: int, box_xyxy, w: int, h: int) -> str:
    """xyxy(像素) → YOLO 归一化 cx cy w h。"""
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def autolabel_dir(images_dir: Path, out_dir: Path, device: str = "cuda"):
    import torch
    from PIL import Image

    processor, model = load_grounding_dino(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = " . ".join(PROMPT_TO_CLASS.keys()) + " ."

    imgs = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    print(f"待标注 {len(imgs)} 张，提示词: {prompt}")

    for img_path in imgs:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[(h, w)],
        )[0]

        lines = []
        for box, label in zip(results["boxes"], results["labels"]):
            cls_id = _match_label(label)
            if cls_id is None:
                continue
            lines.append(to_yolo_line(cls_id, box.tolist(), w, h))

        (out_dir / f"{img_path.stem}.txt").write_text("\n".join(lines))
        print(f"  {img_path.name}: {len(lines)} 个框")

    print(f"\n完成。请人工校对 {out_dir} 后再运行 prepare_dataset.py")


def _match_label(label: str):
    """Grounding DINO 返回的短语可能含多词，做包含匹配。"""
    label = label.lower()
    for phrase, cid in PROMPT_TO_CLASS.items():
        if phrase in label or label in phrase:
            return cid
    return None


def main():
    parser = argparse.ArgumentParser(description="Grounding DINO 零样本自动预标注")
    parser.add_argument("--images", required=True, help="原始图片目录")
    parser.add_argument("--out", required=True, help="YOLO 标注输出目录")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    args = parser.parse_args()
    autolabel_dir(Path(args.images), Path(args.out), args.device)


if __name__ == "__main__":
    main()
