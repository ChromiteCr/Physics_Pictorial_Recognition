"""Set-of-Mark：在照片上画带编号的框，缩放到长边 ≤ 768 px。

图上只画数字（PIL 默认字体没有汉字），器材中文名放进文字图例（legend_text）。
输入框的坐标系与 detect.py 一致：cv2 读图时已按 EXIF 方向转正，所以这里也先 exif_transpose。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .cues import YOLO_CLASS_ZH
from .schema import APPARATUS

MAX_SIDE = 768
JPEG_QUALITY = 85
# 高对比度的循环色（编号 1 用第 0 个颜色）
PALETTE: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40),
)


def load_image(image) -> Image.Image:
    """路径或 PIL 图 → 方向已转正的 RGB 图。"""
    im = image if isinstance(image, Image.Image) else Image.open(image)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def resize_long_side(im: Image.Image, max_side: int = MAX_SIDE) -> tuple[Image.Image, float]:
    """等比缩小到长边 ≤ max_side（不放大），返回 (新图, 缩放比例)。"""
    w, h = im.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return im, s


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # 老版本 Pillow 没有 size 参数
        return ImageFont.load_default()


def draw_marks(image, detections: list[dict], out_path=None, max_side: int = MAX_SIDE):
    """画编号框。

    Args:
        image: 路径或 PIL 图（原尺寸，坐标系与 detections 的 bbox 一致）
        detections: [{"class_name", "bbox": [x1, y1, x2, y2]}, ...]（原图像素坐标）
        out_path: 给出时另存为 JPEG（q85）
    Returns:
        (PIL 图, 图例 [{"id", "class_name", "name_zh", "bbox"}])；bbox 已换到缩放后的坐标
    """
    im, s = resize_long_side(load_image(image), max_side)
    W, H = im.size
    draw = ImageDraw.Draw(im)
    lw = max(2, round(max(W, H) / 256))
    font = _font(max(12, round(max(W, H) / 32)))
    legend = []
    for i, d in enumerate([d for d in detections if d.get("bbox")], start=1):
        x1, y1, x2, y2 = (float(v) * s for v in d["bbox"])
        x1, x2 = sorted((min(max(x1, 0), W - 1), min(max(x2, 0), W - 1)))
        y1, y2 = sorted((min(max(y1, 0), H - 1), min(max(y2, 0), H - 1)))
        color = PALETTE[(i - 1) % len(PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        tag = str(i)
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), tag, font=font)
        tw, th = tx1 - tx0 + 2 * lw, ty1 - ty0 + 2 * lw
        lx, ly = x1, y1 - th if y1 - th >= 0 else y1  # 标签放框的左上角外侧，贴边时放进框里
        draw.rectangle([lx, ly, lx + tw, ly + th], fill=color)
        draw.text((lx + lw - tx0, ly + lw - ty0), tag, fill=(255, 255, 255), font=font)
        name = d["class_name"]
        legend.append({"id": i, "class_name": name,
                       "name_zh": YOLO_CLASS_ZH.get(name) or APPARATUS.get(name, name),
                       "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, "JPEG", quality=JPEG_QUALITY)
    return im, legend


def legend_text(legend: list[dict]) -> str:
    """编号框图例（提示词用）：'1 = 小车（cart）'，每行一个。"""
    if not legend:
        return "（检测器没有找到已知器材，图上没有编号框）"
    return "\n".join(f"{e['id']} = {e['name_zh']}（{e['class_name']}）" for e in legend)
