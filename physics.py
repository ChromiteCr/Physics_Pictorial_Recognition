"""
物理量提取模块（阶段四实现）

当前为骨架，等 YOLOv8 训练完成后在此处填写具体计算逻辑。
接口已定义完毕，可直接被 detect.py 调用。
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Force:
    magnitude: float   # 力的大小，单位 N（牛顿）
    angle_deg: float   # 力的方向，以水平向右为 0°，逆时针为正
    origin_xy: tuple   # 力的作用点（图像坐标，像素）
    label: str         # 力的描述（如 "弹力 F=2.5N"）


@dataclass
class SpringInfo:
    deformation_cm: float    # 形变量（cm）
    natural_length_cm: float # 估计自然长度（cm，若未知则为 None）
    force_n: Optional[float] # 弹力大小（N，若未知则为 None）
    spring_constant: Optional[float]  # 弹簧系数 k（N/m）


def calibrate(detections: list[dict], img_shape: tuple) -> float:
    """
    计算像素/厘米比例（px/cm）。

    当前策略：用弹簧测力计的已知外形尺寸（约 16cm 长）估算。
    后续可改进为识别刻度或使用参考尺。

    Returns:
        px_per_cm: 像素每厘米，若无法标定则返回 None
    """
    DYNAMOMETER_REAL_LENGTH_CM = 16.0

    for det in detections:
        if det["class_name"] == "dynamometer":
            x1, y1, x2, y2 = det["bbox"]
            height_px = abs(y2 - y1)
            if height_px > 20:
                return height_px / DYNAMOMETER_REAL_LENGTH_CM
    return None


def extract_spring_deformation(
    detections: list[dict], px_per_cm: float
) -> Optional[SpringInfo]:
    """
    从弹簧 bounding box 估算形变量。

    假设弹簧竖直放置，bounding box 高度 ≈ 弹簧当前长度。
    自然长度需要在无形变参考图中校准（TODO：允许用户传入）。

    Returns:
        SpringInfo 或 None（若未检测到弹簧）
    """
    # TODO: 在有训练数据后，根据实际标注情况完善此逻辑
    spring_dets = [d for d in detections if d["class_name"] == "spring"]
    if not spring_dets or px_per_cm is None:
        return None

    spring = max(spring_dets, key=lambda d: d["confidence"])
    x1, y1, x2, y2 = spring["bbox"]
    current_length_px = abs(y2 - y1)
    current_length_cm = current_length_px / px_per_cm

    # 暂时用固定值（5cm）表示自然长度，实际需校准
    natural_length_cm = 5.0
    deformation_cm = max(0.0, current_length_cm - natural_length_cm)

    return SpringInfo(
        deformation_cm=round(deformation_cm, 2),
        natural_length_cm=natural_length_cm,
        force_n=None,       # 需要弹簧系数才能算
        spring_constant=None,
    )


def compute_force_diagram(
    detections: list[dict], px_per_cm: float
) -> list[Force]:
    """
    根据检测到的物体推断受力情况，返回力列表。

    简化规则（后续可扩展）：
    - 检测到弹簧 + 木块/铁块 → 弹力（竖直方向）
    - 检测到细绳 + 木块/铁块 → 绳的拉力（根据绳的位置判断方向）
    - 所有物体存在重力（向下）

    Returns:
        list[Force]，若检测结果不足则返回空列表
    """
    # TODO: 根据实际训练后的检测质量完善此逻辑
    forces: list[Force] = []

    block_dets = [
        d for d in detections if d["class_name"] in ("wooden_block", "iron_block")
    ]
    if not block_dets:
        return forces

    block = max(block_dets, key=lambda d: d["confidence"])
    x1, y1, x2, y2 = block["bbox"]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # 重力（向下，270°）
    forces.append(Force(
        magnitude=0.0,   # 需要质量信息，暂为 0
        angle_deg=270.0,
        origin_xy=(cx, cy),
        label="重力 G",
    ))

    # 弹簧弹力
    spring_info = extract_spring_deformation(detections, px_per_cm)
    if spring_info and spring_info.spring_constant:
        f = spring_info.spring_constant * spring_info.deformation_cm / 100
        forces.append(Force(
            magnitude=round(f, 3),
            angle_deg=90.0,
            origin_xy=(cx, y1),
            label=f"弹力 F={f:.2f}N",
        ))

    return forces


def extract_physics(detections: list[dict], img_shape: tuple) -> dict:
    """
    完整物理量提取入口，被 detect.py 调用。

    Args:
        detections: YOLOv8 检测结果列表
        img_shape: (height, width, channels)

    Returns:
        dict，包含所有提取的物理量，可直接序列化为 JSON
    """
    px_per_cm = calibrate(detections, img_shape)
    spring_info = extract_spring_deformation(detections, px_per_cm)
    forces = compute_force_diagram(detections, px_per_cm)

    result = {
        "calibration": {
            "px_per_cm": round(px_per_cm, 3) if px_per_cm else None,
            "note": "基于弹簧测力计尺寸估算" if px_per_cm else "无法标定，缺少弹簧测力计",
        },
        "spring": asdict(spring_info) if spring_info else None,
        "forces": [asdict(f) for f in forces],
        "detected_classes": list({d["class_name"] for d in detections}),
    }
    return result


# ── 简单测试 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    mock_detections = [
        {"class_name": "dynamometer", "class_id": 4, "confidence": 0.92,
         "bbox": [100, 50, 160, 210]},   # 160px 高 ≈ 16cm → 10 px/cm
        {"class_name": "spring", "class_id": 2, "confidence": 0.85,
         "bbox": [110, 220, 150, 290]},  # 70px 高 → 7cm（形变 2cm）
        {"class_name": "wooden_block", "class_id": 0, "confidence": 0.88,
         "bbox": [90, 295, 170, 350]},
    ]
    result = extract_physics(mock_detections, (480, 640, 3))
    print(json.dumps(result, ensure_ascii=False, indent=2))
