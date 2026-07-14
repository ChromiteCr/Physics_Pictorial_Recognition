"""
力学探究实验物理量提取模块（力学版）

管线分两半，各用最合适的工具：
  1) 块状器件（小车/导轨/摆球/弹簧/细线/标定尺）由 YOLO 检测一帧即可，
     用于场景分类 + 像素↔厘米标定
  2) 运动轨迹（位置随时间变化）由经典 CV 逐帧提取——在运动物体上贴
     一块高对比度色块，HSV 阈值 + 质心跟踪，不依赖神经网络，速度快、鲁棒

覆盖实验：
  - 动量定理 / 动量守恒：两小车碰撞前后速度 → 动量是否守恒
  - 机械能守恒：摆球/小车下落或下滑过程中 KE+PE 是否恒定
  - 圆周运动：绳栓球做圆周运动 → 周期、线速度、向心加速度/向心力
  - 振动（高级）：弹簧振子/单摆的振幅、周期、阻尼系数

入口（被 detect.py 调用）：
    track_marker(frames, ...) -> list[TrackPoint]      # 提取轨迹
    extract_physics(mode, tracks, detections, fps, ...) -> dict   # 物理量计算
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import cv2
import numpy as np
from scipy.signal import find_peaks

G = 9.8  # m/s^2


# ── 数据结构 ──────────────────────────────────────────────────────
@dataclass
class TrackPoint:
    t: float  # 秒
    x: float  # 像素
    y: float  # 像素


# ── 0. 标定 ──────────────────────────────────────────────────────
def calibrate(detections: list[dict], ruler_real_length_cm: float = 20.0) -> Optional[float]:
    """
    用标定尺(ruler) bbox 的长边估算 px/cm。

    Returns:
        px_per_cm，若未检测到 ruler 则返回 None
    """
    for det in detections:
        if det["class_name"] == "ruler":
            x1, y1, x2, y2 = det["bbox"]
            length_px = max(abs(x2 - x1), abs(y2 - y1))
            if length_px > 20:
                return length_px / ruler_real_length_cm
    return None


# ── 1. 运动轨迹提取（经典 CV，核心信号） ───────────────────────────
def track_marker(
    frames: list[np.ndarray],
    fps: float,
    color: str = "red",
    min_area: int = 15,
) -> list[TrackPoint]:
    """
    用 HSV 颜色阈值逐帧定位色块质心，得到位置-时间序列。

    拍摄建议：在小车/摆球上贴一块高对比度色块（红/绿），背景尽量干净。

    Args:
        frames: 按时间顺序排列的 BGR 帧列表（来自视频或图片序列）
        fps:    帧率（用于换算时间戳）
        color:  "red" / "green" / "blue"
    Returns:
        TrackPoint 列表（跳过未检出色块的帧）
    """
    ranges = {
        "red": [((0, 70, 70), (10, 255, 255)), ((170, 70, 70), (180, 255, 255))],
        "green": [((35, 60, 60), (85, 255, 255))],
        "blue": [((90, 70, 70), (130, 255, 255))],
    }
    if color not in ranges:
        raise ValueError(f"不支持的标记颜色: {color}")

    points: list[TrackPoint] = []
    for i, frame in enumerate(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = None
        for lo, hi in ranges[color]:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        points.append(TrackPoint(t=i / fps, x=cx, y=cy))

    return points


# ── 2. 动量定理 / 动量守恒 ─────────────────────────────────────────
def _fit_velocity(points: list[TrackPoint], px_per_cm: float) -> tuple[float, float]:
    """对一段轨迹做线性拟合，返回 (vx, vy)，单位 cm/s。"""
    if len(points) < 2:
        return 0.0, 0.0
    t = np.array([p.t for p in points])
    x = np.array([p.x for p in points]) / px_per_cm
    y = np.array([p.y for p in points]) / px_per_cm
    vx = np.polyfit(t, x, 1)[0]
    vy = np.polyfit(t, y, 1)[0]
    return float(vx), float(vy)


def analyze_momentum(
    track1: list[TrackPoint],
    track2: list[TrackPoint],
    mass1_g: float,
    mass2_g: float,
    px_per_cm: float,
    collision_t: float,
) -> dict:
    """
    分析两小车碰撞前后的动量是否守恒。

    Args:
        track1, track2: 两车各自的完整轨迹（碰撞前后都含）
        mass1_g, mass2_g: 两车质量（克）
        collision_t: 碰撞发生的时间戳（秒），用于切分碰撞前后区间
    Returns:
        dict，含碰撞前后速度、动量、守恒误差
    """
    before1 = [p for p in track1 if p.t < collision_t]
    after1 = [p for p in track1 if p.t > collision_t]
    before2 = [p for p in track2 if p.t < collision_t]
    after2 = [p for p in track2 if p.t > collision_t]

    v1_before = _fit_velocity(before1, px_per_cm)
    v1_after = _fit_velocity(after1, px_per_cm)
    v2_before = _fit_velocity(before2, px_per_cm)
    v2_after = _fit_velocity(after2, px_per_cm)

    m1, m2 = mass1_g / 1000, mass2_g / 1000  # kg
    # 矢量动量求和（带符号），不能用速度大小相加——否则两车反向运动时会算错
    px_before = m1 * v1_before[0] / 100 + m2 * v2_before[0] / 100  # cm/s -> m/s
    py_before = m1 * v1_before[1] / 100 + m2 * v2_before[1] / 100
    px_after = m1 * v1_after[0] / 100 + m2 * v2_after[0] / 100
    py_after = m1 * v1_after[1] / 100 + m2 * v2_after[1] / 100
    p_before = math.hypot(px_before, py_before)
    p_after = math.hypot(px_after, py_after)
    error = abs(p_after - p_before) / p_before if p_before > 1e-6 else None

    return {
        "cart1": {
            "v_before_cm_s": [round(v, 2) for v in v1_before],
            "v_after_cm_s": [round(v, 2) for v in v1_after],
        },
        "cart2": {
            "v_before_cm_s": [round(v, 2) for v in v2_before],
            "v_after_cm_s": [round(v, 2) for v in v2_after],
        },
        "momentum_before_kg_m_s": round(p_before, 4),
        "momentum_after_kg_m_s": round(p_after, 4),
        "conservation_error_ratio": round(error, 4) if error is not None else None,
        "verified": (error is not None and error < 0.15),
    }


# ── 3. 机械能守恒 ─────────────────────────────────────────────────
def analyze_energy_conservation(
    track: list[TrackPoint],
    mass_g: float,
    px_per_cm: float,
) -> dict:
    """
    分析单个物体（摆球/小车）运动过程中机械能是否守恒。

    高度 h 取「图像 y 轴最低点为 0」，速度由位置差分得到。
    """
    if len(track) < 3:
        return {"status": "insufficient_points", "note": "轨迹点过少，无法计算速度"}

    t = np.array([p.t for p in track])
    x = np.array([p.x for p in track]) / px_per_cm / 100  # m
    y = np.array([p.y for p in track]) / px_per_cm / 100  # m

    y_ref = y.max()  # 图像 y 向下，最大 y = 最低位置
    h = y_ref - y     # 高度（向上为正）

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    v2 = vx ** 2 + vy ** 2

    m = mass_g / 1000
    ke = 0.5 * m * v2
    pe = m * G * h
    total = ke + pe

    error = (total.max() - total.min()) / total.mean() if total.mean() > 1e-9 else None

    return {
        "time_s": [round(float(v), 3) for v in t],
        "height_m": [round(float(v), 4) for v in h],
        "kinetic_energy_J": [round(float(v), 5) for v in ke],
        "potential_energy_J": [round(float(v), 5) for v in pe],
        "total_energy_J": [round(float(v), 5) for v in total],
        "conservation_error_ratio": round(float(error), 4) if error is not None else None,
        "verified": (error is not None and error < 0.15),
    }


# ── 4. 圆周运动 ───────────────────────────────────────────────────
def analyze_circular_motion(
    track: list[TrackPoint],
    center_xy: tuple[float, float],
    px_per_cm: float,
    mass_g: Optional[float] = None,
) -> dict:
    """
    分析绳栓球做圆周运动：周期、线速度、向心加速度、(可选)向心力。

    Args:
        center_xy: 圆心像素坐标（手动指定或由细线固定端估算）
    """
    if len(track) < 5:
        return {"status": "insufficient_points", "note": "轨迹点过少，无法拟合圆周运动"}

    cx, cy = center_xy
    t = np.array([p.t for p in track])
    x = np.array([p.x for p in track])
    y = np.array([p.y for p in track])

    radius_px = np.hypot(x - cx, y - cy)
    radius_cm = float(radius_px.mean()) / px_per_cm

    angle = np.unwrap(np.arctan2(-(y - cy), x - cx))  # 图像 y 向下，取负号转标准数学方向
    omega = float(np.polyfit(t, angle, 1)[0])  # rad/s
    period_s = 2 * math.pi / abs(omega) if abs(omega) > 1e-6 else None

    v_cm_s = abs(omega) * radius_cm
    a_centripetal_m_s2 = (omega ** 2) * (radius_cm / 100)

    result = {
        "radius_cm": round(radius_cm, 2),
        "angular_velocity_rad_s": round(omega, 3),
        "period_s": round(period_s, 3) if period_s else None,
        "linear_velocity_cm_s": round(v_cm_s, 2),
        "centripetal_acceleration_m_s2": round(a_centripetal_m_s2, 3),
    }
    if mass_g:
        result["centripetal_force_N"] = round((mass_g / 1000) * a_centripetal_m_s2, 4)
    return result


# ── 5. 振动（高级） ───────────────────────────────────────────────
def analyze_vibration(
    track: list[TrackPoint],
    px_per_cm: float,
    axis: str = "x",
) -> dict:
    """
    分析弹簧振子/单摆的振动：振幅、周期、(若有衰减)阻尼系数。

    用 scipy.signal.find_peaks 在位移-时间曲线上找波峰，
    峰值时间差 → 周期；峰值高度随时间衰减 → 阻尼。
    """
    if len(track) < 10:
        return {"status": "insufficient_points", "note": "轨迹点过少，无法分析振动"}

    t = np.array([p.t for p in track])
    raw = np.array([p.x if axis == "x" else p.y for p in track]) / px_per_cm
    signal = raw - raw.mean()  # 去均值，关于平衡位置振荡

    peaks, _ = find_peaks(signal, distance=3)
    if len(peaks) < 2:
        return {"status": "no_peaks", "note": "未找到足够波峰，检查振幅是否过小或采样率不足"}

    peak_times = t[peaks]
    peak_amps = signal[peaks]
    periods = np.diff(peak_times)
    period_s = float(np.mean(periods))
    freq_hz = 1 / period_s if period_s > 0 else None

    damping_ratio = None
    if len(peak_amps) >= 3 and np.all(peak_amps > 0):
        # 对数衰减率：ln(A_n) 随峰序号线性下降，斜率即衰减系数 γ
        log_amp = np.log(peak_amps)
        slope = np.polyfit(peak_times, log_amp, 1)[0]
        damping_ratio = round(float(-slope), 4)

    return {
        "amplitude_cm": round(float(np.abs(peak_amps).mean()), 3),
        "period_s": round(period_s, 3),
        "frequency_hz": round(freq_hz, 3) if freq_hz else None,
        "damping_coefficient": damping_ratio,
        "n_cycles_observed": len(peaks) - 1,
    }


# ── 入口 ─────────────────────────────────────────────────────────
def extract_physics(
    mode: str,
    detections: list[dict],
    tracks: dict,
    fps: float,
    **kwargs,
) -> dict:
    """
    完整力学量提取入口，被 detect.py 调用。

    Args:
        mode: "momentum" | "energy_conservation" | "circular_motion" | "vibration"
        detections: 标定/场景识别用的单帧 YOLO 检测结果
        tracks: {"track1": list[TrackPoint], "track2": list[TrackPoint] (可选)}
        fps: 帧率
        kwargs: 各模式所需的额外参数（mass_g, collision_t, center_xy 等）
    """
    px_per_cm = calibrate(detections, kwargs.get("ruler_real_length_cm", 20.0))
    result: dict = {
        "mode": mode,
        "calibration": {
            "px_per_cm": round(px_per_cm, 3) if px_per_cm else None,
            "note": "基于标定尺(ruler)估算" if px_per_cm else "无法标定，缺少 ruler 检测",
        },
        "detected_classes": sorted({d["class_name"] for d in detections}),
    }
    if px_per_cm is None:
        result["error"] = "缺少标定尺，无法换算物理量"
        return result

    if mode == "momentum":
        result["analysis"] = analyze_momentum(
            tracks["track1"], tracks["track2"],
            kwargs["mass1_g"], kwargs["mass2_g"], px_per_cm, kwargs["collision_t"],
        )
    elif mode == "energy_conservation":
        result["analysis"] = analyze_energy_conservation(
            tracks["track1"], kwargs["mass_g"], px_per_cm,
        )
    elif mode == "circular_motion":
        result["analysis"] = analyze_circular_motion(
            tracks["track1"], kwargs["center_xy"], px_per_cm, kwargs.get("mass_g"),
        )
    elif mode == "vibration":
        result["analysis"] = analyze_vibration(
            tracks["track1"], px_per_cm, kwargs.get("axis", "x"),
        )
    else:
        result["error"] = f"未知模式: {mode}"

    return result


# ══════════════════════════════════════════════════════════════════
# Part 2：实验类型注册表 + 按用户输入计算物理量
#
# [2026-07-11] 架构调整：不再指望 CV 从视频里精确测量物理量（小样本 YOLO
# 精度不够，标定链路也不稳）。真实物理量改成用户看着实物手动输入——这里的
# 计算函数全部吃一个 dict（输入字段 key -> 数值），不碰 TrackPoint/px_per_cm。
# 上面第 0-5 节的 CV 轨迹提取/标定逻辑仍保留（track_marker 可选用于动画画一条
# 参考运动路径），但不再是获得物理量数值的主路径。
# ══════════════════════════════════════════════════════════════════

@dataclass
class InputField:
    """Part 2 动态表单里的一个输入项，网页端用它生成对应的输入控件。"""
    key: str
    label: str
    unit: str = ""
    type: str = "number"          # "number" | "select"
    options: list[str] = field(default_factory=list)  # type="select" 时的选项
    required: bool = True
    default: float | str | None = None
    help: str = ""


def compute_friction(inputs: dict) -> dict:
    """
    探究摩擦力大小的影响因素（初中）。

    实验做法：用测力计沿水平方向匀速拉动木块，此时木块受力平衡，
    测力计示数直接等于摩擦力大小——不需要任何 CV 测量。
    """
    f_reading = float(inputs["dynamometer_reading_n"])
    result = {
        "friction_force_N": round(f_reading, 4),
        "surface_material": inputs.get("surface_material"),
        "formula": "匀速直线运动时二力平衡：摩擦力 f = 测力计示数 F",
    }
    weight_g = inputs.get("weight_g") or 0
    if weight_g:
        normal_force_n = float(weight_g) / 1000 * G
        if normal_force_n > 1e-6:
            result["normal_force_N"] = round(normal_force_n, 4)
            result["mu_estimated"] = round(f_reading / normal_force_n, 4)
            result["formula"] += "；若接触面水平，可估算动摩擦因数 μ = f / G"
    return result


def compute_average_speed(inputs: dict) -> dict:
    """测量平均速度（初中）：v = s / t，直接算，不需要 CV。"""
    distance_m = float(inputs["distance_m"])
    time_s = float(inputs["time_s"])
    if time_s <= 0:
        raise ValueError("时间必须大于 0")
    v = distance_m / time_s
    return {
        "average_speed_m_s": round(v, 4),
        "distance_m": distance_m,
        "time_s": time_s,
        "formula": "v = s / t",
    }


def compute_hooke_law(inputs: dict) -> dict:
    """
    弹簧测力计原理 / 胡克定律初步（初中）。

    支持两种输入：多组 (形变量, 力) 做线性拟合更准确；
    只有一组时直接 k = F / x。
    """
    measurements = inputs.get("measurements")
    if measurements and len(measurements) >= 2:
        x = np.array([m["deformation_cm"] for m in measurements]) / 100  # m
        f = np.array([m["force_n"] for m in measurements])
        k, b = np.polyfit(x, f, 1)
        return {
            "spring_constant_N_per_m": round(float(k), 4),
            "intercept_N": round(float(b), 4),
            "n_measurements": len(measurements),
            "formula": "F = kx（对多组测量数据做线性拟合，k 取斜率）",
        }

    deformation_cm = float(inputs["deformation_cm"])
    force_n = float(inputs["force_n"])
    if deformation_cm <= 0:
        raise ValueError("形变量必须大于 0")
    k = force_n / (deformation_cm / 100)
    return {
        "spring_constant_N_per_m": round(k, 4),
        "deformation_cm": deformation_cm,
        "force_n": force_n,
        "formula": "F = kx",
    }


def compute_newton_second_law(inputs: dict) -> dict:
    """
    验证牛顿第二定律（高中）。

    用户提供质量 + (拉力 或 加速度)：只给一个就推另一个；
    两个都给就对比预测加速度和实测加速度的误差，验证 F=ma。
    """
    mass_kg = float(inputs["mass_g"]) / 1000
    force_n = inputs.get("force_n")
    accel_m_s2 = inputs.get("accel_m_s2")
    force_n = float(force_n) if force_n not in (None, "") else None
    accel_m_s2 = float(accel_m_s2) if accel_m_s2 not in (None, "") else None

    if force_n is not None and accel_m_s2 is not None:
        predicted_a = force_n / mass_kg
        error = abs(predicted_a - accel_m_s2) / predicted_a if predicted_a > 1e-9 else None
        return {
            "mass_kg": round(mass_kg, 4),
            "force_N": round(force_n, 4),
            "measured_accel_m_s2": round(accel_m_s2, 4),
            "predicted_accel_m_s2": round(predicted_a, 4),
            "error_ratio": round(error, 4) if error is not None else None,
            "verified": error is not None and error < 0.15,
            "formula": "F = ma",
        }
    if force_n is not None:
        accel_m_s2 = force_n / mass_kg
        return {"mass_kg": round(mass_kg, 4), "force_N": round(force_n, 4),
                "accel_m_s2": round(accel_m_s2, 4), "derived": "accel_m_s2", "formula": "F = ma"}
    if accel_m_s2 is not None:
        force_n = mass_kg * accel_m_s2
        return {"mass_kg": round(mass_kg, 4), "force_N": round(force_n, 4),
                "accel_m_s2": round(accel_m_s2, 4), "derived": "force_N", "formula": "F = ma"}
    raise ValueError("需要提供 force_n 或 accel_m_s2 中至少一个")


def compute_momentum_conservation(inputs: dict) -> dict:
    """
    验证动量守恒定律（高中）：两车碰撞前后速度由用户直接输入
    （沿导轨方向取正负号表示方向），不再靠 CV 轨迹拟合。
    """
    m1, m2 = float(inputs["mass1_g"]) / 1000, float(inputs["mass2_g"]) / 1000
    v1_before, v2_before = float(inputs["v1_before_m_s"]), float(inputs["v2_before_m_s"])
    v1_after, v2_after = float(inputs["v1_after_m_s"]), float(inputs["v2_after_m_s"])

    p_before = m1 * v1_before + m2 * v2_before
    p_after = m1 * v1_after + m2 * v2_after
    error = abs(p_after - p_before) / abs(p_before) if abs(p_before) > 1e-6 else None

    return {
        "momentum_before_kg_m_s": round(p_before, 4),
        "momentum_after_kg_m_s": round(p_after, 4),
        "conservation_error_ratio": round(error, 4) if error is not None else None,
        "verified": error is not None and error < 0.15,
        "formula": "p = mv，碰撞前后总动量守恒：m₁v₁+m₂v₂ = m₁v₁'+m₂v₂'",
    }


def compute_energy_conservation(inputs: dict) -> dict:
    """验证机械能守恒定律（高中）：下滑高度 + 末速度由用户直接输入。"""
    mass_kg = float(inputs["mass_g"]) / 1000
    height_m = float(inputs["height_m"])
    final_speed_m_s = float(inputs["final_speed_m_s"])

    pe_initial = mass_kg * G * height_m
    ke_final = 0.5 * mass_kg * final_speed_m_s ** 2
    error = abs(ke_final - pe_initial) / pe_initial if pe_initial > 1e-9 else None

    return {
        "initial_potential_energy_J": round(pe_initial, 5),
        "final_kinetic_energy_J": round(ke_final, 5),
        "conservation_error_ratio": round(error, 4) if error is not None else None,
        "verified": error is not None and error < 0.15,
        "formula": "机械能守恒：mgh = ½mv²",
    }


# 实验类型注册表：key 是内部 id，intent_classifier.py 靠 required_components
# 做规则匹配，app.py 靠 input_fields 动态生成表单，compute 是上面对应的函数。
EXPERIMENT_TYPES: dict[str, dict] = {
    "friction": {
        "name": "探究摩擦力大小的影响因素",
        "level": "初中",
        # [2026-07-14] 被拉动的物体可以是小车(cart，在导轨上)也可以是木块
        # (wooden_block，在木板/砂纸/棉布/玻璃等不同接触面上)——两种拍摄场景
        # 都算这个实验。这里只放最基础的 dynamometer，cart/wooden_block 二选一
        # 的判断交给 intent_classifier.classify_intent() 的特判逻辑。
        "required_components": {"dynamometer"},
        "optional_components": {"cart", "wooden_block", "track", "string"},
        "input_fields": [
            InputField("dynamometer_reading_n", "弹簧测力计示数", "N"),
            InputField("surface_material", "接触面材质", type="select",
                       options=["木板", "砂纸", "棉布", "玻璃"], required=False),
            InputField("weight_g", "物块质量(可选，用于估算μ)", "g", required=False, default=0),
        ],
        "compute": compute_friction,
    },
    "average_speed": {
        "name": "测量平均速度",
        "level": "初中",
        "required_components": {"cart", "track"},
        "optional_components": set(),
        "input_fields": [
            InputField("distance_m", "运动距离", "m"),
            InputField("time_s", "运动时间", "s"),
        ],
        "compute": compute_average_speed,
    },
    "hooke_law": {
        "name": "弹簧测力计原理 / 胡克定律初步",
        "level": "初中",
        "required_components": {"dynamometer"},
        "optional_components": set(),
        "input_fields": [
            InputField("deformation_cm", "形变量", "cm"),
            InputField("force_n", "力值", "N"),
        ],
        "compute": compute_hooke_law,
    },
    "newton_second_law": {
        "name": "验证牛顿第二定律",
        "level": "高中",
        "required_components": {"cart", "track"},
        "optional_components": set(),
        "input_fields": [
            InputField("mass_g", "小车质量", "g"),
            InputField("force_n", "所受拉力(可选)", "N", required=False),
            InputField("accel_m_s2", "加速度(可选，与拉力至少填一项)", "m/s²", required=False),
        ],
        "compute": compute_newton_second_law,
    },
    "momentum_conservation": {
        "name": "验证动量守恒定律",
        "level": "高中",
        "required_components": {"cart", "track", "string"},
        "optional_components": set(),
        "input_fields": [
            InputField("mass1_g", "小车1质量", "g"),
            InputField("mass2_g", "小车2质量", "g"),
            InputField("v1_before_m_s", "小车1碰撞前速度(沿导轨方向，含正负号)", "m/s"),
            InputField("v2_before_m_s", "小车2碰撞前速度", "m/s"),
            InputField("v1_after_m_s", "小车1碰撞后速度", "m/s"),
            InputField("v2_after_m_s", "小车2碰撞后速度", "m/s"),
        ],
        "compute": compute_momentum_conservation,
    },
    "energy_conservation_input": {
        "name": "验证机械能守恒定律",
        "level": "高中",
        "required_components": {"cart", "track"},
        "optional_components": set(),
        "input_fields": [
            InputField("mass_g", "小车质量", "g"),
            InputField("height_m", "下滑高度", "m"),
            InputField("final_speed_m_s", "末速度", "m/s"),
        ],
        "compute": compute_energy_conservation,
    },
}


def list_experiment_types() -> list[dict]:
    """返回所有实验类型的概要信息（不含 compute 函数），供网页展示列表用。"""
    return [
        {"id": key, "name": v["name"], "level": v["level"],
         "required_components": sorted(v["required_components"])}
        for key, v in EXPERIMENT_TYPES.items()
    ]


def get_experiment_type(experiment_id: str) -> dict:
    if experiment_id not in EXPERIMENT_TYPES:
        raise KeyError(f"未知实验类型: {experiment_id}")
    return EXPERIMENT_TYPES[experiment_id]


def compute_experiment(experiment_id: str, inputs: dict) -> dict:
    """Part 2 统一入口：按实验类型 id + 用户输入字典算物理量。"""
    exp = get_experiment_type(experiment_id)
    result = exp["compute"](inputs)
    result["experiment_id"] = experiment_id
    result["experiment_name"] = exp["name"]
    return result


# ── 简单测试 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    # 合成一个阻尼振动轨迹做测试
    fps = 30.0
    t = np.arange(0, 4, 1 / fps)
    px_per_cm = 10.0
    x = 100 + 5 * px_per_cm * np.exp(-0.3 * t) * np.cos(2 * math.pi * 1.2 * t)
    mock_track = [TrackPoint(t=float(ti), x=float(xi), y=200.0) for ti, xi in zip(t, x)]

    mock_detections = [
        {"class_name": "ruler", "class_id": 4, "confidence": 0.9, "bbox": [10, 10, 10 + px_per_cm * 20, 30]},
        {"class_name": "spring", "class_id": 2, "confidence": 0.85, "bbox": [80, 150, 120, 250]},
    ]
    out = extract_physics(
        "vibration", mock_detections, {"track1": mock_track}, fps,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # Part 2：按用户输入计算物理量，验证公式实现是否正确
    print("\n--- Part 2 计算函数自测 ---")
    print("摩擦力:", compute_experiment("friction", {
        "dynamometer_reading_n": 1.2, "surface_material": "木板", "weight_g": 500,
    }))
    print("平均速度:", compute_experiment("average_speed", {
        "distance_m": 0.8, "time_s": 2.0,
    }))
    print("胡克定律(单组):", compute_experiment("hooke_law", {
        "deformation_cm": 2.0, "force_n": 1.0,
    }))
    print("胡克定律(多组拟合):", compute_experiment("hooke_law", {
        "measurements": [
            {"deformation_cm": 1.0, "force_n": 0.5},
            {"deformation_cm": 2.0, "force_n": 1.0},
            {"deformation_cm": 3.0, "force_n": 1.5},
        ],
    }))
    print("牛顿第二定律(验证):", compute_experiment("newton_second_law", {
        "mass_g": 200, "force_n": 0.4, "accel_m_s2": 2.05,
    }))
    print("动量守恒:", compute_experiment("momentum_conservation", {
        "mass1_g": 200, "mass2_g": 200,
        "v1_before_m_s": 0.5, "v2_before_m_s": 0.0,
        "v1_after_m_s": 0.24, "v2_after_m_s": 0.26,
    }))
    print("机械能守恒:", compute_experiment("energy_conservation_input", {
        "mass_g": 200, "height_m": 0.3, "final_speed_m_s": 2.4,
    }))
