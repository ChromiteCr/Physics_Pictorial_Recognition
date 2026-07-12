"""
力学实验可视化与动画模块（力学版）

两个入口，均被 detect.py 调用：
  - draw_motion_overlay()   在代表帧上叠加：运动轨迹 + 末态速度箭头 + 关键量标注
  - generate_motion_animation()  按 mode 生成对应的 GIF 动画
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from physics import G

# 中文标题/图例要能正常显示，DejaVu Sans（matplotlib 默认字体）不含 CJK 字形，
# 会画成方框。按常见平台优先级列出候选，第一个装了的就用。
matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "STHeiti", "PingFang SC", "Heiti SC",
    "Noto Sans CJK SC", "SimHei", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

TRACE_COLOR_BGR = (0, 220, 255)    # 黄 - 轨迹
VELOCITY_COLOR_BGR = (0, 0, 255)   # 红 - 速度箭头
TEXT_COLOR_BGR = (40, 40, 220)     # 红 - 文字


def draw_motion_overlay(
    frame_bgr: np.ndarray,
    tracks: dict,
    physics_data: dict,
    save_path: str,
) -> None:
    """在代表帧（建议传入最后一帧）上叠加轨迹与关键物理量。"""
    img = frame_bgr.copy()

    for key, track in tracks.items():
        if not track:
            continue
        pts = np.array([[p.x, p.y] for p in track], dtype=np.int32)
        cv2.polylines(img, [pts], False, TRACE_COLOR_BGR, 2)
        if len(pts) >= 2:
            p0, p1 = pts[-2], pts[-1]
            cv2.arrowedLine(img, tuple(p0), tuple(p1), VELOCITY_COLOR_BGR, 2, tipLength=0.4)

    analysis = physics_data.get("analysis", {})
    lines = [f"mode: {physics_data.get('mode')}"]
    for k, v in analysis.items():
        if isinstance(v, (int, float, bool)) or v is None:
            lines.append(f"{k}: {v}")
    for i, line in enumerate(lines[:8]):
        cv2.putText(img, line, (10, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, TEXT_COLOR_BGR, 2, cv2.LINE_AA)

    cv2.imwrite(save_path, img)


def generate_motion_animation(
    physics_data: dict,
    tracks: dict,
    save_path: str,
    fps: int = 20,
) -> None:
    """按 physics_data['mode'] 分发到对应的动画生成函数。"""
    mode = physics_data.get("mode")
    if mode == "momentum":
        _animate_momentum(physics_data, tracks, save_path, fps)
    elif mode == "energy_conservation":
        _animate_energy(physics_data, save_path, fps)
    elif mode == "circular_motion":
        _animate_circular(physics_data, tracks, save_path, fps)
    elif mode == "vibration":
        _animate_vibration(physics_data, tracks, save_path, fps)
    else:
        raise ValueError(f"未知模式: {mode}")


def _animate_momentum(physics_data, tracks, save_path, fps):
    t1 = tracks.get("track1", [])
    t2 = tracks.get("track2", [])
    if not t1 or not t2:
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    n_frames = min(len(t1), len(t2))

    def update(i):
        ax.clear()
        ax.set_xlim(min(p.x for p in t1 + t2) - 20, max(p.x for p in t1 + t2) + 20)
        ax.set_ylim(0, 2)
        ax.axis("off")
        ax.plot(t1[i].x, 1, "bs", ms=16, label="cart1")
        ax.plot(t2[i].x, 1, "rs", ms=16, label="cart2")
        analysis = physics_data.get("analysis", {})
        ax.set_title(
            f"p_before={analysis.get('momentum_before_kg_m_s')} "
            f"p_after={analysis.get('momentum_after_kg_m_s')} kg·m/s",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=8)

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)


def _animate_energy(physics_data, save_path, fps):
    analysis = physics_data.get("analysis", {})
    t = analysis.get("time_s", [])
    ke = analysis.get("kinetic_energy_J", [])
    pe = analysis.get("potential_energy_J", [])
    total = analysis.get("total_energy_J", [])
    if not t:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    n_frames = len(t)

    def update(i):
        ax.clear()
        ax.set_xlim(0, max(t))
        ax.set_ylim(0, max(total) * 1.2 if total else 1)
        ax.plot(t[: i + 1], ke[: i + 1], "r-", label="KE")
        ax.plot(t[: i + 1], pe[: i + 1], "b-", label="PE")
        ax.plot(t[: i + 1], total[: i + 1], "k--", label="Total")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("Energy (J)")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"机械能守恒  误差={analysis.get('conservation_error_ratio')}", fontsize=10)

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)


def _animate_circular(physics_data, tracks, save_path, fps):
    track = tracks.get("track1", [])
    if not track:
        return
    analysis = physics_data.get("analysis", {})
    fig, ax = plt.subplots(figsize=(5, 5))
    n_frames = len(track)
    xs = [p.x for p in track]
    ys = [p.y for p in track]

    def update(i):
        ax.clear()
        ax.set_aspect("equal")
        ax.plot(xs, ys, "lightgray", lw=1)
        ax.plot(xs[: i + 1], ys[: i + 1], "b-", lw=2)
        ax.plot(xs[i], ys[i], "ro", ms=10)
        ax.invert_yaxis()
        ax.set_title(
            f"T={analysis.get('period_s')}s  v={analysis.get('linear_velocity_cm_s')}cm/s  "
            f"a_c={analysis.get('centripetal_acceleration_m_s2')}m/s²",
            fontsize=9,
        )

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)


def _animate_vibration(physics_data, tracks, save_path, fps):
    track = tracks.get("track1", [])
    if not track:
        return
    analysis = physics_data.get("analysis", {})
    t = [p.t for p in track]
    x = [p.x for p in track]
    x0 = sum(x) / len(x)
    disp = [v - x0 for v in x]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    n_frames = len(track)

    def update(i):
        ax.clear()
        ax.set_xlim(0, max(t))
        ax.set_ylim(min(disp) * 1.2, max(disp) * 1.2)
        ax.plot(t[: i + 1], disp[: i + 1], "g-", lw=1.5)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("displacement (px)")
        ax.set_title(
            f"T={analysis.get('period_s')}s  A={analysis.get('amplitude_cm')}cm  "
            f"γ={analysis.get('damping_coefficient')}",
            fontsize=10,
        )

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Part 2：按实验类型生成理论动画
#
# [2026-07-11] 输入源从"CV提取的真实轨迹"改成"用户输入+physics.compute_experiment()
# 算出的结果"——画的是符合公式的理想化示意动画，不是逐帧还原真实运动。track 参数
# 仍然可选传入（来自 Part 1 的视频，如果有的话），只用来在图上叠加一条参考路径，
# 纯装饰，不影响动画本身依据的物理量。
# ══════════════════════════════════════════════════════════════════

def generate_experiment_animation(
    experiment_id: str,
    computed_result: dict,
    user_inputs: dict,
    save_path: str,
    track: Optional[list] = None,
    fps: int = 20,
    duration_s: float = 2.5,
) -> None:
    """Part 2 统一动画入口，按实验类型 id 分发。"""
    dispatch = {
        "friction": _animate_friction,
        "average_speed": _animate_average_speed,
        "hooke_law": _animate_hooke_law,
        "newton_second_law": _animate_newton_second_law,
        "momentum_conservation": _animate_momentum_v2,
        "energy_conservation_input": _animate_energy_v2,
    }
    fn = dispatch.get(experiment_id)
    if fn is None:
        raise ValueError(f"未知实验类型: {experiment_id}")
    n_frames = int(fps * duration_s)
    fn(computed_result, user_inputs, save_path, fps, n_frames, track)


def _animate_friction(result, inputs, save_path, fps, n_frames, track):
    """木块在测力计拉力下匀速滑动，标注摩擦力大小与方向。"""
    f = result["friction_force_N"]
    fig, ax = plt.subplots(figsize=(6, 3))

    def update(i):
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.axis("off")
        x = 1 + (i / n_frames) * 6  # 匀速滑动
        rect = plt.Rectangle((x, 0.8), 1.2, 0.8, facecolor="peru", edgecolor="black")
        ax.add_patch(rect)
        ax.plot([0, 10], [0.8, 0.8], "k-", lw=2)  # 接触面
        # 施加的拉力（沿运动方向，右指）
        ax.annotate("", xy=(x + 1.8, 1.2), xytext=(x + 1.2, 1.2),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2))
        ax.text(x + 1.9, 1.2, f"F拉={f}N", color="blue", fontsize=10, va="center")
        # 摩擦力（反方向，左指）
        ax.annotate("", xy=(x - 0.3, 1.0), xytext=(x + 0.3, 1.0),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))
        ax.text(x - 1.3, 1.0, f"f摩={f}N", color="red", fontsize=10, va="center")
        material = inputs.get("surface_material")
        title = f"匀速直线运动：F拉 = f摩 = {f} N"
        if material:
            title += f"（接触面：{material}）"
        ax.set_title(title, fontsize=10)

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)


def _animate_average_speed(result, inputs, save_path, fps, n_frames, track):
    """木块匀速移动 distance_m，用时 time_s，动画展示 s-t 关系。"""
    distance = result["distance_m"]
    total_t = result["time_s"]
    v = result["average_speed_m_s"]
    fig, ax = plt.subplots(figsize=(6, 3.5))

    def update(i):
        ax.clear()
        t_now = (i / n_frames) * total_t
        x_now = v * t_now
        ax.set_xlim(0, max(total_t, 1e-6))
        ax.set_ylim(0, max(distance, 1e-6) * 1.2)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("s (m)")
        ts = np.linspace(0, t_now, max(2, i + 1))
        ax.plot(ts, v * ts, "b-", lw=2)
        ax.plot(t_now, x_now, "ro", ms=8)
        ax.set_title(f"平均速度 v = s/t = {distance}/{total_t} = {v} m/s", fontsize=10)

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)


def _animate_hooke_law(result, inputs, save_path, fps, n_frames, track):
    """弹簧从自然长度逐渐拉伸到目标形变量，同步显示 F=kx 直线。"""
    k = result["spring_constant_N_per_m"]
    measurements = inputs.get("measurements")
    if measurements:
        target_x_cm = max(m["deformation_cm"] for m in measurements)
        target_f = max(m["force_n"] for m in measurements)
    else:
        target_x_cm = result.get("deformation_cm", 5.0)
        target_f = result.get("force_n", k * target_x_cm / 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    def update(i):
        frac = i / n_frames
        x_cm = target_x_cm * frac
        f = k * x_cm / 100

        ax1.clear()
        block_h = 1.0
        natural_len = 3.0
        total_len = natural_len + x_cm  # 弹簧从固定端到挂钩的长度（含形变）
        ax1.set_xlim(0, 2)
        ax1.set_ylim(0, natural_len + target_x_cm + block_h + 1)
        ax1.axis("off")
        # 固定端在顶部，弹簧往下垂到 y=block_h（挂钩处），质量块占 [0, block_h]
        ax1.plot([0.5, 1.5], [block_h + total_len, block_h + total_len], "k-", lw=3)  # 固定端
        n_coils = 8
        y_pts = np.linspace(block_h + total_len, block_h, n_coils * 2 + 2)
        x_pts = np.array([1.0] + [0.65 if j % 2 == 0 else 1.35 for j in range(n_coils * 2)] + [1.0])
        ax1.plot(x_pts, y_pts, "b-", lw=1.5)
        rect = plt.Rectangle((0.6, 0), 0.8, block_h, facecolor="peru", edgecolor="black")
        ax1.add_patch(rect)
        ax1.set_title(f"x={x_cm:.1f}cm  F={f:.2f}N", fontsize=10)

        ax2.clear()
        xs = np.linspace(0, target_x_cm * 1.2, 50)
        ax2.plot(xs, k * xs / 100, "k--", lw=1, label=f"F=kx (k={k:.1f}N/m)")
        if measurements:
            ax2.scatter([m["deformation_cm"] for m in measurements],
                       [m["force_n"] for m in measurements], color="green", zorder=3, label="测量点")
        ax2.plot(x_cm, f, "ro", ms=8)
        ax2.set_xlabel("x (cm)")
        ax2.set_ylabel("F (N)")
        ax2.legend(loc="upper left", fontsize=8)

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)


def _animate_newton_second_law(result, inputs, save_path, fps, n_frames, track):
    """
    小车在恒定拉力下做匀加速运动，v-t 图线性增长，斜率=a。

    compute_newton_second_law() 的返回字段随输入组合而变：只给一个量时是
    accel_m_s2/force_N（推算出另一个）；两个都给时是 measured_accel_m_s2/
    predicted_accel_m_s2（对比验证）。动画统一用"预测/推算出的加速度"作图。
    """
    a = result.get("accel_m_s2", result.get("predicted_accel_m_s2"))
    f = result["force_N"]
    mass_kg = result["mass_kg"]
    total_t = 2.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    def update(i):
        t_now = (i / n_frames) * total_t
        v_now = a * t_now
        x_now = 0.5 * a * t_now ** 2

        ax1.clear()
        ax1.set_xlim(0, max(0.5 * a * total_t ** 2, 1.0))
        ax1.set_ylim(0, 2)
        ax1.axis("off")
        rect = plt.Rectangle((x_now, 0.8), 1.0, 0.6, facecolor="steelblue", edgecolor="black")
        ax1.add_patch(rect)
        ax1.annotate("", xy=(x_now + 1.6, 1.1), xytext=(x_now + 1.0, 1.1),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2))
        ax1.text(x_now + 1.7, 1.1, f"F={f:.2f}N", color="blue", fontsize=9, va="center")
        ax1.set_title(f"m={mass_kg*1000:.0f}g  a={a:.2f}m/s²", fontsize=10)

        ax2.clear()
        ts = np.linspace(0, t_now, max(2, i + 1))
        ax2.plot(ts, a * ts, "b-", lw=2)
        ax2.set_xlim(0, total_t)
        ax2.set_ylim(0, a * total_t * 1.2 if a > 0 else 1)
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("v (m/s)")
        ax2.set_title("v-t 图（斜率=a）", fontsize=10)

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)


def _animate_momentum_v2(result, inputs, save_path, fps, n_frames, track):
    """两车碰撞前后位置动画，标注碰撞前后动量对比（Part 2 用户输入驱动版）。"""
    v1b, v2b = inputs["v1_before_m_s"], inputs["v2_before_m_s"]
    v1a, v2a = inputs["v1_after_m_s"], inputs["v2_after_m_s"]
    half = n_frames // 2
    fig, ax = plt.subplots(figsize=(6, 3))

    def update(i):
        ax.clear()
        ax.set_xlim(-3, 3)
        ax.set_ylim(0, 2)
        ax.axis("off")
        if i < half:
            frac = i / half
            x1 = -2 + v1b * frac
            x2 = 1 - v2b * frac if v2b else 1
        else:
            frac = (i - half) / max(n_frames - half, 1)
            x1 = -2 + v1b + v1a * frac
            x2 = 1 - v2b + v2a * frac
        ax.plot(x1, 1, "bs", ms=18, label="小车1")
        ax.plot(x2, 1, "rs", ms=18, label="小车2")
        ax.set_title(
            f"p碰前={result['momentum_before_kg_m_s']}  p碰后={result['momentum_after_kg_m_s']} kg·m/s"
            f"  误差={result.get('conservation_error_ratio')}",
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=8)

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)


def _animate_energy_v2(result, inputs, save_path, fps, n_frames, track):
    """小车从高度 h 下滑到底部，KE/PE 此消彼长（Part 2 用户输入驱动版）。"""
    mass_kg = inputs["mass_g"] / 1000
    h0 = inputs["height_m"]
    v_final = inputs["final_speed_m_s"]
    fig, ax = plt.subplots(figsize=(6, 4))

    def update(i):
        frac = i / n_frames
        h = h0 * (1 - frac)
        v = v_final * frac
        pe = mass_kg * G * h
        ke = 0.5 * mass_kg * v ** 2

        ax.clear()
        ax.bar(["PE", "KE", "Total"], [pe, ke, pe + ke], color=["steelblue", "orange", "gray"])
        ax.set_ylim(0, result["initial_potential_energy_J"] * 1.3 + 1e-6)
        ax.set_ylabel("Energy (J)")
        ax.set_title(
            f"PE₀={result['initial_potential_energy_J']}J  "
            f"KE末={result['final_kinetic_energy_J']}J  "
            f"误差={result.get('conservation_error_ratio')}",
            fontsize=9,
        )

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)


# ── 简单测试 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from physics import TrackPoint
    import math

    fps = 30.0
    t = [i / fps for i in range(120)]
    x = [100 + 50 * math.exp(-0.3 * ti) * math.cos(2 * math.pi * 1.2 * ti) for ti in t]
    track = [TrackPoint(t=ti, x=xi, y=200.0) for ti, xi in zip(t, x)]

    mock_physics = {
        "mode": "vibration",
        "analysis": {"period_s": 0.83, "amplitude_cm": 5.0, "damping_coefficient": 0.3},
    }
    _animate_vibration(mock_physics, {"track1": track}, "/tmp/test_vibration.gif", fps=20)
    print("动画已保存到 /tmp/test_vibration.gif")
