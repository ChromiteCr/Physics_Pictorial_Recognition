"""
力学实验可视化与动画模块（力学版）

两个入口，均被 detect.py 调用：
  - draw_motion_overlay()   在代表帧上叠加：运动轨迹 + 末态速度箭头 + 关键量标注
  - generate_motion_animation()  按 mode 生成对应的 GIF 动画
"""

from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
