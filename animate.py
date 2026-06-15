"""
动画与力图叠加生成模块（阶段五实现）

当前为骨架，接口已定义，具体渲染逻辑在阶段五填充。
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


# 箭头颜色（BGR for OpenCV，RGB for Matplotlib）
FORCE_COLOR_BGR = (0, 64, 255)     # 红色
FORCE_COLOR_RGB = (1.0, 0.25, 0.0)


def draw_force_overlay(
    image_path: str | Path,
    physics_data: dict,
    save_path: str,
    arrow_scale: float = 30.0,
) -> None:
    """
    在原图上叠加力箭头，保存结果。

    Args:
        image_path:  原始实验照片路径
        physics_data: extract_physics() 返回的字典
        save_path:   输出图片路径
        arrow_scale: 力箭头长度缩放系数（像素/牛顿）
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    forces = physics_data.get("forces", [])
    if not forces:
        cv2.imwrite(save_path, img)
        return

    for force in forces:
        magnitude = force["magnitude"]
        angle_deg = force["angle_deg"]
        ox, oy = force["origin_xy"]
        label = force["label"]

        angle_rad = np.deg2rad(angle_deg)
        length = max(magnitude * arrow_scale, 30)  # 最短 30px
        ex = int(ox + length * np.cos(angle_rad))
        ey = int(oy - length * np.sin(angle_rad))  # 图像 y 轴向下

        cv2.arrowedLine(img, (int(ox), int(oy)), (ex, ey),
                        FORCE_COLOR_BGR, 2, tipLength=0.25)
        cv2.putText(img, label, (ex + 5, ey),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, FORCE_COLOR_BGR, 2)

    cv2.imwrite(save_path, img)


def generate_spring_animation(
    spring_info: dict,
    save_path: str,
    fps: int = 15,
    duration_s: float = 2.0,
) -> None:
    """
    生成弹簧从自然长度伸长到当前形变量的 GIF 动画，
    同步显示胡克定律公式 F = kx。

    Args:
        spring_info:  extract_spring_deformation() 返回的 SpringInfo 字典
        save_path:    输出 GIF 路径
        fps:          帧率
        duration_s:   动画总时长（秒）
    """
    natural_len = spring_info.get("natural_length_cm", 5.0)
    deformation = spring_info.get("deformation_cm", 0.0)
    k = spring_info.get("spring_constant")
    total_len = natural_len + deformation

    n_frames = int(fps * duration_s)
    fig, ax = plt.subplots(figsize=(3, 5))

    def _draw_spring(ax, length_cm: float, deform_cm: float):
        ax.clear()
        ax.set_xlim(0, 2)
        ax.set_ylim(0, total_len * 1.3)
        ax.axis("off")

        # 固定端
        ax.plot([0.5, 1.5], [length_cm, length_cm], "k-", lw=3)
        ax.plot([1.0, 1.0], [length_cm, length_cm + 0.3], "k-", lw=2)

        # 弹簧（锯齿线）
        n_coils = 8
        y_pts = np.linspace(length_cm + 0.3, 0, n_coils * 2 + 2)
        x_pts = np.array([1.0] + [0.65 if i % 2 == 0 else 1.35
                                   for i in range(n_coils * 2)] + [1.0])
        ax.plot(x_pts, y_pts, "b-", lw=1.5)

        # 质量块
        rect = plt.Rectangle((0.6, -0.5), 0.8, 0.5,
                              linewidth=1.5, edgecolor="gray", facecolor="peru")
        ax.add_patch(rect)

        # 公式
        f_val = k * deform_cm / 100 if k and deform_cm > 0 else 0.0
        formula = (f"F = k·x\n"
                   f"x = {deform_cm:.1f} cm\n"
                   f"F = {f_val:.2f} N" if k else f"x = {deform_cm:.1f} cm")
        ax.text(0.05, 0.05, formula, transform=ax.transAxes,
                fontsize=9, verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    def update(frame):
        t = frame / n_frames
        # 缓入缓出
        ease = t * t * (3 - 2 * t)
        current_deform = deformation * ease
        current_len = natural_len + current_deform
        _draw_spring(ax, current_len, current_deform)

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)


# ── 简单测试 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    spring_data = {
        "deformation_cm": 2.5,
        "natural_length_cm": 5.0,
        "force_n": 1.25,
        "spring_constant": 50.0,  # N/m
    }
    generate_spring_animation(spring_data, "/tmp/test_spring.gif")
    print("动画已保存到 /tmp/test_spring.gif")
