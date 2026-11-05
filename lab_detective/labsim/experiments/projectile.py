"""平抛：小球从斜槽上不同高度 H 滚下，从高 h 的水平槽口飞出，测落地点的水平距离 x（留出实验，远迁移）。

学生的做法（人教版必修第二册 第五章"实验：探究平抛运动的特点"的器材，加上"用平抛测初速度"的方法）：
斜槽固定在桌边，末端（槽口）调成水平；用重垂线在地面标出槽口正下方的点 O，卷尺量出槽口到地面的高度 h。
地面铺白纸、上面盖复写纸。小球从斜槽上标好的高度 H = 5、10、15、20、25、30 cm 处由静止释放（H 从槽口量起），
每个高度放 5 次，用尽量小的圆把 5 个落点圈住，圆心作为平均落点，量出它到 O 的距离 x。

物理（g 最后约掉）：
- 实心球纯滚动：动能 = 平动 ½mv² + 转动 ½Iω²，I = (2/5)mR²，ω = v/R
  ⇒ mgH = ½mv²(1 + 2/5) = (7/10)mv² ⇒ v₀² = (10/7)gH（只有 5/7 的势能变成平动动能）。
- 平抛：t = √(2h/g)，x = v₀t ⇒ x² = v₀²·2h/g = (20/7)·h·H = 4hH·(5/7)。
- 一般地，转动动能与平动动能之比 β = I/(mr_c²)，r_c 是球心到接触点连线的转动半径；
  平底槽 r_c = R，β = 2/5；双轨/V 形槽两接触线相距 2s 时 r_c = √(R² − s²)，β = (2/5)R²/(R² − s²) 更大
  ⇒ x² = 4hH/(1 + β)（构造参数 rail_gap_cm = 2s；这是物理本身，不是故障）。
- v₀² ≫ gR（约 1.4 vs 0.1 m²/s²），小球离开槽口时直接飞出，不会绕棱角转一段。
- 空气阻力：2 cm 钢球、v ≈ 1.7 m/s，阻力加速度约 0.008 m/s²，飞行 0.4 s 内使 x 少约 0.06 cm，忽略。

面板坐标：横轴 = 释放高度 H（cm），纵轴 = x²（cm²），理想斜率 4h/(1+β)（h 用记录值），截距 0。
"""

from __future__ import annotations

import numpy as np

from ..core import (
    G,
    Cause,
    Dataset,
    Experiment,
    LinearData,
    MenuItem,
    ObsKind,
    Prior,
    Protocol,
    Reading,
)

SINGLE_DROP_SIGMA = 0.7  # cm，单次落点的散布（释放、槽面、球的微小差别）


class Projectile(Experiment):
    id = "projectile"
    name_zh = "平抛运动：落地距离与释放高度的关系"
    name_en = "Projectile: landing distance vs release height"
    held_out = True
    transfer = "far"

    nuisance = {}
    linear = ()

    causes = (
        Cause(
            "projectile.random_only", "只有随机误差", "random error only",
            "装置调整正确、各量测量正确，小球在槽上纯滚动，数据偏离只来自落点和读数的随机误差。",
            "The setup and all measurements are correct and the ball rolls without slipping; "
            "deviations come only from random scatter of the landing points and readings.",
            "none", ("none",),
            fix_zh="每个高度多放几次，用小圆圈出平均落点。",
            fix_en="Release more times at each height and circle the average landing point.",
        ),
        # 推导：沿斜槽滚动阻力（槽面粗糙、有灰尘）做功 μ_r·mg·cosα·(H/sinα) ∝ H；
        # 斜面与水平段的拐角不圆滑时，小球撞向水平段损失法向速度，损失的动能也正比于 H。
        # ⇒ (1+β)·½mv₀² = mgH(1 − f)，f 与 H 无关 ⇒ x² = 4hH(1 − f)/(1+β)，整体按比例变小。
        Cause(
            "projectile.ramp_loss", "斜槽上有能量损失", "energy lost on the ramp",
            "斜槽面粗糙、有灰尘，或斜面与末端水平段的连接处不圆滑，小球在拐角处磕了一下，滚下来时有一部分能量变成了热。",
            "The ramp surface is rough or dusty, or the bend between the slope and the horizontal end "
            "is not smooth so the ball knocks against it; part of the energy turns into heat.",
            "y_scale", ("slope_low",),
            magnitude=Prior.uniform(0.02, 0.25, 47),
            magnitude_label_zh="损失的能量占 mgH 的比例", magnitude_unit="",
            fix_zh="擦干净斜槽，换连接处圆滑过渡的斜槽；或在槽口装光电门直接测出射速度。",
            fix_en="Clean the ramp or use one with a smooth bend; or measure the exit speed directly.",
        ),
        # 推导：小球离地落到地面，球心下降 = 槽口处球底到落点平面的高度 h_真。
        # 学生量到了别的位置（桌面而槽口垫高了、量到球心或球顶、落地面垫了板），记录值 h = h_真 − δ。
        # x² = 4h_真·H/(1+β) 按真实高度，而理想斜率按记录的 h 算 ⇒ 斜率比 = h_真/h = 1 + δ/h。
        Cause(
            "projectile.height_wrong", "抛出高度量错了位置", "launch height measured to the wrong point",
            "量抛出高度时没有量槽口处小球底部到落点所在平面的竖直距离：量的是桌面（槽口其实垫高了）、量到了球心或球顶，或者落点处垫了木板却仍从地面量起。",
            "The launch height was not measured from the bottom of the ball at the lip down to the "
            "landing surface: it was measured to the table top (the lip is raised), to the ball's "
            "centre or top, or from the floor although the ball lands on a board.",
            "x_scale", ("slope_high", "slope_low"),
            magnitude=Prior.sym_uniform(0.5, 4.0, 36),
            magnitude_label_zh="真实高度 − 记录高度", magnitude_unit="cm",
            fix_zh="用重垂线从槽口处小球的最低点量到落点所在的平面。",
            fix_en="Measure with a plumb line from the bottom of the ball at the lip to the landing surface.",
        ),
        # 推导：槽口向上（下）倾 θ，速度大小不变（由能量决定），方向变了：
        # 竖直 h = −v₀sinθ·t + ½gt² ⇒ t = [v₀sinθ + √(v₀²sin²θ + 2gh)]/g，x = v₀cosθ·t。
        # 小角度 x ≈ v₀t₀ + v₀²θ/g，v₀² ∝ H ⇒ x² ≈ aH + bθ·H^{3/2}：放得越高偏得越多。
        Cause(
            "projectile.end_tilted", "槽口不水平", "ramp end not horizontal",
            "斜槽末端没有调水平，小球离开槽口时速度方向略微向上（或向下）。",
            "The end of the ramp is not horizontal, so the ball leaves the lip moving slightly "
            "upwards (or downwards).",
            # H^{3/2} 项用直线拟合：斜率与截距反向偏离（上倾斜率大、截距负），弯曲一般不显著
            "bend", ("slope_high", "intercept_neg", "slope_low", "intercept_pos"),
            magnitude=Prior.sym_uniform(0.5, 6.0, 44),
            magnitude_label_zh="槽口倾角（正 = 向上）", magnitude_unit="°",
            fix_zh="把小球放在槽口水平段上，调到它在任何位置都能静止为止。",
            fix_en="Adjust the end until a ball placed anywhere on it stays at rest.",
        ),
        # 推导：滚下时若静摩擦不够（槽面太光滑、斜面太陡），小球边滚边滑，转得不够快。
        # 打滑阶段摩擦力 F < 纯滚所需 F_c = β/(1+β)·mg·sinα，记 s = 1 − F/F_c ∈ (0, 1]：
        # ½mv₀² = mgH − F·L ⇒ v₀² = 2gH(1 + sβ)/(1+β)，介于 2gH/(1+β)（纯滚）与 2gH（无摩擦滑下）之间。
        # 末端水平段很短，来不及重新变成纯滚 ⇒ x² 整体按比例变大。
        Cause(
            "projectile.slipping", "小球边滚边滑", "ball slips on the ramp",
            "斜槽太光滑或太陡，小球滚下时摩擦力不够，一边滚一边滑，转动得比纯滚动时慢。",
            "The ramp is too smooth or too steep, so friction cannot keep the ball rolling; it "
            "slides as it rolls and spins more slowly than in pure rolling.",
            "y_scale", ("slope_high",),
            magnitude=Prior.uniform(0.05, 1.0, 39),
            magnitude_label_zh="打滑程度（0 = 纯滚，1 = 无摩擦滑下）", magnitude_unit="",
            fix_zh="降低斜槽倾角，或换表面稍粗糙的斜槽，保证小球纯滚动。",
            fix_en="Use a less steep or slightly rougher ramp so the ball rolls without slipping.",
        ),
        # 推导：每个平均落点由 n 次落地取圆心，σ₀ = σ_单次/√n；白纸移动、释放位置或手法不一致
        # 是与之独立的另一份随机误差 δ ⇒ 平均落点的标准差 σ = √(σ₀² + δ²)。
        Cause(
            "projectile.landing_scatter", "白纸被碰动或释放不一致", "paper shifted or release not repeatable",
            "几次落地之间白纸或复写纸被碰动了，或者每次释放的位置、放手方式不一样（有时推了一下）。",
            "The paper or carbon paper moved between drops, or the ball was released from a "
            "slightly different place or given a push some of the time.",
            "scatter", ("scatter_excess",),
            magnitude=Prior.uniform(0.6, 2.5, 39),
            magnitude_label_zh="平均落点的额外随机误差", magnitude_unit="cm",
            fix_zh="用胶带把白纸和复写纸固定好；用挡板定位释放点，每次都由静止放手。",
            fix_en="Tape the papers down; use a stop to fix the release point and release from rest.",
        ),
        # 推导：释放高度的零点量错（从桌面量起而槽口高于桌面，或量到球顶），H_真 = H + δ。
        # x² = 4h(H + δ)/(1+β) ⇒ 斜率不变，截距 = 4hδ/(1+β)。
        Cause(
            "projectile.release_zero", "释放高度的起点量错了", "release height measured from the wrong zero",
            "释放高度 H 不是从槽口处小球的最低点量起的：从桌面量起，或者量的是球顶到槽口的高度，每个 H 都差了同样一段。",
            "The release height was not measured from the bottom of the ball at the lip: it was "
            "measured from the table top, or to the top of the ball, so every H is off by the same amount.",
            "x_offset", ("intercept_pos", "intercept_neg"),
            magnitude=Prior.sym_uniform(0.2, 3.0, 56),
            magnitude_label_zh="真实释放高度 − 记录值", magnitude_unit="cm",
            fix_zh="H 统一从槽口处小球的最低点量到释放处小球的最低点。",
            fix_en="Measure H from the ball's lowest point at the lip to its lowest point at release.",
        ),
    )

    protocols = {
        "single": Protocol("single", "一个高度", "one height", (("main", 15.0),)),
        "standard": Protocol(
            "standard", "课本方案（6 个释放高度，每个放 5 次取平均落点）",
            "worksheet (6 release heights, 5 drops each)",
            tuple(("main", H) for H in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)),
        ),
    }

    menu = (
        MenuItem("h_check", "用重垂线和卷尺重新量槽口处球底到落点平面的高度", "re-measure the launch height",
                 2.0, (("h_check", 0.0),)),
        # 挡光长度按直径 2R 算：光束偏离球心 3 mm（R = 1 cm）弦就短约 5%，算出的 v₀² 偏大约 10%，会被误读成"打滑"
        MenuItem("exit_gate", "先用游标卡尺量出小球直径；在槽口装光电门、光束对准球心，测小球飞出时的挡光时间（H = 10、20 cm 各一次）",
                 "caliper the ball diameter; photogate at the lip with the beam through the ball's centre: "
                 "blocking time at H = 10 and 20 cm",
                 5.0, (("exit_gate", 10.0), ("exit_gate", 20.0))),
        MenuItem("end_level", "用手机水平仪测槽口倾角", "measure the lip angle with a phone level",
                 2.0, (("end_angle", 0.0),)),
        MenuItem("extra_high", "再从 H = 40 cm 放 5 次", "add H = 40 cm (5 drops)", 3.0, (("main", 40.0),)),
        MenuItem("repeat_mid", "在 H = 15 cm 重新放 5 次", "repeat H = 15 cm (5 drops)", 3.0, (("main", 15.0),)),
    )

    def __init__(self, h_cm: float = 80.0, ball_radius_cm: float = 1.0, rail_gap_cm: float = 0.0,
                 drops_per_point: int = 5):
        R, s = float(ball_radius_cm), 0.5 * float(rail_gap_cm)
        assert 0.0 <= s < R, "两轨间距必须小于小球直径"
        self.h = float(h_cm)  # 记录的抛出高度，cm（也是 random_only 下的真值）
        self.R = R
        self.beta = 0.4 * R**2 / (R**2 - s**2)  # 转动动能 / 平动动能
        self.drops = int(drops_per_point)
        sig = SINGLE_DROP_SIGMA / np.sqrt(self.drops)
        self.obs_kinds = {
            "main": ObsKind(
                "main", "从高度 H 释放，量平均落点到 O 的距离", "release from H, measure the landing distance",
                (Reading("x", "水平距离", "landing distance", "cm", resolution=0.1, sigma=float(sig)),),
                setting_label_zh="释放高度 H", setting_label_en="release height", setting_unit="cm",
            ),
            "h_check": ObsKind(
                "h_check", "重量抛出高度", "re-measure the launch height",
                (Reading("h", "抛出高度", "launch height", "cm", resolution=0.1, sigma=0.15),),
            ),
            "exit_gate": ObsKind(
                "exit_gate", "槽口光电门", "photogate at the lip",
                (Reading("dt", "小球挡光时间", "ball blocking time", "ms", resolution=0.1, sigma=0.1),),
                setting_label_zh="释放高度 H", setting_label_en="release height", setting_unit="cm",
            ),
            "end_angle": ObsKind(
                "end_angle", "槽口倾角", "angle of the ramp end",
                (Reading("angle", "槽口倾角（向上为正）", "lip angle (up positive)", "°", resolution=0.1, sigma=0.3),),
            ),
        }

    # ---- 物理 ----

    def _launch(self, H, cause_id, mag):
        """返回 (v₀ m/s, 槽口倾角 rad, 真实抛出高度 m)。H: cm。"""
        if cause_id == "projectile.release_zero":
            H = H + mag
        H = np.maximum(H, 0.0) / 100.0
        factor = 1.0 / (1.0 + self.beta)
        if cause_id == "projectile.ramp_loss":
            factor = factor * (1.0 - mag)
        if cause_id == "projectile.slipping":
            factor = factor * (1.0 + mag * self.beta)
        v0 = np.sqrt(2.0 * G * H * factor)
        theta = np.deg2rad(mag) if cause_id == "projectile.end_tilted" else 0.0 * mag
        h = self.h + (mag if cause_id == "projectile.height_wrong" else 0.0)
        return v0, theta, h / 100.0

    def predict(self, kind, settings, order, params, cause_id):
        P = params["mag"].shape[0]
        n = settings.size
        L = len(self.linear_params(cause_id))
        U = np.zeros((P, n, 1, L))
        mag = params["mag"]  # (P, 1)
        H = settings[None, :]
        c = np.zeros((P, n, 1))
        v0, th, h = self._launch(H, cause_id, mag)
        if kind == "main":
            vy = v0 * np.sin(th)  # 向上为正
            t = (vy + np.sqrt(vy**2 + 2.0 * G * h)) / G
            c[:, :, 0] = 100.0 * v0 * np.cos(th) * t
            return c, U
        if kind == "h_check":
            c[:, :, 0] = 100.0 * h
            return c, U
        if kind == "exit_gate":
            # 光束穿过球心时挡光长度 = 直径 2R（偏离球心时弦更短，这里不计）
            c[:, :, 0] = 1000.0 * (2.0 * self.R / 100.0) / np.maximum(v0, 1e-3)
            return c, U
        if kind == "end_angle":
            c[:, :, 0] = np.rad2deg(th)
            return c, U
        raise KeyError(kind)

    def reading_sigma(self, kind, params, cause_id):
        base = super().reading_sigma(kind, params, cause_id)
        if kind == "main" and cause_id == "projectile.landing_scatter":
            return np.sqrt(base**2 + params["mag"][:, :, None] ** 2)
        return base

    def linearise(self, ds: Dataset) -> LinearData:
        idx = [i for i, o in enumerate(ds.observations) if o.kind == "main"]
        H = np.array([ds.observations[i].setting for i in idx])
        x = np.array([ds.observations[i].values[0] for i in idx])
        r = self.obs_kinds["main"].readings[0]
        sx = float(np.sqrt(r.sigma**2 + r.resolution**2 / 12))
        sy = 2.0 * np.abs(x) * sx  # σ(x²) = 2x·σ(x)
        slope = 4.0 * self.h / (1.0 + self.beta)  # = (20/7)h（实心球、平底槽）
        return LinearData(H, x**2, sy, np.array(idx), "释放高度 H (cm)", "水平距离的平方 x² (cm²)", slope, 0.0)
