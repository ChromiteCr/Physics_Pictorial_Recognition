"""自由落体：用光电门测小球下落时间，研究 h = ½gt²（高中，留出实验·近迁移）。

方案说明：人教版必修第一册第二章第 4 节用打点计时器（重物 + 纸带）研究自由落体；学校里测 g
更常用"自由落体仪"，这里按后者建模（11/7 审物理卡时确认）：
铁架台立柱上端的电磁铁吸住小钢球，立柱上装一个光电门。断开电磁铁电源的同时数字计时器开始计时，
小球下落、球的下沿挡住光电门光束时停止计时。从立柱刻度读出小球下沿到光束的高度 h，
光电门依次放在 20、40、60、80、100、120 cm 处，各测一次下落时间 t。
"single" 方案只在 100 cm 处测一次。

物理：理想时 h = ½gt² ⇒ t = β·√h，β = √(2/g)。g 按课本取 9.8 m/s²（各地相差不到 0.3%），
所以 β 的先验很窄、理想斜率已知。

面板坐标：横轴 = √h（√m），纵轴 = 下落时间 t（s）。理想直线 t = √(2/g)·√h，过原点、斜率已知。

计时起点可选（构造参数 start）：
- "magnet"（默认）：电磁铁断电即开始计时，如上；
- "gate"：上下两个光电门，小球经过上方光电门开始计时、经过下方光电门停止计时，h 是两束光之间的距离。
  这时电磁铁延迟和"高度量错起点"都不影响结果，换成"上方光电门离小球太远"这一原因。

读数误差：数字计时器分辨率 0.1 ms；随机误差 0.6 ms（刻度尺读高度 ±1 mm 换算成时间 δh/v 约 0.2~0.5 ms，
加上每次释放的微小差别），属于假设值。
"""

from __future__ import annotations

import math

import numpy as np

from ..core import (
    G,
    Cause,
    Dataset,
    Experiment,
    LinearData,
    LinearParam,
    MenuItem,
    ObsKind,
    Prior,
    Protocol,
    Reading,
)

BETA0 = math.sqrt(2.0 / G)  # 理想斜率 √(2/g)，s/√m

T_GATE = Reading("t", "下落时间", "fall time", "s", resolution=0.0001, sigma=0.0006)

# 光电门位置读数误差换算成时间误差用的代表速度：标准方案高度中点 0.7 m 处的 v = √(2gh)
V_MID = math.sqrt(2 * G * 0.7)

# 轻球的浮力比例 ε = ρ_空气V/m 与阻力系数 κ = ρ_空气C_dA/(2m) 之比 = 8r/(3C_d)，取 r = 2 cm、C_d = 0.47
BUOY_PER_KAPPA = 8 * 0.02 / (3 * 0.47)  # m

RANDOM = "free_fall.random_only"
DELAY = "free_fall.release_delay"
TOP = "free_fall.height_to_top"
MARK = "free_fall.gate_mark"
DRAG = "free_fall.air_drag"
SCATTER = "free_fall.gate_scatter"
SPEED = "free_fall.release_speed"

TIMED_KINDS = ("main", "steel", "shim")

_MAGNET_ONLY_CAUSES = (DELAY, TOP, MARK)
_GATE_ONLY_CAUSES = (SPEED,)
_MAGNET_ONLY_KINDS = ("shim", "height_check")
_GATE_ONLY_KINDS = ("gap_check",)


def _drag_factor(kappa: np.ndarray, h: np.ndarray) -> np.ndarray:
    """有二次阻力和浮力时 t / √(2h/g) 的倍数。

    推导：m dv/dt = mg(1−ε) − ½ρC_dAv²，令 κ = ρC_dA/(2m)、g' = g(1−ε)：
      dv/dt = g' − κv² ⇒ h(t) = ln cosh(√(g'κ)·t)/κ ⇒ t = arccosh(e^{κh})/√(g'κ)
      = √(2h/g') · arccosh(e^{u})/√(2u)，u = κh；小 u 时 ≈ √(2h/g)(1 + u/6)(1 + ε/2)。
    """
    u = np.maximum(kappa * h, 1e-9)
    phi = np.arccosh(np.exp(u)) / np.sqrt(2.0 * u)
    eps = BUOY_PER_KAPPA * kappa
    return phi / np.sqrt(1.0 - eps)


def _obs_kinds() -> dict[str, ObsKind]:
    h_lab = dict(setting_label_zh="下落高度 h", setting_label_en="drop height", setting_unit="cm")
    return {
        "main": ObsKind("main", "光电门放在高度 h 处，测下落时间", "photogate at height h, read fall time",
                        (T_GATE,), **h_lab),
        "steel": ObsKind("steel", "换成同样大小的实心钢球，测下落时间",
                         "drop a solid steel ball of the same size", (T_GATE,), **h_lab),
        "shim": ObsKind("shim", "在电磁铁和小球之间垫一张薄纸，再测下落时间",
                        "put a thin paper between magnet and ball, then time the drop", (T_GATE,), **h_lab),
        "height_check": ObsKind(
            "height_check", "用刻度尺从小球下沿量到光电门光束中心，与表中记录的高度比较",
            "measure from the ball's bottom to the beam centre and compare with the table",
            (Reading("h_diff", "原记录高度 − 重量高度", "recorded − re-measured height", "cm",
                     resolution=0.1, sigma=0.05),),
            **h_lab,
        ),
        "gap_check": ObsKind(
            "gap_check", "量小球下沿到上方光电门光束的距离", "measure ball bottom to the upper beam",
            (Reading("gap", "小球下沿到上方光束的距离", "ball bottom to upper beam", "cm",
                     resolution=0.1, sigma=0.05),),
        ),
    }


class FreeFall(Experiment):
    id = "free_fall"
    name_zh = "研究自由落体运动（光电门测下落时间）"
    name_en = "Free fall timed with a photogate"
    held_out = True
    transfer = "near"

    obs_kinds = _obs_kinds()
    nuisance = {}

    # β = √(2/g)：先验宽度对应 g 相差约 ±0.02 m/s²（±1σ）
    linear = (LinearParam("beta", BETA0, 0.0005, "√(2/g)", "s/√m"),)

    causes = (
        Cause(
            RANDOM, "只有随机误差", "random error only",
            "装置和步骤都正确，数据偏离只来自高度读数和每次释放的微小差别。",
            "Setup and procedure are correct; deviations come only from small reading and release "
            "differences.",
            "none", ("none",),
            fix_zh="每个高度多测几次取平均。", fix_en="Repeat each drop and average.",
        ),
        # 推导：断电后线圈电流按 L/R 衰减，铁芯和小球还有剩磁，磁力要过 δ 才小于重力，
        #   小球比计时起点晚 δ 出发：t_记 = β√h + δ（δ 几毫秒到二十几毫秒，假设）。
        Cause(
            DELAY, "电磁铁断电后小球没有立刻下落", "ball released late after the magnet is switched off",
            "断开电源时计时器立刻开始计时，但电磁铁的磁性不会瞬间消失，剩磁还要把小球吸住一小会儿，"
            "小球比计时起点晚一点才开始下落。",
            "The timer starts the instant the magnet is switched off, but the magnetism does not "
            "vanish at once; residual magnetism holds the ball for a moment before it falls.",
            "y_offset", ("intercept_pos",),
            linear=(LinearParam("release_delay", 0.012, 0.004, "释放延迟", "s"),),
            fix_zh="在电磁铁和小球之间垫一张薄纸减小剩磁，或改用两个光电门计时。",
            fix_en="Put a thin paper spacer on the magnet, or time between two photogates.",
        ),
        # 推导：光束被球的下沿挡住时停止计时，真正的下落距离 = 释放时下沿到光束。
        #   从球顶（电磁铁下表面）量起多记一个直径 D，从球心量起多记 D/2：h_记 = h + e。
        #   t = β√(h_记 − e)：低处时间偏小得多，高处偏小得少，整体看起来"g 偏大"。
        Cause(
            TOP, "高度从小球顶端量起", "height measured from the top of the ball",
            "量高度时从电磁铁下表面（也就是小球顶端）或球心量到光电门，"
            "而小球要到它的下沿碰到光束才停止计时，小球实际下落的距离比记录的少一截。",
            "The height was measured from the magnet face (the top of the ball) or the ball's centre, "
            "but the timer stops when the ball's bottom reaches the beam, so the ball actually falls "
            "a little less than recorded.",
            "x_offset", ("slope_high", "curve_down", "intercept_neg"),
            magnitude=Prior.uniform(0.5, 3.0, 121),
            magnitude_label_zh="多记的高度", magnitude_unit="cm",
            fix_zh="从小球下沿量到光电门的光束中心。",
            fix_en="Measure from the bottom of the ball to the centre of the beam.",
        ),
        # 推导：光电门位置按门框上沿对立柱刻度读，光束实际在上沿下方 b 处：h_记 = h − b。
        #   t = β√(h_记 + b)：每个时间都比按记录高度算的长，低处长得更多。
        Cause(
            MARK, "光电门位置读在门框上沿", "photogate position read at the frame edge",
            "光电门的位置是按门框上沿对着立柱刻度读的，可是光束在两臂中间的小孔处，"
            "比上沿低一段，小球实际下落的距离比记录的多。",
            "The photogate position was read at the top edge of its frame, but the beam passes "
            "through small holes lower down, so the ball actually falls farther than recorded.",
            "x_offset", ("slope_low", "intercept_pos", "curve_up"),
            magnitude=Prior.uniform(0.3, 2.0, 101),
            magnitude_label_zh="光束低于读数位置", magnitude_unit="cm",
            fix_zh="读光电门位置时对准光束中心（两臂上小孔的位置）。",
            fix_en="Read the photogate position at the centre of the beam holes.",
        ),
        # 推导见 _drag_factor：t = √(2h/g) · arccosh(e^{κh})/√(2κh) / √(1−ε)，κ = ρC_dA/(2m)。
        #   乒乓球大小的轻球 κ ≈ 0.1~0.2 /m，1 m 处时间长约 2%~3%；实心钢球 κ ≈ 0.003 /m，可忽略。
        #   浮力 ε = ρ_空气/ρ_球 也让表观 g 变小（乒乓球约 1.5%）。
        Cause(
            DRAG, "小球太轻，空气阻力不可忽略", "ball too light: air drag matters",
            "用的不是实心钢球，而是顶上嵌了小铁片、能被电磁铁吸住的塑料球或空心球；"
            "它很轻，下落时空气阻力和浮力相对它的重力不能忽略，下落得越快阻力越大。",
            "Instead of a solid steel ball, a light plastic or hollow ball with an iron cap (so the "
            "magnet can hold it) was used; air drag and buoyancy are not negligible compared with "
            "its weight, and drag grows as it speeds up.",
            "bend", ("slope_high", "curve_up"),
            magnitude=Prior.loguniform(0.015, 0.3, 601),
            magnitude_label_zh="阻力系数 κ = ρC_dA/(2m)", magnitude_unit="1/m",
            fix_zh="换用实心小钢球（密度大、体积小）。",
            fix_en="Use a small solid steel ball.",
        ),
        # 推导：每次挪动光电门后读位置有 ±σ_h 的误差（视差、门没拧紧下滑），
        #   时间误差 δt = δh/v，v = √(2gh)；用高度中点处的速度近似成常数：σ_t' = σ_h / v_mid。
        Cause(
            SCATTER, "光电门位置每次都读得不准", "photogate position misread each time",
            "每次移动光电门以后，读它在立柱上的位置时眼睛高低不一，或者螺丝没拧紧、门稍微滑下来一点，"
            "每个高度都有大小、方向不同的误差。",
            "Each time the photogate is moved, its position is read from a different eye height, or "
            "the clamp slips a little, so every height carries its own error.",
            "scatter", ("scatter_excess",),
            magnitude=Prior.uniform(0.2, 1.5, 66),
            magnitude_label_zh="高度读数的额外误差", magnitude_unit="cm",
            fix_zh="拧紧光电门后再读数，视线与光束小孔平齐。",
            fix_en="Tighten the clamp and read with your eye level with the beam holes.",
        ),
        # 推导（仅 start="gate"）：上方光电门放在小球下沿下方 d0 处，小球经过它时已有速度 √(2gd0)。
        #   t = β(√(h + d0) − √d0)，就像从静止开始下落的前一段没被计时。
        Cause(
            SPEED, "上方光电门离小球太远", "upper photogate too far below the ball",
            "上方光电门没有紧贴在小球下面，而是低了一段，小球经过它开始计时的时候已经在往下运动了。",
            "The upper photogate is not just below the ball but some distance lower, so the ball is "
            "already moving when it starts the timer.",
            "bend", ("intercept_neg", "slope_low", "curve_up"),
            magnitude=Prior.loguniform(0.1, 3.0, 601),
            magnitude_label_zh="小球下沿到上方光束的距离", magnitude_unit="cm",
            fix_zh="把上方光电门移到小球正下方、光束刚好贴着小球下沿。",
            fix_en="Move the upper gate so its beam is just below the ball.",
        ),
    )

    protocols = {
        "single": Protocol("single", "只测一次（100 cm）", "single drop (100 cm)", (("main", 100.0),)),
        "standard": Protocol(
            "standard", "光电门放在 6 个高度，各测一次", "photogate at 6 heights, one drop each",
            tuple(("main", h) for h in (20.0, 40.0, 60.0, 80.0, 100.0, 120.0)),
        ),
    }

    menu = (
        MenuItem("height_check", "从小球下沿重新量到光束中心（60 cm 这一组）",
                 "re-measure the 60 cm height from ball bottom to beam centre", 2.0,
                 (("height_check", 60.0),)),
        MenuItem("steel_ball", "换成同样大小的实心钢球，在 100 cm 处测 3 次",
                 "drop a solid steel ball from 100 cm, 3 times", 3.0, (("steel", 100.0),) * 3),
        MenuItem("paper_shim", "在电磁铁上垫一张薄纸，在 60 cm 处测 2 次",
                 "put a paper spacer on the magnet; drop from 60 cm twice", 2.0, (("shim", 60.0),) * 2),
        MenuItem("reposition_80", "把光电门移开再重新对准 80 cm，测一次，重复 3 次",
                 "move the gate away and back to 80 cm, drop once; do this 3 times", 4.0,
                 (("main", 80.0),) * 3),
        MenuItem("gap_check", "量小球下沿到上方光电门光束的距离", "measure the gap to the upper beam",
                 1.0, (("gap_check", 0.0),)),
    )

    def __init__(self, start: str = "magnet"):
        if start not in ("magnet", "gate"):
            raise ValueError("start 只能是 'magnet' 或 'gate'")
        self.start = start
        drop_causes = _GATE_ONLY_CAUSES if start == "magnet" else _MAGNET_ONLY_CAUSES
        drop_kinds = _GATE_ONLY_KINDS if start == "magnet" else _MAGNET_ONLY_KINDS
        self.causes = tuple(c for c in type(self).causes if c.id not in drop_causes)
        self.obs_kinds = {k: v for k, v in type(self).obs_kinds.items() if k not in drop_kinds}
        self.menu = tuple(m for m in type(self).menu if all(k in self.obs_kinds for k, _ in m.plan))

    def relevant_nuisance(self, cause_id, kinds):
        return ()

    def predict(self, kind, settings, order, params, cause_id):
        P = params["mag"].shape[0]
        n = settings.size
        L = len(self.linear_params(cause_id))
        mag = params["mag"]  # (P, 1)
        c = np.zeros((P, n, 1))
        U = np.zeros((P, n, 1, L))
        if kind in TIMED_KINDS:
            h = np.broadcast_to(settings[None, :] / 100.0, (P, n))  # m，记录的高度
            if cause_id == TOP:
                h = np.maximum(h - mag / 100.0, 1e-6)
            elif cause_id == MARK:
                h = h + mag / 100.0
            if cause_id == SPEED:
                d0 = mag / 100.0
                g = np.sqrt(h + d0) - np.sqrt(d0)
            elif cause_id == DRAG and kind != "steel":
                g = np.sqrt(h) * _drag_factor(mag, h)
            else:
                g = np.sqrt(h)
            U[:, :, 0, 0] = g
            if cause_id == DELAY and kind != "shim":  # 垫纸后剩磁吸力很小，延迟近似为 0（假设）
                U[:, :, 0, 1] = 1.0
            return c, U
        if kind == "height_check":
            if cause_id == TOP:
                c[:] = mag[:, :, None]
            elif cause_id == MARK:
                c[:] = -mag[:, :, None]
            return c, U
        if kind == "gap_check":
            if cause_id == SPEED:
                c[:] = mag[:, :, None]
            return c, U
        raise KeyError(kind)

    def reading_sigma(self, kind, params, cause_id):
        base = super().reading_sigma(kind, params, cause_id)
        if kind == "main" and cause_id == SCATTER:
            extra = params["mag"][:, :, None] / 100.0 / V_MID
            return np.sqrt(base**2 + extra**2)
        return base

    def linearise(self, ds: Dataset) -> LinearData:
        idx = [i for i, o in enumerate(ds.observations) if o.kind == "main"]
        h = np.array([ds.observations[i].setting for i in idx]) / 100.0
        y = np.array([ds.observations[i].values[0] for i in idx])
        sy = np.full(y.shape, float(np.sqrt(T_GATE.sigma**2 + T_GATE.resolution**2 / 12)))
        return LinearData(np.sqrt(h), y, sy, np.array(idx), "√h (√m)", "下落时间 t (s)", BETA0, 0.0)
