"""用单摆测量重力加速度（留出实验，远迁移，高中）。

学生的做法（人教版选择性必修第一册第二章第 5 节）：细线一端固定在铁架台上，另一端系小钢球；
用刻度尺量摆线长、游标卡尺量摆球直径，摆长 L = 线长 + 半径（悬点到球心）；把摆球拉开不超过 5° 释放，
在竖直平面内摆动，从摆球经过最低点开始计时，测 N = 30 次全振动的总时间 t，T = t/N。
依次取 L = 40、55、70、85、100 cm。计时默认用手机传感器（如 phyphox 的光电/磁力计秒表，摆球经过最低点自动触发，
不含人的反应时间）；也可选手按秒表（"stopwatch"，每次按键约 0.1 s 的随机误差）。

面板坐标：横轴 x = 记录的摆长 L（m），纵轴 y = T²（s²）。理想直线 T² = (4π²/g)·L，截距 0，
斜率已知（g = 9.8 m/s²）；这里的理想斜率含 5° 摆角的周期修正 (1/AGM(1, cos2.5°))² ≈ 1.00095。

读数噪声的说明：总时间 t 的高斯部分合并了手机触发抖动（约 0.01 s）和量摆长的误差（约 1 mm），
后者折算到 t：δt = ½t·δL/L，在 40~100 cm 内约 0.03~0.05 s，取 0.04 s（接口要求噪声与控制量无关）。

已计入模型的小量：球形摆球的复摆修正 L_eff = L + 2r²/(5L)（r = 1 cm、L = 40 cm 时 0.025%）；
5° 摆角的周期修正。忽略的小量：摆线质量（m_线/m_球 ~ 0.4% 时 T² 约小 m_线/(6m_球) ≈ 0.07%）、
空气浮力和附加质量（钢球 ρ_空气/ρ_球 ~ 1.5×10⁻⁴）、30 次全振动中振幅的衰减（取平均振幅）。
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

N_OSC = 30  # 课本：测 30~50 次全振动
N_SHORT = 10  # 菜单：只数 10 次
THETA_STD = 5.0  # 课本要求的最大摆角（度），标准方案按 5° 计算周期修正
R_NOM_CM = 1.0  # 复摆修正用的摆球半径（该修正 ~10⁻⁴，用标称值）
LENGTHS_CM = (40.0, 55.0, 70.0, 85.0, 100.0)

# t 的高斯误差（s）：(计时器本身, 摆长误差折算到 30 次全振动)
TIMER_SIGMA = {"phone": 0.01, "stopwatch": math.sqrt(2) * 0.1}
LENGTH_SIGMA_T30 = 0.039

RANDOM = "pendulum.random_only"
TOP = "pendulum.top_of_bob"
AMP = "pendulum.large_amplitude"
MISCOUNT = "pendulum.miscount"
CONICAL = "pendulum.conical"
TIMING = "pendulum.timing_offset"

TIMING_KINDS = ("main", "main10", "careful")


def _period_factor(theta_rad):
    """有限摆角周期修正 T/T₀ = (2/π)K(sin(θ₀/2)) = 1/AGM(1, cos(θ₀/2))（精确，算术-几何平均收敛极快）。

    推导：能量守恒 θ̇² = 2(g/L)(cosθ − cosθ₀)，T = 4∫₀^θ₀ dθ/θ̇ = 4√(L/g)·K(k)，k = sin(θ₀/2)；
    K(k) = (π/2)(1 + k²/4 + …) 且 k² ≈ θ₀²/4，故 T ≈ T₀(1 + θ₀²/16)。
    """
    a = np.ones_like(np.asarray(theta_rad, dtype=float))
    b = np.cos(np.asarray(theta_rad, dtype=float) / 2.0)
    for _ in range(8):
        a, b = 0.5 * (a + b), np.sqrt(a * b)
    return 1.0 / a


K_STD = float(_period_factor(math.radians(THETA_STD)))


def _t_reading(n: int, timer: str) -> Reading:
    sig = math.sqrt(TIMER_SIGMA[timer] ** 2 + (LENGTH_SIGMA_T30 * n / N_OSC) ** 2)
    return Reading(f"t{n}", f"{n} 次全振动的总时间", f"time for {n} oscillations", "s",
                   resolution=0.01, sigma=round(sig, 4))


def _obs_kinds(timer: str, n_osc: int) -> dict:
    how = "手机传感器" if timer == "phone" else "秒表"
    return {
        "main": ObsKind(
            "main", f"拉开小角度释放，用{how}测 {n_osc} 次全振动的总时间", f"time {n_osc} oscillations",
            (_t_reading(n_osc, timer),),
            setting_label_zh="摆长 L", setting_label_en="pendulum length L", setting_unit="cm",
        ),
        "main10": ObsKind(
            "main10", f"同样释放，只测 {N_SHORT} 次全振动的总时间", f"time {N_SHORT} oscillations",
            (_t_reading(N_SHORT, timer),),
            setting_label_zh="摆长 L", setting_label_en="pendulum length L", setting_unit="cm",
        ),
        "careful": ObsKind(
            "careful", f"摆角小于 5°、在竖直平面内轻轻释放，测 {n_osc} 次全振动的总时间",
            f"release gently in a vertical plane below 5°, time {n_osc} oscillations",
            (_t_reading(n_osc, timer),),
            setting_label_zh="摆长 L", setting_label_en="pendulum length L", setting_unit="cm",
        ),
        "bob_d": ObsKind(
            "bob_d", "用游标卡尺测摆球直径", "measure the bob diameter with a vernier caliper",
            (Reading("bob_d", "摆球直径", "bob diameter", "cm", resolution=0.002, sigma=0.002),),
        ),
        "length_check": ObsKind(
            "length_check", "从悬点量到摆球球心，重新测量这一组的摆长", "re-measure the length to the bob centre",
            (Reading("L_center", "悬点到球心的长度（重测）", "re-measured length to the centre", "cm",
                     resolution=0.1, sigma=0.1),),
            setting_label_zh="原记录的摆长", setting_label_en="recorded length", setting_unit="cm",
        ),
    }


CAUSES = (
    Cause(
        RANDOM, "只有随机误差", "random error only",
        "摆长从悬点量到球心，摆角很小，摆球在竖直平面内摆动，计数和计时都正确，数据偏离只来自读数的随机误差。",
        "Length to the bob centre, small swings in a vertical plane, correct counting and timing; "
        "deviations come only from random reading error.",
        "none", ("none",),
        fix_zh="多测几组、取平均即可。", fix_en="Take more readings and average.",
    ),
    # 推导：记录摆长 L' = L − r（量到球顶），T² = (4π²/g)(L' + r)，斜率不变、截距 +(4π²/g)·r。
    Cause(
        TOP, "摆长量到了摆球顶端", "length measured to the top of the bob",
        "量摆长时只量了从悬点到摆球顶端（摆线下端）的长度，没有加上摆球的半径。",
        "The length was measured from the pivot to the top of the bob (end of the string) and the bob "
        "radius was not added.",
        "x_offset", ("intercept_pos",),
        magnitude=Prior.point(1.0),
        fix_zh="用游标卡尺测摆球直径 d，摆长 = 线长 + d/2（悬点到球心）。",
        fix_en="Measure the bob diameter d with a caliper; length = string length + d/2.",
    ),
    # 推导：见 _period_factor，T = T₀/AGM(1, cos(θ₀/2)) ≈ T₀(1 + θ₀²/16)；
    # 每次都从同样的角度 θ₀ 释放，修正因子与摆长无关 → T² 整体乘 (1 + θ₀²/16)²（30° 时 +3.5%）。
    # （若每次拉开同样的水平距离 A，θ₀ = A/L 随摆长变，图线会弯曲；这里按固定角度建模。）
    Cause(
        AMP, "摆角太大", "amplitude too large",
        "释放摆球时拉开的角度远大于 5°，每次都从差不多同样大的角度释放；摆角大时回复力比 mgθ 小，摆得更慢。",
        "The bob was pulled far beyond 5° before release, about the same angle every time; at large "
        "angles the restoring force is weaker than mgθ, so the swing is slower.",
        "y_scale", ("slope_high",),
        magnitude=Prior.uniform(15.0, 45.0, 31),
        magnitude_label_zh="释放时的摆角 θ₀", magnitude_unit="°",
        fix_zh="把摆球拉开不超过 5°（摆长 1 m 时水平拉开不超过约 9 cm）再释放。",
        fix_en="Release from no more than 5° (about 9 cm sideways for a 1 m pendulum).",
    ),
    # 推导：计时内实际完成 N + m 次全振动，t = (N + m)T，记录 T' = t/N = (1 + m/N)T，
    # T'² = (1 + m/N)²·T²：N = 30、m = −1（开始计时时就数"1"）时斜率 ×0.934，m = +1 时 ×1.068。
    Cause(
        MISCOUNT, "全振动次数数错了", "oscillations miscounted",
        "数全振动次数时，开始计时那一下就数了\"1\"（或者结束时多数、少数了一次），计时的这段时间里摆球实际摆动的次数"
        "和记录的次数差一次，算周期时却仍用 T = t/N。",
        "When counting, the student said \"one\" at the moment timing started (or counted one too many/few "
        "at the end), so the timed interval holds one oscillation more or less than the recorded N, yet "
        "T = t/N was used.",
        "y_scale", ("slope_low", "slope_high"),
        magnitude=Prior.sym_uniform(1.0, 1.0, 2),
        magnitude_label_zh="实际次数 − 记录次数", magnitude_unit="次",
        fix_zh="摆球第一次经过最低点时数\"0\"并开始计时，之后每经过一次（同方向）数一次。",
        fix_en="Say \"zero\" as the bob first passes the lowest point and start timing then.",
    ),
    # 推导：圆锥摆，拉力 F：F cosα = mg，F sinα = mω²·L sinα → ω² = g/(L cosα)，
    # T² = (4π²L/g)·cosα，比单摆短（α = 20° 时 T² 小 6%）。
    # 细长椭圆轨道（短半轴角 b 很小）逐次计数时周期只改变 b² 的二阶量（Olsson 1981 进动球摆），可忽略。
    Cause(
        CONICAL, "摆球做了圆锥摆运动", "bob moved as a conical pendulum",
        "释放时手给了摆球一个侧向的速度，摆球没有在竖直平面内来回摆动，而是绕着竖直线在水平面内转圈（圆锥摆），"
        "学生按摆球回到同一位置数一次全振动。",
        "The hand gave the bob a sideways push at release, so instead of swinging in a vertical plane it "
        "circled around the vertical (a conical pendulum); one revolution was counted as one oscillation.",
        "y_scale", ("slope_low",),
        magnitude=Prior.uniform(8.0, 25.0, 35),
        magnitude_label_zh="摆线与竖直方向的夹角 α", magnitude_unit="°",
        fix_zh="释放前让摆球静止，松手时不要带侧向力，确认摆球在竖直平面内摆动后再开始计时。",
        fix_en="Hold the bob still and let go without a sideways push; check it swings in one plane.",
    ),
    # 推导：t' = N·T + τ，T' = T + τ/N，T'² = T² + 2Tτ/N + τ²/N²；T ∝ √L，
    # 在 40~100 cm 内 √L ≈ 0.39 + 0.61·L（m），所以 2Tτ/N 同时抬高截距和斜率（√L 的弯曲很弱）：
    # τ > 0 → 截距偏正且斜率偏大；τ < 0 → 截距偏负且斜率偏小（N = 30、τ = 0.6 s 时截距约 +0.03 s²、斜率约 +1%）。
    Cause(
        TIMING, "计时的起止时刻没有对准", "timing start/stop misaligned",
        "开始计时和停止计时的时刻与摆球经过最低点的时刻没有对齐，而且每次都偏同一个方向（例如先按下开始键再松手释放、"
        "用秒表时开始和停止的反应延迟不同），每次记录的总时间都多了（或少了）差不多同样的一段时间。",
        "Timing did not start and stop exactly as the bob passed the lowest point, and the mismatch was "
        "the same each time (e.g. pressing start before releasing the bob, or different reaction delays "
        "at start and stop), so every total time is off by about the same amount.",
        "y_offset", ("intercept_pos", "slope_high", "intercept_neg", "slope_low"),
        linear=(LinearParam("tau", 0.0, 0.3, "总时间的固定偏差", "s"),),
        fix_zh="摆球经过最低点时开始计时（同一位置停止）；用手机传感器自动触发计时。",
        fix_en="Start and stop timing as the bob passes the lowest point, ideally with a sensor trigger.",
    ),
)


class Pendulum(Experiment):
    id = "pendulum"
    name_zh = "用单摆测量重力加速度"
    name_en = "Measuring g with a simple pendulum"
    held_out = True
    transfer = "far"

    obs_kinds = _obs_kinds("phone", N_OSC)

    # 摆球半径（cm）：学校常用直径 1.5~3 cm 的钢球
    nuisance = {"r_cm": Prior.uniform(0.75, 1.5, 16)}

    # 1/√g（s/√m）：g = 9.80 ± 0.01 m/s²（当地 g 已知到这个程度），只给"理想斜率"留一点余地；
    # 斜率的统计误差约 0.2%，所以这里不能放宽到各地 g 的全部差别（±0.3%），否则干净数据也会显得斜率偏
    linear = (LinearParam("inv_sqrt_g", 1.0 / math.sqrt(G), 0.5 * 0.001 / math.sqrt(G), "1/√g", "s/√m"),)

    causes = CAUSES

    protocols = {
        "single": Protocol("single", "单次测量（L = 85 cm）", "single run (L = 85 cm)", (("main", 85.0),)),
        "standard": Protocol(
            "standard", "课本标准方案（5 个摆长）", "textbook worksheet (5 lengths)",
            tuple(("main", L) for L in LENGTHS_CM),
        ),
    }

    menu = (
        MenuItem("caliper_bob", "用游标卡尺测摆球直径", "measure the bob diameter", 2.0, (("bob_d", 0.0),)),
        MenuItem("length_check", "从悬点量到球心，重测 85 cm 那一组的摆长", "re-measure the 85 cm length to the bob centre",
                 3.0, (("length_check", 85.0),)),
        MenuItem("count10", f"在 100 cm 处只数 {N_SHORT} 次全振动再计时一次", f"time {N_SHORT} oscillations at 100 cm",
                 2.0, (("main10", 100.0),)),
        MenuItem("small_angle", "在 100 cm 处摆角小于 5°、在竖直平面内轻轻释放后重测",
                 "repeat at 100 cm with a gentle planar release below 5°", 3.0, (("careful", 100.0),)),
        MenuItem("repeat_40", "在 40 cm 处重复测一次", "repeat the 40 cm measurement", 2.0, (("main", 40.0),)),
    )

    def __init__(self, timer: str = "phone", n_osc: int = N_OSC):
        assert timer in TIMER_SIGMA, timer
        self.timer = timer
        self.n_osc = int(n_osc)
        if timer != "phone" or self.n_osc != N_OSC:
            self.obs_kinds = _obs_kinds(timer, self.n_osc)

    def relevant_nuisance(self, cause_id, kinds):
        if cause_id == TOP or "bob_d" in kinds:
            return ("r_cm",)
        return ()

    def predict(self, kind, settings, order, params, cause_id):
        P = params["mag"].shape[0]
        n = settings.size
        L = len(self.linear_params(cause_id))
        U = np.zeros((P, n, 1, L))
        mag = params["mag"]
        if kind in TIMING_KINDS:
            N = N_SHORT if kind == "main10" else self.n_osc
            Lc = np.broadcast_to(settings[None, :] / 100.0, (P, n))
            if cause_id == TOP:
                Lc = Lc + params["r_cm"] / 100.0
            L_eff = Lc + 0.4 * (R_NOM_CM / 100.0) ** 2 / Lc  # 球形摆球的复摆修正
            k = K_STD
            if cause_id == AMP and kind != "careful":
                k = _period_factor(np.radians(mag))
            cone = 1.0
            if cause_id == CONICAL and kind != "careful":
                cone = np.sqrt(np.cos(np.radians(mag)))
            m = mag if cause_id == MISCOUNT else 0.0
            # t = (N + m)·2π√(L_eff)·K·√cosα·(1/√g) + τ
            U[:, :, 0, 0] = (N + m) * 2.0 * math.pi * np.sqrt(L_eff) * k * cone
            if cause_id == TIMING:
                U[:, :, 0, 1] = 1.0
            return np.zeros((P, n, 1)), U
        if kind == "bob_d":
            return np.broadcast_to(2.0 * params["r_cm"], (P, n))[:, :, None], U
        if kind == "length_check":
            Lc = np.broadcast_to(settings[None, :], (P, n))
            if cause_id == TOP:
                Lc = Lc + params["r_cm"]
            return np.broadcast_to(Lc, (P, n))[:, :, None], U
        raise KeyError(kind)

    def linearise(self, ds: Dataset) -> LinearData:
        idx = [i for i, o in enumerate(ds.observations) if o.kind == "main"]
        x = np.array([ds.observations[i].setting for i in idx]) / 100.0
        t = np.array([ds.observations[i].values[0] for i in idx])
        T = t / self.n_osc
        r = self.obs_kinds["main"].readings[0]
        sd_t = float(np.sqrt(r.sigma**2 + r.resolution**2 / 12))
        sy = 2.0 * T * sd_t / self.n_osc
        ideal = 4.0 * math.pi**2 / G * K_STD**2
        return LinearData(x, T**2, sy, np.array(idx), "摆长 L (m)", "T² (s²)", ideal, 0.0)
