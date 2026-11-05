"""斜面上测动摩擦因数（高中，留出实验，近迁移）。

人教版没有把它列为学生必做实验；这里按必修一"用光电门测速度"和匀变速规律 v² − v₀² = 2ax 的方法组织，
作为"standard"方案：
木块（顶上装宽 1.00 cm 的遮光片）从斜面上方由静止释放，依次经过相距 d = 50.0 cm 的光电门 A、B，
数字计时器给出挡光时间 t_A、t_B（0.1 ms），学生算 v = Δx/t，a = (v_B² − v_A²)/(2d)，记到 0.01 m/s²。
量角器读倾角 θ = 30°、35°、40°、45°、50°，每个角度测 3 次。"single" 方案：只在 40° 测 3 次。
构造参数 timer = "ticker_tape" 时改用打点计时器（纸带逐差法求 a），原因清单换掉与光电门有关的两条。

物理：a = g(sin θ − μ cos θ)。两边除以 cos θ：a/cos θ = g·tan θ − μg。
面板坐标：横轴 = tan θ（θ 为量角器读数），纵轴 = a/cos θ（m/s²）；理想斜率 = g（已知），
截距 = −μg 含待测量 μ，没有"理想值"（ideal_intercept = nan，面板不检验截距）。

几点物理细节：
- 两光电门测速法与释放方式无关：v_B² − v_A² = 2ad 对任何初速度都成立，所以"推了一下才放手"不影响 a。
- v = Δx/t 是挡光期间的平均速度（中间时刻速度），比经过光电门中点时的速度略小；代入后
  a 偏大 a²(τ_A² − τ_B²)/(8d)，τ 为挡光时间，相对误差约 1e−4，忽略。
- 要能由静止滑下，需 tan θ > μ_静；μ 先验上限约 0.45，最小角 30°（tan 30° = 0.58）留了余量。
- 读数随机误差：木板各处摩擦不均（δμ ≈ 0.005 → δa ≈ g cos θ·δμ ≈ 0.04 m/s²，近似与角度无关），
  再加上计时器取整，合计取常数 σ_a = 0.05 m/s²。注意 0.1 ms 取整的影响随角度变大：δa ≈ (v_B²/d)·(δt/t_B)，
  遮光片 1 cm、在 A 上方约 15 cm 处释放时，30° 约 0.03 m/s²，50° 约 0.1 m/s²（与摩擦不均合成约 0.06~0.12）。
  引擎的读数噪声不能随控制量变化，所以这里取常数；σ_a = 0.05 相当于计时器分辨率 0.01 ms，
  用 0.1 ms 计时器的真实数据在大角度处会显得"散布偏大"。
"""

from __future__ import annotations

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

D_REC = 50.0  # 记录的光电门间距（cm）
FLAG_REC = 1.000  # 记录的遮光片宽度（cm，标称值）
L_BOARD = 100.0  # 转轴到斜面顶端测高点的板长（cm）
THETA_REF = 40.0  # 压力参考角（°）：μ(N) 中的 N_ref = m g cos 40°
ANGLES = (30.0, 35.0, 40.0, 45.0, 50.0)
SIGMA_A = 0.05


def _acc_readings(prefix: str, label: str) -> tuple[Reading, ...]:
    return tuple(
        Reading(f"{prefix}{i}", f"{label}（第 {i} 次）", f"{prefix}{i}", "m/s²", resolution=0.01, sigma=SIGMA_A)
        for i in (1, 2, 3)
    )


_COMMON_KINDS = {
    "main": ObsKind(
        "main", "在该倾角下由静止释放木块，测 3 次加速度", "release from rest, measure the acceleration 3 times",
        _acc_readings("a", "加速度"),
        setting_label_zh="斜面倾角（量角器读数）", setting_label_en="incline angle", setting_unit="°",
    ),
    "heavy": ObsKind(
        "heavy", "木块上加一个等质量砝码，测 3 次加速度", "double the block mass, measure 3 times",
        _acc_readings("ah", "加重后加速度"),
        setting_label_zh="斜面倾角（量角器读数）", setting_label_en="incline angle", setting_unit="°",
    ),
    "height": ObsKind(
        "height", "用刻度尺量斜面顶端的高度（板长 100.0 cm）", "measure the height of the raised end (board 100.0 cm)",
        (Reading("height", "斜面顶端高度", "raised-end height", "cm", resolution=0.1, sigma=0.1),),
        setting_label_zh="量角器读数", setting_label_en="protractor reading", setting_unit="°",
    ),
}

_PHOTOGATE_KINDS = {
    "flag": ObsKind(
        "flag", "用游标卡尺测遮光片宽度", "measure the flag width with a vernier caliper",
        (Reading("flag_width", "遮光片宽度", "flag width", "cm", resolution=0.002, sigma=0.001),),
    ),
    "spacing": ObsKind(
        "spacing", "对准两个光电门的光线位置，重量间距", "re-measure the gate spacing beam to beam",
        (Reading("gate_spacing", "两光线间距", "beam-to-beam spacing", "cm", resolution=0.1, sigma=0.05),),
    ),
}

_C_RANDOM = Cause(
    "incline_friction.random_only", "只有随机误差", "random error only",
    "测量步骤正确，数据偏离只来自木板各处摩擦不均和计时取整等随机因素。",
    "The procedure is correct; deviations come only from random factors such as uneven friction.",
    "none", ("none",),
    fix_zh="每个角度多测几次取平均即可。", fix_en="Repeat each angle and average.",
)

# 量角器零点偏差 δ（真实倾角 = 读数 + δ）：a/(g cos θ_r) = (cos δ + μ sin δ)·tan θ_r + (sin δ − μ cos δ)。
# 仍是严格的直线：斜率变为 g(cos δ + μ sin δ) ≈ g(1 + μδ)（二阶小量，μ = 0.3、δ = 4° 时约 +2%），
# 截距给出的 μ 却变成 μ cos δ − sin δ ≈ μ − δ（4° → 偏小 0.07，误差很大）。量斜面高度 h = L sin(θ_r + δ) 可直接暴露。
# 只建模量角器自身的零点偏差：若是桌面倾斜，从桌面量的高度给出的仍是 sin θ_r，量高度查不出来，所以不归入本原因。
_C_PROTRACTOR = Cause(
    "incline_friction.protractor_zero", "量角器零点不准", "protractor zero offset",
    "量角器（或角度指针）的零刻度没有对准水平，读出的倾角与斜面真实倾角差一个固定角度。",
    "The protractor's zero is not aligned with the horizontal, so every angle "
    "reading differs from the true incline angle by the same amount.",
    "x_offset", ("slope_high", "slope_low"),
    magnitude=Prior.sym_uniform(1.0, 5.0, 10),
    magnitude_label_zh="真实倾角 − 量角器读数", magnitude_unit="°",
    fix_zh="用刻度尺量斜面高度 h 和板长 L，用 sin θ = h/L 校准角度。",
    fix_en="Measure the height and length of the incline and use sin θ = h/L.",
)

# 遮光片宽度记错：v = Δx_记/t，a ∝ v² → a_记 = (Δx_记/Δx_真)²·a = (1 + ε)²·a，纯比例。
# 斜率 = (1+ε)² g；截距也乘同一个因子，所以用"截距/拟合斜率"求 μ 反而不受影响。
_C_FLAG = Cause(
    "incline_friction.flag_width", "遮光片宽度用错", "wrong flag width",
    "计算速度时用的是遮光片的标称宽度（或用刻度尺粗量的宽度），与遮光片的实际宽度不一样。",
    "The speed was computed with the nominal (or roughly measured) flag width, which differs from the "
    "actual width.",
    "y_scale", ("slope_high", "slope_low"),
    magnitude=Prior.sym_uniform(0.02, 0.10, 10),
    magnitude_label_zh="记录宽度/实际宽度 − 1", magnitude_unit="",
    fix_zh="用游标卡尺测遮光片宽度；或用 μ = −截距/拟合斜率，比例误差会约掉。",
    fix_en="Measure the flag with a caliper, or take μ = −intercept/fitted slope so the scale cancels.",
)

# 光电门间距从 A 的前侧面量到 B 的后侧面：d_记 = d + w（w = 一个光电门的厚度），
# a_记 = (v_B² − v_A²)/(2 d_记) = a·(d_记 − w)/d_记，纯比例、总是偏小。
_C_SPACING = Cause(
    "incline_friction.gate_spacing", "光电门间距量多了", "gate spacing measured front-to-rear",
    "量两个光电门的距离时，从第一个光电门的前侧面量到第二个光电门的后侧面，多量了一个光电门的厚度；"
    "真正起作用的是两条光线之间的距离。",
    "The gate spacing was measured from the front face of the first gate to the back face of the second, "
    "adding one gate thickness; what matters is the distance between the two light beams.",
    "y_scale", ("slope_low",),
    magnitude=Prior.uniform(1.0, 4.0, 7),
    magnitude_label_zh="多量的长度", magnitude_unit="cm",
    fix_zh="对准两个光电门的光线（发射孔中心）量间距。",
    fix_en="Measure between the two light beams (centres of the emitter holes).",
)

# μ 随压力变化（不严格满足 f = μN）：μ = μ₀·(N/N_ref)^(−γ)，N = m g cos θ。
# 倾角越大压力越小、μ 越大 → 大 tan θ 处 a/cos θ 偏低，主要表现为斜率偏小。
# 质量加倍时按 f = μN 加速度不变（与质量无关），这里 μ 变为 2^(−γ) 倍 → a 变大，可以暴露。
_C_LOAD = Cause(
    "incline_friction.mu_load", "摩擦因数随压力变化", "friction coefficient depends on load",
    "木块和木板之间的摩擦并不严格与压力成正比：倾角变大时木块对木板的压力变小，"
    "而这对接触面在压力小时摩擦因数略大。",
    "Friction is not exactly proportional to the normal force: at larger angles the block presses less "
    "on the board, and for this pair of surfaces the coefficient is slightly larger at small loads.",
    "bend", ("slope_low",),
    magnitude=Prior.uniform(0.05, 0.33, 8),
    magnitude_label_zh="μ ∝ N^(−γ) 中的 γ", magnitude_unit="",
    fix_zh="换压力不同（加砝码）的木块在同一角度比较；结论里注明 μ 只在所测压力范围内近似为常数。",
    fix_en="Compare different loads at the same angle; state that μ is only approximately constant.",
)

_C_SCATTER = Cause(
    "incline_friction.release_scatter", "每次释放方式不一样", "inconsistent release",
    "每次释放的位置和姿势都不一样：有时木块歪着滑下、蹭到斜面侧边，有时遮光片没有正对光线。",
    "The block was released differently each time: sometimes it slid crookedly and rubbed the side "
    "rail, sometimes the flag did not cross the beam squarely.",
    "scatter", ("scatter_excess",),
    magnitude=Prior.uniform(0.10, 0.40, 10),
    magnitude_label_zh="额外的加速度散布", magnitude_unit="m/s²",
    fix_zh="在同一位置、让木块摆正后由静止释放，必要时加导向挡板。",
    fix_en="Release from the same mark with the block square to the rail; add a guide if needed.",
)

# 纸带与限位孔、振针间的阻力 f（打点计时器方案）：a_记 = g(sin θ − μ cos θ) − f/m。
# 除以 cos θ 后多出 −(f/m)/cos θ = −(f/m)√(1 + tan²θ)，大角度处更低 → 斜率偏小；质量加倍时影响减半。
_C_TAPE = Cause(
    "incline_friction.tape_drag", "纸带受到阻力", "paper-tape drag",
    "纸带穿过打点计时器的限位孔时与孔壁摩擦，振针打点时也拖住纸带，木块下滑时多受一个向上的阻力。",
    "The paper tape rubs on the timer's guide slot and is held back by the striker, adding a backward "
    "force on the block.",
    "y_offset", ("slope_low",),
    magnitude=Prior.uniform(0.1, 0.6, 11),
    magnitude_label_zh="阻力产生的减速度 f/m", magnitude_unit="m/s²",
    fix_zh="让纸带与限位孔对正、不扭曲；或改用光电门。",
    fix_en="Align the tape with the guide slot, or use photogates instead.",
)

_MENU_COMMON = (
    MenuItem("measure_height", "量斜面顶端高度，用 sin θ = h/L 核对 30° 的读数",
             "measure the raised-end height at the 30° setting", 3.0, (("height", 30.0),)),
    MenuItem("heavy_block", "木块上加一个等质量砝码，在 40° 测 3 次",
             "double the block mass and measure 3 times at 40°", 4.0, (("heavy", 40.0),)),
    MenuItem("repeat_40", "在 40° 再测 3 次", "repeat 3 runs at 40°", 3.0, (("main", 40.0),)),
)
_MENU_PHOTOGATE = (
    MenuItem("caliper_flag", "用游标卡尺测遮光片宽度", "measure the flag width with a caliper", 2.0,
             (("flag", 0.0),)),
    MenuItem("remeasure_spacing", "对准光线位置，重量两个光电门的间距", "re-measure the gate spacing beam to beam",
             2.0, (("spacing", 0.0),)),
)


class InclineFriction(Experiment):
    id = "incline_friction"
    name_zh = "斜面上测动摩擦因数"
    name_en = "Coefficient of kinetic friction on an incline"
    held_out = True
    transfer = "near"

    obs_kinds = {**_COMMON_KINDS, **_PHOTOGATE_KINDS}
    nuisance = {}

    # 木块与木板间的动摩擦因数：0.15~0.45（±3σ）
    linear = (LinearParam("mu", 0.30, 0.05, "动摩擦因数 μ", ""),)

    causes = (_C_RANDOM, _C_PROTRACTOR, _C_FLAG, _C_SPACING, _C_LOAD, _C_SCATTER)

    protocols = {
        "single": Protocol("single", "单个角度（40°，测 3 次）", "single angle (40°, 3 runs)", (("main", 40.0),)),
        "standard": Protocol(
            "standard", "5 个倾角，每个测 3 次", "5 angles, 3 runs each", tuple(("main", t) for t in ANGLES),
        ),
    }

    menu = _MENU_PHOTOGATE + _MENU_COMMON

    def __init__(self, timer: str = "photogate"):
        if timer not in ("photogate", "ticker_tape"):
            raise ValueError(timer)
        self.timer = timer
        if timer == "ticker_tape":
            self.obs_kinds = dict(_COMMON_KINDS)
            self.causes = (_C_RANDOM, _C_PROTRACTOR, _C_TAPE, _C_LOAD, _C_SCATTER)
            self.menu = _MENU_COMMON

    def relevant_nuisance(self, cause_id, kinds):
        return ()

    def predict(self, kind, settings, order, params, cause_id):
        P = params["mag"].shape[0]
        n = settings.size
        L = len(self.linear_params(cause_id))
        U = np.zeros((P, n, 1, L))
        c = np.zeros((P, n, 1))
        name = cause_id.split(".", 1)[1]
        mag = params["mag"]  # (P, 1)
        if kind == "flag":
            c[:, :, 0] = FLAG_REC / (1.0 + mag) if name == "flag_width" else FLAG_REC
            return c, U
        if kind == "spacing":
            c[:, :, 0] = D_REC - mag if name == "gate_spacing" else D_REC
            return c, U
        th = np.deg2rad(settings)[None, :]  # 量角器读数
        if name == "protractor_zero":
            th = th + np.deg2rad(mag)  # 真实倾角
        if kind == "height":
            c[:, :, 0] = np.broadcast_to(L_BOARD * np.sin(th), (P, n))
            return c, U
        if kind not in ("main", "heavy"):
            raise KeyError(kind)
        km = 2.0 if kind == "heavy" else 1.0  # 质量倍数
        s = 1.0  # 读数比例（遮光片、间距）
        if name == "flag_width":
            s = (1.0 + mag) ** 2
        elif name == "gate_spacing":
            s = (D_REC - mag) / D_REC
        load = 1.0  # μ 的压力修正因子
        if name == "mu_load":
            load = (km * np.cos(th) / np.cos(np.deg2rad(THETA_REF))) ** (-mag)
        c[:, :, 0] = np.broadcast_to(s * G * np.sin(th), (P, n))
        if name == "tape_drag":
            c[:, :, 0] -= mag / km
        U[:, :, 0, 0] = np.broadcast_to(-s * G * np.cos(th) * load, (P, n))
        return c, U

    def reading_sigma(self, kind, params, cause_id):
        base = super().reading_sigma(kind, params, cause_id)
        if kind in ("main", "heavy") and cause_id == "incline_friction.release_scatter":
            return np.sqrt(base**2 + params["mag"][:, :, None] ** 2)
        return base

    def linearise(self, ds: Dataset) -> LinearData:
        idx = [i for i, o in enumerate(ds.observations) if o.kind == "main"]
        th = np.deg2rad([ds.observations[i].setting for i in idx])
        a = np.array([np.mean(ds.observations[i].values) for i in idx])
        rd = self.obs_kinds["main"].readings
        sa = float(np.sqrt(rd[0].sigma**2 + rd[0].resolution**2 / 12) / np.sqrt(len(rd)))
        x = np.tan(th)
        y = a / np.cos(th)
        sy = sa / np.cos(th)
        return LinearData(x, y, sy, np.array(idx), "tan θ", "a/cos θ (m/s²)", G, float("nan"))
