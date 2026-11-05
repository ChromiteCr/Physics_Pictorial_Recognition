"""验证机械能守恒定律：小车沿倾斜轨道下滑（训练实验，高中）。

学生的做法（人教版必修第二册第八章第 5 节"倾斜导轨 + 光电门"方案，器材按 app 换成小车 + 轨道）：
轨道一端垫高、倾角固定；小车上装宽 d 的遮光片，从光电门上方竖直高度 h 处由静止释放，
光电门测出遮光时间 Δt，v = d/Δt；依次取 h = 5、10、15、20、25 cm，各测一次。
默认不记录轨道倾角。仪器可选：光电门（默认，计时分辨率 0.1 ms）或打点计时器（"ticker"，
一条纸带上取 5 个点，v_n = (x_{n+1} − x_{n−1})/(2T)，T = 0.02 s；匀加速时中间时刻速度公式是精确的）。

面板坐标：横轴 x = g·h（J/kg，h 是记录的下落高度），纵轴 y = ½v²（J/kg，v 是记录的速度）。
理想直线 y = x：斜率 1、截距 0（质量约掉，验证的是单位质量的能量）。

读数噪声的说明：光电门读数 Δt 的高斯部分 0.025 ms 合并了计时抖动、刻度尺量 h 的误差和释放位置
的微小差别（后两者折算到 Δt：δΔt = ½Δt·δh/h），取整 0.1 ms 由引擎另算。接口要求噪声与控制量无关，
实际上 h 越小，h 的相对误差折算到 Δt 越大，这里取 5~25 cm 的折中值。

忽略的小量（量级都远小于读数误差）：
- 遮光片测的是挡光期间的平均速度 = 中间时刻的速度，比遮光片中心过光电门时的速度略小，
  v_c² − v̄² = (aΔt/2)²，相对差 ~ 10⁻⁴；
- 空气阻力；遮光片中心与小车其他点下落高度相同（刚体平动），用哪一点量 h 都行，但两端必须是同一点。
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

FLAG_MM = 10.0  # 遮光片标称宽度（学生计算速度时用的值）
T_TICK = 0.02  # 打点计时器周期（50 Hz）
CART_KG = 0.30  # 小车（含遮光片、车轮）总质量
LOAD_KG = 0.50  # 菜单里加的配重
HEIGHTS_CM = (5.0, 10.0, 15.0, 20.0, 25.0)

DT = Reading("dt", "挡光时间", "gate blocking time", "ms", resolution=0.1, sigma=0.025)
SPAN = Reading("dot_span", "该点前后相邻两点的间距", "distance between the two neighbouring dots", "cm",
               resolution=0.1, sigma=0.03)

MAIN_KINDS = ("main", "loaded", "string_release")

RANDOM = "energy.random_only"
FLAG = "energy.flag_width"
SLANT = "energy.slant_length"
FRICTION = "energy.track_friction"
INERTIA = "energy.wheel_inertia"
WRONG_POINT = "energy.wrong_height_point"
PUSH = "energy.push"


def _obs_kinds(timer: str) -> dict:
    r = DT if timer == "photogate" else SPAN
    how = "光电门测挡光时间" if timer == "photogate" else "纸带上量相邻两点间距"
    kinds = {
        "main": ObsKind(
            "main", f"从高度 h 处由静止释放小车，{how}", "release from height h, record the speed reading", (r,),
            setting_label_zh="下落高度 h", setting_label_en="drop height h", setting_unit="cm",
        ),
        "loaded": ObsKind(
            "loaded", f"小车上加 {LOAD_KG * 1000:.0f} g 配重后，从高度 h 处释放", "same, with a load on the cart", (r,),
            setting_label_zh="下落高度 h", setting_label_en="drop height h", setting_unit="cm",
        ),
        "string_release": ObsKind(
            "string_release", "用细线拴住小车，剪断细线释放（手不碰小车）", "release by cutting a string", (r,),
            setting_label_zh="下落高度 h", setting_label_en="drop height h", setting_unit="cm",
        ),
        "height_check": ObsKind(
            "height_check", "用铅垂线和刻度尺重新竖直测量释放点到测速点的高度差",
            "re-measure the vertical drop with a plumb line",
            (Reading("h_vertical", "竖直高度差（重测）", "re-measured vertical drop", "cm", resolution=0.1, sigma=0.05),),
            setting_label_zh="原记录的高度", setting_label_en="recorded height", setting_unit="cm",
        ),
        "coast": ObsKind(
            "coast", "把轨道调水平，轻推小车，测它滑行时的减速度", "level the track and measure the coasting deceleration",
            (Reading("decel", "水平滑行减速度", "coasting deceleration", "m/s²", resolution=0.001, sigma=0.01),),
        ),
    }
    if timer == "photogate":
        kinds["flag_width"] = ObsKind(
            "flag_width", "用游标卡尺测遮光片宽度", "measure the flag width with a vernier caliper",
            (Reading("flag_width", "遮光片宽度", "flag width", "mm", resolution=0.02, sigma=0.01),),
        )
    return kinds


def _causes(timer: str) -> tuple[Cause, ...]:
    speed_how = "v = d/Δt" if timer == "photogate" else "v = (x_{n+1} − x_{n−1})/(2T)"
    causes = [
        Cause(
            RANDOM, "只有随机误差", "random error only",
            "测量步骤正确，摩擦和车轮转动的影响小到可以忽略，数据偏离只来自读数的随机误差。",
            "The procedure is correct and friction and wheel rotation are negligible; deviations come only "
            "from random reading error.",
            "none", ("none",),
            fix_zh="多测几组、取平均即可。", fix_en="Take more readings and average.",
        ),
        # 推导：记录速度 v' = d₀/Δt，真实 v = d/Δt，令 k = d₀/d，则 ½v'² = k²·½v² = k²·g h，
        # 所以 ½v'² 对 g h 的斜率是 k²（d 真实偏大 → k < 1 → 斜率小于 1）。
        # Δt = d/v 对真实宽度 d 是线性的：d = d₀ + Δd，Δd 作线性参数。
        Cause(
            FLAG, "遮光片宽度用错了", "wrong flag width",
            f"用 {speed_how} 算速度时，遮光片宽度 d 用的是标称值或用刻度尺粗量的值，和遮光片的真实宽度不一样。",
            "The flag width used in v = d/Δt is the nominal value or a rough ruler reading, not the true width.",
            "y_scale", ("slope_low", "slope_high"),
            linear=(LinearParam("w_err", 0.0, 0.4, "遮光片真实宽度 − 所用宽度", "mm"),),
            fix_zh="用游标卡尺测出遮光片的真实宽度，用实测值重新计算速度。",
            fix_en="Measure the flag width with a vernier caliper and recompute the speeds.",
        ),
        # 推导：学生把沿轨道的距离 s 记成 h，真实下落高度 h = s·sinθ，
        # ½v² = g s sinθ = sinθ·(g s)，所以斜率 = sinθ（倾角 12°~30° → 0.21~0.5）；记录了倾角时 sinθ 已知，没有自由参数。
        Cause(
            SLANT, "把斜面上的距离当成了下落高度", "slant distance used as height",
            "没有测竖直高度差，而是把小车沿轨道滑过的距离当成了下落高度 h，也没有记录轨道的倾角。",
            "The distance travelled along the track was recorded as the drop height h; the incline angle "
            "was not recorded.",
            "x_scale", ("slope_low",),
            magnitude=Prior.point(1.0),
            fix_zh="用铅垂线和刻度尺量竖直高度差；或量出轨道倾角，用 h = s·sinθ 换算。",
            fix_en="Measure the vertical drop, or record the angle and use h = s·sinθ.",
        ),
        # 推导：滚动阻力 f = μN = μMg cosθ，滑过距离 s = h/sinθ，动能定理
        # Mgh − μMg cosθ·h/sinθ = ½Mv² → ½v² = (1 − μ cotθ)·g h，斜率 1 − μ cotθ（倾角固定时是比例）。
        # 水平滑行时只有阻力：a = μg（菜单里的减速度测量直接测 μ）。
        Cause(
            FRICTION, "轨道和车轮之间有摩擦", "rolling friction on the track",
            "车轮的滚动阻力和轮轴里的摩擦在小车下滑的全过程中都在做负功，一部分机械能变成了内能"
            + ("。" if timer == "photogate" else "；纸带和限位孔之间、振针和纸带之间也有阻力。"),
            "Rolling resistance and axle friction do negative work all the way down, turning some mechanical "
            "energy into heat" + ("." if timer == "photogate" else "; the tape also drags in the timer."),
            "y_scale", ("slope_low",),
            magnitude=Prior.loguniform(0.004, 0.03, 25),
            magnitude_label_zh="等效滚动阻力系数 μ", magnitude_unit="",
            fix_zh="清洁轨道和车轮、给轮轴加润滑；或换气垫导轨；或者测出阻力，在能量里扣除阻力做的功。",
            fix_en="Clean the track and wheels, lubricate the axles or use an air track; or measure the drag "
            "and account for its work.",
        ),
        # 推导：4 个均匀圆盘车轮总质量 m_w，每个 I = ½m r²，纯滚动 ω = v/r，
        # 转动动能 = 4·½·(½m r²)(v/r)² = ¼ m_w v² = (m_w/2M)·½Mv²（M 为含车轮的总质量）。
        # Mgh = ½Mv²(1 + ε)，ε = m_w/(2M) → 斜率 1/(1+ε)；车轮占总质量 10% → ε = 5%。
        # 加配重 Δm 后 ε' = m_w/(2(M+Δm)) = ε·M/(M+Δm)，摩擦（μ 不随载荷变）和宽度误差都不变。
        Cause(
            INERTIA, "车轮转动分走了动能", "wheel rotational energy",
            "小车下滑时四个车轮在转动，减少的重力势能有一部分变成了车轮的转动动能，计算动能时只算了小车平动的 ½Mv²。",
            "The four wheels spin as the cart rolls down, so part of the lost potential energy becomes "
            "rotational energy of the wheels, while only the translational ½Mv² is counted.",
            "y_scale", ("slope_low",),
            magnitude=Prior.uniform(0.015, 0.08, 27),
            magnitude_label_zh="车轮转动动能与平动动能之比 m_w/(2M)", magnitude_unit="",
            fix_zh="在小车上加配重（车轮所占质量比例变小），或者把车轮的转动动能 ¼m_w v² 算进去。",
            fix_en="Load the cart so the wheels are a smaller fraction of the mass, or include ¼m_w v².",
        ),
        # 推导：记录高度 h' = h + δ（δ 是两次读数参考点的高度差），½v² = g h = g h' − gδ，
        # 斜率不变、截距 −gδ（δ 可正可负）。
        Cause(
            WRONG_POINT, "高度量到了错误的位置", "height measured to the wrong point",
            "量下落高度时，释放处和测速处用的不是小车上的同一个参考点（例如一处量到车顶、一处量到轨道面），"
            "或者量到了轨道末端而不是测速点的位置，所以每次记录的高度都差一个固定的量。",
            "The two height readings were not taken at the same point of the cart (e.g. top of the cart at "
            "release, track surface at the gate), or were taken to the end of the track instead of the "
            "speed-measuring point, so every recorded height is off by the same amount.",
            "x_offset", ("intercept_pos", "intercept_neg"),
            magnitude=Prior.sym_uniform(0.5, 4.0, 36),
            magnitude_label_zh="记录高度 − 真实高度", magnitude_unit="cm",
            fix_zh="释放点和测速点都量小车上同一个点（如遮光片中心）到桌面的高度，再相减。",
            fix_en="Measure both heights to the same point of the cart (e.g. the flag centre) and subtract.",
        ),
        # 推导：½Mv² = ½Mv₀² + Mgh → ½v² = g h + ½v₀²，斜率不变、截距 +½v₀²。
        Cause(
            PUSH, "释放时推了小车", "cart pushed at release",
            "释放小车时手没有直接松开，而是顺势推了一下（或手指带了小车一下），小车离开出发点时已经有了速度。",
            "The cart was not simply let go: the hand gave it a small push, so it already had some speed "
            "when it left the start.",
            "y_offset", ("intercept_pos",),
            magnitude=Prior.uniform(0.2, 0.6, 21),
            magnitude_label_zh="释放时的初速度 v₀", magnitude_unit="m/s",
            fix_zh="用挡板或细线固定小车，抽开挡板（剪断细线）释放，手不碰小车。",
            fix_en="Hold the cart with a stop or a string and release it without touching it.",
        ),
    ]
    if timer != "photogate":
        causes = [c for c in causes if c.id != FLAG]
    return tuple(causes)


def _menu(timer: str) -> tuple[MenuItem, ...]:
    items = [
        MenuItem("height_check", "用铅垂线和刻度尺重测 25 cm 那次的竖直高度差", "re-measure the 25 cm drop vertically",
                 2.0, (("height_check", 25.0),)),
        MenuItem("coast_test", "把轨道调水平，轻推小车，测滑行减速度", "measure the coasting deceleration on a level track",
                 6.0, (("coast", 0.0),)),
        MenuItem("string_release", "用细线拴住小车、剪断细线释放，在 5 cm 和 15 cm 处重测",
                 "release by cutting a string at 5 cm and 15 cm", 3.0,
                 (("string_release", 5.0), ("string_release", 15.0))),
        MenuItem("add_load", f"小车上加 {LOAD_KG * 1000:.0f} g 配重，在 15 cm 和 25 cm 处重测",
                 "add a load to the cart and repeat at 15 cm and 25 cm", 4.0,
                 (("loaded", 15.0), ("loaded", 25.0))),
    ]
    if timer == "photogate":
        items.insert(0, MenuItem("caliper_flag", "用游标卡尺测遮光片宽度", "measure the flag width with a caliper",
                                 2.0, (("flag_width", 0.0),)))
    return tuple(items)


class Energy(Experiment):
    id = "energy"
    name_zh = "验证机械能守恒定律（小车沿倾斜轨道下滑）"
    name_en = "Conservation of mechanical energy: cart on an inclined track"

    obs_kinds = _obs_kinds("photogate")

    # 轨道倾角（度）：默认不记录，取学校轨道常见范围；构造时给出倾角则为定值
    nuisance = {"theta_deg": Prior.uniform(12.0, 30.0, 19)}

    # 遮光片真实宽度（mm）：游标卡尺精度 0.02 mm，所有原因共用
    linear = (LinearParam("w_true", FLAG_MM, 0.02, "遮光片真实宽度", "mm"),)

    causes = _causes("photogate")

    protocols = {
        "single": Protocol("single", "单次测量（h = 20 cm）", "single run (h = 20 cm)", (("main", 20.0),)),
        "standard": Protocol(
            "standard", "课本标准方案（5 个高度）", "textbook worksheet (5 heights)",
            tuple(("main", h) for h in HEIGHTS_CM),
        ),
    }

    menu = _menu("photogate")

    def __init__(self, timer: str = "photogate", angle_deg: float | None = None,
                 flag_mm: float = FLAG_MM, cart_kg: float = CART_KG):
        assert timer in ("photogate", "ticker"), timer
        self.timer = timer
        self.flag_mm = float(flag_mm)
        self.cart_kg = float(cart_kg)
        if timer != "photogate" or flag_mm != FLAG_MM:
            self.obs_kinds = _obs_kinds(timer)
            self.causes = _causes(timer)
            self.menu = _menu(timer)
            self.linear = (LinearParam("w_true", self.flag_mm, 0.02, "遮光片真实宽度", "mm"),) if timer == "photogate" else ()
        if angle_deg is not None:
            self.nuisance = {"theta_deg": Prior.point(angle_deg)}

    def relevant_nuisance(self, cause_id, kinds):
        if cause_id in (SLANT, FRICTION):
            return ("theta_deg",)
        return ()

    def _v2(self, kind, h_cm, params, cause_id):
        """测速点处真实速度的平方 (P, n)。"""
        P = params["mag"].shape[0]
        mag = params["mag"]
        h = h_cm[None, :] / 100.0
        if cause_id == SLANT:
            h = h * np.sin(np.radians(params["theta_deg"]))
        elif cause_id == WRONG_POINT:
            h = h - mag / 100.0
        kappa = 1.0
        if cause_id == FRICTION:
            kappa = 1.0 - mag / np.tan(np.radians(params["theta_deg"]))
        elif cause_id == INERTIA:
            eps = mag * (self.cart_kg / (self.cart_kg + LOAD_KG) if kind == "loaded" else 1.0)
            kappa = 1.0 / (1.0 + eps)
        v2 = 2.0 * G * kappa * h
        if cause_id == PUSH and kind != "string_release":
            v2 = v2 + mag**2
        return np.maximum(np.broadcast_to(v2, (P, h_cm.size)), 1e-6)

    def predict(self, kind, settings, order, params, cause_id):
        P = params["mag"].shape[0]
        n = settings.size
        L = len(self.linear_params(cause_id))
        U = np.zeros((P, n, 1, L))
        if kind in MAIN_KINDS:
            v = np.sqrt(self._v2(kind, settings, params, cause_id))
            if self.timer == "photogate":
                # Δt(ms) = d(mm)/v(m/s)，d = w_true (+ w_err)，对宽度类线性参数都是 1/v
                U[:, :, 0, :] = (1.0 / v)[:, :, None]
                return np.zeros((P, n, 1)), U
            return (200.0 * T_TICK * v)[:, :, None], U  # 两点间距 = 2T·v，单位 cm
        if kind == "flag_width":
            U[:, :, 0, :] = 1.0
            return np.zeros((P, n, 1)), U
        if kind == "height_check":
            h = np.broadcast_to(settings[None, :], (P, n))
            if cause_id == SLANT:
                h = h * np.sin(np.radians(params["theta_deg"]))
            elif cause_id == WRONG_POINT:
                h = h - params["mag"]
            return np.broadcast_to(h, (P, n))[:, :, None], U
        if kind == "coast":
            a = params["mag"] * G if cause_id == FRICTION else np.zeros((P, 1))
            return np.broadcast_to(a, (P, n))[:, :, None], U
        raise KeyError(kind)

    def _main_sd(self) -> float:
        r = self.obs_kinds["main"].readings[0]
        return float(np.sqrt(r.sigma**2 + r.resolution**2 / 12))

    def linearise(self, ds: Dataset) -> LinearData:
        idx = [i for i, o in enumerate(ds.observations) if o.kind == "main"]
        h = np.array([ds.observations[i].setting for i in idx]) / 100.0
        raw = np.array([ds.observations[i].values[0] for i in idx])
        if self.timer == "photogate":
            v = self.flag_mm / raw  # mm/ms = m/s
        else:
            v = raw / 100.0 / (2.0 * T_TICK)
        y = 0.5 * v**2
        sy = y * 2.0 * self._main_sd() / raw  # ½v² ∝ 1/Δt² 或 ∝ 间距²，相对误差翻倍
        return LinearData(G * h, y, sy, np.array(idx), "g·h (J/kg)", "½v² (J/kg)", 1.0, 0.0)
