"""物理组件库：物体、连接件、表面。每个组件带自己的未知参数、已知量、误差机制和附加测量。

参数命名 <组件名>.<量>（hanger.hook_g、spring.inv_k、block.mu）；误差机制 key = <组件名>_<机制>。
默认先验和幅度范围取自精选实验（labsim/experiments/）里已经推导、核对过的值，镜像时数值完全一致。

物体：Cart（小车，含车轮转动惯量）、Block（木块）、Ball（小球）、Bob（摆球）、Hanger（挂钩/小盘 + 钩码）。
连接件：Spring（弹簧）、StringLine（细线）、Pulley（定滑轮）、HandPull（手拉测力计的方式）、Release（释放方式）。
表面：Track（水平导轨，可垫高平衡摩擦）、Incline（斜面：倾角、阻力、斜面长度当高度）、Board（水平木板，可能不平）、
Lever（杠杆，见 templates.TorqueBalance）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..labsim.core import G, LinearParam, Prior, Reading
from .base import AuxMeasurement, Component, Ctx, Effect, OpenQuestion, Variant

# ── 物体 ──────────────────────────────────────────────────────────────


@dataclass
class Cart(Component):
    """小车（或滑块）。质量 M 由天平称出（已知）；车轮滚动时的转动动能常被忽略。

    车轮转动：4 个均匀圆盘轮总质量 m_w，I = ½m r²、ω = v/r → 转动动能 = ¼m_w v²，
    等效于平动质量多了 J = Σ I/r² = m_w/2。幅度 ε = J/M（M 为空车质量），加配重 Δm 后 J 不变、
    ε' = J/(M + Δm)。load_g：菜单"加配重"用的质量。
    """

    label_zh: str = "小车"
    mass_g: float = 500.0
    load_g: float = 500.0
    wheel_prior: Prior = Prior.uniform(0.015, 0.08, 27)

    def all_effects(self):
        return [
            Effect(
                "wheel_inertia", "车轮转动分走了动能", "wheel rotational energy",
                "小车运动时车轮在转动，一部分能量（或一部分拉力）用来让车轮转起来，计算时只按小车平动处理。",
                "The wheels spin as the cart moves, so part of the energy (or of the pull) goes into "
                "spinning them, while the analysis treats the cart as purely translating.",
                "y_scale", ("slope_low",), self.wheel_prior,
                "车轮等效质量与空车质量之比 ΣI/(r²M)", "",
                "在小车上加配重（车轮所占比例变小），或把车轮的转动动能算进去。",
                "Load the cart so the wheels are a smaller fraction of the mass, or include their rotation.",
            )
        ]

    def mass(self, ctx: Ctx):
        """平动质量（kg），"加配重"变体时加上配重。"""
        m = self.mass_g / 1000.0
        return m + self.load_g / 1000.0 if ctx.flag("loaded") else m

    def rot_inertia(self, ctx: Ctx):
        """车轮等效质量 J = ε·M_空车（kg）。"""
        if self.has("wheel_inertia") and ctx.on(self.key("wheel_inertia")):
            return ctx.mag * (self.mass_g / 1000.0)
        return 0.0

    def variants(self, setup):
        if not self.has("wheel_inertia"):
            return []
        s = setup.standard
        pick = (s[len(s) // 2], s[-1])
        return [Variant(
            "loaded", f"小车上加 {self.load_g:.0f} g 配重后再测", "same, with a load on the cart",
            frozenset({"loaded"}),
            menu=(("add_load", f"小车上加 {self.load_g:.0f} g 配重，在 {pick[0]:g} 和 {pick[1]:g} 处重测",
                   "add a load to the cart and repeat at two settings", 4.0, pick),),
        )]

    def questions(self):
        return [OpenQuestion(self.pname("mass_g"), "小车（含车上砝码、遮光片）的质量是多少？用天平称。", "g", self.mass_g)]


@dataclass
class Block(Component):
    """木块。动摩擦因数 μ 未知，作线性参数（摩擦力 = μN 对 μ 是线性的）。

    mu_pull：同一个 μ 出现在斜拉公式分母里时用的小网格（见 Setup.sync 与 HandPull 的推导）。
    """

    label_zh: str = "木块"
    weight_n: float = 2.0
    mu_mean: float = 0.30
    mu_sd: float = 0.07
    mu_linear: bool = True

    def linear(self):
        if not self.mu_linear:
            return ()
        return (LinearParam(self.pname("mu"), self.mu_mean, self.mu_sd, "动摩擦因数 μ", ""),)

    def nuisance(self):
        if not self.mu_linear:
            return {}
        return {self.pname("mu_pull"): Prior.normal(self.mu_mean, self.mu_sd, 15)}

    def questions(self):
        return [OpenQuestion(self.pname("weight_n"), "木块的重力是多少？可以用测力计竖直挂着称。", "N", self.weight_n)]


@dataclass
class Ball(Component):
    """小球（自由落体、平抛）。直径用游标卡尺量出（已知）。

    空气阻力（轻球）：m dv/dt = mg − ½ρC_dA v²，令 κ = ρC_dA/(2m)，沿下落距离积分
    v dv/dx = g − κv² → v² = (g/κ)(1 − e^{−2κh}) ≈ 2gh(1 − κh)：越往下越偏离 v² = 2gh。
    钢球 κ ~ 10⁻³ /m 可忽略；乒乓球、塑料球 κ ~ 0.05~0.2 /m。
    """

    label_zh: str = "小球"
    diameter_cm: float = 2.0
    drag_prior: Prior = Prior.loguniform(0.02, 0.15, 17)

    def all_effects(self):
        return [
            Effect(
                "air_drag", "小球太轻，空气阻力不能忽略", "air drag on a light ball",
                "用的是空心或塑料的轻球，下落时受到的空气阻力与重力相比不算小，速度越大阻力越大。",
                "A hollow or plastic ball was used; air drag is not small compared with its weight and grows with speed.",
                "bend", ("curve_down", "slope_low"), self.drag_prior,
                "阻力系数 κ = ρC_dA/(2m)", "1/m",
                "换用同样大小的实心钢球。", "Use a solid steel ball of the same size.",
            )
        ]

    def drag_coef(self, ctx: Ctx):
        """空气阻力系数 κ（1/m）；不适用时 None。"""
        if self.has("air_drag") and ctx.on(self.key("air_drag")):
            return ctx.mag
        return None

    def questions(self):
        return [OpenQuestion(self.pname("diameter_cm"), "小球的直径是多少？用游标卡尺量。", "cm", self.diameter_cm)]


@dataclass
class Bob(Component):
    """单摆摆球。半径未知（网格；用游标卡尺可测）。

    摆长量到球顶：记录 L' = L − r → 真实摆长 = 记录值 + r。
    大摆角：T = T₀/AGM(1, cos(θ₀/2)) ≈ T₀(1 + θ₀²/16)（见 templates.period_factor）。
    圆锥摆：F cosα = mg、F sinα = mω²L sinα → ω² = g/(L cosα)，T = 2π√(L cosα/g)（精确，不再乘小角修正）。
    "careful" 变体（小于 5°、竖直平面内轻放）里后两种效应不起作用。
    """

    label_zh: str = "摆球"
    r_prior: Prior = Prior.uniform(0.75, 1.5, 16)  # 半径 cm
    amp_prior: Prior = Prior.uniform(15.0, 45.0, 31)
    cone_prior: Prior = Prior.uniform(8.0, 25.0, 35)

    def nuisance(self):
        return {self.pname("r_cm"): self.r_prior}

    def all_effects(self):
        r = self.pname("r_cm")
        return [
            Effect(
                "top_of_bob", "摆长量到了摆球顶端", "length measured to the top of the bob",
                "量摆长时只量了从悬点到摆球顶端（摆线下端）的长度，没有加上摆球的半径。",
                "The length was measured from the pivot to the top of the bob (end of the string) and the bob "
                "radius was not added.",
                "x_offset", ("intercept_pos",), Prior.point(1.0),
                fix_zh="用游标卡尺测摆球直径 d，摆长 = 线长 + d/2（悬点到球心）。",
                fix_en="Measure the bob diameter d with a caliper; length = string length + d/2.",
                uses=(r,),
            ),
            Effect(
                "large_amplitude", "摆角太大", "amplitude too large",
                "释放摆球时拉开的角度远大于 5°，每次都从差不多同样大的角度释放；摆角大时回复力比 mgθ 小，摆得更慢。",
                "The bob was pulled far beyond 5° before release, about the same angle every time; at large "
                "angles the restoring force is weaker than mgθ, so the swing is slower.",
                "y_scale", ("slope_high",), self.amp_prior, "释放时的摆角 θ₀", "°",
                "把摆球拉开不超过 5°（摆长 1 m 时水平拉开不超过约 9 cm）再释放。",
                "Release from no more than 5° (about 9 cm sideways for a 1 m pendulum).",
            ),
            Effect(
                "conical", "摆球做了圆锥摆运动", "bob moved as a conical pendulum",
                "释放时手给了摆球一个侧向的速度，摆球没有在竖直平面内来回摆动，而是绕着竖直线在水平面内转圈（圆锥摆），"
                "学生按摆球回到同一位置数一次全振动。",
                "The hand gave the bob a sideways push at release, so instead of swinging in a vertical plane it "
                "circled around the vertical (a conical pendulum); one revolution was counted as one oscillation.",
                "y_scale", ("slope_low",), self.cone_prior, "摆线与竖直方向的夹角 α", "°",
                "释放前让摆球静止，松手时不要带侧向力，确认摆球在竖直平面内摆动后再开始计时。",
                "Hold the bob still and let go without a sideways push; check it swings in one plane.",
            ),
        ]

    def control(self, ctx, x, quantity):
        if quantity == "length" and self.has("top_of_bob") and ctx.on(self.key("top_of_bob")):
            return x + ctx.p(self.pname("r_cm")) / 100.0
        return x

    def release_angle(self, ctx: Ctx):
        """大摆角时的 θ₀（度）；不适用时 None。"""
        if self.has("large_amplitude") and ctx.on(self.key("large_amplitude")) and not ctx.flag("careful"):
            return ctx.mag
        return None

    def cone_angle(self, ctx: Ctx):
        if self.has("conical") and ctx.on(self.key("conical")) and not ctx.flag("careful"):
            return ctx.mag
        return None

    def variants(self, setup):
        if not (self.has("large_amplitude") or self.has("conical")):
            return []
        L = setup.standard[-1]
        return [Variant(
            "careful", "摆角小于 5°、在竖直平面内轻轻释放后计时", "gentle planar release below 5°",
            frozenset({"careful"}),
            menu=(("small_angle", f"在 {L:g} 处摆角小于 5°、在竖直平面内轻轻释放后重测",
                   "repeat with a gentle planar release below 5°", 3.0, (L,)),),
        )]

    def aux(self, setup):
        r = self.pname("r_cm")
        return [AuxMeasurement(
            "bob_d", "用游标卡尺测摆球直径", "measure the bob diameter with a vernier caliper",
            Reading("bob_d", "摆球直径", "bob diameter", "cm", resolution=0.002, sigma=0.002),
            lambda ctx: np.broadcast_to(2.0 * ctx.p(r), (ctx.P, ctx.n)), uses=(r,), menu_label_zh="用游标卡尺测摆球直径",
        )]

    def questions(self):
        return [OpenQuestion(self.pname("r_cm"), "摆球的直径是多少？用游标卡尺量。", "cm", None)]


@dataclass
class Hanger(Component):
    """挂钩 / 小盘 + 钩码。

    mass_g = None：悬挂质量就是控制量（钩码 + 小盘的记录值）；否则是已知的固定质量。
    机制 omitted（漏算挂钩质量）：挂钩自身有质量，原长在挂上挂钩之前读，记录的悬挂质量只算了钩码
      → 真实悬挂质量 = 记录值 + 挂钩质量（干扰量 hook_g，用天平可称）。
    机制 heavy（槽码质量没有远小于小车质量）：课本近似"细线拉力 = mg"忽略了槽码自身的惯性；
      精确的连接体动力学里槽码也要加速：等效质量多一个 ρ²m（见 templates.UniformAccel）。
      inertial=True 时惯性总是计入（精确基线，不再列为原因）。
    """

    label_zh: str = "挂钩和钩码"
    mass_g: float | None = None
    hook_prior: Prior = Prior.uniform(5.0, 50.0, 181)
    inertial: bool = False

    def nuisance(self):
        return {self.pname("hook_g"): self.hook_prior} if self.has("omitted") else {}

    def all_effects(self):
        effs = [
            Effect(
                "omitted", "漏算挂钩质量", "hanger mass left out",
                "钩码挂在一个有质量的挂钩上；算拉力（或振子质量）时只用了钩码的质量，没把挂钩的质量算进去。",
                "The masses hang on a hanger with its own mass, but only the slotted masses were counted.",
                "x_offset", ("intercept_pos",), Prior.point(1.0),
                fix_zh="称出挂钩质量，按（钩码 + 挂钩）的总质量计算；或挂上挂钩后再量原长。",
                fix_en="Weigh the hanger and include it, or read the unloaded length with the hanger attached.",
                uses=(self.pname("hook_g"),),
            ),
        ]
        if not self.inertial:
            effs.append(Effect(
                "heavy", "槽码质量没有远小于小车质量", "hanging mass not much less than cart mass",
                "槽码和小车一起加速，槽码自身也要一部分重力来获得加速度，细线拉力比槽码的重力小；"
                "槽码越重，拉力和槽码重力差得越多。",
                "The hanging mass accelerates together with the cart, so part of its weight goes into "
                "accelerating itself; the string tension is less than its weight, more so for heavier loads.",
                "bend", ("slope_low", "curve_down"),
                fix_zh="用槽码重力代替拉力的相对误差约为 m/(M + m)。改进：在小车上加砝码增大 M；或改用力传感器直接测细线拉力；"
                "或把槽码先放在小车上、每次从车上取下挂到小盘里，保持 M + m 不变，以整体为研究对象。",
                fix_en="The relative error of using the hanging weight as the pull is about m/(M + m). Add mass to the "
                "cart, use a force sensor, or move masses from the cart to the hanger so that M + m stays constant.",
            ))
        return effs

    def mass(self, ctx: Ctx):
        """悬挂质量（kg）。"""
        if self.mass_g is None:
            return ctx.x
        return self.mass_g / 1000.0

    def is_inertial(self, ctx: Ctx) -> bool:
        return self.inertial or (self.has("heavy") and ctx.on(self.key("heavy")))

    def control(self, ctx, x, quantity):
        if quantity == "mass" and self.mass_g is None and self.has("omitted") and ctx.on(self.key("omitted")):
            return x + ctx.p(self.pname("hook_g")) / 1000.0
        return x

    def aux(self, setup):
        if not self.has("omitted"):
            return []
        h = self.pname("hook_g")
        return [AuxMeasurement(
            "hanger_mass", "用天平称挂钩质量", "weigh the hanger on a balance",
            Reading("hanger_mass", "挂钩质量", "hanger mass", "g", resolution=0.1, sigma=0.1),
            lambda ctx: np.broadcast_to(ctx.p(h), (ctx.P, ctx.n)), uses=(h,), menu_label_zh="用天平称挂钩质量",
        )]

    def questions(self):
        if self.mass_g is None:
            return [OpenQuestion(self.pname("masses"), "每次挂的钩码（含小盘）总质量分别是多少？", "g", None)]
        return [OpenQuestion(self.pname("mass_g"), "悬挂重物（含小盘）的质量是多少？", "g", self.mass_g)]


# ── 连接件 ────────────────────────────────────────────────────────────


@dataclass
class Spring(Component):
    """弹簧。劲度系数 k 未知，按用途取不同的线性参数：
    静态（mode="static"）：伸长量 = (1/k)·F，线性参数 1/k；
    振动（mode="oscillation"）：T = 2π√(m/k) = 2π√m·(1/√k)，线性参数 1/√k。
    mass_g：弹簧自身质量（学生称出，振动时计入等效质量 m_s/3）。

    超出弹性限度：大拉力处更容易被拉长，伸长量乘 (1 + β(F/F_ref)²)，β 为 F = F_ref 处的额外伸长比例。
    """

    label_zh: str = "弹簧"
    mode: str = "static"
    inv_k_mean: float = 0.05
    inv_k_sd: float = 0.0115
    mass_g: float = 0.0
    f_ref_n: float = 0.25 * G
    limit_prior: Prior = Prior.uniform(0.05, 0.5, 46)
    scatter_prior: Prior = Prior.uniform(0.15, 0.6, 46)  # 读数单位（cm）

    def linear(self):
        if self.mode == "static":
            return (LinearParam(self.pname("inv_k"), self.inv_k_mean, self.inv_k_sd, "1/k", "m/N"),)
        # 1/√k 的先验：由 1/k 的均值和相对标准差换算（δ(1/√k)/(1/√k) = ½·δ(1/k)/(1/k)）
        m = math.sqrt(self.inv_k_mean)
        return (LinearParam(self.pname("inv_sqrt_k"), m, 0.5 * m * self.inv_k_sd / self.inv_k_mean,
                            "1/√k", "√(m/N)"),)

    def all_effects(self):
        if self.mode != "static":
            return []
        return [
            Effect(
                "beyond_limit", "超出弹性限度", "beyond the elastic limit",
                "所挂钩码太重，弹簧被拉过了弹性限度，大拉力时弹簧变得更容易被拉长。",
                "The load was too large; past its elastic limit the spring stretches more easily at large forces.",
                "bend", ("curve_up",), self.limit_prior, "大拉力处的额外伸长比例", "",
                "减小钩码总质量，只用弹性限度内的数据作图。",
                "Use smaller loads and plot only data within the elastic limit.",
            ),
            Effect(
                "not_settled", "读数时弹簧没有静止", "read while still oscillating",
                "弹簧还在上下振动时就读数，或者每次读数时眼睛的高度都不一样。",
                "Readings were taken while the spring was still bouncing, or from a different eye height each time.",
                "scatter", ("scatter_excess",), self.scatter_prior, "额外读数误差", "cm",
                "等弹簧完全静止后再读数，视线与刻度平齐。",
                "Wait until the spring is still and keep your eye level with the scale.",
                scatter=True,
            ),
        ]

    def bend(self, ctx: Ctx, F):
        if self.has("beyond_limit") and ctx.on(self.key("beyond_limit")):
            return 1.0 + ctx.mag * (F / self.f_ref_n) ** 2
        return 1.0

    def extra_sigma(self, ctx):
        if self.has("not_settled") and ctx.on(self.key("not_settled")) and ctx.kind == "main":
            return ctx.mag
        return None

    def questions(self):
        if self.mode == "oscillation":
            return [OpenQuestion(self.pname("mass_g"), "弹簧自身的质量是多少？用天平称。", "g", self.mass_g)]
        return []


@dataclass
class StringLine(Component):
    """细线。应与导轨平行：滑轮高度不对时细线与导轨成 φ 角。

    φ 视为不变：小车位移 x 时细线收短 x·cosφ，槽码速度 = v cosφ，速度比 ρ = cosφ；
    沿导轨的拉力分量 T cosφ。垂直分量减小正压力、阻力少 μT sinφ（μ ≈ 0.01 的滚动阻力，二阶小量，略去）。
    span_cm：量高度差时小车挂钩到滑轮顶端的水平距离。
    """

    label_zh: str = "细线"
    span_cm: float = 40.0
    angle_prior: Prior = Prior.uniform(8.0, 25.0, 18)

    def all_effects(self):
        return [
            Effect(
                "not_parallel", "细线与轨道不平行", "string not parallel to the track",
                "定滑轮的高度没有调好，细线斜着拉小车，只有沿轨道方向的分力使小车加速。",
                "The pulley height was not adjusted, so the string pulls the cart at an angle and only its "
                "component along the track accelerates the cart.",
                "x_scale", ("slope_low",), self.angle_prior, "细线与轨道的夹角", "°",
                "调节定滑轮的高度，使细线与轨道平行。",
                "Adjust the pulley height so that the string is parallel to the track.",
            )
        ]

    def ratio(self, ctx: Ctx):
        if self.has("not_parallel") and ctx.on(self.key("not_parallel")):
            return np.cos(np.radians(ctx.mag))
        return 1.0

    def aux(self, setup):
        if not self.has("not_parallel"):
            return []
        key, span = self.key("not_parallel"), self.span_cm

        def rise(ctx):
            if ctx.on(key):
                return np.broadcast_to(span * np.tan(np.radians(ctx.mag)), (ctx.P, ctx.n))
            return np.zeros((ctx.P, ctx.n))

        return [AuxMeasurement(
            "string_height", "用刻度尺量细线两端离轨道的高度差",
            "measure the height difference of the string ends above the track",
            Reading("string_rise", "细线两端离轨道的高度差", "height difference of string ends", "cm",
                    resolution=0.1, sigma=0.2),
            rise, minutes=1.0, menu_label_zh="用刻度尺量细线两端离轨道的高度差",
        )]


@dataclass
class Pulley(Component):
    """定滑轮。滑轮两侧拉力差带动它转动并克服轴摩擦：(T1 − T2) = (I/r²)·a + f_p。
    I/r² 并入等效质量（J，网格，g），轴摩擦 f_p 是细线上一个恒定阻力（线性参数，N）。

    mode="combined"：两者合成一个原因（精选牛顿第二定律的做法，J 里还含车轮）；
    mode="separate"：转动惯量 inertia、轴摩擦 axle 各是一个原因（axle_linear=False 时轴摩擦放网格，
    用于速度要开方的能量类模板）。
    """

    label_zh: str = "滑轮"
    mode: str = "separate"
    j_prior: Prior = Prior.uniform(5.0, 25.0, 16)  # g
    axle_mean: float = 0.02
    axle_sd: float = 0.006
    axle_linear: bool = True
    axle_prior: Prior = Prior.uniform(0.005, 0.04, 15)  # N（网格版）

    def all_effects(self):
        if self.mode == "combined":
            return [Effect(
                "drag", "滑轮转动惯量和轴摩擦", "pulley inertia and axle friction",
                "定滑轮较重或轴不灵活：细线要带动滑轮（以及小车车轮）转起来，还要克服滑轮轴的摩擦；"
                "平衡摩擦力时细线没有挂在滑轮上，这部分阻力没有被补偿。",
                "The pulley is heavy or its axle is stiff: the string must spin up the pulley (and the cart's "
                "wheels) and overcome axle friction, which the friction-compensation step never included.",
                "x_offset", ("intercept_neg", "slope_low"), self.j_prior, "滑轮和车轮的等效质量 I/r²", "g",
                "换用轻质、转动灵活的滑轮（如带轴承的铝滑轮），并检查滑轮轴是否卡滞。",
                "Use a light, free-running pulley (e.g. with a ball bearing) and check the axle.",
                linear=(LinearParam(self.pname("f_axle"), self.axle_mean, self.axle_sd, "滑轮轴摩擦力", "N"),),
            )]
        axle = Effect(
            "axle", "滑轮轴有摩擦", "pulley axle friction",
            "滑轮的轴不灵活，转动时有摩擦力矩，相当于在细线上加了一个与运动方向相反的恒定阻力。",
            "The pulley axle is stiff; its friction torque acts like a constant force on the string opposing the motion.",
            "x_offset", ("intercept_neg",),
            fix_zh="给滑轮轴上润滑油或换带轴承的滑轮。", fix_en="Lubricate the axle or use a ball-bearing pulley.",
        )
        if self.axle_linear:
            axle = _replace(axle, linear=(LinearParam(self.pname("f_axle"), self.axle_mean, self.axle_sd,
                                                      "滑轮轴摩擦力", "N"),))
        else:
            axle = _replace(axle, magnitude=self.axle_prior, magnitude_label_zh="轴摩擦的等效阻力",
                            magnitude_unit="N")
        return [
            Effect(
                "inertia", "滑轮的转动惯量不能忽略", "pulley moment of inertia",
                "滑轮比较重，运动时细线还要带动滑轮越转越快，一部分拉力用来让滑轮转动起来。",
                "The pulley is heavy; the string must also spin it up, so part of the pull goes into rotating it.",
                "y_scale", ("slope_low",), self.j_prior, "滑轮等效质量 I/r²", "g",
                "换用轻质滑轮；或称出滑轮质量，按等效质量 I/r²（均匀圆盘为 m_轮/2）修正。",
                "Use a lighter pulley, or correct with its equivalent mass I/r² (uniform disc: m/2).",
            ),
            axle,
        ]

    def J(self, ctx: Ctx):
        """转动等效质量（kg）。"""
        for loc in ("drag", "inertia"):
            if self.has(loc) and ctx.on(self.key(loc)):
                return ctx.mag / 1000.0
        return 0.0

    def axle(self, ctx: Ctx):
        """轴摩擦等效阻力（N）：线性参数或网格。"""
        if self.has("drag") and ctx.on(self.key("drag")):
            return ctx.lin(self.pname("f_axle"))
        if self.has("axle") and ctx.on(self.key("axle")):
            return ctx.lin(self.pname("f_axle")) if self.axle_linear else ctx.mag
        return 0.0


@dataclass
class HandPull(Component):
    """用手通过测力计拉物体（摩擦力实验的拉法）。

    斜向上拉（与水平成 a）、木块匀速：F cos a = μ(N − F sin a) + … → F = (μN + …)/(cos a + μ sin a)。
    分母里的 μ 用网格 mu_pull（与 Block 的线性参数 μ 同步真值）。
    读数时还在加速：F − μN = m·a_acc，m = N/g → 多出 N·a_acc/g。
    "fixed" 变体（测力计水平固定、抽动木板）：木块静止，读数 = μN，与拉法无关。
    """

    label_zh: str = "手拉测力计"
    block: str = "block"
    angle_prior: Prior = Prior.uniform(5.0, 45.0, 9)
    acc_prior: Prior = Prior.uniform(0.2, 1.0, 9)

    def all_effects(self):
        return [
            Effect(
                "pull_angle", "拉力斜向上", "pulled at an upward angle",
                "拉测力计的手比木块高，拉力斜向上而不是沿水平方向；拉力的一部分向上提木块，"
                "木块对木板的压力变小，同时只有水平分力在克服摩擦。",
                "The hand was higher than the block, so the pull pointed upward; part of the pull lifts the block "
                "(less normal force) and only the horizontal part overcomes friction.",
                "y_scale", ("none",), self.angle_prior, "拉力与水平方向的夹角", "°",
                "让测力计和细线与木板平行，沿水平方向拉。",
                "Keep the scale and string parallel to the board and pull horizontally.",
                uses=(f"{self.block}.mu_pull",),
            ),
            Effect(
                "accelerating", "没有匀速拉动", "not pulled at constant speed",
                "读数时木块还在越拉越快，拉力除了克服摩擦，还要让木块加速。",
                "The block was still speeding up when the scale was read, so the pull both overcame friction "
                "and accelerated the block.",
                "y_scale", ("none",), self.acc_prior, "读数时木块的加速度", "m/s²",
                "等木块匀速运动后再读数；更好的办法是固定测力计、抽动木板。",
                "Read only once the block moves steadily, or fix the scale and pull the board instead.",
            ),
        ]

    def hand(self, ctx: Ctx) -> bool:
        return not ctx.flag("fixed")

    def denominator(self, ctx: Ctx):
        if self.hand(ctx) and self.has("pull_angle") and ctx.on(self.key("pull_angle")):
            a = np.deg2rad(ctx.mag)
            return np.cos(a) + ctx.p(f"{self.block}.mu_pull") * np.sin(a)
        return None

    def acceleration(self, ctx: Ctx):
        if self.hand(ctx) and self.has("accelerating") and ctx.on(self.key("accelerating")):
            return ctx.mag
        return None

    def variants(self, setup):
        s = setup.standard
        pick = (s[0], s[len(s) // 2], s[-1])
        return [Variant(
            "fixed", "测力计水平固定、抽动木板，读示数", "fix the scale, pull the board out, read the scale",
            frozenset({"fixed"}),
            menu=(("fixed_scale", f"改用固定测力计、抽动木板的方法，在 {pick[0]:g}、{pick[1]:g}、{pick[2]:g} N 压力下各读一次",
                   "fix the scale and pull the board out at three normal forces", 6.0, pick),),
        )]


@dataclass
class Release(Component):
    """释放方式。手推了一下：小车离开出发点时已有速度 v₀，½Mv² = ½Mv₀² + … → v² 多出 v₀²。
    "no_push" 变体：用细线拴住、剪断细线释放（手不碰小车）。
    机制 scatter：每次释放的位置和姿势都不一样（只加大散布）。
    """

    label_zh: str = "释放"
    push_prior: Prior = Prior.uniform(0.2, 0.6, 21)
    scatter_prior: Prior = Prior.uniform(0.005, 0.03, 11)
    scatter_unit: str = "s"

    def all_effects(self):
        return [
            Effect(
                "push", "释放时推了一下", "pushed at release",
                "释放时手没有直接松开，而是顺势推了一下（或手指带了一下），物体离开出发点时已经有了速度。",
                "The object was not simply let go: the hand gave it a small push, so it already had some speed "
                "when it left the start.",
                "y_offset", ("intercept_pos",), self.push_prior, "释放时的初速度 v₀", "m/s",
                "用挡板或细线固定物体，抽开挡板（剪断细线）释放，手不碰物体。",
                "Hold it with a stop or a string and release without touching it.",
            ),
            Effect(
                "scatter", "每次释放方式不一样", "inconsistent release",
                "每次释放的位置和姿势都不一样：有时物体歪着滑下、蹭到侧边，有时没有从标记处由静止放开。",
                "The release differed from run to run: sometimes the object slid crookedly or was not let go "
                "from rest at the mark.",
                "scatter", ("scatter_excess",), self.scatter_prior, "额外的读数散布", self.scatter_unit,
                "在同一标记处、摆正后由静止释放，必要时加导向挡板。",
                "Release from the same mark, squared up and from rest; add a guide if needed.",
                scatter=True,
            ),
        ]

    def v0(self, ctx: Ctx):
        if self.has("push") and ctx.on(self.key("push")) and not ctx.flag("no_push"):
            return ctx.mag
        return 0.0

    def extra_sigma(self, ctx):
        if self.has("scatter") and ctx.on(self.key("scatter")):
            return ctx.mag
        return None

    def variants(self, setup):
        if not self.has("push"):
            return []
        s = setup.standard
        pick = (s[0], s[len(s) // 2])
        return [Variant(
            "string_release", "用细线拴住、剪断细线释放（手不碰物体）", "release by cutting a string",
            frozenset({"no_push"}),
            menu=(("string_release", f"用细线拴住、剪断细线释放，在 {pick[0]:g} 和 {pick[1]:g} 处重测",
                   "release by cutting a string at two settings", 3.0, pick),),
        )]


# ── 表面 ──────────────────────────────────────────────────────────────


@dataclass
class Track(Component):
    """水平导轨 / 长木板（小车在上面运动）。

    compensated=True（牛顿第二定律的阻力补偿法）：一端垫高让小车拖着纸带匀速下滑。
      沿板方向小车受 Mg sinθ − f；没垫够时残余力 F_res < 0，垫太高时 F_res > 0（线性参数，N）。
      "detached" 变体（取下细线、轻推小车测它自己的加速度）：a₀ = F_res/M。
    compensated=False：没有补偿，滚动阻力 f = μ_r·N（μ_r 网格，"rolling" 机制），
      附加测量"水平滑行减速度"直接测 μ_r·g。
    """

    label_zh: str = "导轨"
    compensated: bool = True
    unbal_mean: float = -0.05
    over_mean: float = 0.05
    res_sd: float = 0.015
    rolling_prior: Prior = Prior.loguniform(0.004, 0.03, 25)

    def all_effects(self):
        if self.compensated:
            return [
                Effect(
                    "unbalanced", "没有平衡摩擦力或平衡不足", "friction not (fully) compensated",
                    "导轨没有垫高或垫得太低，小车受到的摩擦力和纸带阻力没有被重力沿斜面的分力抵消，"
                    "细线拉力有一部分用来克服阻力。",
                    "The track was not raised, or not raised enough, so friction and drag are not cancelled by the "
                    "component of gravity along it; part of the pull goes into overcoming them.",
                    "x_offset", ("intercept_neg",),
                    fix_zh="不挂槽码，把导轨不带滑轮的一端垫高，直到轻推后小车做匀速运动（纸带点迹间隔相等）。",
                    fix_en="With nothing hanging, raise the far end until a nudged cart moves at constant speed.",
                    linear=(LinearParam(self.pname("f_unbal"), self.unbal_mean, self.res_sd,
                                        "未被抵消的阻力（负值）", "N"),),
                ),
                Effect(
                    "overbalanced", "平衡摩擦力过度", "friction over-compensated",
                    "导轨垫得太高，重力沿斜面的分力超过了阻力，即使不挂槽码，小车也会沿导轨加速下滑。",
                    "The track was raised too much: gravity along it exceeds friction, so the cart speeds up even "
                    "with nothing hanging.",
                    "x_offset", ("intercept_pos",),
                    fix_zh="把垫块往外移一点，减小倾角，直到轻推小车后纸带点迹间隔相等。",
                    fix_en="Lower the raised end until the nudged cart moves at constant speed.",
                    linear=(LinearParam(self.pname("f_over"), self.over_mean, self.res_sd, "多出来的下滑力", "N"),),
                ),
            ]
        return [Effect(
            "rolling", "轨道阻力不可忽略", "track resistance not negligible",
            "小车运动时车轮的滚动阻力和轮轴摩擦一直在做负功，一部分能量变成了内能。",
            "Rolling resistance and axle friction act all along the track, turning part of the energy into heat.",
            "y_scale", ("slope_low",), self.rolling_prior, "等效滚动阻力系数 μ", "",
            "清洁轨道和车轮、给轮轴加润滑；或者先测出阻力，在计算里扣除阻力做的功。",
            "Clean the track and wheels, lubricate the axles, or measure the drag and account for its work.",
        )]

    def residual(self, ctx: Ctx):
        for loc, nm in (("unbalanced", "f_unbal"), ("overbalanced", "f_over")):
            if self.has(loc) and ctx.on(self.key(loc)):
                return ctx.lin(self.pname(nm))
        return 0.0

    def resistance(self, ctx: Ctx, normal):
        """滚动阻力（N），normal = 正压力。"""
        if self.has("rolling") and ctx.on(self.key("rolling")):
            return ctx.mag * normal
        return 0.0

    def variants(self, setup):
        if not self.compensated:
            return []
        return [Variant(
            "balance_check", "取下细线，轻推小车，测小车自己的加速度",
            "unhook the string, nudge the cart, measure its own acceleration",
            frozenset({"detached"}), reading=None,
            menu=(("balance_check", "取下细线，轻推小车，测小车自己的加速度",
                   "unhook the string, nudge the cart and measure its own acceleration", 3.0, (0.0,)),),
        )]

    def aux(self, setup):
        if self.compensated:
            return []
        return [_coast_aux(self.key("rolling"))]


def _coast_aux(key: str) -> AuxMeasurement:
    """把轨道调水平、轻推小车测滑行减速度：只有阻力时 a = μg。"""

    def decel(ctx):
        if ctx.on(key):
            return np.broadcast_to(ctx.mag * G, (ctx.P, ctx.n))
        return np.zeros((ctx.P, ctx.n))

    return AuxMeasurement(
        "coast", "把轨道调水平，轻推小车，测它滑行时的减速度", "level the track and measure the coasting deceleration",
        Reading("decel", "水平滑行减速度", "coasting deceleration", "m/s²", resolution=0.001, sigma=0.01),
        decel, minutes=6.0, menu_label_zh="把轨道调水平，轻推小车，测滑行减速度",
    )


@dataclass
class Incline(Component):
    """斜面（倾斜导轨）。

    angle_deg：记录的倾角；None 表示没记录（真值放在网格 theta_deg 上，先验 angle_prior）。
    protractor：量角器组件名（量角器零点不准时真实倾角 = 记录值 + δ）。
    friction："grid"（滚动阻力系数 μ 放网格，能量类模板要开方，只能这样）、"linear"（μ 作线性参数）或 None。
      阻力 f = μN = μMg cosθ；沿斜面滑过 s = h/sinθ，动能定理里阻力功 = μMg cosθ·h/sinθ → 能量按 (1 − μ cotθ) 打折。
    slant：把小车沿斜面滑过的距离当成了竖直高度（只在没记录倾角时列入）：真实高度 = 记录值 × sinθ。
    """

    label_zh: str = "斜面"
    angle_deg: float | None = None
    angle_prior: Prior = Prior.uniform(12.0, 30.0, 19)
    protractor: str | None = None
    friction: str | None = "grid"
    mu_prior: Prior = Prior.loguniform(0.004, 0.03, 25)
    mu_mean: float = 0.015
    mu_sd: float = 0.005
    board_cm: float = 100.0

    @property
    def theta_name(self) -> str:
        return self.pname("theta_deg")

    def nuisance(self):
        return {} if self.angle_deg is not None else {self.theta_name: self.angle_prior}

    def _angle_uses(self) -> tuple[str, ...]:
        return () if self.angle_deg is not None else (self.theta_name,)

    def all_effects(self):
        effs = []
        if self.friction == "grid":
            effs.append(Effect(
                "friction", "轨道和车轮之间有摩擦", "rolling friction on the track",
                "车轮的滚动阻力和轮轴里的摩擦在下滑的全过程中都在做负功，一部分机械能变成了内能。",
                "Rolling resistance and axle friction do negative work all the way down, turning some mechanical "
                "energy into heat.",
                "y_scale", ("slope_low",), self.mu_prior, "等效滚动阻力系数 μ", "",
                "清洁轨道和车轮、给轮轴加润滑；或换气垫导轨；或者测出阻力，在能量里扣除阻力做的功。",
                "Clean the track and wheels, lubricate the axles or use an air track; or measure the drag and "
                "account for its work.",
                uses=self._angle_uses(),
            ))
        elif self.friction == "linear":
            effs.append(Effect(
                "friction", "斜面阻力不可忽略", "track resistance not negligible",
                "小车沿斜面运动时一直受到滚动阻力和轮轴摩擦，这个阻力与小车对斜面的压力成正比。",
                "Rolling resistance and axle friction act all along the incline, proportional to the normal force.",
                "y_offset", ("intercept_neg", "slope_low"),
                fix_zh="清洁轨道和车轮、给轮轴加润滑；或先测出阻力系数，在合力里扣除。",
                fix_en="Clean and lubricate, or measure the drag coefficient and subtract it from the net force.",
                linear=(LinearParam(self.pname("mu"), self.mu_mean, self.mu_sd, "滚动阻力系数", ""),),
            ))
        if self.angle_deg is None:
            effs.append(Effect(
                "slant", "把斜面上的距离当成了下落高度", "slant distance used as height",
                "没有测竖直高度差，而是把物体沿斜面滑过的距离当成了下落高度 h，也没有记录斜面的倾角。",
                "The distance travelled along the incline was recorded as the drop height h; the angle was not recorded.",
                "x_scale", ("slope_low",), Prior.point(1.0),
                fix_zh="用铅垂线和刻度尺量竖直高度差；或量出斜面倾角，用 h = s·sinθ 换算。",
                fix_en="Measure the vertical drop, or record the angle and use h = s·sinθ.",
                uses=(self.theta_name,),
            ))
        return effs

    def angle(self, ctx: Ctx):
        """真实倾角（弧度），(P, 1) 或标量。"""
        th = ctx.p(self.theta_name) if self.angle_deg is None else self.angle_deg
        if self.protractor is not None:
            th = th + ctx.part(self.protractor).offset_deg(ctx)
        return np.radians(th)

    def angle_rec(self) -> float:
        if self.angle_deg is None:
            raise ValueError("没有记录倾角")
        return math.radians(self.angle_deg)

    def control(self, ctx, x, quantity):
        if quantity == "height" and self.has("slant") and ctx.on(self.key("slant")):
            return x * np.sin(np.radians(ctx.p(self.theta_name)))
        return x

    def loss_factor(self, ctx: Ctx):
        """能量打折因子 1 − μ cotθ（网格摩擦时）；否则 1。"""
        if self.friction == "grid" and ctx.on(self.key("friction")):
            return 1.0 - ctx.mag / np.tan(self.angle(ctx))
        return 1.0

    def resistance(self, ctx: Ctx, normal):
        """沿斜面的阻力（N）：线性摩擦时是 Affine。"""
        if self.friction == "linear" and ctx.on(self.key("friction")):
            return ctx.lin(self.pname("mu")) * normal
        if self.friction == "grid" and ctx.on(self.key("friction")):
            return ctx.mag * normal
        return 0.0

    def aux(self, setup):
        out = []
        if self.friction == "grid":
            out.append(_coast_aux(self.key("friction")))
        if self.protractor is not None and self.angle_deg is not None:
            L, rec = self.board_cm, self.angle_deg

            def height(ctx, inc=self):
                th = inc.angle(ctx)
                return np.broadcast_to(L * np.sin(th), (ctx.P, ctx.n))

            out.append(AuxMeasurement(
                "incline_height", f"用刻度尺量斜面顶端的高度（板长 {L:.1f} cm）",
                "measure the height of the raised end",
                Reading("incline_height", "斜面顶端高度", "raised-end height", "cm", resolution=0.1, sigma=0.1),
                height, minutes=3.0, setting=rec, setting_label_zh="量角器读数", setting_unit="°",
                menu_label_zh="量斜面顶端高度，用 sin θ = h/L 核对量角器读数",
            ))
        return out

    def questions(self):
        return [OpenQuestion(self.pname("angle_deg"), "斜面（轨道）的倾角是多少？用量角器量，或量高度和长度换算。", "°",
                             self.angle_deg)]


@dataclass
class Board(Component):
    """水平长木板（滑动摩擦实验）。木板（或桌面）有一点倾斜时（倾角 β，正 = 向高处拉）：
    F = μN cosβ + N sinβ；掉头反向拉 F' = μN cosβ − N sinβ。"reverse" 变体就是掉头拉。
    """

    label_zh: str = "长木板"
    tilt_prior: Prior = Prior.sym_uniform(1.0, 6.0, 12)

    def all_effects(self):
        return [Effect(
            "tilt", "木板没有放水平", "board not level",
            "长木板（或桌面）有一点倾斜，木块是沿着一个很缓的斜坡被拉动的；重力沿木板方向有一个分力，"
            "向高处拉时要多用力，向低处拉时少用力。",
            "The board (or table) is slightly tilted, so the block is pulled along a gentle slope; a part of its "
            "weight acts along the board, adding to or subtracting from the pull.",
            "y_scale", ("none",), self.tilt_prior, "木板倾角（正 = 向高处拉）", "°",
            "用水平仪或小球检查木板是否水平；或沿两个方向各拉一次取平均。",
            "Level the board (spirit level or a marble), or pull both ways and average.",
        )]

    def tilt(self, ctx: Ctx):
        """(倾角弧度, 方向符号)；不适用时 None。"""
        if self.has("tilt") and ctx.on(self.key("tilt")):
            return np.deg2rad(ctx.mag), (-1.0 if ctx.flag("reverse") else 1.0)
        return None

    def variants(self, setup):
        if not self.has("tilt"):
            return []
        s = setup.standard
        pick = (s[0], s[len(s) // 2], s[-1])
        return [Variant(
            "reverse", "木块掉头，沿木板反方向匀速拉动，读示数", "pull the block the other way along the board",
            frozenset({"reverse"}),
            menu=(("reverse_pull", f"木块掉头，沿木板反方向匀速拉动，在 {pick[0]:g}、{pick[1]:g}、{pick[2]:g} N 压力下各读一次",
                   "pull the block the other way along the board at three normal forces", 4.0, pick),),
        )]


@dataclass
class Lever(Component):
    """带刻度的杠杆。支点不在杠杆重心（没有先调平衡）时，杠杆自重产生一个附加力矩 M₀：
    F₁l₁ = F₂l₂ + M₀。机制 not_balanced：M₀ 作线性参数（N·m）。
    """

    label_zh: str = "杠杆"
    m0_sd: float = 0.02

    def all_effects(self):
        return [Effect(
            "not_balanced", "挂钩码前杠杆没有调平衡", "lever not balanced before loading",
            "挂钩码之前没有调节平衡螺母让杠杆水平静止，杠杆自身的重力对支点有一个力矩。",
            "The lever was not balanced with its nuts before loading, so its own weight exerts a torque about the pivot.",
            "y_offset", ("intercept_pos", "intercept_neg"),
            fix_zh="挂钩码前先调节两端的平衡螺母，使杠杆在水平位置平衡。",
            fix_en="Balance the empty lever with its nuts before hanging any masses.",
            linear=(LinearParam(self.pname("m0"), 0.0, self.m0_sd, "杠杆自重的附加力矩", "N·m"),),
        )]

    def extra_torque(self, ctx: Ctx):
        if self.has("not_balanced") and ctx.on(self.key("not_balanced")):
            return ctx.lin(self.pname("m0"))
        return 0.0


def _replace(e: Effect, **kw) -> Effect:
    from dataclasses import replace

    return replace(e, **kw)
