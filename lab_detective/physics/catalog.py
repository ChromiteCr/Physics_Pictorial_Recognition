"""装置目录：用组件组合出来的装置。精选实验只是起点，其余装置都由物理组件组合而成。

两类条目：
1. 镜像（mirror_of = 精选实验 id）：用组件重新搭出精选实验，原因后缀和观测种类名沿用原实验，
   用来验证组件层——同一组参数下预测与精选实验一致（tests/test_physics_mirror.py）。
   与精选实验的约定一致：牛顿第二定律的"只有随机误差"是课本模型 a = F/M，其余原因都带精确的连接体动力学
   （Setup.exact_in_faults = 槽码惯性）。
2. 新组合（训练集里没有，默认 held_out=True，可作迁移测试）：
   斜面上用悬挂重物拉小车的牛顿第二定律、悬挂重物带动小车的系统机械能守恒、弹簧振子周期与质量的关系、
   光电门测自由落体 g、斜面上滑块由静止下滑的匀加速运动。

每个条目带看图建模用的器材组（vision/schema.APPARATUS 的 id）、构型线索词和先验常见程度（from_photo 用），
以及"标准方案下能看出来的大故障"（测试和可辨识性说明用）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

from ..labsim.core import G, MenuItem, Prior
from .base import Control, Setup
from .components import Ball, Block, Board, Bob, Cart, HandPull, Hanger, Incline, Pulley, Release, Spring, StringLine, Track
from .compose import ComposedExperiment
from .instruments import PhoneTimer, Photogate, Protractor, Ruler, SpringScale, Stopwatch, TickerTimer
from .templates import (
    EnergyIncline,
    FreeFallGate,
    FrictionPull,
    PendulumPeriod,
    SlideFromRest,
    SpringOscillator,
    StaticSpring,
    SystemEnergy,
    UniformAccel,
)

# ── 镜像：精选实验 ────────────────────────────────────────────────────


def hooke_setup() -> Setup:
    return Setup(
        id="hooke_composed", name_zh="探究弹簧弹力与形变量的关系（组件组合）",
        name_en="Hooke's law (composed from components)",
        template=StaticSpring(spring="spring"),
        parts=[Spring("spring"), Hanger("hanger", enabled=("omitted",)), Ruler("ruler", role="reading")],
        instrument="ruler",
        control=Control("mass", "钩码质量", "slotted mass", "g", 1e-3),
        standard=(50.0, 100.0, 150.0, 200.0, 250.0), single=200.0,
        random_zh="测量步骤正确，数据偏离只来自读数的随机误差。",
        extra_menu=(
            MenuItem("zero_load", "只挂挂钩、不挂钩码，读伸长量", "read the extension with the hanger only", 1.0,
                     (("main", 0.0),)),
            MenuItem("heavy_load", "再挂一组更重的钩码（350 g）", "add a heavier load (350 g)", 2.0, (("main", 350.0),)),
            MenuItem("repeat_150", "在 150 g 处重复读一次", "repeat the 150 g reading", 1.0, (("main", 150.0),)),
        ),
        held_out=False, transfer=None,
        cause_alias={"hanger_omitted": "hanger_omitted", "ruler_zero_offset": "zero_offset",
                     "ruler_total_length": "total_length", "spring_beyond_limit": "beyond_limit",
                     "spring_not_settled": "reading_scatter"},
        knowledge_points=("kp.hooke_law", "kp.elastic_limit", "kp.zero_adjustment", "kp.parallax"),
    )


def friction_setup() -> Setup:
    standard = (2.0, 3.0, 4.0, 5.0, 6.0)
    return Setup(
        id="friction_composed", name_zh="探究影响滑动摩擦力大小的因素（组件组合）",
        name_en="Sliding friction vs normal force (composed)",
        template=FrictionPull(block="block", board="board", pull="hand"),
        parts=[Block("block"), Board("board"), HandPull("hand", block="block"),
               SpringScale("scale", orientation="horizontal")],
        instrument="scale",
        control=Control("force", "压力（木块与钩码总重）", "normal force", "N", 1.0),
        standard=standard, single=4.0,
        extra_protocols={"fixed_scale": ("改进方案：固定测力计、抽动木板（5 组压力）", "fixed scale, pull the board",
                                         tuple(("fixed", w) for w in standard))},
        extra_menu=(MenuItem("repeat_4", "压力 4 N 时再拉两次", "repeat the 4 N pull twice", 2.0,
                             (("main", 4.0), ("main", 4.0))),),
        held_out=False, transfer=None,
        sync={"block.mu_pull": "block.mu"},
        cause_alias={"scale_zero_shift": "zero_vertical", "hand_pull_angle": "pull_angle",
                     "hand_accelerating": "accelerating", "board_tilt": "board_tilt",
                     "scale_pointer_scatter": "reading_scatter"},
        knowledge_points=("kp.kinetic_friction", "kp.friction_measurement", "kp.equilibrium_two_force",
                          "kp.zero_adjustment"),
    )


def newton2_setup(timer: str = "ticker") -> Setup:
    if timer == "ticker":
        inst = TickerTimer("timer", mode="accel", enabled=())
    elif timer == "photogate":
        inst = Photogate("timer", mode="accel", enabled=(), sigma=0.006, resolution=0.001)
    else:
        raise ValueError(f"timer 只能是 ticker / photogate：{timer}")
    return Setup(
        id="newton2_composed", name_zh="探究加速度与力的关系（组件组合）",
        name_en="Newton's second law: acceleration vs force (composed)",
        template=UniformAccel(body="cart", track="track", link="string", pulley="pulley", hanger="hanger"),
        parts=[Cart("cart", mass_g=500.0, enabled=()), Track("track", compensated=True), StringLine("string"),
               Pulley("pulley", mode="combined"), Hanger("hanger", enabled=("heavy",)), inst],
        instrument="timer",
        control=Control("mass", "小盘和槽码总质量", "hanging mass", "g", 1e-3),
        standard=(10.0, 20.0, 30.0, 40.0, 50.0), single=30.0,
        random_zh="已平衡摩擦力，细线与轨道平行，槽码质量远小于小车质量，数据偏离只来自测加速度的随机误差。",
        extra_menu=(
            MenuItem("repeat_10", "在 10 g 处重复测一次", "repeat the 10 g run", 2.0, (("main", 10.0),)),
            MenuItem("repeat_50", "在 50 g 处重复测一次", "repeat the 50 g run", 2.0, (("main", 50.0),)),
        ),
        held_out=False, transfer=None,
        exact_in_faults=("hanger_heavy",),
        cause_alias={"track_unbalanced": "unbalanced", "track_overbalanced": "overbalanced",
                     "hanger_heavy": "heavy_hanging", "string_not_parallel": "string_angle",
                     "pulley_drag": "pulley_drag"},
        knowledge_points=("kp.newton_second_law", "kp.balancing_friction", "kp.connected_bodies", "kp.pulley"),
    )


def energy_setup(timer: str = "photogate", angle_deg: float | None = None) -> Setup:
    if timer == "photogate":
        inst = Photogate("timer", mode="speed", flag_mm=10.0, width_linear=True)
    elif timer == "ticker":
        inst = TickerTimer("timer", mode="speed", enabled=())
    else:
        raise ValueError(f"timer 只能是 photogate / ticker：{timer}")
    return Setup(
        id="energy_composed", name_zh="验证机械能守恒定律（小车沿倾斜轨道下滑，组件组合）",
        name_en="Conservation of mechanical energy on an incline (composed)",
        template=EnergyIncline(body="cart", incline="incline", release="release"),
        parts=[Cart("cart", mass_g=300.0, load_g=500.0), Incline("incline", angle_deg=angle_deg, friction="grid"),
               Release("release", enabled=("push",)),
               Ruler("ruler", role="control", recheck_kind="height_check",
                     recheck_label_zh="用铅垂线和刻度尺重新竖直测量释放点到测速点的高度差"),
               inst],
        instrument="timer",
        control=Control("height", "下落高度 h", "drop height h", "cm", 1e-2),
        standard=(5.0, 10.0, 15.0, 20.0, 25.0), single=20.0,
        random_zh="测量步骤正确，摩擦和车轮转动的影响小到可以忽略，数据偏离只来自读数的随机误差。",
        held_out=False, transfer=None,
        cause_alias={"timer_flag_width": "flag_width", "incline_slant": "slant_length",
                     "incline_friction": "track_friction", "cart_wheel_inertia": "wheel_inertia",
                     "ruler_wrong_point": "wrong_height_point", "release_push": "push"},
        knowledge_points=("kp.mechanical_energy_conservation", "kp.kinetic_energy", "kp.gravitational_pe",
                          "kp.photogate", "kp.energy_dissipation"),
    )


def pendulum_setup(timer: str = "phone") -> Setup:
    common = dict(count=30, short_count=10, extra_sigma_ref=0.039, tau_sd=0.3)
    if timer == "phone":
        inst = PhoneTimer("timer", **common)
    elif timer == "stopwatch":
        inst = Stopwatch("timer", **common)
    else:
        raise ValueError(f"timer 只能是 phone / stopwatch：{timer}")
    return Setup(
        id="pendulum_composed", name_zh="用单摆测量重力加速度（组件组合）",
        name_en="Measuring g with a simple pendulum (composed)",
        template=PendulumPeriod(bob="bob"),
        parts=[Bob("bob"), inst,
               Ruler("ruler", role="control", enabled=(), recheck_kind="length_check",
                     recheck_label_zh="从悬点量到摆球球心，重新测量这一组的摆长",
                     recheck_reading_zh="悬点到球心的长度（重测）", recheck_sigma=0.1, recheck_at=85.0)],
        instrument="timer",
        control=Control("length", "摆长 L", "pendulum length L", "cm", 1e-2),
        standard=(40.0, 55.0, 70.0, 85.0, 100.0), single=85.0,
        random_zh="摆长从悬点量到球心，摆角很小，摆球在竖直平面内摆动，计数和计时都正确，数据偏离只来自读数的随机误差。",
        extra_menu=(MenuItem("repeat_40", "在 40 cm 处重复测一次", "repeat the 40 cm measurement", 2.0,
                             (("main", 40.0),)),),
        held_out=True, transfer="far",
        cause_alias={"bob_top_of_bob": "top_of_bob", "bob_large_amplitude": "large_amplitude",
                     "timer_miscount": "miscount", "bob_conical": "conical", "timer_timing_offset": "timing_offset"},
        knowledge_points=("kp.simple_pendulum", "kp.pendulum_g", "kp.shm", "kp.multiple_measurements"),
    )


# ── 新组合（不在精选实验里） ──────────────────────────────────────────


def incline_newton2_setup() -> Setup:
    """斜面上用悬挂重物拉小车：倾角 5° 的轨道，滑轮装在高端，槽码从车上取下挂到小盘里（M_t = 1 kg 不变），
    两个光电门测加速度。精确关系 a·M_t = mg − (M_t − m)g sinθ − 阻力，学生画 a 对合力，理想斜率 1/M_t。"""
    return Setup(
        id="incline_newton2", name_zh="斜面上用悬挂重物拉小车，探究加速度与合力的关系",
        name_en="Newton's second law with a hanging mass pulling a cart up an incline",
        template=UniformAccel(body="cart", incline="incline", link="string", pulley="pulley", hanger="hanger",
                              conserve_total=True, uphill=True, x_mode="net"),
        parts=[Cart("cart", mass_g=1000.0, enabled=()),
               Incline("incline", angle_deg=5.0, protractor="protractor", friction="linear"),
               Protractor("protractor"), StringLine("string"),
               Pulley("pulley", mode="separate", j_prior=Prior.uniform(5.0, 40.0, 15)),
               Hanger("hanger", inertial=True, enabled=()),
               Photogate("gate", mode="accel", sigma=0.008, resolution=0.001)],
        instrument="gate",
        control=Control("mass", "挂在小盘里的槽码（含小盘）质量 m", "hanging mass", "g", 1e-3),
        standard=(100.0, 150.0, 200.0, 250.0, 300.0), single=200.0,
        main_label_zh="从车上取槽码挂到小盘里，由静止释放，测小车沿斜面向上的加速度",
        main_label_en="move masses from the cart to the hanger, measure the cart's acceleration up the incline",
        random_zh="倾角读得准，阻力可以忽略，细线与斜面平行，滑轮轻而灵活，数据偏离只来自测加速度的随机误差。",
        extra_menu=(MenuItem("repeat_100", "在 100 g 处重复测一次", "repeat the 100 g run", 2.0, (("main", 100.0),)),),
        held_out=True, transfer="near",
        cause_alias={"incline_friction": "track_resistance", "protractor_zero": "protractor_zero",
                     "string_not_parallel": "string_angle", "pulley_inertia": "pulley_inertia",
                     "pulley_axle": "pulley_axle", "gate_flag_width": "flag_width", "gate_gate_spacing": "gate_spacing"},
        knowledge_points=("kp.newton_second_law", "kp.incline_forces", "kp.connected_bodies", "kp.photogate"),
    )


def system_energy_setup() -> Setup:
    """水平导轨上的小车（500 g）被跨过定滑轮的 50 g 重物拉动；重物由静止下落 h 后，光电门测小车速度。"""
    return Setup(
        id="system_energy", name_zh="悬挂重物带动小车，验证系统机械能守恒",
        name_en="Mechanical energy of a hanging mass pulling a cart",
        template=SystemEnergy(body="cart", track="track", hanger="hanger", pulley="pulley", release="release"),
        parts=[Cart("cart", mass_g=500.0, load_g=500.0), Track("track", compensated=False),
               Pulley("pulley", mode="separate", enabled=("inertia",), j_prior=Prior.uniform(5.0, 40.0, 15)),
               Hanger("hanger", mass_g=50.0, inertial=True, enabled=()),
               Release("release", enabled=("push",)),
               Ruler("ruler", role="control", recheck_kind="height_check",
                     recheck_label_zh="用刻度尺重新量重物从释放到小车过光电门时下落的高度"),
               Photogate("gate", mode="speed", flag_mm=10.0, width_linear=True)],
        instrument="gate",
        control=Control("height", "重物下落高度 h", "drop of the hanging mass h", "cm", 1e-2),
        standard=(10.0, 20.0, 30.0, 40.0, 50.0), single=30.0,
        main_label_zh="重物由静止下落 h 后，读小车上遮光片的挡光时间",
        main_label_en="release, read the cart's gate time after the weight drops h",
        random_zh="阻力和滑轮、车轮的转动可以忽略，由静止释放，高度量得正确，数据偏离只来自读数的随机误差。",
        held_out=True, transfer="near",
        cause_alias={"cart_wheel_inertia": "wheel_inertia", "track_rolling": "track_resistance",
                     "pulley_inertia": "pulley_inertia", "release_push": "push", "ruler_wrong_point": "wrong_height_point",
                     "gate_flag_width": "flag_width"},
        knowledge_points=("kp.mechanical_energy_conservation", "kp.connected_bodies", "kp.kinetic_energy",
                          "kp.energy_dissipation", "kp.photogate"),
    )


def spring_oscillator_setup() -> Setup:
    """竖直弹簧下挂钩码上下振动（弹簧 20 g 已称出），手机传感器测 20 次全振动的时间。"""
    return Setup(
        id="spring_oscillator", name_zh="探究弹簧振子的周期与振子质量的关系",
        name_en="Period of a spring oscillator vs mass",
        template=SpringOscillator(spring="spring"),
        parts=[Spring("spring", mode="oscillation", mass_g=20.0), Hanger("hanger", enabled=("omitted",)),
               PhoneTimer("timer", count=20, short_count=10, tau_sd=0.2)],
        instrument="timer",
        control=Control("mass", "钩码质量 m", "slotted mass", "g", 1e-3),
        standard=(50.0, 100.0, 150.0, 200.0, 250.0), single=150.0,
        random_zh="振子质量算全了（含挂钩和弹簧的 1/3），振幅不大，计数和计时都正确，数据偏离只来自计时的随机误差。",
        held_out=True, transfer="far",
        cause_alias={"hanger_omitted": "hanger_omitted", "timer_miscount": "miscount",
                     "timer_timing_offset": "timing_offset"},
        knowledge_points=("kp.shm", "kp.hooke_law", "kp.multiple_measurements", "kp.graph_method"),
    )


def photogate_free_fall_setup() -> Setup:
    """电磁铁吸住直径 2.00 cm 的钢球，正下方 h 处有光电门；小球穿过光束的挡光时间给出速度，验证 v² = 2gh 并测 g。"""
    return Setup(
        id="photogate_free_fall", name_zh="用光电门测小球速度，研究自由落体并测重力加速度",
        name_en="Free fall: ball speed from a photogate",
        template=FreeFallGate(ball="ball"),
        parts=[Ball("ball", diameter_cm=2.0),
               Ruler("ruler", role="control", wrong_prior=Prior.sym_uniform(0.5, 3.0, 26),
                     recheck_kind="height_check", recheck_label_zh="从释放时的球心量到光束中心，重量这一组的下落高度"),
               Photogate("gate", mode="speed", flag_mm=20.0, width_linear=False, round_object=True,
                         sigma=0.02, resolution=0.1)],
        instrument="gate",
        control=Control("height", "下落高度 h（球心到光束）", "drop height", "cm", 1e-2),
        standard=(20.0, 40.0, 60.0, 80.0, 100.0), single=80.0,
        main_label_zh="小球从高度 h 处由静止下落，读光电门的挡光时间",
        main_label_en="drop the ball from height h, read the gate blocking time",
        random_zh="用的是实心钢球，光束对准球心，直径和高度都量得正确，数据偏离只来自计时的随机误差。",
        held_out=True, transfer="near",
        cause_alias={"ball_air_drag": "air_drag", "ruler_wrong_point": "height_reference",
                     "gate_flag_width": "diameter_wrong", "gate_off_center": "beam_off_center"},
        knowledge_points=("kp.free_fall", "kp.instantaneous_velocity", "kp.photogate", "kp.graph_method"),
    )


def incline_slide_setup() -> Setup:
    """木块在倾角 30° 的木板上由静止下滑（抬起挡板释放，同时启动计时），到距离 s 处的光电门停止计时。"""
    return Setup(
        id="incline_slide", name_zh="研究木块沿斜面由静止下滑的匀加速运动",
        name_en="Uniformly accelerated slide down an incline",
        template=SlideFromRest(body="block"),
        parts=[Block("block", mu_linear=False), Incline("incline", angle_deg=30.0, friction=None),
               Release("release", enabled=("scatter",)),
               Ruler("ruler", role="control", recheck_kind="distance_check",
                     recheck_label_zh="从木块前端的释放位置量到光束，重量这一组的距离",
                     recheck_reading_zh="释放点到光束的距离（重测）"),
               Photogate("gate", mode="timer", sigma=0.002, resolution=0.0001)],
        instrument="gate",
        control=Control("distance", "释放点到光电门的距离 s", "distance along the incline", "cm", 1e-2),
        standard=(20.0, 40.0, 60.0, 80.0, 100.0), single=60.0,
        main_label_zh="抬起挡板由静止释放木块，读到达光电门的时间",
        main_label_en="lift the stop to release the block, read the time to the gate",
        random_zh="木块从同一位置由静止释放，距离量得正确，计时与运动同步，数据偏离只来自计时的随机误差。",
        held_out=True, transfer="near",
        cause_alias={"release_scatter": "release_scatter", "ruler_wrong_point": "start_mark",
                     "gate_start_delay": "start_delay"},
        knowledge_points=("kp.uniform_acceleration", "kp.incline_forces", "kp.kinetic_friction", "kp.photogate"),
    )


# ── 目录 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CatalogEntry:
    """目录条目。

    requires：必需器材组（每组里有一个就算有），optional：可选器材；cue_words：构型线索里出现就加分的词；
    prior：常见程度（同样匹配时课本必做实验优先）；options(apparatus, cues)：由照片推出构造选项（计时器等）。
    param_map：镜像时 组合参数名 → 精选实验参数名（只给测试用）。
    large_fault：(原因后缀, simulate 参数) —— 标准方案下面板上看得出来的大故障。
    constant：(说明, 由面板斜率和截距算出物理常数的函数, 真值)。
    """

    id: str
    name_zh: str
    build: Callable[..., Setup]
    mirror_of: str | None
    held_out: bool
    requires: tuple[frozenset, ...]
    optional: frozenset = frozenset()
    cue_words: tuple[str, ...] = ()
    prior: float = 0.0
    options: Callable[[set, list], dict] = field(default=lambda app, cues: {})
    param_map: Mapping[str, str] = field(default_factory=dict)
    large_fault: tuple[str, dict] | None = None
    constant: tuple | None = None

    def setup(self, **opts) -> Setup:
        return self.build(**opts)

    def experiment(self, **opts) -> ComposedExperiment:
        return ComposedExperiment(self.build(**opts))


def _fs(*xs) -> frozenset:
    return frozenset(xs)


def _timer_accel(app, cues):
    return {"timer": "photogate"} if "photogate" in app else {}


def _timer_speed(app, cues):
    if "photogate" in app:
        return {"timer": "photogate"}
    if {"ticker_timer", "paper_tape"} & app:
        return {"timer": "ticker"}
    return {}


def _timer_pendulum(app, cues):
    return {"timer": "stopwatch"} if "stopwatch" in app else {}


BALL = _fs("ball", "iron_block")
TRACKLIKE = _fs("track", "incline")

CATALOG: dict[str, CatalogEntry] = {e.id: e for e in (
    CatalogEntry(
        "hooke_composed", "探究弹簧弹力与形变量的关系", hooke_setup, "hooke", False,
        requires=(_fs("spring"),), optional=_fs("slotted_mass", "ruler", "iron_stand"),
        cue_words=("竖直", "伸长", "刻度尺", "挂钩码", "静止"), prior=0.3,
        param_map={"hanger.hook_g": "hanger_g", "spring.inv_k": "inv_k", "ruler.zero_offset": "zero_offset",
                   "ruler.unloaded_length": "unloaded_length"},
        large_fault=("hanger_omitted", dict(nuisance={"hanger.hook_g": 45.0})),
        constant=("劲度系数 k (N/m) = 100/斜率", lambda slope, icpt: 100.0 / slope, 20.0),
    ),
    CatalogEntry(
        "friction_composed", "探究影响滑动摩擦力大小的因素", friction_setup, "friction", False,
        requires=(_fs("dynamometer"), _fs("wooden_block")), optional=_fs("track", "slotted_mass", "string"),
        cue_words=("水平", "匀速", "拉", "木板"), prior=0.3,
        param_map={"block.mu": "mu", "block.mu_pull": "mu_pull", "scale.hook_weight": "hook_weight"},
        large_fault=("zero_vertical", dict(linear={"scale.hook_weight": 0.3})),
        constant=("动摩擦因数 μ = 斜率", lambda slope, icpt: slope, 0.30),
    ),
    CatalogEntry(
        "newton2_composed", "探究加速度与力的关系（牛顿第二定律）", newton2_setup, "newton2", False,
        requires=(_fs("cart"), TRACKLIKE, _fs("string")),
        optional=_fs("pulley", "slotted_mass", "ticker_timer", "paper_tape", "photogate"),
        cue_words=("滑轮", "槽码", "钩码", "细线连着小车", "平衡摩擦"), prior=0.3, options=_timer_accel,
        param_map={"track.f_unbal": "f_unbal", "track.f_over": "f_over", "pulley.f_axle": "f_pulley"},
        large_fault=("overbalanced", dict(linear={"track.f_over": 0.08})),
        constant=("小车质量 M (kg) = 1/斜率", lambda slope, icpt: 1.0 / slope, 0.5),
    ),
    CatalogEntry(
        "energy_composed", "验证机械能守恒定律（小车沿倾斜轨道下滑）", energy_setup, "energy", False,
        requires=(_fs("cart"), TRACKLIKE), optional=_fs("photogate", "incline", "ticker_timer", "paper_tape", "metal_stop"),
        cue_words=("倾斜", "斜面", "释放", "下滑", "高度"), prior=0.25, options=_timer_speed,
        param_map={"incline.theta_deg": "theta_deg", "timer.w_true": "w_true", "timer.w_err": "w_err"},
        large_fault=("push", dict(magnitude=0.5)),
        constant=("½v² 对 gh 的斜率（应为 1）", lambda slope, icpt: slope, 1.0),
    ),
    CatalogEntry(
        "pendulum_composed", "用单摆测量重力加速度", pendulum_setup, "pendulum", True,
        requires=(_fs("string"), BALL), optional=_fs("iron_stand", "stopwatch", "ruler", "caliper"),
        cue_words=("摆", "细线下方挂着小球", "来回", "最低点"), prior=0.25, options=_timer_pendulum,
        param_map={"bob.r_cm": "r_cm", "gravity.inv_sqrt_g": "inv_sqrt_g", "timer.tau": "tau"},
        large_fault=("top_of_bob", dict(nuisance={"bob.r_cm": 1.5})),
        constant=("g (m/s²) = 4π²K(5°)²/斜率", lambda slope, icpt: 4 * math.pi**2 * PendulumPeriod().k_std**2 / slope, G),
    ),
    CatalogEntry(
        "incline_newton2", "斜面上用悬挂重物拉小车，探究加速度与合力的关系", incline_newton2_setup, None, True,
        requires=(_fs("cart"), TRACKLIKE, _fs("string")),
        optional=_fs("incline", "pulley", "slotted_mass", "photogate", "protractor"),
        cue_words=("斜面", "倾斜", "向上拉", "高端", "滑轮在上"), prior=0.05,
        large_fault=("flag_width", dict(magnitude=0.08)),
        constant=("系统总质量 M_t (kg) = 1/斜率", lambda slope, icpt: 1.0 / slope, 1.0),
    ),
    CatalogEntry(
        "system_energy", "悬挂重物带动小车，验证系统机械能守恒", system_energy_setup, None, True,
        requires=(_fs("cart"), TRACKLIKE, _fs("string")),
        optional=_fs("pulley", "slotted_mass", "heavy_weight", "photogate"),
        cue_words=("下落", "能量", "机械能", "重物"), prior=0.1,
        large_fault=("push", dict(magnitude=0.5)),
        constant=("ΔE_k 对 ΔE_p 的斜率（应为 1）", lambda slope, icpt: slope, 1.0),
    ),
    CatalogEntry(
        "spring_oscillator", "探究弹簧振子的周期与振子质量的关系", spring_oscillator_setup, None, True,
        requires=(_fs("spring"),), optional=_fs("slotted_mass", "stopwatch", "iron_stand"),
        cue_words=("振动", "振子", "周期", "上下", "计时"), prior=0.1,
        large_fault=("hanger_omitted", dict(nuisance={"hanger.hook_g": 45.0})),
        constant=("劲度系数 k (N/m) = 4π²/斜率", lambda slope, icpt: 4 * math.pi**2 / slope, 20.0),
    ),
    CatalogEntry(
        "photogate_free_fall", "用光电门测小球速度，研究自由落体", photogate_free_fall_setup, None, True,
        requires=(BALL, _fs("photogate")), optional=_fs("iron_stand", "ruler", "caliper"),
        cue_words=("下落", "自由落体", "电磁铁", "竖直"), prior=0.15,
        large_fault=("height_reference", dict(magnitude=3.0)),
        constant=("g (m/s²) = 斜率/2", lambda slope, icpt: slope / 2.0, G),
    ),
    CatalogEntry(
        "incline_slide", "研究木块沿斜面由静止下滑的匀加速运动", incline_slide_setup, None, True,
        requires=(_fs("wooden_block", "cart"), _fs("incline", "track"), _fs("photogate", "stopwatch")),
        optional=_fs("protractor", "ruler", "metal_stop"),
        cue_words=("斜面", "下滑", "由静止", "倾斜"), prior=0.1,
        large_fault=("start_mark", dict(magnitude=4.0)),
        constant=("加速度 a (m/s²) = 1/斜率²", lambda slope, icpt: 1.0 / slope**2, 2.35),
    ),
)}


def mirrors() -> list[CatalogEntry]:
    return [e for e in CATALOG.values() if e.mirror_of]


def new_compositions() -> list[CatalogEntry]:
    return [e for e in CATALOG.values() if not e.mirror_of]


def get(entry_id: str) -> CatalogEntry:
    return CATALOG[entry_id]


def experiment(entry_id: str, **opts) -> ComposedExperiment:
    return CATALOG[entry_id].experiment(**opts)
