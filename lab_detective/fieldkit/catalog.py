"""真实故障注入实验的目录：每个实验的基础做法、可注入的条件（配方）、追加测量怎么做、仪器配置。

约定：
- 只记录器材读数，不记录任何关于人的信息；只有学生本人测量，成年人只负责安全监督和按密封卡设置故障。
- 每个条件（配方）给出：谁来设置（adult = 学生看不到，算"盲"；student = 学生自己照卡片做，算"未盲"）、
  具体动作（卡片内页上的文字，只写动作，不写原因名）、幅度选项（都在对应原因的先验范围内）、安全提示。
- 真值写成"引擎 → 原因后缀（按优先级）"：第一个在该引擎原因表里存在的后缀就是真值；
  用到后面的后缀时记为"等价"（例如通用通道斜率未知时，比例类误差被斜率吸收，等价于只有随机误差）。
- 会损坏器材的故障一律不做（例如弹簧超过弹性限度，只在模拟里做），见 EXCLUDED。
- 精选的几个实验只是起点：其他装置走"其他实验"通道，由通用关系族和物理组件拼出来（单摆这里就走这条通道）。
"""

from __future__ import annotations

import copy
import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from ..labsim.core import Experiment

FIELDKIT_VERSION = 1

# 真实器材上永远不做的原因（只在模拟里做）
EXCLUDED = {
    "hooke.beyond_limit": "超过弹性限度会永久拉坏弹簧，甚至让弹簧崩断；只在模拟里做。",
}


@dataclass(frozen=True)
class Magnitude:
    value: float  # 数值（与原因的幅度同单位，方向用正负号）
    text: str  # 卡片上给成年人/学生看的具体做法


@dataclass(frozen=True)
class Recipe:
    key: str  # 条件键（只出现在密封文件里）
    truth: dict  # 引擎名 → 原因后缀元组（按优先级）
    setter: str  # "adult" | "student"
    steps: tuple[str, ...]  # 成年人照做的动作（含假动作），可带 {mag}
    student_text: str = ""  # setter == "student" 时交给学生的话，可带 {mag}
    magnitudes: tuple[Magnitude, ...] = ()
    visible: bool = False  # 成年人设置但学生能看出来（记为未盲）
    safety: str = ""
    note: str = ""  # 给开发者/分析看的物理说明（不上卡片）

    @property
    def blind(self) -> bool:
        return self.setter == "adult" and not self.visible


@dataclass(frozen=True)
class FieldExperiment:
    key: str
    name_zh: str
    primary: str  # 主引擎名："twin" | "generic"
    mainline_b: bool  # 是否进入主线 B 的合并配对检验（课内实验）
    config: dict  # 仪器配置默认值
    base_howto: str
    recipes: tuple[Recipe, ...]  # 第一条是"不设故障"
    howto: dict  # 菜单项 id → 怎么做
    unsafe: dict = field(default_factory=dict)  # 菜单项 id → 真实器材上不做的理由（配置可解锁的见 unlock）
    unlock: dict = field(default_factory=dict)  # 菜单项 id → (配置键, 最小值)：配置达到才允许
    setting_note: str = ""  # 基础方案的控制量怎么填


def _deg_height(length_m: float, deg: float) -> float:
    """板长 length_m 时倾角 deg 对应的垫高（cm）。"""
    return 100.0 * length_m * math.sin(math.radians(abs(deg)))


# ── 胡克定律 ─────────────────────────────────────────────────────────────

HOOKE = FieldExperiment(
    key="hooke",
    name_zh="探究弹簧弹力与形变量的关系（胡克定律）",
    primary="twin",
    mainline_b=True,
    config={"max_load_g": 250.0},
    base_howto=(
        "弹簧竖直挂在铁架台上，刻度尺竖直固定在旁边。先读原长 L0：装置保持你到场时的样子，不增减任何东西，"
        "等弹簧静止，视线与指针平齐读数。读完原长后，如果挂钩不在弹簧上就把它挂上。然后依次在挂钩上挂 50、100、150、"
        "200、250 g 钩码，每次等弹簧静止后读指针位置 L，记录伸长量 x = L − L0（cm，读到 0.1 cm）。拉力只按钩码质量算。"
    ),
    setting_note="setting 列 = 钩码质量（g）",
    recipes=(
        Recipe(
            "clean", {"twin": ("random_only",)}, "adult",
            ("学生到场前：把挂钩挂在弹簧下端。",
             "学生读完原长后：把刻度尺的夹子拧松一下，再在原来的位置拧紧（刻度尺不要移动）。"),
        ),
        Recipe(
            "ruler_shift", {"twin": ("zero_offset",)}, "adult",
            ("学生到场前：把挂钩挂在弹簧下端。",
             "学生读完原长后：把刻度尺的夹子拧松，把刻度尺{mag}，再拧紧。"),
            magnitudes=(Magnitude(-0.6, "竖直向上平移 0.6 cm"), Magnitude(0.8, "竖直向下平移 0.8 cm")),
            note="原长读完后刻度尺平移 s：之后每个读数整体差 s，等价于原长记录偏了 s（先验 N(0, 0.4²) cm 内）。"
                 "方向的正负取决于刻度尺数字的朝向，分析只用原因，不用符号。",
        ),
        Recipe(
            "hanger_late", {"twin": ("hanger_omitted",)}, "adult",
            ("学生到场前：把挂钩取下，放在铁架台底座旁边（用质量约 10~40 g 的挂钩）。",
             "学生读完原长后：把刻度尺的夹子拧松一下，再在原来的位置拧紧（刻度尺不要移动）。"),
            visible=True,
            note="读原长时挂钩不在弹簧上，之后挂上却不计入拉力，即挂钩质量漏算（先验 5~50 g）。学生能看到挂钩在不在，记为未盲。",
        ),
        Recipe(
            "unsteady_read", {"twin": ("reading_scatter",)}, "student",
            ("学生到场前：把挂钩挂在弹簧下端。",
             "学生读完原长后：把刻度尺的夹子拧松一下，再在原来的位置拧紧（刻度尺不要移动）。"),
            student_text="本次每挂上一组钩码后，不要等弹簧静止，心里数到 1 就立刻读数。",
            note="读数时弹簧仍在振动 → 读数额外散布（先验 0.15~0.6 cm）。幅度不受控，学生自己做，记为未盲。",
        ),
    ),
    howto={
        "weigh_hanger": "把挂钩取下，用天平称它的质量（g，读到 0.1 g），再挂回去。",
        "zero_check": "取下全部钩码，把装置恢复到读原长时的样子（读原长时挂钩在不在弹簧上，就保持一样），等弹簧静止后重读指针，"
                      "记录：原来记下的原长 − 现在读到的值（cm）。",
        "zero_load": "只挂挂钩、不挂钩码，等弹簧静止后读指针，记录伸长量 x = L − L0（cm）。",
        "heavy_load": "挂 350 g 钩码，等弹簧静止后读伸长量（只有确认这根弹簧的最大允许负载 ≥ 350 g 才做）。",
        "repeat_150": "取下钩码，再重新挂上 150 g，等弹簧静止后读伸长量。",
    },
    unsafe={"heavy_load": "350 g 可能超过学校弹簧的弹性限度，会永久拉坏弹簧。"},
    unlock={"heavy_load": ("max_load_g", 350.0)},
)

# ── 滑动摩擦力 ───────────────────────────────────────────────────────────

_FRICTION_ZERO_H = "学生到场前：把测力计水平放在桌上（挂钩不挂物体），用调零装置把指针调到零刻度线。"
_FRICTION_FLAT = "学生到场前：确认木板平放在桌面上，下面不垫任何东西。"

FRICTION = FieldExperiment(
    key="friction",
    name_zh="探究影响滑动摩擦力大小的因素",
    primary="twin",
    mainline_b=True,
    config={"scale_division": 0.1, "board_length_m": 1.0},
    base_howto=(
        "木板放在桌上，木块放在木板上。测力计已由成年人调好零点，你不要再调。用测力计沿木板拉木块，使它做匀速直线运动，"
        "读测力计示数 F（N，读到分度值）。在木块上依次加 0~4 个钩码，使压力 N 约为 2、3、4、5、6 N（压力 = 木块和钩码的总重，"
        "按实际值填），各读一次。每次都从木板的 A 端拉向 B 端（A、B 两端在木板上用铅笔标好）。"
    ),
    setting_note="setting 列 = 压力 N（N），按木块和钩码的实际总重填写",
    recipes=(
        Recipe("clean", {"twin": ("random_only",)}, "adult", (_FRICTION_ZERO_H, _FRICTION_FLAT)),
        Recipe(
            "vertical_zero", {"twin": ("zero_vertical",)}, "adult",
            ("学生到场前：把测力计竖直挂起来（挂钩朝下，不挂物体），在竖直状态下用调零装置把指针调到零刻度线，然后水平放回桌上。",
             _FRICTION_FLAT),
            note="竖直调零后水平使用：挂钩、拉杆和弹簧自重造成的伸长被当成零点，水平读数整体偏小。",
        ),
        Recipe(
            "board_shim", {"twin": ("board_tilt",)}, "adult",
            (_FRICTION_ZERO_H, "学生到场前：{mag}；垫块尽量放在不显眼的位置。"),
            magnitudes=(Magnitude(2.5, "在木板 B 端（拉向的那一端）下面垫高 {h25} cm（约 2.5°）"),
                        Magnitude(-3.5, "在木板 A 端（开始拉的那一端）下面垫高 {h35} cm（约 3.5°）")),
            visible=True,
            note="木板倾斜 β：向高处拉多用 W sinβ（正号），向低处拉少用 W sinβ（负号）；先验 ±1°~6°。"
                 "学生可能注意到木板不平，记为未盲。",
        ),
        Recipe(
            "upward_pull", {"twin": ("pull_angle",)}, "student",
            (_FRICTION_ZERO_H, _FRICTION_FLAT),
            student_text="本次拉动时，让测力计和细线斜向上，与木板成{mag}（拉的手比木块高；可以用量角器先比一下）。",
            magnitudes=(Magnitude(40.0, "约 40°"), Magnitude(45.0, "约 45°")),
            note="斜向上拉：F = μN/(cosα + μ sinα)；40°~45° 时示数比 μN 大几个百分点（先验 5°~45°）。学生自己做，未盲。",
        ),
    ),
    howto={
        "fixed_scale": "把测力计水平固定（拉环挂在钉子或铁架台上），细线连住木块；用手把木板向外抽动，木块相对桌面静止，"
                       "读示数。压力 2、4、6 N 各读一次。",
        "reverse_pull": "木块掉头，从 B 端拉向 A 端，沿木板匀速拉动，压力 2、4、6 N 各读一次。",
        "zero_h": "把测力计水平放在桌上、不挂任何物体，读示数（指针在零刻度线另一侧时记负数）。",
        "repeat_4": "压力约 4 N 时再拉两次，各读一次。",
    },
)

# ── 牛顿第二定律（没有摩擦力实验器材时替换摩擦力） ─────────────────────────────

_N2_SHAM = "学生平衡好摩擦力之后：用手摸一下垫块、把它往里推一推再放回原位（高度不变）。"

NEWTON2 = FieldExperiment(
    key="newton2",
    name_zh="探究加速度与力的关系（牛顿第二定律）",
    primary="twin",
    mainline_b=True,
    config={"cart_mass_g": 500.0, "timer": "ticker", "board_length_m": 1.0},
    base_howto=(
        "长木板一端装定滑轮，另一端垫高。先由你平衡摩擦力：不挂槽码，轻推拖着纸带的小车，调垫块直到纸带点迹间隔均匀；"
        "然后请成年人看一眼装置（他会按卡片做一个动作），之后你不要再动垫块和滑轮。用天平称小车（含车上砝码）质量 M，填在表头。"
        "细线跨过定滑轮挂小盘，小盘和槽码总质量 m 依次为 10、20、30、40、50 g；每次先接通打点计时器再释放小车，"
        "用逐差法求加速度 a（m/s²，记到 0.01）。"
    ),
    setting_note="setting 列 = 小盘和槽码总质量 m（g）",
    recipes=(
        Recipe(
            "clean", {"twin": ("heavy_hanging",)}, "adult", (_N2_SHAM,),
            note="孪生里的'只有随机误差'是课本理想模型 a = F/M；真实装置总有连接体效应 a = mg/(M + m)，"
                 "所以不设故障时的真值是'槽码质量没有远小于小车质量'。",
        ),
        Recipe(
            "shim_low", {"twin": ("unbalanced",)}, "adult",
            ("学生平衡好摩擦力之后：把垫块换薄（或往外抽），使木板高的一端降低约 {mag}；动作尽量和平时摸垫块一样。",),
            magnitudes=(Magnitude(-0.05, "{h05} cm"),),
            note="平衡不足：未被抵消的阻力约 0.05 N（先验 N(−0.05, 0.015²) N）；M = 500 g、板长 1 m 时约降低 1.0 cm。",
        ),
        Recipe(
            "shim_high", {"twin": ("overbalanced",)}, "adult",
            ("学生平衡好摩擦力之后：在垫块下再加垫，使木板高的一端升高约 {mag}；动作尽量和平时摸垫块一样。",),
            magnitudes=(Magnitude(0.05, "{h05} cm"),),
            note="平衡过度：多出的下滑力约 0.05 N（先验 N(0.05, 0.015²) N）。",
        ),
        Recipe(
            "pulley_height", {"twin": ("string_angle",)}, "adult",
            ("学生平衡好摩擦力之后：把定滑轮调高，使细线与木板成{mag}，再把垫块摸一下放回原位。",),
            magnitudes=(Magnitude(12.0, "约 12°（小车离滑轮 40 cm 时，细线两端离木板的高度差约 8.5 cm）"),
                        Magnitude(18.0, "约 18°（小车离滑轮 40 cm 时，细线两端离木板的高度差约 13 cm）")),
            visible=True,
            note="细线与木板成 φ：a = mg·cosφ/(M + m cos²φ)（先验 8°~25°）。滑轮高度看得出来，记为未盲。",
        ),
    ),
    howto={
        "balance_check": "取下细线，轻推小车，用纸带（逐差法）测小车自己的加速度 a0（m/s²），记录。",
        "string_height": "把小车放在离滑轮 40 cm 处，量细线在小车挂钩处和滑轮顶端处离木板的高度，记录两者之差（cm）。",
        "repeat_10": "m = 10 g 时重测一次加速度。",
        "repeat_50": "m = 50 g 时重测一次加速度。",
    },
)

# ── 单摆（走"其他实验"通道；留出孪生作对照引擎） ──────────────────────────────

PENDULUM = FieldExperiment(
    key="pendulum",
    name_zh="用单摆测量重力加速度",
    primary="generic",
    mainline_b=False,
    config={"timer": "phone", "n_osc": 30, "lengths_cm": [40.0, 55.0, 70.0, 85.0, 100.0], "slope_known": False},
    base_howto=(
        "细线一端夹在铁架台的铁夹上，另一端系小钢球。量摆长 L：从悬点量到摆球球心（线长 + 摆球半径，读到 0.1 cm），"
        "按实际值填写。把摆球拉开一个小角度（不超过 5°），由静止释放，确认在竖直平面内摆动；用手机传感器计时（如 phyphox），"
        "测 30 次全振动的总时间 t（s，记到 0.01 s）。摆长按表里的顺序（往返顺序）各测一次。"
    ),
    setting_note="setting 列 = 摆长 L（cm），按实际量得的值填写",
    recipes=(
        Recipe(
            "clean", {"generic": ("random_only",), "twin": ("random_only",)}, "adult",
            ("学生开始前：检查铁夹是否夹紧、铁架台是否放稳（不改变摆长和其他任何东西）。",),
        ),
        Recipe(
            "top_length", {"generic": ("x_offset",), "twin": ("top_of_bob",)}, "student",
            ("学生开始前：检查铁夹是否夹紧、铁架台是否放稳（不改变摆长和其他任何东西）。",),
            student_text="本次量摆长时只量到摆球顶端（摆线下端），不要加摆球半径，按这个值填 setting。",
            note="记录摆长 = 真实摆长 − 摆球半径 → 通道里的'控制量起点有偏差'（x_offset，先验约 0.3~1.8 cm）。",
        ),
        Recipe(
            "wide_swing", {"generic": ("scale", "random_only"), "twin": ("large_amplitude",)}, "student",
            ("学生开始前：检查铁夹是否夹紧、铁架台是否放稳（不改变摆长和其他任何东西）。",),
            student_text="本次每次都把摆球拉开{mag}再释放（用量角器比一下，每次角度尽量一样）。",
            magnitudes=(Magnitude(20.0, "约 20°"), Magnitude(30.0, "约 30°")),
            safety="摆角大时摆球速度大：人站在摆动平面之外，铁架台底座压稳。",
            note="固定角度释放：周期整体乘 1/AGM(1, cos(θ₀/2))，与摆长无关，是纯比例；斜率未知时通用通道看不出来（等价于只有随机误差），"
                 "留出孪生（斜率已知）能看出来。",
        ),
        Recipe(
            "count_slip", {"generic": ("scale", "random_only"), "twin": ("miscount",)}, "student",
            ("学生开始前：检查铁夹是否夹紧、铁架台是否放稳（不改变摆长和其他任何东西）。",),
            student_text="本次计时开始的那一下就数\"1\"，数到\"30\"时停止计时，仍按 30 次全振动记录总时间。",
            magnitudes=(Magnitude(-1.0, "少一次"),),
            note="实际只摆了 29 次却按 30 次记：周期整体乘 29/30，纯比例；通用通道斜率未知时等价于只有随机误差。",
        ),
    ),
    howto={
        # 通用通道的菜单
        "x_zero_check": "在 40 cm 那一组上：用游标卡尺测摆球直径，用米尺重量悬点到摆球顶端的线长，算出悬点到球心的实际摆长，"
                        "记录：实际摆长 − 原来记录的摆长（cm）。",
        "y_zero_check": "不做任何计时，直接读手机计时器的初始示数（s）。",
        "y_calibrate": "用另一个可信的计时器（例如电脑上的秒表）标出一段已知时长（约 50 s），用同一部手机、同样的传感器计时方法测它；"
                       "已知时长填 setting，手机测得的值填 value。",
        "repeat_mid": "摆长 70 cm 处重复测 3 次（每次都重新拉开、释放）。",
        "repeat_first": "最后再在 40 cm 处重测一次。",
        "extend_range": "把摆长加到约 130 cm 再测一次（铁架台不够高就在 value 填 skip）。",
        # 留出孪生的菜单（只在加做"孪生 EIG"时用）
        "caliper_bob": "用游标卡尺测摆球直径（cm，读到 0.002 cm）。",
        "length_check": "把摆长调回 85 cm 那一组，从悬点量到摆球球心重测摆长（cm）。",
        "count10": "摆长 100 cm 处只数 10 次全振动再计时一次。",
        "small_angle": "摆长 100 cm 处摆角小于 5°、在竖直平面内轻轻释放后重测 30 次全振动。",
        "repeat_40": "摆长 40 cm 处重复测一次。",
    },
)

FIELD_EXPERIMENTS: dict[str, FieldExperiment] = {e.key: e for e in (HOOKE, FRICTION, NEWTON2, PENDULUM)}
DEFAULT_EXPERIMENTS = ("hooke", "friction", "pendulum")
DEFAULT_SLOTS = ("eig", "best_fixed", "repeat")
OPTIONAL_SLOTS = {"pendulum": ("twin_eig",)}  # 计划 §1 第 3 条：单摆加做留出孪生的 EIG 选择（默认不做）
SLOT_LABEL = {"eig": "信息增益推荐", "best_fixed": "最佳固定测量", "repeat": "简单重复", "twin_eig": "留出孪生推荐"}
TWIN_PREFIX = "twin."  # 只有留出孪生认识的观测种类在记录表里加这个前缀


# ── 引擎 ─────────────────────────────────────────────────────────────────


def pendulum_spec(config: dict) -> dict:
    """单摆走"其他实验"通道时的说明：实验卡 → spec_from_card，补上这台装置的单位、摆长和读数噪声。"""
    from ..knowledge import load_card
    from ..labsim.generic import spec_from_card

    n_osc = int(config.get("n_osc", 30))
    slope = n_osc * 2.0 * math.pi / math.sqrt(980.0)  # t = N·2π√(L/g)，L 用 cm
    over = dict(
        x_var={"name": "摆长（悬点到摆球球心）", "symbol": "L", "unit": "cm", "tool": "米尺", "resolution": 0.1},
        y_var={"name": f"{n_osc} 次全振动的总时间", "symbol": "t", "unit": "s", "tool": "手机传感器计时",
               "resolution": 0.01},
        settings=[float(v) for v in config.get("lengths_cm", [40.0, 55.0, 70.0, 85.0, 100.0])],
        sigma={"x": 0.1, "y": 0.04},  # 与留出孪生同一噪声：手机触发约 0.01 s，量摆长 1 mm 折算到 t 约 0.039 s
        slope_guess=slope,
    )
    if config.get("slope_known"):
        over["ideal_slope"] = slope
    return spec_from_card(load_card("pendulum"), **over)


def build_engines(key: str, config: dict, spec: dict | None = None) -> dict[str, Experiment]:
    """按仪器配置实例化引擎。单摆给出 spec 时直接用（spec 冻结在随机表里）。"""
    if key == "hooke":
        from ..labsim.experiments.hooke import Hooke

        return {"twin": Hooke()}
    if key == "friction":
        from ..labsim.experiments.friction import Friction

        return {"twin": Friction(float(config.get("scale_division", 0.1)))}
    if key == "newton2":
        from ..labsim.experiments.newton2 import Newton2

        return {"twin": Newton2(float(config.get("cart_mass_g", 500.0)), str(config.get("timer", "ticker")))}
    if key == "pendulum":
        from ..labsim.experiments.pendulum import Pendulum
        from ..labsim.generic import GenericExperiment

        return {"generic": GenericExperiment(spec if spec is not None else pendulum_spec(config)),
                "twin": Pendulum(str(config.get("timer", "phone")), int(config.get("n_osc", 30)))}
    raise KeyError(key)


def resolve_truth(engine: Experiment, suffixes: tuple[str, ...]) -> tuple[str, bool]:
    """→ (真值原因 id, 是否用了等价后缀)。"""
    ids = set(engine.cause_ids())
    for i, s in enumerate(suffixes):
        cid = f"{engine.id}.{s}"
        if cid in ids:
            return cid, i > 0
    raise KeyError(f"{engine.id}: 没有 {suffixes} 中的任何原因")


def allowed_items(fexp: FieldExperiment, engine: Experiment, config: dict) -> tuple[str, ...]:
    """真实器材上允许做的菜单项（危险项除非配置解锁）。"""
    out = []
    for m in engine.menu:
        if m.id in fexp.unsafe:
            k, need = fexp.unlock.get(m.id, (None, None))
            if k is None or float(config.get(k, 0.0)) < need:
                continue
        out.append(m.id)
    return tuple(out)


def best_fixed(engine: Experiment, allowed: tuple[str, ...], protocol: str = "standard") -> str:
    """最佳固定追加测量：与 eig._best_fixed_cached 同一个校准集（同一种子），只在允许的菜单项里挑。"""
    from ..eig import best_fixed_item, protocol_key

    e = copy.copy(engine)
    e.menu = tuple(m for m in engine.menu if m.id in allowed)
    seed = zlib.crc32(f"{engine.id}|{protocol_key(protocol)}|best_fixed".encode())
    return best_fixed_item(e, protocol, np.random.default_rng(seed))


def howto(fexp: FieldExperiment, engine: Experiment, item_id: str) -> str:
    text = fexp.howto.get(item_id)
    return text if text else engine.menu_item(item_id).label_zh


def fill_magnitude(text: str, config: dict) -> str:
    """卡片文字里的垫高占位符按板长和小车质量换算。"""
    L = float(config.get("board_length_m", 1.0))
    M = float(config.get("cart_mass_g", 500.0)) / 1000.0
    rep = {
        "{h25}": f"{_deg_height(L, 2.5):.1f}",
        "{h35}": f"{_deg_height(L, 3.5):.1f}",
        "{h05}": f"{100.0 * 0.05 * L / (M * 9.8):.1f}",  # Δh = f·L/(M g)
    }
    for k, v in rep.items():
        text = text.replace(k, v)
    return text
