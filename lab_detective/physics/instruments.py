"""测量仪器：把真实物理量映射成学生记下的读数（尽量仿射，线性参数才能保留），带分辨率、随机误差和仪器误差。

每个仪器：
- reading(setup, flags, n_count)：主读数的 Reading 规格（分辨率、随机误差标准差）；
- read(ctx, template)：向模板要它需要的物理量（wants），返回读数均值（数组或 Affine）；
- student(raw)：学生由读数算出的物理量（SI）及其对读数的导数，给模板的线性化做误差传递。

刻度尺 Ruler、弹簧测力计 SpringScale（竖直/水平调零偏移）、秒表 Stopwatch（反应时间）、
手机计时 PhoneTimer（数错次数）、光电门 Photogate（测速度/加速度/计时，含遮光片宽度误差）、
打点计时器 TickerTimer、天平 Balance、量角器 Protractor、游标卡尺 Caliper。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ..labsim.core import LinearParam, Prior, Reading
from .base import Affine, AuxMeasurement, Component, Ctx, Effect, OpenQuestion, Variant


@dataclass
class Instrument(Component):
    """仪器基类。wants：向模板要的物理量名（空 = 模板的主量）。"""

    wants: str = ""

    def reading(self, setup, flags: frozenset = frozenset(), n_count: int | None = None) -> Reading:
        raise NotImplementedError

    def read(self, ctx: Ctx, tmpl):
        raise NotImplementedError

    def quantity(self, ctx: Ctx, tmpl):
        return tmpl.quantity(ctx, self.wants or tmpl.primary)

    def student(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """读数 → 学生算出的物理量（SI）和 d(物理量)/d(读数)。默认：读数本身就是 SI 量。"""
        raw = np.asarray(raw, dtype=float)
        return raw, np.ones_like(raw)


# ── 刻度尺 ────────────────────────────────────────────────────────────


@dataclass
class Ruler(Instrument):
    """刻度尺（cm）。

    role="reading"：读主测量（如弹簧伸长量）。误差：
      zero_offset——原长读数偏差 δ（线性参数），伸长量 = 读数 − 原长，原长记大 δ → 每个伸长量都小 δ；
      total_length——记了总长没减原长：读数多出原长 L₀（线性参数）。
    role="control"：量控制量（高度、距离、摆长）。误差 wrong_point——两端用的不是同一个参考点：
      记录值 = 真实值 + δ（网格，可正可负）→ 真实值 = 记录值 − δ。
      附加测量 recheck：用正确方法重量一次，读数 = 真实控制量（记录单位），能直接暴露各种"量错"原因。
    """

    label_zh: str = "刻度尺"
    role: str = "reading"
    resolution: float = 0.1
    sigma: float = 0.05
    key_name: str = "extension"
    label: str = "伸长量"
    zero_sd: float = 0.4
    total_mean: float = 12.0
    total_sd: float = 2.0
    wrong_prior: Prior = Prior.sym_uniform(0.5, 4.0, 36)
    recheck_kind: str = "height_check"
    recheck_label_zh: str = "用铅垂线和刻度尺重新竖直测量这一组的高度"
    recheck_reading_zh: str = "竖直高度差（重测）"
    recheck_sigma: float = 0.05
    recheck_at: float | None = None

    def all_effects(self):
        if self.role == "reading":
            return [
                Effect(
                    "zero_offset", "原长读数有偏差", "unloaded length misread",
                    "读起始（原长）读数时视线没有与刻度平齐，或看错了刻度线，之后每个读数都用这个有偏差的起点去减。",
                    "The starting (unloaded) reading was misread (parallax or wrong mark), and every later value was "
                    "computed from it.",
                    "y_offset", ("intercept_pos", "intercept_neg"),
                    fix_zh="重读起始读数，眼睛与指针（或刻度）平齐。",
                    fix_en="Re-read the starting value with your eye level with the pointer.",
                    linear=(LinearParam(self.pname("zero_offset"), 0.0, self.zero_sd, "原长读数偏差", "cm"),),
                ),
                Effect(
                    "total_length", "记录了总长而不是伸长量", "total length recorded",
                    "表格里记的是挂重物后的总长度，没有减去原长。",
                    "The table records the total stretched length instead of the extension.",
                    "y_offset", ("intercept_pos",),
                    fix_zh="伸长量 = 挂钩码后的长度 − 原长。", fix_en="Extension = stretched length − unloaded length.",
                    linear=(LinearParam(self.pname("unloaded_length"), self.total_mean, self.total_sd, "弹簧原长", "cm"),),
                ),
            ]
        return [Effect(
            "wrong_point", "长度量到了错误的位置", "measured to the wrong point",
            "量这个长度时两端用的不是同一个参考点（例如一端量到物体顶部、另一端量到底部或轨道面），"
            "每次记录的值都差一个固定的量。",
            "The two ends of this length were not measured to the same reference point (e.g. top of the object at "
            "one end, bottom or track surface at the other), so every recorded value is off by the same amount.",
            "x_offset", ("intercept_pos", "intercept_neg"), self.wrong_prior, "记录值 − 真实值", "cm",
            "两端都量到物体上同一个点（例如遮光片中心或球心），再相减。",
            "Measure both ends to the same point of the object and subtract.",
        )]

    def reading(self, setup, flags=frozenset(), n_count=None):
        return Reading(self.key_name, self.label, self.key_name, "cm", resolution=self.resolution, sigma=self.sigma)

    def read(self, ctx, tmpl):
        q = 100.0 * self.quantity(ctx, tmpl)  # m → cm
        if self.role == "reading":
            if self.has("zero_offset") and ctx.on(self.key("zero_offset")):
                q = q - ctx.lin(self.pname("zero_offset"))
            if self.has("total_length") and ctx.on(self.key("total_length")):
                q = q + ctx.lin(self.pname("unloaded_length"))
        return q

    def student(self, raw):
        raw = np.asarray(raw, dtype=float)
        return raw / 100.0, np.full_like(raw, 0.01)

    def control(self, ctx, x, quantity):
        if (self.role == "control" and quantity in ("height", "distance", "length")
                and self.has("wrong_point") and ctx.on(self.key("wrong_point"))):
            return x - ctx.mag / 100.0
        return x

    def aux(self, setup):
        if self.role == "reading":
            if not self.has("zero_offset"):
                return []
            key, p = self.key("zero_offset"), self.pname("zero_offset")

            def diff(ctx):
                return ctx.lin(p) if ctx.on(key) else np.zeros((ctx.P, ctx.n))

            return [AuxMeasurement(
                "zero_check", "取下重物后重读起始读数，与原记录比较", "re-read the unloaded length",
                Reading("zero_diff", "原长原记录 − 重读值", "recorded L0 − re-read L0", "cm",
                        resolution=self.resolution, sigma=self.sigma),
                diff, menu_label_zh="取下重物，重读原长",
            )]
        at = self.recheck_at if self.recheck_at is not None else setup.standard[-1]
        to_si = setup.control.to_si

        def recheck(ctx):
            return np.broadcast_to(ctx.x / to_si, (ctx.P, ctx.n))

        return [AuxMeasurement(
            self.recheck_kind, self.recheck_label_zh, "re-measure the control quantity properly",
            Reading(self.recheck_kind.replace("_check", "") + "_re", self.recheck_reading_zh, "re-measured value",
                    "cm", resolution=self.resolution, sigma=self.recheck_sigma),
            recheck, setting=at, setting_label_zh="原记录值", setting_unit=setup.control.unit,
            menu_label_zh=f"{self.recheck_label_zh}（原记录 {at:g} {setup.control.unit}）",
        )]


# ── 弹簧测力计 ────────────────────────────────────────────────────────


@dataclass
class SpringScale(Instrument):
    """弹簧测力计（N）。

    调零方向与使用方向不同：竖直挂着时挂钩、拉杆和弹簧下半段的重力 w 已把弹簧拉长 w/k，指针此时被调到 0；
    横放后这部分重力不再沿弹簧轴向，弹簧缩回 → 水平使用时示数 = F − w（水平空载读 −w）。
    反过来水平调零、竖直使用：示数 = F + w。w 是读数里的常数项，作线性参数。
    指针晃动时读数（手拉、木块走走停停）：只加大散布；"fixed" 变体（测力计固定）指针稳定。
    """

    label_zh: str = "弹簧测力计"
    orientation: str = "horizontal"
    division: float = 0.1
    sigma: float = 0.05
    sigma_fixed: float = 0.02
    sigma_zero: float = 0.02
    hook_mean: float = 0.15
    hook_sd: float = 0.05
    scatter_prior: Prior = Prior.uniform(0.10, 0.35, 10)

    @property
    def _zero_sign(self) -> float:
        return -1.0 if self.orientation == "horizontal" else 1.0

    def all_effects(self):
        if self.orientation == "horizontal":
            zero = Effect(
                "zero_shift", "竖直调零后水平使用", "zeroed vertically, used horizontally",
                "测力计是竖直挂着调的零，实验时却水平拉；竖直时挂钩、拉杆和弹簧自身的重力把弹簧拉长了一点，"
                "这一点在调零时被当成了零，横过来以后这部分重力不再拉弹簧。",
                "The scale was zeroed hanging vertically but used horizontally; when vertical, the weight of the hook, "
                "pull rod and spring itself stretches it a little, and that stretch was taken as zero.",
                "y_offset", ("intercept_neg",),
                fix_zh="在水平方向上给测力计调零（或先水平放置检查指针是否指零）再拉。",
                fix_en="Zero the scale in the horizontal position before pulling.",
                linear=(LinearParam(self.pname("hook_weight"), self.hook_mean, self.hook_sd,
                                    "挂钩、拉杆等的有效重力", "N"),),
            )
        else:
            zero = Effect(
                "zero_shift", "水平调零后竖直使用", "zeroed horizontally, used vertically",
                "测力计是平放着调的零，实验时却竖直挂着用；挂钩、拉杆和弹簧自身的重力会把弹簧再拉长一点。",
                "The scale was zeroed lying flat but used hanging vertically; the weight of the hook, rod and spring "
                "stretches it a little more.",
                "y_offset", ("intercept_pos",),
                fix_zh="按使用方向（竖直）拿着测力计调零。", fix_en="Zero the scale in the vertical position it is used in.",
                linear=(LinearParam(self.pname("hook_weight"), self.hook_mean, self.hook_sd,
                                    "挂钩、拉杆等的有效重力", "N"),),
            )
        return [zero, Effect(
            "pointer_scatter", "指针还在晃动时读数", "read while the pointer was moving",
            "拉动时物体走走停停，测力计指针一直在晃，学生在指针晃动时随手读了一个数。",
            "The object moved in jerks and the pointer kept swinging; the reading was taken at a random moment.",
            "scatter", ("scatter_excess",), self.scatter_prior, "额外读数误差", "N",
            "尽量匀速拉，指针稳定后再读；或改用固定测力计、抽动木板的方法。",
            "Pull steadily and read when the pointer is still, or fix the scale and pull the board.",
            scatter=True,
        )]

    def reading(self, setup, flags=frozenset(), n_count=None):
        if "fixed" in flags:
            return Reading("fixed_force", "测力计示数（物体静止）", "fixed_force", "N",
                           resolution=self.division, sigma=self.sigma_fixed)
        return Reading("pull_force", "拉力（测力计示数）", "pull_force", "N", resolution=self.division, sigma=self.sigma)

    def read(self, ctx, tmpl):
        F = self.quantity(ctx, tmpl)
        if self.has("zero_shift") and ctx.on(self.key("zero_shift")):
            F = F + self._zero_sign * ctx.lin(self.pname("hook_weight"))
        return F

    def extra_sigma(self, ctx):
        if (self.has("pointer_scatter") and ctx.on(self.key("pointer_scatter"))
                and ctx.kind in ("main", "reverse") and not ctx.flag("fixed")):
            return ctx.mag
        return None

    def aux(self, setup):
        key, p, s = self.key("zero_shift"), self.pname("hook_weight"), self._zero_sign
        where = "水平" if self.orientation == "horizontal" else "竖直"

        def zero(ctx):
            return s * ctx.lin(p) if ctx.on(key) else np.zeros((ctx.P, ctx.n))

        return [AuxMeasurement(
            "zero_h" if self.orientation == "horizontal" else "zero_v",
            f"测力计{where}放置、不挂物体，读示数", "hold the scale as used with no load, read it",
            Reading("zero_h" if self.orientation == "horizontal" else "zero_v", f"{where}空载示数", "no-load reading",
                    "N", resolution=self.division, sigma=self.sigma_zero),
            zero, minutes=1.0, menu_label_zh=f"把测力计{where}放置、不挂物体，读示数",
        )]


# ── 计时器 ────────────────────────────────────────────────────────────


@dataclass
class _Timer(Instrument):
    """计时器（s）。count = N：数 N 次全振动的总时间（周期类模板给 T）；None：单次计时（模板给 t）。

    数错次数：计时内实际完成 N + m 次全振动，t = (N + m)T，记录的 T' = t/N = (1 + m/N)T。
    起止时刻没对准：t' = (N + m)T + τ（τ 作线性参数）。
    extra_sigma_ref：其他量（如摆长）的随机误差折算到总时间后的标准差，按 n/extra_ref_count 缩放。
    """

    label_zh: str = "计时器"
    count: int | None = None
    short_count: int | None = None
    sigma: float = 0.01
    resolution: float = 0.01
    extra_sigma_ref: float = 0.0
    extra_ref_count: int = 30
    tau_mean: float = 0.0
    tau_sd: float = 0.3

    def _offset_effect(self) -> Effect:
        raise NotImplementedError

    def all_effects(self):
        effs = []
        if self.count is not None:
            effs.append(Effect(
                "miscount", "全振动次数数错了", "oscillations miscounted",
                "数全振动次数时，开始计时那一下就数了\"1\"（或者结束时多数、少数了一次），计时的这段时间里实际摆动的次数"
                "和记录的次数差一次，算周期时却仍用 T = t/N。",
                "When counting, the student said \"one\" at the moment timing started (or counted one too many/few at "
                "the end), so the timed interval holds one oscillation more or less than recorded.",
                "y_scale", ("slope_low", "slope_high"), Prior.sym_uniform(1.0, 1.0, 2), "实际次数 − 记录次数", "次",
                "第一次经过最低点（平衡位置）时数\"0\"并开始计时，之后每经过一次（同方向）数一次。",
                "Say \"zero\" as it first passes the equilibrium point and start timing then.",
            ))
        effs.append(self._offset_effect())
        return effs

    def n_of(self, n_count: int | None) -> int | None:
        return n_count if n_count is not None else self.count

    def reading(self, setup, flags=frozenset(), n_count=None):
        n = self.n_of(n_count)
        if n is None:
            return Reading("t", "时间", "time", "s", resolution=self.resolution, sigma=self.sigma)
        sig = math.sqrt(self.sigma**2 + (self.extra_sigma_ref * n / self.extra_ref_count) ** 2)
        return Reading(f"t{n}", f"{n} 次全振动的总时间", f"time for {n} oscillations", "s",
                       resolution=self.resolution, sigma=round(sig, 4))

    def read(self, ctx, tmpl):
        n = self.n_of(ctx.n_count)
        if n is None:
            t = self.quantity(ctx, tmpl) if self.wants else tmpl.quantity(ctx, "t")
        else:
            m = ctx.mag if (self.has("miscount") and ctx.on(self.key("miscount"))) else 0.0
            t = (n + m) * tmpl.quantity(ctx, "T")
        if self.has("timing_offset") and ctx.on(self.key("timing_offset")):
            t = t + ctx.lin(self.pname("tau"))
        return t

    def variants(self, setup):
        if self.count is None or self.short_count is None:
            return []
        n, L = self.short_count, setup.standard[-1]
        return [Variant(
            f"main{n}", f"同样释放，只测 {n} 次全振动的总时间", f"time {n} oscillations",
            frozenset(), n_count=n,
            menu=((f"count{n}", f"在 {L:g} {setup.control.unit} 处只数 {n} 次全振动再计时一次",
                   f"time {n} oscillations once", 2.0, (L,)),),
        )]


@dataclass
class Stopwatch(_Timer):
    """手按秒表。每次计时有开始、停止两次人的反应，随机误差 σ ≈ √2 × 0.1 s；
    起止反应不对称（看到运动才开始、听到落地才停止）时每次都偏同一段时间 τ。"""

    label_zh: str = "秒表"
    sigma: float = math.sqrt(2) * 0.1
    tau_mean: float = 0.0
    tau_sd: float = 0.15

    def _offset_effect(self):
        return Effect(
            "timing_offset", "计时与运动不同步（反应时间）", "timing offset (reaction time)",
            "按秒表的时刻与运动开始、结束的时刻不同步：例如看到物体动了才开始计时，或听到落地声才停止，"
            "每次计时都差不多偏了同样一段时间。",
            "The stopwatch was not started/stopped in sync with the motion (started after seeing it move, stopped "
            "after hearing it land), so every time is off by about the same amount.",
            "y_offset", ("intercept_pos", "intercept_neg"),
            fix_zh="由释放物体的同学自己同时开始计时，眼睛盯住终点；或改用光电门计时。",
            fix_en="Let the person releasing also start the watch and watch the finish, or use a photogate.",
            linear=(LinearParam(self.pname("tau"), self.tau_mean, self.tau_sd, "计时偏差", "s"),),
        )


@dataclass
class PhoneTimer(_Timer):
    """手机传感器计时（如 phyphox 光电/磁力计秒表，经过平衡位置自动触发，不含人的反应时间），σ ≈ 0.01 s。"""

    label_zh: str = "手机计时"
    sigma: float = 0.01

    def _offset_effect(self):
        return Effect(
            "timing_offset", "计时的起止时刻没有对准", "timing start/stop misaligned",
            "开始计时和停止计时的时刻与经过平衡位置（最低点）的时刻没有对齐，而且每次都偏同一个方向，"
            "每次记录的总时间都多了（或少了）差不多同样的一段时间。",
            "Timing did not start and stop exactly as it passed the equilibrium point, and the mismatch was the "
            "same each time, so every total time is off by about the same amount.",
            "y_offset", ("intercept_pos", "slope_high", "intercept_neg", "slope_low"),
            fix_zh="经过平衡位置时开始计时（同一位置停止）；让传感器自动触发计时。",
            fix_en="Start and stop as it passes the equilibrium point, ideally with a sensor trigger.",
            linear=(LinearParam(self.pname("tau"), self.tau_mean, self.tau_sd, "总时间的固定偏差", "s"),),
        )


# ── 光电门 ────────────────────────────────────────────────────────────


@dataclass
class Photogate(Instrument):
    """光电门 + 数字计时器。

    mode="speed"：单门测速度，读挡光时间 Δt（ms）= 挡光宽度（mm）/ v（m/s）。学生算 v = w₀/Δt（w₀ 为所用宽度）。
      width_linear=True：真实宽度作线性参数 w_true（速度要开方的模板里只能这样），宽度用错 = 多一个线性参数 w_err；
      width_linear=False：宽度已知，"宽度用错"按比例 ε = w₀/w_真 − 1 放网格（此时模板里的 1/v 才能含线性参数）。
      round_object=True（挡光的是小球）：光束不过球心时挡光长度是一条弦，c = d√(1 − e²)（e = 偏离/半径），
      算出的速度偏大 1/√(1 − e²) 倍。
      忽略：挡光期间的平均速度 = 中间时刻速度，比中心过光束时略小，v_c² − v̄² = (aΔt/2)² ~ 10⁻⁴ 相对差。
    mode="accel"：两门测加速度 a = (v₂² − v₁²)/(2d)（m/s²）。宽度用错 → a 乘 (1 + ε)²；
      间距从前一个门的前侧面量到后一个门的后侧面 → d_记 = d + w，a 乘 (d_记 − w)/d_记。
    mode="timer"：释放装置启动计时、物体到达光束停止（s）。释放装置延迟 τ：t_记 = t + τ（线性参数）。
    """

    label_zh: str = "光电门"
    mode: str = "speed"
    flag_mm: float = 10.0
    width_linear: bool = True
    width_sd: float = 0.02
    err_sd: float = 0.4
    round_object: bool = False
    ratio_prior: Prior = Prior.sym_uniform(0.02, 0.10, 10)
    offcenter_prior: Prior = Prior.uniform(0.2, 0.6, 9)
    spacing_cm: float = 50.0
    thickness_prior: Prior = Prior.uniform(1.0, 4.0, 7)
    sigma: float | None = None
    resolution: float | None = None
    delay_mean: float = 0.02
    delay_sd: float = 0.008

    _DEFAULTS = {"speed": (0.025, 0.1), "accel": (0.006, 0.001), "timer": (0.0006, 0.0001)}

    def __post_init__(self):
        if self.mode not in self._DEFAULTS:
            raise ValueError(f"光电门 mode 只能是 {tuple(self._DEFAULTS)}")
        s, r = self._DEFAULTS[self.mode]
        self.sigma = s if self.sigma is None else self.sigma
        self.resolution = r if self.resolution is None else self.resolution

    def linear(self):
        if self.mode == "speed" and self.width_linear:
            return (LinearParam(self.pname("w_true"), self.flag_mm, self.width_sd, "遮光片真实宽度", "mm"),)
        return ()

    def all_effects(self):
        effs = []
        if self.mode in ("speed", "accel"):
            what = "小球直径" if self.round_object else "遮光片宽度"
            flag = Effect(
                "flag_width", f"{what}用错了", f"wrong {'ball diameter' if self.round_object else 'flag width'}",
                f"用 v = d/Δt 算速度时，{what} d 用的是标称值或用刻度尺粗量的值，和真实值不一样。",
                "The width used in v = d/Δt is a nominal value or a rough ruler reading, not the true width.",
                "y_scale", ("slope_low", "slope_high"),
                fix_zh=f"用游标卡尺测出{what}的真实值，用实测值重新计算速度。",
                fix_en="Measure the width with a vernier caliper and recompute the speeds.",
            )
            if self.mode == "speed" and self.width_linear:
                flag = replace(flag, linear=(LinearParam(self.pname("w_err"), 0.0, self.err_sd,
                                                         "真实宽度 − 所用宽度", "mm"),))
            else:
                flag = replace(flag, magnitude=self.ratio_prior, magnitude_label_zh="所用宽度/真实宽度 − 1")
            effs.append(flag)
        if self.mode == "speed" and self.round_object:
            effs.append(Effect(
                "off_center", "光束没有通过球心", "beam misses the ball centre",
                "光电门的光束没有对准小球的球心，小球挡光的长度是一条比直径短的弦，算速度时却按直径计算。",
                "The beam does not pass through the centre of the ball, so the ball blocks it along a chord shorter "
                "than the diameter, yet the speed is computed with the diameter.",
                "y_scale", ("slope_high",), self.offcenter_prior, "光束离球心的距离 / 半径", "",
                "调节光电门高度和位置，使光束正对球心（小球静止挂在光束处检查）。",
                "Adjust the gate so the beam passes through the ball's centre.",
            ))
        if self.mode == "accel":
            effs.append(Effect(
                "gate_spacing", "光电门间距量多了", "gate spacing measured front-to-rear",
                "量两个光电门的距离时，从第一个光电门的前侧面量到第二个光电门的后侧面，多量了一个光电门的厚度；"
                "真正起作用的是两条光线之间的距离。",
                "The spacing was measured from the front face of the first gate to the back face of the second, "
                "adding one gate thickness; what matters is the distance between the two beams.",
                "y_scale", ("slope_low",), self.thickness_prior, "多量的长度", "cm",
                "对准两个光电门的光线（发射孔中心）量间距。", "Measure between the two light beams.",
            ))
        if self.mode == "timer":
            effs.append(Effect(
                "start_delay", "计时开始后物体才开始运动", "motion starts after the timer",
                "释放装置一动作计时器就开始计时，但物体要过一小会儿才真正开始运动（挡板抬起、剩磁、手指松开都需要时间）。",
                "The timer starts the instant the release is triggered, but the object only starts moving a moment "
                "later (the stop lifting, residual magnetism, fingers opening).",
                "y_offset", ("intercept_pos",),
                fix_zh="改用两个光电门计时（物体经过第一个门开始计时），或减小释放装置的延迟。",
                fix_en="Time between two gates, or reduce the release delay.",
                linear=(LinearParam(self.pname("delay"), self.delay_mean, self.delay_sd, "释放延迟", "s"),),
            ))
        return effs

    # 物理

    @property
    def _lin_width(self) -> bool:
        return self.mode == "speed" and self.width_linear

    def _width(self, ctx: Ctx):
        """挡光宽度的真实值（mm）。"""
        if self._lin_width:
            w = ctx.lin(self.pname("w_true"))
            if self.has("flag_width") and ctx.on(self.key("flag_width")):
                w = w + ctx.lin(self.pname("w_err"))
            return w
        w = self.flag_mm
        if self.has("flag_width") and ctx.on(self.key("flag_width")):
            w = w / (1.0 + ctx.mag)
        return w

    def _chord(self, ctx: Ctx):
        if self.has("off_center") and ctx.on(self.key("off_center")):
            return np.sqrt(1.0 - ctx.mag**2)
        return 1.0

    def reading(self, setup, flags=frozenset(), n_count=None):
        if self.mode == "speed":
            return Reading("dt", "挡光时间", "gate blocking time", "ms", resolution=self.resolution, sigma=self.sigma)
        if self.mode == "accel":
            return Reading("acceleration", "加速度", "acceleration", "m/s²", resolution=self.resolution, sigma=self.sigma)
        return Reading("t", "计时器读数", "timer reading", "s", resolution=self.resolution, sigma=self.sigma)

    def read(self, ctx, tmpl):
        if self.mode == "speed":
            inv_v = tmpl.quantity(ctx, "inv_v")  # s/m
            w = self._width(ctx)
            if not self._lin_width:
                w = w * self._chord(ctx)
            return w * inv_v  # mm·s/m = ms
        if self.mode == "accel":
            a = tmpl.quantity(ctx, "a")
            if self.has("flag_width") and ctx.on(self.key("flag_width")):
                a = a * (1.0 + ctx.mag) ** 2
            if self.has("gate_spacing") and ctx.on(self.key("gate_spacing")):
                a = a * ((self.spacing_cm - ctx.mag) / self.spacing_cm)
            return a
        t = tmpl.quantity(ctx, "t")
        if self.has("start_delay") and ctx.on(self.key("start_delay")):
            t = t + ctx.lin(self.pname("delay"))
        return t

    def questions(self):
        if self.mode == "timer" or self.round_object:  # 小球直径由小球组件来问
            return []
        what = "小球直径" if self.round_object else "遮光片宽度"
        return [OpenQuestion(self.pname("flag_mm"), f"{what}是多少？最好用游标卡尺量。", "mm", self.flag_mm)]

    def student(self, raw):
        raw = np.asarray(raw, dtype=float)
        if self.mode == "speed":
            v = self.flag_mm / raw  # mm/ms = m/s
            return v, -v / raw
        return raw, np.ones_like(raw)

    def aux(self, setup):
        out = []
        if self.mode in ("speed", "accel") and self.has("flag_width"):
            what = "小球直径" if self.round_object else "遮光片宽度"

            def width(ctx, g=self):  # 卡尺量的是真实宽度（直径），不受光束偏离影响
                w = g._width(ctx)
                return w if isinstance(w, Affine) else np.broadcast_to(as_arr(w), (ctx.P, ctx.n))

            out.append(AuxMeasurement(
                "flag_width", f"用游标卡尺测{what}", "measure the width with a vernier caliper",
                Reading("flag_width", what, "flag width", "mm", resolution=0.02, sigma=0.01),
                width, menu_id="caliper_flag", menu_label_zh=f"用游标卡尺测{what}",
            ))
        if self.mode == "accel" and self.has("gate_spacing"):
            key, d = self.key("gate_spacing"), self.spacing_cm

            def spacing(ctx):
                v = d - ctx.mag if ctx.on(key) else np.full((ctx.P, 1), d)
                return np.broadcast_to(v, (ctx.P, ctx.n))

            out.append(AuxMeasurement(
                "spacing", "对准两个光电门的光线位置，重量间距", "re-measure the gate spacing beam to beam",
                Reading("gate_spacing", "两光线间距", "beam-to-beam spacing", "cm", resolution=0.1, sigma=0.05),
                spacing, menu_id="remeasure_spacing", menu_label_zh="对准光线位置，重量两个光电门的间距",
            ))
        return out


def as_arr(x):
    return np.asarray(x, dtype=float)


# ── 打点计时器 ────────────────────────────────────────────────────────


@dataclass
class TickerTimer(Instrument):
    """打点计时器（50 Hz，T = 0.02 s）。

    mode="accel"：每 5 个点取一个计数点（0.1 s），6 段逐差 a = [(x4+x5+x6) − (x1+x2+x3)]/(9T²)；
      点位读数 σ_p ≈ 0.4 mm → σ_a ≈ 0.011，再加运动起伏，取 σ_a = 0.015 m/s²，记到 0.01。
    mode="speed"：量测速点前后相邻两点的间距 = 2T·v（匀加速时中间时刻速度公式精确），记到 0.1 cm。
    纸带阻力：纸带与限位孔、振针之间的摩擦，相当于物体多受一个恒定阻力 f（网格，N）。
    """

    label_zh: str = "打点计时器"
    mode: str = "accel"
    period: float = 0.02
    drag_prior: Prior = Prior.uniform(0.02, 0.12, 11)

    def all_effects(self):
        return [Effect(
            "tape_drag", "纸带受到阻力", "paper-tape drag",
            "纸带穿过打点计时器的限位孔时与孔壁摩擦，振针打点时也拖住纸带，物体运动时多受一个阻力。",
            "The paper tape rubs on the timer's guide slot and is held back by the striker, adding a backward force.",
            "x_offset", ("intercept_neg",), self.drag_prior, "纸带阻力", "N",
            "让纸带与限位孔对正、不扭曲；或改用光电门。", "Align the tape with the guide slot, or use photogates.",
        )]

    def drag(self, ctx: Ctx):
        if self.has("tape_drag") and ctx.on(self.key("tape_drag")):
            return ctx.mag
        return 0.0

    def reading(self, setup, flags=frozenset(), n_count=None):
        if self.mode == "accel":
            return Reading("acceleration", "加速度", "acceleration", "m/s²", resolution=0.01, sigma=0.015)
        return Reading("dot_span", "该点前后相邻两点的间距", "distance between the two neighbouring dots", "cm",
                       resolution=0.1, sigma=0.03)

    def read(self, ctx, tmpl):
        if self.mode == "accel":
            return tmpl.quantity(ctx, "a")
        return 200.0 * self.period * tmpl.quantity(ctx, "v")  # 2T·v，m → cm

    def student(self, raw):
        raw = np.asarray(raw, dtype=float)
        if self.mode == "accel":
            return raw, np.ones_like(raw)
        k = 1.0 / (100.0 * 2.0 * self.period)
        return raw * k, np.full_like(raw, k)


# ── 天平、量角器、游标卡尺 ──────────────────────────────────────────────


@dataclass
class Balance(Instrument):
    """天平（g）：读数 = 质量，分辨率 0.1 g。作主测量仪器时模板给质量（kg）。"""

    label_zh: str = "天平"
    resolution: float = 0.1
    sigma: float = 0.1

    def reading(self, setup, flags=frozenset(), n_count=None):
        return Reading("mass", "质量", "mass", "g", resolution=self.resolution, sigma=self.sigma)

    def read(self, ctx, tmpl):
        return 1000.0 * self.quantity(ctx, tmpl)

    def student(self, raw):
        raw = np.asarray(raw, dtype=float)
        return raw / 1000.0, np.full_like(raw, 1e-3)


@dataclass
class Protractor(Instrument):
    """量角器（°）。零刻度没对准水平：真实倾角 = 读数 + δ（网格，可正可负）。
    斜面组件通过 offset_deg() 取这个偏差；附加测量"量斜面高度"在 Incline 里。"""

    label_zh: str = "量角器"
    resolution: float = 1.0
    sigma: float = 0.5
    zero_prior: Prior = Prior.sym_uniform(1.0, 5.0, 10)

    def all_effects(self):
        return [Effect(
            "zero", "量角器零点不准", "protractor zero offset",
            "量角器（或角度指针）的零刻度没有对准水平，读出的倾角与真实倾角差一个固定角度。",
            "The protractor's zero is not aligned with the horizontal, so every angle reading is off by the same amount.",
            "x_offset", ("slope_high", "slope_low"), self.zero_prior, "真实倾角 − 量角器读数", "°",
            "用刻度尺量斜面高度 h 和板长 L，用 sin θ = h/L 校准角度。",
            "Measure the height and length of the incline and use sin θ = h/L.",
        )]

    def offset_deg(self, ctx: Ctx):
        if self.has("zero") and ctx.on(self.key("zero")):
            return ctx.mag
        return 0.0

    def reading(self, setup, flags=frozenset(), n_count=None):
        return Reading("angle", "角度", "angle", "°", resolution=self.resolution, sigma=self.sigma)

    def read(self, ctx, tmpl):
        return np.degrees(self.quantity(ctx, tmpl)) - self.offset_deg(ctx)


@dataclass
class Caliper(Instrument):
    """游标卡尺（cm，分度 0.002 cm）。作主测量仪器时模板给长度（m）。"""

    label_zh: str = "游标卡尺"
    resolution: float = 0.002
    sigma: float = 0.002

    def reading(self, setup, flags=frozenset(), n_count=None):
        return Reading("length", "长度", "length", "cm", resolution=self.resolution, sigma=self.sigma)

    def read(self, ctx, tmpl):
        return 100.0 * self.quantity(ctx, tmpl)

    def student(self, raw):
        raw = np.asarray(raw, dtype=float)
        return raw / 100.0, np.full_like(raw, 0.01)
