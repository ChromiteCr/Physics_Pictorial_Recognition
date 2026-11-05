"""规律模板：用组件算出"真实的物理观测量"，并给出面板用的线性化。

每个模板：
- quantity(ctx, what)：真实物理量（SI），what 是仪器要的量名（伸长量 ext、拉力 F、加速度 a、速度 v、
  1/v（inv_v）、周期 T、时间 t、水平距离 x……）；含线性参数时返回 Affine；
- linear(setup) / nuisance(setup)：模板自己的未知量（例如 1/√g）；
- linearise(exp, ds)：学生的作图坐标（横轴、纵轴、纵轴标准差、理想斜率和截距）。

模板只描述规律本身，误差机制都在组件和仪器里，通过 ctx.on(...) 判断是否打开。

静态弹簧 StaticSpring、拉动摩擦 FrictionPull、匀加速系统 UniformAccel（等效质量 M + ΣI/r² + ρ²m、广义合力）、
斜面机械能 EnergyIncline、系统机械能 SystemEnergy、单摆周期 PendulumPeriod、弹簧振子周期 SpringOscillator、
光电门测速自由落体 FreeFallGate、由静止匀加速 SlideFromRest、斜槽平抛 ProjectileRamp、杠杆平衡 TorqueBalance、
一维碰撞 Collision1D。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..labsim.core import G, Dataset, LinearData, LinearParam
from .base import Ctx


def period_factor(theta_rad):
    """有限摆角周期修正 T/T₀ = 1/AGM(1, cos(θ₀/2))（精确，算术-几何平均收敛极快）。

    能量守恒 θ̇² = 2(g/L)(cosθ − cosθ₀)，T = 4√(L/g)·K(k)，k = sin(θ₀/2)；K = (π/2)/AGM(1, √(1−k²))，
    小角度 T ≈ T₀(1 + θ₀²/16)。
    """
    a = np.ones_like(np.asarray(theta_rad, dtype=float))
    b = np.cos(np.asarray(theta_rad, dtype=float) / 2.0)
    for _ in range(8):
        a, b = 0.5 * (a + b), np.sqrt(a * b)
    return 1.0 / a


def _main(ds: Dataset, kind: str = "main"):
    idx = [i for i, o in enumerate(ds.observations) if o.kind == kind]
    s = np.array([ds.observations[i].setting for i in idx], dtype=float)
    raw = np.array([ds.observations[i].values[0] for i in idx], dtype=float)
    return np.array(idx), s, raw


def _sd(exp, kind: str = "main") -> float:
    r = exp.obs_kinds[kind].readings[0]
    return float(np.sqrt(r.sigma**2 + r.resolution**2 / 12.0))


def _unit(exp, kind: str = "main") -> str:
    return exp.obs_kinds[kind].readings[0].unit


class Template:
    """模板基类。primary：仪器没指定时默认要的量。"""

    id = ""
    name_zh = ""
    primary = ""
    main_label_zh = "测量"
    main_label_en = "measure"

    def linear(self, setup) -> tuple[LinearParam, ...]:
        return ()

    def nuisance(self, setup) -> dict:
        return {}

    def always_uses(self, setup) -> tuple[str, ...]:
        """所有原因下都影响预测的干扰量（尽量没有：网格是笛卡儿积，维度越少越快）。"""
        return ()

    def quantity(self, ctx: Ctx, what: str):
        raise NotImplementedError

    def linearise(self, exp, ds: Dataset) -> LinearData:
        raise NotImplementedError

    def _bad(self, what):
        return KeyError(f"模板 {self.id} 不提供物理量 {what}")


# ── 静态弹簧 ──────────────────────────────────────────────────────────


@dataclass
class StaticSpring(Template):
    """弹簧静态平衡：kx = F → 伸长量 x = (1/k)·F，F = 悬挂质量 × g。"""

    spring: str = "spring"
    id = "static_spring"
    name_zh = "弹簧弹力与伸长量（静态平衡）"
    primary = "ext"
    main_label_zh = "挂钩码读伸长量"
    main_label_en = "hang masses, read extension"

    def quantity(self, ctx, what):
        if what != "ext":
            raise self._bad(what)
        sp = ctx.part(self.spring)
        F = ctx.x * G  # 真实拉力（N）
        return ctx.lin(sp.pname("inv_k")) * (F * sp.bend(ctx, ctx.x_rec * G))

    def linearise(self, exp, ds):
        idx, s, raw = _main(ds)
        x = s * exp.setup.control.to_si * G
        sy = np.full(raw.shape, _sd(exp))
        return LinearData(x, raw, sy, idx, "拉力 F (N)", f"伸长量 x ({_unit(exp)})", None, 0.0)


# ── 拉动摩擦 ──────────────────────────────────────────────────────────


@dataclass
class FrictionPull(Template):
    """用测力计拉木块在木板上匀速运动，测力计示数 = 滑动摩擦力 f = μN。

    一般情形（木板倾角 β，拉力与水平成 a，读数时加速度 a_acc；单故障时只有一项不为零）：
      F cos a = μ(N cos β − F sin a) ± N sin β + (N/g)·a_acc
      → F = [μN cos β ± N sin β + N·a_acc/g] / (cos a + μ sin a)。
    分子里的 μ 是线性参数，分母里的 μ 用同步真值的小网格（见 HandPull）。
    """

    block: str = "block"
    board: str = "board"
    pull: str = "pull"
    id = "friction_pull"
    name_zh = "拉动木块测滑动摩擦力"
    primary = "F"
    main_label_zh = "手拉木块匀速运动，读测力计示数"
    main_label_en = "pull the block steadily by hand, read the scale"

    def quantity(self, ctx, what):
        if what != "F":
            raise self._bad(what)
        W = np.broadcast_to(ctx.x, (ctx.P, ctx.n))  # 压力（= 木块与钩码总重，N）
        blk, board = ctx.part(self.block), ctx.part(self.board)
        pull = ctx.part(self.pull) if self.pull else None
        mu = ctx.lin(blk.pname("mu"))
        tilt = board.tilt(ctx)
        if tilt is None:
            num = mu * W
        else:
            b, sgn = tilt
            num = mu * (W * np.cos(b)) + sgn * W * np.sin(b)
        if pull is not None:
            acc = pull.acceleration(ctx)
            if acc is not None:
                num = num + W * acc / G
            den = pull.denominator(ctx)
            if den is not None:
                num = num / den
        return num

    def linearise(self, exp, ds):
        kind = "main" if any(o.kind == "main" for o in ds.observations) else "fixed"
        idx, s, raw = _main(ds, kind)
        sy = np.full(raw.shape, _sd(exp, kind))
        y_label = "拉力 F (N)" if kind == "main" else "测力计示数 F (N)"
        return LinearData(s * exp.setup.control.to_si, raw, sy, idx, "压力 N (N)", y_label, None, 0.0)


# ── 匀加速系统（广义坐标） ──────────────────────────────────────────────


@dataclass
class UniformAccel(Template):
    """小车 + 轨道/斜面 + 细线 + 滑轮 + 悬挂重物 组成的匀加速系统，取小车沿轨道的位移为广义坐标。

    各物体速度与小车速度之比 ρ_i（细线平行时槽码 ρ = 1，斜拉时 ρ = cosφ），拉格朗日方程给出
      a = F_g / M_g，F_g = Σρ_i F_i（各外力沿运动方向的分量）− 阻力，M_g = Σρ_i² m_i + ΣI/r²（滑轮、车轮）。
    具体地：F_g = ρmg ± Mg sinθ − f_轨道 − f_轴 − f_纸带 (+ 残余下滑力)，M_g = M + J_轮 + ρ²m + J_滑轮。
    "detached" 变体（取下细线）：只剩小车和轨道，a₀ = F_res/M。
    conserve_total：槽码先放在车上、每次取下挂到小盘里（车 + 槽码总质量 M_t 不变），车质量 = M_t − m。
    uphill：有斜面时被悬挂重物沿斜面向上拉（否则沿斜面向下运动）。
    x_mode："mg" 横轴 = mg（课本：拉力近似 mg，理想斜率 1/M）；
            "net" 横轴 = 学生算的合力 mg ∓ (M_t − m)g sinθ_记（理想斜率 1/M_t，精确）。
    """

    body: str = "cart"
    track: str | None = None
    incline: str | None = None
    link: str | None = None
    pulley: str | None = None
    hanger: str | None = None
    conserve_total: bool = False
    uphill: bool = True
    x_mode: str = "mg"
    id = "uniform_accel"
    name_zh = "匀加速运动的系统（牛顿第二定律）"
    primary = "a"
    main_label_zh = "挂槽码测小车加速度"
    main_label_en = "hang slotted masses, measure the cart's acceleration"

    def terms(self, ctx: Ctx):
        """(广义合力 F_g（N，可为 Affine）, 等效质量 M_g（kg）)。"""
        cart = ctx.part(self.body)
        detached = ctx.flag("detached")
        hanger = ctx.part(self.hanger) if self.hanger else None
        m = hanger.mass(ctx) if hanger is not None else 0.0
        M = cart.mass(ctx)
        if self.conserve_total:
            M = M - m
        F = 0.0
        Mg = M + cart.rot_inertia(ctx)
        if self.track:
            tr = ctx.part(self.track)
            F = F + tr.residual(ctx) - tr.resistance(ctx, M * G)
        if self.incline:
            inc = ctx.part(self.incline)
            th = inc.angle(ctx)
            sgn = -1.0 if self.uphill else 1.0
            F = F + sgn * M * G * np.sin(th) - inc.resistance(ctx, M * G * np.cos(th))
        if hanger is not None and not detached:
            rho = ctx.part(self.link).ratio(ctx) if self.link else 1.0
            F = F + rho * m * G
            if hanger.is_inertial(ctx):
                Mg = Mg + rho**2 * m
        if self.pulley and not detached:
            pl = ctx.part(self.pulley)
            Mg = Mg + pl.J(ctx)
            F = F - pl.axle(ctx)
        for p in ctx.setup.parts:
            drag = getattr(p, "drag", None)
            if callable(drag):
                F = F - drag(ctx)
        return F, Mg

    def quantity(self, ctx, what):
        if what != "a":
            raise self._bad(what)
        F, Mg = self.terms(ctx)
        return F / Mg

    def ideal_slope(self, setup) -> float:
        return 1.0 / (setup.part(self.body).mass_g / 1000.0)

    def x_axis(self, setup, m: np.ndarray) -> np.ndarray:
        if self.x_mode == "mg":
            return m * G
        Mt = setup.part(self.body).mass_g / 1000.0
        th = setup.part(self.incline).angle_rec() if self.incline else 0.0
        sgn = -1.0 if self.uphill else 1.0
        return m * G + sgn * (Mt - m) * G * math.sin(th)

    def linearise(self, exp, ds):
        idx, s, raw = _main(ds)
        setup = exp.setup
        x = self.x_axis(setup, s * setup.control.to_si)
        y, dy = setup.main_instrument.student(raw)
        sy = np.abs(dy) * _sd(exp)
        x_label = "拉力 F = mg (N)" if self.x_mode == "mg" else "合力 F = mg − (M − m)g·sinθ (N)"
        return LinearData(x, y, sy, idx, x_label, "加速度 a (m/s²)", self.ideal_slope(setup), 0.0)


# ── 斜面机械能 ────────────────────────────────────────────────────────


@dataclass
class EnergyIncline(Template):
    """小车从斜面上竖直高度 h 处由静止释放，测经过光电门（或纸带测速点）时的速度。

    动能定理（广义坐标 s = h/sinθ）：½(M + J)v² = ½(M + J)v₀² + (Mg sinθ − μMg cosθ)·h/sinθ
      → v² = v₀² + 2gh·(1 − μ cotθ)·M/(M + J)。理想（无阻力、不计车轮）时 ½v² = gh。
    """

    body: str = "cart"
    incline: str = "incline"
    release: str | None = "release"
    id = "energy_incline"
    name_zh = "斜面上的机械能守恒"
    primary = "v"
    main_label_zh = "从高度 h 处由静止释放，测速度"
    main_label_en = "release from height h, record the speed reading"

    def v2(self, ctx: Ctx) -> np.ndarray:
        cart, inc = ctx.part(self.body), ctx.part(self.incline)
        M = cart.mass(ctx)
        J = cart.rot_inertia(ctx)
        v2 = 2.0 * G * (M / (M + J)) * inc.loss_factor(ctx) * ctx.x
        if self.release:
            v2 = v2 + ctx.part(self.release).v0(ctx) ** 2
        return np.maximum(np.broadcast_to(v2, (ctx.P, ctx.n)), 1e-6)

    def quantity(self, ctx, what):
        if what == "v":
            return np.sqrt(self.v2(ctx))
        if what == "inv_v":
            return 1.0 / np.sqrt(self.v2(ctx))
        raise self._bad(what)

    def linearise(self, exp, ds):
        idx, s, raw = _main(ds)
        h = s * exp.setup.control.to_si
        v, dv = exp.setup.main_instrument.student(raw)
        y = 0.5 * v**2
        sy = np.abs(v * dv) * _sd(exp)
        return LinearData(G * h, y, sy, idx, "g·h (J/kg)", "½v² (J/kg)", 1.0, 0.0)


# ── 系统机械能 ────────────────────────────────────────────────────────


@dataclass
class SystemEnergy(Template):
    """悬挂重物通过跨过定滑轮的细线带动水平导轨上的小车，重物下落 h 时小车也走了 h（细线不可伸长）。

    系统动能定理：½M_g v² = ½M_g v₀² + (ρmg − f − f_轴)·h，M_g = M + J_轮 + ρ²m + J_滑轮。
    学生验证 ΔE_p = mgh 与 ΔE_k = ½(M + m)v²：理想直线斜率 1、截距 0。
    阻力和惯性都在开方里，只能放网格（滑轮轴摩擦用 axle_linear=False）。
    """

    body: str = "cart"
    track: str | None = "track"
    hanger: str = "hanger"
    pulley: str | None = "pulley"
    link: str | None = None
    release: str | None = "release"
    id = "system_energy"
    name_zh = "系统机械能守恒（悬挂重物带动小车）"
    primary = "v"
    main_label_zh = "重物由静止下落 h 后测小车速度"
    main_label_en = "release, measure the cart's speed after the weight drops h"

    def v2(self, ctx):
        cart = ctx.part(self.body)
        M = cart.mass(ctx)
        m = ctx.part(self.hanger).mass(ctx)
        rho = ctx.part(self.link).ratio(ctx) if self.link else 1.0
        F = rho * m * G
        Mg = M + cart.rot_inertia(ctx) + rho**2 * m
        if self.track:
            F = F - ctx.part(self.track).resistance(ctx, M * G)
        if self.pulley:
            pl = ctx.part(self.pulley)
            F = F - pl.axle(ctx)
            Mg = Mg + pl.J(ctx)
        v2 = 2.0 * F * ctx.x / Mg
        if self.release:
            v2 = v2 + ctx.part(self.release).v0(ctx) ** 2
        return np.maximum(np.broadcast_to(v2, (ctx.P, ctx.n)), 1e-6)

    def quantity(self, ctx, what):
        if what == "v":
            return np.sqrt(self.v2(ctx))
        if what == "inv_v":
            return 1.0 / np.sqrt(self.v2(ctx))
        raise self._bad(what)

    def linearise(self, exp, ds):
        idx, s, raw = _main(ds)
        setup = exp.setup
        h = s * setup.control.to_si
        M = setup.part(self.body).mass_g / 1000.0
        m = setup.part(self.hanger).mass_g / 1000.0
        v, dv = setup.main_instrument.student(raw)
        y = 0.5 * (M + m) * v**2
        sy = (M + m) * np.abs(v * dv) * _sd(exp)
        return LinearData(m * G * h, y, sy, idx, "重物减少的势能 mgh (J)", "系统增加的动能 ½(M+m)v² (J)", 1.0, 0.0)


# ── 单摆 ──────────────────────────────────────────────────────────────


@dataclass
class PendulumPeriod(Template):
    """单摆周期 T = 2π√(L_eff/g)·K(θ₀)·√cosα（圆锥摆时 K = 1）。

    球形摆球的复摆修正 L_eff = L + 2r²/(5L)（用标称半径，~10⁻⁴）；课本 5° 摆角的周期修正 K(5°) ≈ 1.0005。
    线性参数 1/√g：当地 g 已知到 ±0.1%（g_rel_sd），只给理想斜率留一点余地。
    """

    bob: str = "bob"
    r_nominal_cm: float = 1.0
    theta_std_deg: float = 5.0
    g_rel_sd: float = 0.001
    id = "pendulum_period"
    name_zh = "单摆周期"
    primary = "T"
    main_label_zh = "拉开小角度释放，测多次全振动的总时间"
    main_label_en = "release at a small angle, time several oscillations"

    @property
    def k_std(self) -> float:
        return float(period_factor(math.radians(self.theta_std_deg)))

    def linear(self, setup):
        s = 1.0 / math.sqrt(G)
        return (LinearParam("gravity.inv_sqrt_g", s, 0.5 * self.g_rel_sd * s, "1/√g", "s/√m"),)

    def quantity(self, ctx, what):
        if what != "T":
            raise self._bad(what)
        bob = ctx.part(self.bob)
        Lc = np.broadcast_to(ctx.x, (ctx.P, ctx.n))
        L_eff = Lc + 0.4 * (self.r_nominal_cm / 100.0) ** 2 / Lc
        k, cone = self.k_std, 1.0
        amp = bob.release_angle(ctx)
        if amp is not None:
            k = period_factor(np.radians(amp))
        alpha = bob.cone_angle(ctx)
        if alpha is not None:
            k, cone = 1.0, np.sqrt(np.cos(np.radians(alpha)))
        return ctx.lin("gravity.inv_sqrt_g") * (2.0 * math.pi * np.sqrt(L_eff) * k * cone)

    def linearise(self, exp, ds):
        idx, s, t = _main(ds)
        N = exp.setup.main_instrument.count
        T = t / N
        sy = 2.0 * T * _sd(exp) / N
        ideal = 4.0 * math.pi**2 / G * self.k_std**2
        return LinearData(s * exp.setup.control.to_si, T**2, sy, idx, "摆长 L (m)", "T² (s²)", ideal, 0.0)


# ── 弹簧振子 ──────────────────────────────────────────────────────────


@dataclass
class SpringOscillator(Template):
    """竖直弹簧振子：T = 2π√(m_eff/k)，m_eff = 悬挂质量 + m_s/3。

    m_s/3：弹簧各段速度沿长度线性分布 v(y) = v·y/ℓ，动能 ∫½(m_s/ℓ)v²y²/ℓ² dy = ½(m_s/3)v²。
    竖直悬挂时重力只移动平衡位置，不改周期。T² = 4π²(m + m_s/3)/k：斜率 4π²/k（k 未知，斜率自由）。
    """

    spring: str = "spring"
    id = "spring_oscillator"
    name_zh = "弹簧振子的周期"
    primary = "T"
    main_label_zh = "挂上钩码、拉开一点释放，测多次全振动的总时间"
    main_label_en = "pull down slightly, release, time several oscillations"

    def quantity(self, ctx, what):
        if what != "T":
            raise self._bad(what)
        sp = ctx.part(self.spring)
        m_eff = np.broadcast_to(ctx.x + sp.mass_g / 3000.0, (ctx.P, ctx.n))
        return ctx.lin(sp.pname("inv_sqrt_k")) * (2.0 * math.pi * np.sqrt(m_eff))

    def linearise(self, exp, ds):
        idx, s, t = _main(ds)
        setup = exp.setup
        N = setup.main_instrument.count
        T = t / N
        sy = 2.0 * T * _sd(exp) / N
        m_eff = s * setup.control.to_si + setup.part(self.spring).mass_g / 3000.0
        return LinearData(m_eff, T**2, sy, idx, "振子质量 m + m_弹簧/3 (kg)", "T² (s²)", None, 0.0)


# ── 光电门测速度的自由落体 ────────────────────────────────────────────


@dataclass
class FreeFallGate(Template):
    """小球由静止下落 h（球心到光束）后穿过光电门，挡光时间给出速度：v² = 2gh。

    有空气阻力时 v² = (g/κ)(1 − e^{−2κh})（见 Ball）。v = √g·√s，s = 2h 或 (1 − e^{−2κh})/κ，
    所以 1/v = (1/√g)/√s 对线性参数 1/√g 是线性的。学生画 v²–h，理想斜率 2g。
    """

    ball: str = "ball"
    g_rel_sd: float = 0.002
    id = "free_fall_gate"
    name_zh = "光电门测速度验证自由落体"
    primary = "inv_v"
    main_label_zh = "小球从高度 h 处由静止下落，读光电门挡光时间"
    main_label_en = "drop the ball from height h, read the gate blocking time"

    def linear(self, setup):
        s = 1.0 / math.sqrt(G)
        return (LinearParam("gravity.inv_sqrt_g", s, 0.5 * self.g_rel_sd * s, "1/√g", "s/√m"),)

    def quantity(self, ctx, what):
        if what != "inv_v":
            raise self._bad(what)
        h = np.maximum(np.broadcast_to(ctx.x, (ctx.P, ctx.n)), 1e-9)
        kappa = ctx.part(self.ball).drag_coef(ctx)
        s = 2.0 * h if kappa is None else (1.0 - np.exp(-2.0 * kappa * h)) / kappa
        return ctx.lin("gravity.inv_sqrt_g") * (1.0 / np.sqrt(s))

    def linearise(self, exp, ds):
        idx, s, raw = _main(ds)
        h = s * exp.setup.control.to_si
        v, dv = exp.setup.main_instrument.student(raw)
        sy = 2.0 * np.abs(v * dv) * _sd(exp)
        return LinearData(h, v**2, sy, idx, "下落高度 h (m)", "v² (m²/s²)", 2.0 * G, 0.0)


# ── 由静止匀加速 ──────────────────────────────────────────────────────


@dataclass
class SlideFromRest(Template):
    """物体沿斜面由静止匀加速滑下：s = ½at² → t = (1/√a)·√(2s)，a = g(sinθ − μcosθ) 未知。

    线性参数 1/√a（先验按 a_mean、相对标准差 a_rel_sd 换算）。学生画 t–√(2s)，斜率 1/√a，截距 0；
    由 a 和记录的倾角求 μ = (g sinθ − a)/(g cosθ)。
    """

    body: str = "block"
    a_mean: float = 2.35
    a_rel_sd: float = 0.18
    id = "slide_from_rest"
    name_zh = "斜面上由静止下滑的匀加速运动"
    primary = "t"
    main_label_zh = "从起点由静止释放，测到达光电门的时间"
    main_label_en = "release from rest, time the arrival at the gate"

    def linear(self, setup):
        u = 1.0 / math.sqrt(self.a_mean)
        return (LinearParam("motion.inv_sqrt_a", u, 0.5 * self.a_rel_sd * u, "1/√a", "s/√m"),)

    def quantity(self, ctx, what):
        if what != "t":
            raise self._bad(what)
        s = np.maximum(np.broadcast_to(ctx.x, (ctx.P, ctx.n)), 1e-9)
        return ctx.lin("motion.inv_sqrt_a") * np.sqrt(2.0 * s)

    def linearise(self, exp, ds):
        idx, s, t = _main(ds)
        x = np.sqrt(2.0 * s * exp.setup.control.to_si)
        sy = np.full(t.shape, _sd(exp))
        return LinearData(x, t, sy, idx, "√(2s) (√m)", "时间 t (s)", None, 0.0)


# ── 斜槽平抛 ──────────────────────────────────────────────────────────


@dataclass
class ProjectileRamp(Template):
    """小球从斜槽上高 H 处由静止滚下，从高 h 的水平槽口飞出，落地水平距离 x。

    纯滚动：mgH = ½mv₀²(1 + β)，β = I/(m r_c²) = (2/5)R²/(R² − s²)（双轨间距 2s，平底槽 s = 0 → β = 2/5）；
    平抛 t = √(2h/g)，x = v₀t → x² = 4hH/(1 + β)（g 约掉）。学生画 x²–H，理想斜率 4h/(1+β)。
    """

    ball: str = "ball"
    launch_h_cm: float = 80.0
    rail_gap_cm: float = 0.0
    id = "projectile_ramp"
    name_zh = "斜槽平抛"
    primary = "x"
    main_label_zh = "从斜槽高度 H 处释放，量落地点的水平距离"
    main_label_en = "release from height H, measure the landing distance"

    def beta(self, setup) -> float:
        R = setup.part(self.ball).diameter_cm / 2.0
        s = self.rail_gap_cm / 2.0
        return 0.4 * R**2 / (R**2 - s**2)

    def quantity(self, ctx, what):
        if what != "x":
            raise self._bad(what)
        H = np.maximum(np.broadcast_to(ctx.x, (ctx.P, ctx.n)), 0.0)
        v0 = np.sqrt(2.0 * G * H / (1.0 + self.beta(ctx.setup)))
        return v0 * math.sqrt(2.0 * self.launch_h_cm / 100.0 / G)

    def linearise(self, exp, ds):
        idx, s, x = _main(ds)
        sy = 2.0 * x * _sd(exp)
        ideal = 4.0 * self.launch_h_cm / (1.0 + self.beta(exp.setup))
        return LinearData(s, x**2, sy, idx, "释放高度 H (cm)", "x² (cm²)", ideal, 0.0)


# ── 杠杆平衡 ──────────────────────────────────────────────────────────


@dataclass
class TorqueBalance(Template):
    """杠杆一侧在 L2 处挂重 F2 的钩码，另一侧 L1（控制量）处用测力计竖直拉到水平平衡：
    F1·L1 = F2·L2 + M₀（M₀ = 没调平衡时杠杆自重的力矩）→ F1 = (F2·L2 + M₀)/L1。学生画 F1–1/L1，理想斜率 F2·L2。
    """

    lever: str = "lever"
    load_g: float = 200.0
    load_arm_cm: float = 10.0
    id = "torque_balance"
    name_zh = "杠杆的平衡条件"
    primary = "F"
    main_label_zh = "测力计在 L1 处竖直拉到杠杆水平，读示数"
    main_label_en = "pull the scale down at L1 until level, read it"

    @property
    def torque(self) -> float:
        return self.load_g / 1000.0 * G * self.load_arm_cm / 100.0

    def quantity(self, ctx, what):
        if what != "F":
            raise self._bad(what)
        L1 = np.broadcast_to(ctx.x, (ctx.P, ctx.n))
        tau = self.torque + ctx.part(self.lever).extra_torque(ctx)
        return tau * (1.0 / L1)

    def linearise(self, exp, ds):
        idx, s, F = _main(ds)
        x = 1.0 / (s * exp.setup.control.to_si)
        return LinearData(x, F, np.full(F.shape, _sd(exp)), idx, "1/L1 (1/m)", "测力计示数 F1 (N)", self.torque, 0.0)


# ── 一维碰撞 ──────────────────────────────────────────────────────────


@dataclass
class Collision1D(Template):
    """A 车（挡光时间 Δt₁，控制量，ms）撞静止的 B 车后粘在一起，一起经过第二个光电门。

    守恒的是 Σ(m + J)v（J = 车轮 ΣI/R²：轨道对轮的静摩擦冲量改变轮的转速，J_f R = IΔω → J_f = −(I/R²)Δv）：
      (m_A + J_A)v₁ = (m_A + m_B + J_A + J_B)v₂ → 1/v₂ = (Δt₁/w)·(m_A + m_B + J_A + J_B)/(m_A + J_A)。
    第二个光电门读 Δt₂ = w/v₂（遮光片宽度 w 已知，两门共用时宽度约掉）。学生画 p' = (m_A+m_B)w/Δt₂ 对 p = m_A w/Δt₁。
    """

    cart_a: str = "cart_a"
    cart_b: str = "cart_b"
    gate: str = "gate"
    id = "collision_1d"
    name_zh = "一维完全非弹性碰撞"
    primary = "inv_v"
    main_label_zh = "推 A 撞静止的 B，读碰后经过光电门 2 的挡光时间"
    main_label_en = "push A into B at rest, read the blocking time at gate 2"

    def quantity(self, ctx, what):
        if what != "inv_v":
            raise self._bad(what)
        A, B = ctx.part(self.cart_a), ctx.part(self.cart_b)
        mA = A.mass(ctx) + A.rot_inertia(ctx)
        mAB = mA + B.mass(ctx) + B.rot_inertia(ctx)
        w = ctx.part(self.gate).flag_mm
        dt1 = np.broadcast_to(ctx.rec, (ctx.P, ctx.n))  # 控制量就是记录的 Δt₁（ms）
        return dt1 / w * (mAB / mA)  # ms/mm = s/m

    def linearise(self, exp, ds):
        idx, dt1, dt2 = _main(ds)
        setup = exp.setup
        w = setup.part(self.gate).flag_mm
        mA = setup.part(self.cart_a).mass_g / 1000.0
        mB = setup.part(self.cart_b).mass_g / 1000.0
        p = mA * w / dt1
        p2 = (mA + mB) * w / dt2
        sy = p2 * _sd(exp) / dt2
        return LinearData(p, p2, sy, idx, "碰前动量 p (kg·m/s)", "碰后动量 p' (kg·m/s)", 1.0, 0.0)
