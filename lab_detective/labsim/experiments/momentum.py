"""验证动量守恒定律：导轨上两车碰撞，用光电门测碰撞前后的速度。

学生的做法（人教版选择性必修第一册 第一章"实验：验证动量守恒定律"参考案例 1，默认方案）：
气垫导轨调水平，两滑块（下文统称"小车"）上各装宽 w 的遮光片，天平称出质量 m_A、m_B。
B 静止放在两个光电门之间，推 A 去撞 B：A 先经过光电门 1（挡光时间 Δt₁），碰后两车粘在一起
（A 前端的撞针扎进 B 上的橡皮泥）经过光电门 2（挡光时间 Δt₂）。换不同力度推 5 次。
速度 v = w/Δt，碰前动量 p = m_A·w/Δt₁，碰后动量 p' = (m_A + m_B)·w/Δt₂。
构造参数 collision="elastic" 时改用弹性碰撞架：碰后 B、A 先后经过光电门 2（Δt_B'、Δt_A'），
p' = m_A·w/Δt_A' + m_B·w/Δt_B'（要求 m_A > m_B，A 碰后继续向前）。

建模要点：
- 控制量取学生记下的 Δt₁（不是真实推速）：学生并不知道真实速度，引擎也不应知道。
  理想粘合碰撞 m_A·v₁ = (m_A+m_B)·v₂ ⇒ Δt₂ = Δt₁·(m_A+m_B)/m_A，w 完全约掉——
  这正是"两门共用的遮光片宽度误差"在本方案里测不出来的物理原因。
- Δt₁ 本身也有计时误差，按误差传递并进 Δt₂ 的读数标准差：σ² = σ_碰² + σ_门² + k²(σ_门² + d²/12)，
  k = Δt₂/Δt₁（变量含误差问题）。
- 光束有一定粗细时，有效挡光长度 ≈ w + 光束直径，两门相同，等价于共同的宽度比例误差，同样约掉。
- 车轮转动惯量（构造参数 wheel_inertia_g = 每车 ΣI/R²，单位 g；气垫导轨滑块为 0）：
  车轮纯滚，轨道对轮的静摩擦冲量 J_f 使轮子转速改变：J_f·R = I·Δω = I·Δv/R ⇒ J_f = −(I/R²)Δv，
  于是 (m + ΣI/R²)·Δv = 两车间的冲量，守恒的是 Σ(m + ΣI/R²)·v 而不是 Σm·v。
  两车相同轮子、质量不同时，按天平质量算的动量比会偏离 1 约零点几个百分点。
- 面板坐标：横轴 = 碰前动量 p（kg·m/s），纵轴 = 碰后动量 p'（kg·m/s），理想斜率 1、截距 0。
- 空气阻力：0.5 m/s 的滑块受气膜黏滞与空气阻力很小，归入"导轨摩擦"原因，不单独建模。
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

DT_RES = 0.1  # 数字计时器分辨率 0.1 ms（= 0.0001 s）
GATE_SIGMA = 0.1  # 单次挡光时间的随机误差（遮光片边缘、光敏元件抖动），ms
COLL_SIGMA = 0.25  # 每次碰撞不完全一样（没正碰、橡皮泥形变不同）带来的额外散布，ms
T12_SIGMA = 2.0  # 两门间时间：主要来自两门间距用米尺量的误差（±1 mm / 40 cm），ms
V_MIN = 0.01  # m/s，车停下时的下限，只为避免除零（此时挡光时间极长，似然自然很小）


def _gate_reading(key, zh, en):
    return Reading(key, zh, en, "ms", resolution=DT_RES, sigma=GATE_SIGMA)


class Momentum(Experiment):
    id = "momentum"
    name_zh = "验证动量守恒定律（两车碰撞）"
    name_en = "Conservation of momentum: two-cart collision"

    # 所有原因共用的线性参数：无（光电门计时偏差是原因自带的线性参数）
    linear = ()

    causes = (
        Cause(
            "momentum.random_only", "只有随机误差", "random error only",
            "导轨水平、摩擦可忽略，质量和遮光片宽度都记录正确，数据偏离只来自计时和碰撞的随机误差。",
            "Level track, negligible friction, masses and flag width recorded correctly; deviations "
            "come only from random timing and collision variability.",
            "none", ("none",),
            fix_zh="多推几次、取平均即可。", fix_en="Repeat the collision a few more times and average.",
        ),
        # 推导：沿运动方向下坡角 θ，a = g·sinθ ≈ g·Δh/L（Δh/L 记作每米高度差，单位 mm/m）。
        # 光电门 1 到碰撞点 d₁：v_c² = v₁² + 2a·d₁；粘合：v' = m_A·v_c/(m_A+m_B)；
        # 碰撞点到光电门 2 距离 d₂：v₂² = v'² + 2a·d₂。重力分力的冲量 Mg·sinθ·t ∝ 距离/速度，
        # 所以慢的那几次偏得最多：下坡时 p' > p，上坡时 p' < p（上坡与摩擦在正向碰撞里完全等价）。
        Cause(
            "momentum.track_tilt", "导轨不水平", "track not level",
            "导轨一端略高，小车在两个光电门之间运动时，重力沿导轨的分力一直在推它（下坡）或拉它（上坡）。",
            "One end of the track is slightly higher, so the component of gravity along the track keeps "
            "pushing (downhill) or holding back (uphill) the carts between the two gates.",
            "bend", ("curve_up", "slope_low", "curve_down", "slope_high"),
            magnitude=Prior.sym_uniform(0.3, 4.0, 48),
            magnitude_label_zh="每米导轨的高度差（正 = 沿运动方向下坡）", magnitude_unit="mm/m",
            fix_zh="接通气源后调节导轨底脚螺丝，使单独推出的小车经过两个光电门的挡光时间相等。",
            fix_en="Adjust the levelling screws until a lone cart has equal blocking times at both gates.",
        ),
        # 推导：动摩擦（滑块与导轨局部接触、气孔堵塞或车轮轴承发涩）给恒定减速度 a = μg。
        # v_c² = v₁² − 2μg·d₁，v₂² = v'² − 2μg·d₂；损失的动量 = μMg·t，路程越长、车越慢，损失越多。
        Cause(
            "momentum.friction", "两门之间有摩擦", "friction between the gates",
            "气源风量不够或气孔堵了，滑块局部擦着导轨走（小车则是轴承发涩），从光电门 1 到光电门 2 的一路上一直受摩擦力。",
            "The air supply is weak or some holes are blocked, so the slider rubs on the track (for a "
            "wheeled cart: stiff bearings); friction acts all the way from gate 1 to gate 2.",
            "bend", ("slope_high",),
            magnitude=Prior.loguniform(0.0003, 0.004, 40),
            magnitude_label_zh="等效动摩擦因数", magnitude_unit="",
            fix_zh="检查气源和气孔，让滑块浮起来；两个光电门尽量靠近碰撞点，推得快一些。",
            fix_en="Check the air supply so the slider floats; put both gates close to the collision point.",
        ),
        # 推导：B 的真实质量 m_B + δ（δ 为没称进去的橡皮泥/配重），粘合时 Δt₂ = Δt₁·(m_A+m_B+δ)/m_A，
        # 表格却用 m_A+m_B 算 p'，于是 p'/p = (m_A+m_B)/(m_A+m_B+δ)，与速度无关的纯比例误差。
        Cause(
            "momentum.mass_entry", "质量记录有误", "a cart's mass recorded wrongly",
            "记录的 B 车质量和实际不一样：粘上的橡皮泥、加的配重片没称进去，或抄错了数。",
            "The recorded mass of cart B differs from its real mass: the plasticine or an added weight "
            "was not weighed, or the number was copied wrongly.",
            "y_scale", ("slope_low", "slope_high"),
            magnitude=Prior.sym_uniform(5.0, 60.0, 44),
            magnitude_label_zh="B 车真实质量 − 记录质量", magnitude_unit="g",
            fix_zh="把装好橡皮泥、配重和遮光片的两辆车重新放到天平上称一遍。",
            fix_en="Re-weigh both carts with plasticine, weights and flags attached.",
        ),
        # 推导：所有速度都用同一个记录宽度 w 计算，真实宽度 w(1+ε)。p = m_A·w/Δt₁ 与 p' = M·w/Δt₂
        # 被同一个因子 1/(1+ε) 缩放，p'/p 不变；而且 Δt₂/Δt₁ = M/m_A 与 w 无关。
        # 标准方案的碰撞数据对它完全不敏感（特征 none），只有直接量宽度或另测一个独立的速度才能发现。
        Cause(
            "momentum.flag_width", "遮光片宽度用错了", "wrong flag width used",
            "计算速度时用的遮光片宽度不是实际宽度：直接抄了标称值没有测量，或游标卡尺读错了；两车用的是同一个错数。",
            "The flag width used to compute speeds is not the real width (the nominal value was copied "
            "or the caliper was misread); the same wrong value is used for both carts.",
            "aux_only", ("none",),
            magnitude=Prior.sym_uniform(0.02, 0.15, 26),
            magnitude_label_zh="真实宽度相对记录值的偏差", magnitude_unit="",
            fix_zh="用游标卡尺测量遮光片宽度，按实测值重新计算速度。",
            fix_en="Measure the flag width with a vernier caliper and recompute the speeds.",
        ),
        # 推导：光敏元件遮光和复光的反应快慢不同，每次挡光时间都多记了固定的 τ：Δt_记 = w/v + τ。
        # 由 Δt₁ 反推 v₁ = w/(Δt₁−τ)，于是 Δt₂ = k(Δt₁−τ) + τ（k = M/m_A），对 τ 线性。
        # p'/p = kΔt₁/Δt₂ = 1/[1 − (k−1)τ/(kΔt₁)] ⇒ 推得越快（Δt₁ 越小）偏得越多，τ > 0 时 p' 偏大。
        Cause(
            "momentum.timing_offset", "光电门计时有固定偏差", "photogate timing offset",
            "光电门（尤其是自制或老旧的光敏电阻式光电门）遮光时和复光时反应快慢不一样，每次记下的挡光时间都多（或少）了同样一小段时间。",
            "The photogate sensor responds at different speeds when the beam is blocked and when it "
            "is restored (common in home-made or old photoresistor gates), so every blocking time is "
            "too long (or too short) by the same small amount.",
            "bend", ("slope_high", "slope_low"),
            linear=(LinearParam("tau", 0.0, 1.0, "挡光时间的固定偏差", "ms"),),
            fix_zh="换用响应快的光电门，或用两门间时间等独立方法标定计时偏差。",
            fix_en="Use a faster photogate, or calibrate the offset with an independent timing method.",
        ),
    )

    protocols = {
        "single": Protocol("single", "推一次", "one collision", (("main", 24.0),)),
        "standard": Protocol(
            "standard", "课本标准方案（不同力度推 5 次）", "textbook worksheet (5 pushes)",
            tuple(("main", t) for t in (15.0, 19.0, 24.0, 30.0, 38.0)),
        ),
    }

    menu = (
        MenuItem("reweigh", "把两辆车（含橡皮泥、配重、遮光片）重新称一遍", "re-weigh both carts",
                 2.0, (("mass_check", 0.0),)),
        MenuItem("caliper", "用游标卡尺测遮光片宽度", "measure the flag width with a caliper",
                 2.0, (("flag_check", 0.0),)),
        MenuItem("coast_forward", "拿走 B，单独推 A 正向通过两个光电门（快、慢各一次），记两门挡光时间和两门间时间",
                 "remove B and coast A forward through both gates (fast and slow), record both "
                 "blocking times and the gate-to-gate time",
                 3.0, (("solo_fwd", 15.0), ("solo_fwd", 30.0))),
        MenuItem("coast_reverse", "单独推 A 反向通过两个光电门（快、慢各一次）",
                 "coast A backwards through both gates (fast and slow)",
                 3.0, (("solo_rev", 15.0), ("solo_rev", 30.0))),
        MenuItem("fast_push", "再做一次推得很快的碰撞（Δt₁ 约 11 ms）", "one more, very fast collision",
                 1.0, (("main", 11.0),)),
    )

    def __init__(self, collision: str = "sticky", m_a_g: float = 300.0, m_b_g: float = 200.0,
                 flag_cm: float = 1.00, d1_cm: float = 20.0, d2_cm: float = 20.0,
                 wheel_inertia_g: float = 0.0):
        assert collision in ("sticky", "elastic"), collision
        if collision == "elastic":
            assert m_a_g > m_b_g, "弹性碰撞方案要求 m_A > m_B，A 碰后继续向前"
        self.collision = collision
        self.m_a, self.m_b = float(m_a_g), float(m_b_g)  # g，记录值（也是 random_only 下的真值）
        self.w_cm = float(flag_cm)
        self.w = flag_cm / 100.0  # m
        self.d1, self.d2 = d1_cm / 100.0, d2_cm / 100.0  # m
        self.D = self.d1 + self.d2  # 两光电门间距，m
        self.j_wheel = float(wheel_inertia_g)
        if collision == "sticky":
            after = (_gate_reading("dt2", "碰后挡光时间 Δt₂", "post-collision blocking time"),)
            self.nuisance = {}
        else:
            after = (
                _gate_reading("dt_b", "碰后 B 的挡光时间 Δt_B'", "B's post-collision blocking time"),
                _gate_reading("dt_a", "碰后 A 的挡光时间 Δt_A'", "A's post-collision blocking time"),
            )
            # 弹性碰撞架的恢复系数（动量守恒与它无关，但决定动量怎样分给两车）
            self.nuisance = {"e": Prior.uniform(0.80, 1.0, 21)}
        self.obs_kinds = {
            "main": ObsKind(
                "main", "推 A 撞 B，记碰后挡光时间", "push A into B, record post-collision times", after,
                setting_label_zh="碰前 A 的挡光时间 Δt₁", setting_label_en="A's pre-collision blocking time",
                setting_unit="ms",
            ),
            "solo_fwd": ObsKind(
                "solo_fwd", "单独推 A 正向通过两门", "coast A forward through both gates",
                (_gate_reading("dt_out", "光电门 2 的挡光时间", "blocking time at gate 2"),
                 Reading("t12", "两门间时间", "gate-to-gate time", "ms", resolution=DT_RES, sigma=T12_SIGMA)),
                setting_label_zh="光电门 1 的挡光时间", setting_label_en="blocking time at gate 1",
                setting_unit="ms",
            ),
            "solo_rev": ObsKind(
                "solo_rev", "单独推 A 反向通过两门", "coast A backwards through both gates",
                (_gate_reading("dt_out", "光电门 1 的挡光时间", "blocking time at gate 1"),
                 Reading("t12", "两门间时间", "gate-to-gate time", "ms", resolution=DT_RES, sigma=T12_SIGMA)),
                setting_label_zh="光电门 2 的挡光时间", setting_label_en="blocking time at gate 2",
                setting_unit="ms",
            ),
            "mass_check": ObsKind(
                "mass_check", "天平重称两车", "re-weigh both carts on a balance",
                (Reading("m_a", "A 车质量", "mass of A", "g", resolution=0.1, sigma=0.1),
                 Reading("m_b", "B 车质量", "mass of B", "g", resolution=0.1, sigma=0.1)),
            ),
            "flag_check": ObsKind(
                "flag_check", "游标卡尺测遮光片宽度", "caliper reading of the flag width",
                (Reading("flag_w", "遮光片宽度", "flag width", "cm", resolution=0.002, sigma=0.001),),
            ),
        }

    # ---- 物理 ----

    def _accel(self, cause_id, mag, direction):
        """沿运动方向的加速度（m/s²）。direction = +1 正向（A 撞 B 的方向），−1 反向。"""
        if cause_id == "momentum.track_tilt":
            return direction * G * mag / 1000.0
        if cause_id == "momentum.friction":
            return -G * mag
        return 0.0

    def _after_ratios(self, cause_id, params):
        """碰后各读数对应物体的速度 / 碰撞瞬间 A 的速度（按真实质量、含车轮等效质量）。"""
        mag = params["mag"]
        m_b = self.m_b + (mag if cause_id == "momentum.mass_entry" else 0.0)
        ma = self.m_a + self.j_wheel
        mb = m_b + self.j_wheel
        M = ma + mb
        if self.collision == "sticky":
            return [ma / M + 0.0 * mag]
        e = params["e"]
        return [(1.0 + e) * ma / M + 0.0 * mag, (ma - e * mb) / M + 0.0 * mag]

    @staticmethod
    def _coast(v, a, d):
        return np.sqrt(np.maximum(v**2 + 2.0 * a * d, V_MIN**2))

    def predict(self, kind, settings, order, params, cause_id):
        P = params["mag"].shape[0]
        n = settings.size
        L = len(self.linear_params(cause_id))
        r = len(self.obs_kinds[kind].readings)
        c = np.zeros((P, n, r))
        U = np.zeros((P, n, r, L))
        mag = params["mag"]  # (P, 1)
        t_in = settings[None, :]  # ms
        w = self.w * (1.0 + mag) if cause_id == "momentum.flag_width" else self.w + 0.0 * mag
        offset = cause_id == "momentum.timing_offset"
        if kind == "main":
            a = self._accel(cause_id, mag, +1)
            v1 = w / (t_in / 1000.0)
            vc = self._coast(v1, a, self.d1)
            for i, k in enumerate(self._after_ratios(cause_id, params)):
                v2 = self._coast(k * vc, a, self.d2)
                dt = 1000.0 * w / v2  # ms
                c[:, :, i] = dt
                if offset:  # 无加速度时 Δt = (Δt₁ − τ)·q + τ，q = Δt/Δt₁|τ=0 ⇒ ∂Δt/∂τ = 1 − q
                    U[:, :, i, 0] = 1.0 - dt / t_in
            return c, U
        if kind in ("solo_fwd", "solo_rev"):
            a = self._accel(cause_id, mag, +1 if kind == "solo_fwd" else -1)
            v_in = w / (t_in / 1000.0)
            v_out = self._coast(v_in, a, self.D)
            c[:, :, 0] = 1000.0 * w / v_out
            c[:, :, 1] = 1000.0 * 2.0 * self.D / (v_in + v_out)  # 匀变速：t = 2D/(v_in + v_out)
            if offset:
                # 第二门挡光时间 = w/v + τ = 第一门读数（对 τ 不敏感）；
                # 两门间时间按两次"开始挡光"计，延迟相同而抵消：T = D/v = D(Δt_in − τ)/w
                U[:, :, 1, 0] = -self.D / self.w
            return c, U
        if kind == "mass_check":
            c[:, :, 0] = self.m_a
            c[:, :, 1] = self.m_b + (mag if cause_id == "momentum.mass_entry" else 0.0)
            return c, U
        if kind == "flag_check":
            c[:, :, 0] = 100.0 * w
            return c, U
        raise KeyError(kind)

    def reading_sigma(self, kind, params, cause_id):
        base = super().reading_sigma(kind, params, cause_id)
        if kind == "main":
            # Δt₁ 当作控制量，它的计时误差按 k = Δt/Δt₁ 传进每个碰后读数
            p = dict(params)
            p["mag"] = np.zeros_like(params["mag"])
            ks = [1.0 / np.asarray(k, dtype=float) for k in self._after_ratios("momentum.random_only", p)]
            s1 = GATE_SIGMA**2 + DT_RES**2 / 12.0
            sig = [np.sqrt(COLL_SIGMA**2 + GATE_SIGMA**2 + k**2 * s1) for k in ks]
            return np.stack(np.broadcast_arrays(*sig), axis=-1)  # (P, 1, r) 或 (r,)
        if kind in ("solo_fwd", "solo_rev"):
            s_out = np.sqrt(2 * GATE_SIGMA**2 + DT_RES**2 / 12.0)
            return np.array([s_out, T12_SIGMA])
        return base

    def relevant_nuisance(self, cause_id, kinds):
        if self.collision == "elastic" and "main" in kinds:
            return ("e",)
        return ()

    def linearise(self, ds: Dataset) -> LinearData:
        idx = [i for i, o in enumerate(ds.observations) if o.kind == "main"]
        t1 = np.array([ds.observations[i].setting for i in idx]) / 1000.0  # s
        vals = np.array([ds.observations[i].values for i in idx], dtype=float) / 1000.0  # s
        ma, mb = self.m_a / 1000.0, self.m_b / 1000.0  # kg
        x = ma * self.w / t1
        st = np.sqrt(GATE_SIGMA**2 + DT_RES**2 / 12.0) / 1000.0
        sp = np.sqrt(COLL_SIGMA**2 + GATE_SIGMA**2 + DT_RES**2 / 12.0) / 1000.0
        if self.collision == "sticky":
            parts = [(ma + mb) * self.w / vals[:, 0]]
            ts = [vals[:, 0]]
        else:
            parts = [mb * self.w / vals[:, 0], ma * self.w / vals[:, 1]]
            ts = [vals[:, 0], vals[:, 1]]
        y = np.sum(parts, axis=0)
        var_y = np.sum([(p * sp / t) ** 2 for p, t in zip(parts, ts)], axis=0)
        var_x = (x * st / t1) ** 2
        sy = np.sqrt(var_y + var_x)  # 理想斜率为 1，x 的误差直接并入 y
        return LinearData(x, y, sy, np.array(idx), "碰前动量 p (kg·m/s)", "碰后动量 p' (kg·m/s)", 1.0, 0.0)
