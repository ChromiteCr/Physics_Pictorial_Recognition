"""阿特伍德机：用两个跨过定滑轮的重物验证牛顿第二定律（留出实验，近迁移）。

人教版没有把阿特伍德机列为学生必做实验，这里的"标准方案"采用高中常见的拓展做法：
两侧挂总质量 M_t = m1 + m2 = 600 g 的重物，把小槽码从一侧移到另一侧，使质量差 Δm = m1 − m2 依次为
10~50 g（总质量不变）；较重的 m1 从离地 h = 0.80 m 处由静止释放，m2 从地面开始上升；
用秒表测 m1 落地所用时间 t（每个 Δm 测 4 次），算 a = 2h/t²。

面板坐标：横轴 = Δm·g（N，视为准确），纵轴 = a = 2h/t²（m/s²）。
理想直线 a = Δm·g/M_t，截距 0，斜率 1/M_t 已知。
推导：m1 g − T1 = m1 a，T2 − m2 g = m2 a，理想滑轮 T1 = T2 → a = (m1 − m2) g/(m1 + m2)。

建模约定（单故障世界）："只有随机误差" = 轻滑轮、无轴摩擦、细线质量可忽略、计时与测高都正确；
每个原因 = 理想模型 + 一个被忽略的效应或操作错误。学生记录的是原始读数 t，所以计时偏差是 t 上的线性参数，
其余效应通过 t = √(2h/a) 非线性地进入，放在网格上。

仪器：
- 秒表（默认）：分辨率 0.01 s，人工起停计时的随机误差 σ ≈ 0.10 s（Faux & Godolphin 2019）。
- 光电门（timer="photogate"）：电磁铁释放同时开始计时，落地处光电门停止，分辨率 0.1 ms，σ ≈ 2 ms。
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

TIMERS = {
    # timer: (σ_t, 分辨率, 计时偏差先验 (均值, 标准差))
    "stopwatch": (0.10, 0.01, (0.0, 0.15)),
    "photogate": (0.002, 0.0001, (0.02, 0.007)),
}
PG_SIGMA, PG_RES = 0.002, 0.0001  # 追加测量里的光电门
PULLEY_ABOVE_START = 0.4  # 释放时 m1 在滑轮下方 0.4 m，细线总长 ≈ h + 0.8 m

STRING_IDS = ("atwood.string_mass",)


def _obs_kinds(timer: str) -> dict[str, ObsKind]:
    sig, res, _ = TIMERS[timer]
    t_main = Reading("fall_time", "下落时间", "fall time", "s", resolution=res, sigma=sig)
    return {
        "main": ObsKind(
            "main", "释放重物，测较重一侧落地所用时间", "release and time the heavier mass to the floor", (t_main,),
            setting_label_zh="两侧质量差 Δm", setting_label_en="mass difference", setting_unit="g",
        ),
        "thin_thread": ObsKind(
            "thin_thread", "换用很细的尼龙线后计时", "time the fall after switching to a very thin thread", (t_main,),
            setting_label_zh="两侧质量差 Δm", setting_label_en="mass difference", setting_unit="g",
        ),
        "photogate_run": ObsKind(
            "photogate_run", "用电磁铁释放、光电门计时", "release with an electromagnet, time with a photogate",
            (Reading("gate_time", "光电门测得的下落时间", "photogate fall time", "s",
                     resolution=PG_RES, sigma=PG_SIGMA),),
            setting_label_zh="两侧质量差 Δm", setting_label_en="mass difference", setting_unit="g",
        ),
        "remeasure_h": ObsKind(
            "remeasure_h", "从重物底面量到落地点，重测下落高度", "re-measure the drop from the bottom of the mass",
            (Reading("h_diff", "原记录高度 − 重测高度", "recorded h − re-measured h", "cm",
                     resolution=0.1, sigma=0.2),),
        ),
    }


def _causes(timer: str) -> tuple[Cause, ...]:
    _, _, (dt_mu, dt_sd) = TIMERS[timer]
    return (
        Cause(
            "atwood.random_only", "只有随机误差", "random error only",
            "滑轮轻而灵活，细线很轻，下落高度从重物底面量到落地点，计时的开始和结束与运动同步，数据偏离只来自计时的随机误差。",
            "The pulley is light and free-running, the thread is light, the drop is measured from the bottom "
            "of the mass and timing starts and stops with the motion; deviations are random timing error only.",
            "none", ("none",),
            fix_zh="每个质量差多测几次、取平均即可。", fix_en="Time each setting several times and average.",
        ),
        # 推导：滑轮转动 (T1 − T2) r = I α，α = a/r → T1 − T2 = (I/r²) a = J a。
        # 与 m1 g − T1 = m1 a、T2 − m2 g = m2 a 联立 → a = Δm g/(M_t + J)：斜率按 M_t/(M_t + J) 变小。
        Cause(
            "atwood.pulley_inertia", "滑轮的转动惯量不能忽略", "pulley moment of inertia",
            "滑轮比较重，重物下落时细线还要带动滑轮越转越快，一部分重力用来让滑轮转动起来。",
            "The pulley is heavy; as the masses move the string must also spin it up, so part of the "
            "driving weight goes into rotating the pulley.",
            "y_scale", ("slope_low",),
            magnitude=Prior.uniform(10.0, 80.0, 29),
            magnitude_label_zh="滑轮等效质量 I/r²", magnitude_unit="g",
            fix_zh="换用轻质滑轮；或称出滑轮质量，按 a = Δm·g/(M_t + I/r²) 修正（均匀圆盘 I/r² = m_轮/2）。",
            fix_en="Use a lighter pulley, or correct with a = Δm·g/(M_t + I/r²) (uniform disc: I/r² = m_pulley/2).",
        ),
        # 推导：轴摩擦力矩 τ 等效为细线上的恒定阻力 f = τ/r：(M_t) a = Δm g − f，
        # Δm g 要先超过 f 才会加速。
        Cause(
            "atwood.pulley_friction", "滑轮轴有摩擦", "pulley axle friction",
            "滑轮的轴不灵活，转动时有摩擦力矩，相当于在细线上加了一个与运动方向相反的恒定阻力。",
            "The pulley axle is stiff; its friction torque acts like a constant force on the string opposing the motion.",
            "x_offset", ("intercept_neg",),
            magnitude=Prior.uniform(0.01, 0.06, 26),
            magnitude_label_zh="等效阻力 τ/r", magnitude_unit="N",
            fix_zh="给滑轮轴上润滑油或换带轴承的滑轮；也可以先找出刚好能匀速运动的质量差，从 Δm 中扣除。",
            fix_en="Lubricate the axle or use a ball-bearing pulley; or find the mass difference that just gives "
            "uniform motion and subtract it.",
        ),
        # 推导：细线线密度 λ。m1 一侧线长 L1、m2 一侧 L2，开始时 m2 在地面，L2 − L1 = h；
        # m1 下落 x 后 L1 − L2 = 2x − h，线的重力差 λg(2x − h)。整根线随重物一起运动，惯性加 m_s = λ(h + 0.8 m)。
        # (M_t + m_s) ẍ = Δm g − λgh + 2λg x，即 ẍ = A + B x（A = (Δm − λh)g/M_eff，B = 2λg/M_eff），
        # 从静止出发 x = (A/B)(cosh√B t − 1) → t = arcosh(1 + Bh/A)/√B。
        # 一阶近似 a_测 = 2h/t² ≈ (Δm g − (2/3)λgh)/M_eff：前期的反向重力差权重更大，等效一个恒定阻力。
        Cause(
            "atwood.string_mass", "细线质量不能忽略", "string mass not negligible",
            "用的是比较粗重的绳子。开始时轻物一侧的绳子更长，绳子自身的重力差阻碍运动，下落过程中这个差逐渐反转；"
            "整根绳子也要一起被加速。",
            "A thick, heavy cord is used. At the start the cord on the lighter side is longer, so its weight "
            "opposes the motion, reversing as the masses move; the whole cord must also be accelerated.",
            "x_offset", ("intercept_neg",),
            magnitude=Prior.uniform(1.5, 6.0, 19),
            magnitude_label_zh="绳子的线密度", magnitude_unit="g/m",
            fix_zh="换用很细的尼龙线（线密度 < 0.3 g/m），或让两侧绳长相等时再释放。",
            fix_en="Use a very thin nylon thread (< 0.3 g/m), or release when both sides have equal cord length.",
        ),
        # 推导：测得时间 t_测 = t + δ（开始按得晚 δ < 0，停止按得晚 δ > 0），
        # a_测 = 2h/(t + δ)² ≈ a(1 − 2δ/t)，t 越短（Δm 越大）相对偏差越大，斜率被整体抬高或压低。
        Cause(
            "atwood.timing_offset", "计时与运动不同步", "timing offset",
            "按秒表的时刻与重物开始运动、落地的时刻不同步：例如看到重物下落才开始计时（开始得晚），"
            "或听到落地声才停止计时（结束得晚），每次计时都差不多偏了同样一段时间。",
            "The stopwatch is not started/stopped in sync with the motion, e.g. started only after seeing the "
            "mass move or stopped after hearing it land, so every time is off by about the same amount.",
            "y_scale", ("slope_high", "slope_low"),
            linear=(LinearParam("dt", dt_mu, dt_sd, "计时偏差", "s"),),
            fix_zh="由释放重物的同学自己同时开始计时，眼睛盯住落地点；或改用电磁铁释放 + 光电门计时。",
            fix_en="Let the person releasing the mass start the watch, watch the landing point, or use an "
            "electromagnet release with a photogate.",
        ),
        # 推导：重物底面实际离地 h_真 = h_记 − Δ（例如从重物顶端量起），a_测 = 2h_记/t² = a·h_记/h_真：
        # 所有 a 按同一比例偏大。幅度上限 18 cm ≈ 6 个 50 g 钩码叠起来连挂钩约 12 cm + 缓冲垫约 5 cm。
        Cause(
            "atwood.wrong_distance", "下落高度量错了位置", "drop height measured to the wrong point",
            "下落高度没有从重物底面量到落地点，例如从重物顶端量起，或忘了减去地上缓冲垫的厚度，记录的高度比重物实际下落的距离长。",
            "The drop was not measured from the bottom of the mass to the landing point (e.g. measured from the "
            "top of the mass, or ignoring the cushion on the floor), so the recorded height is longer than the real drop.",
            "y_scale", ("slope_high",),
            magnitude=Prior.uniform(3.0, 18.0, 25),
            magnitude_label_zh="多量的高度", magnitude_unit="cm",
            fix_zh="从重物底面量到它实际落到的位置（缓冲垫上表面）。",
            fix_en="Measure from the bottom of the mass to where it actually lands (top of the cushion).",
        ),
    )


class Atwood(Experiment):
    id = "atwood"
    name_zh = "阿特伍德机验证牛顿第二定律"
    name_en = "Atwood machine: Newton's second law"
    held_out = True
    transfer = "near"

    obs_kinds = _obs_kinds("stopwatch")
    nuisance = {}
    linear = ()
    causes = _causes("stopwatch")

    protocols = {
        "single": Protocol("single", "单次计时", "single timing", (("main", 30.0),)),
        "standard": Protocol(
            "standard", "常见方案（5 个质量差，各测 4 次）", "common worksheet (5 mass differences × 4 timings)",
            tuple(("main", dm) for dm in (10.0, 20.0, 30.0, 40.0, 50.0) for _ in range(4)),
        ),
    }

    menu = (
        MenuItem("remeasure_h", "从重物底面量到落地点，重测下落高度", "re-measure the drop height properly",
                 2.0, (("remeasure_h", 0.0),)),
        MenuItem("photogate_run", "Δm = 30 g 时用电磁铁释放、光电门计时一次", "one photogate-timed run at Δm = 30 g",
                 5.0, (("photogate_run", 30.0),)),
        MenuItem("thin_thread", "换成很细的尼龙线，Δm = 10 g 时计时 3 次", "switch to thin thread, time 3 runs at Δm = 10 g",
                 5.0, (("thin_thread", 10.0),) * 3),
        MenuItem("repeat_10", "Δm = 10 g 时再计时 3 次", "three more timings at Δm = 10 g", 3.0, (("main", 10.0),) * 3),
    )

    def __init__(self, total_mass_g: float = 600.0, drop_m: float = 0.8, timer: str = "stopwatch"):
        if timer not in TIMERS:
            raise ValueError(f"timer must be one of {tuple(TIMERS)}")
        self.total_mass_g = float(total_mass_g)
        self.drop_m = float(drop_m)
        self.timer = timer
        self.obs_kinds = _obs_kinds(timer)
        self.causes = _causes(timer)

    def relevant_nuisance(self, cause_id, kinds):
        return ()

    def _fall_time(self, kind: str, dm_g: np.ndarray, params, cause_id: str) -> np.ndarray:
        """真实下落时间 t（不含计时偏差），形状 (P, n)。"""
        Mt = self.total_mass_g / 1000.0
        h = self.drop_m
        dm = dm_g[None, :] / 1000.0
        mag = params["mag"]  # (P, 1)
        if cause_id == "atwood.string_mass" and kind != "thin_thread":
            lam = mag / 1000.0  # kg/m
            Meff = Mt + lam * (h + 2 * PULLEY_ABOVE_START)
            A = np.maximum((dm - lam * h) * G / Meff, 1e-4)
            B = 2 * lam * G / Meff
            return np.arccosh(1.0 + B * h / A) / np.sqrt(B)
        if cause_id == "atwood.pulley_inertia":
            a = dm * G / (Mt + mag / 1000.0)
        elif cause_id == "atwood.pulley_friction":
            a = np.maximum((dm * G - mag) / Mt, 1e-4)  # Δm g ≤ f 时实际不动；先验保证不会发生
        else:
            a = np.broadcast_to(dm * G / Mt, mag.shape[:1] + dm.shape[1:])
        if cause_id == "atwood.wrong_distance":
            h = h - mag / 100.0
        return np.sqrt(2 * h / a)

    def predict(self, kind, settings, order, params, cause_id):
        P = params["mag"].shape[0]
        n = settings.size
        L = len(self.linear_params(cause_id))
        U = np.zeros((P, n, 1, L))
        if kind in ("main", "thin_thread", "photogate_run"):
            t = np.broadcast_to(self._fall_time(kind, settings, params, cause_id), (P, n))
            if cause_id == "atwood.timing_offset" and kind != "photogate_run":
                U[:, :, 0, 0] = 1.0
            return t[:, :, None], U
        if kind == "remeasure_h":
            c = np.zeros((P, n))
            if cause_id == "atwood.wrong_distance":
                c[:] = params["mag"]
            return c[:, :, None], U
        raise KeyError(kind)

    def linearise(self, ds: Dataset) -> LinearData:
        idx = [i for i, o in enumerate(ds.observations) if o.kind == "main"]
        dm = np.array([ds.observations[i].setting for i in idx])
        t = np.array([ds.observations[i].values[0] for i in idx])
        x = dm / 1000.0 * G
        y = 2 * self.drop_m / t**2
        r = self.obs_kinds["main"].readings[0]
        sd_t = float(np.sqrt(r.sigma**2 + r.resolution**2 / 12))
        # σ_a = |da/dt|·σ_t = (4h/t³)σ_t。t 用同一 Δm 各次计时的平均值：用每次自己的 t 定权重，
        # 测得偏慢（a 偏小）的点权重偏大，斜率系统偏低约 3%（干净数据上 z 偏 −0.75）
        t_bar = np.array([t[dm == d].mean() for d in dm])
        sy = 4 * self.drop_m * sd_t / t_bar**3
        Mt = self.total_mass_g / 1000.0
        return LinearData(x, y, sy, np.array(idx), "质量差产生的力 Δm·g (N)", "加速度 a = 2h/t² (m/s²)",
                          1.0 / Mt, 0.0)
