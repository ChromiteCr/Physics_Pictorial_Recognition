"""牛顿第二定律：探究加速度与力的关系（小车 + 槽码，保持小车质量不变）。

学生的做法（人教版必修一第四章"实验：探究加速度与力、质量的关系"，阻力补偿法）：
长木板一端装定滑轮，另一端垫高；先不挂槽码，让拖着纸带的小车在木板上做匀速运动（平衡摩擦力）；
再用细线跨过定滑轮挂上小盘和槽码，总质量 m 依次取 10~50 g，小车（含车上砝码）质量 M 用天平称出、保持不变。
用打点计时器打出纸带，逐差法求加速度 a（也可选光电门）。拉力近似取 F = m·g。

面板坐标：横轴 = 记录的拉力 F = m·g（N，视为准确），纵轴 = 记录的加速度 a（m/s²）。
理想直线 a = F/M，截距 0，斜率 1/M 已知（M 由构造参数给出，相当于学生称出并填写的值）。

建模约定（单故障世界，与胡克定律样板一致）：
- "只有随机误差" = 课本理想模型 a = F/M 成立（细线拉力 = mg、无未补偿阻力、滑轮和车轮转动惯量可忽略）。
- 每个原因 = 理想模型 + 一个被忽略的物理效应或一个操作错误。
- 精确的连接体动力学 a = mg/(M + m) 单独作为"槽码质量没有远小于小车质量"这一原因。

仪器：
- 打点计时器（默认）：T = 0.1 s 的计数点，6 段逐差 a = [(x4+x5+x6) − (x1+x2+x3)]/(9T²)；
  点位读数 σ_p ≈ 0.4 mm → σ_a = √6·σ_p/(9T²) ≈ 0.011 m/s²，再加小车运动本身的起伏，取 σ_a = 0.015 m/s²，记录到 0.01 m/s²。
- 光电门（timer="photogate"）：遮光条宽 b，两门间距 d，a = (v2² − v1²)/(2d)，计时分辨率 0.1 ms；取 σ_a = 0.006 m/s²，记录到 0.001。
  细节：b/Δt 是通过时间中点的速度，v̄² = v中位² − a²τ²/4（τ 为挡光时间），代入后 a 的相对偏差 ≈ a(τ1² − τ2²)/(8d) ~ 1e-4，可忽略。
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
    # timer: (σ_a, 记录分辨率)
    "ticker": (0.015, 0.01),
    "photogate": (0.006, 0.001),
}

STRING_SPAN_CM = 40.0  # 量细线高度差时，小车挂钩到滑轮顶端的水平距离


def _obs_kinds(timer: str) -> dict[str, ObsKind]:
    sig, res = TIMERS[timer]
    acc = Reading("acceleration", "加速度", "acceleration", "m/s²", resolution=res, sigma=sig)
    return {
        "main": ObsKind(
            "main", "挂槽码测小车加速度", "hang slotted masses, measure the cart's acceleration", (acc,),
            setting_label_zh="小盘和槽码总质量", setting_label_en="hanging mass", setting_unit="g",
        ),
        "balance_check": ObsKind(
            "balance_check", "取下细线，轻推小车，测小车自己的加速度",
            "unhook the string, nudge the cart, measure its own acceleration",
            (Reading("free_acc", "不挂槽码时小车的加速度", "acceleration with no hanging mass", "m/s²",
                     resolution=res, sigma=sig),),
        ),
        "string_height": ObsKind(
            "string_height", "用刻度尺量细线两端离木板的高度差",
            "measure the height difference of the string ends above the board",
            (Reading("string_rise", "细线两端离木板的高度差", "height difference of string ends", "cm",
                     resolution=0.1, sigma=0.2),),
        ),
    }


class Newton2(Experiment):
    id = "newton2"
    name_zh = "探究加速度与力的关系（牛顿第二定律）"
    name_en = "Newton's second law: acceleration vs force"

    obs_kinds = _obs_kinds("ticker")
    nuisance = {}
    linear = ()

    causes = (
        Cause(
            "newton2.random_only", "只有随机误差", "random error only",
            "已平衡摩擦力，细线与木板平行，槽码质量远小于小车质量，数据偏离只来自测加速度的随机误差。",
            "Friction is compensated, the string is parallel to the board and the hanging mass is much "
            "smaller than the cart; deviations come only from random error in the measured acceleration.",
            "none", ("none",),
            fix_zh="多测几组、取平均即可。", fix_en="Take more runs and average.",
        ),
        # 推导：沿板方向 M a = T + (Mg sinθ − f)。未平衡时 F_res = Mg sinθ − f < 0，T ≈ mg
        # → a = (mg + F_res)/M：拉力要先超过 |F_res| 才有加速度，F 轴上截距为 −F_res > 0。
        Cause(
            "newton2.unbalanced", "没有平衡摩擦力或平衡不足", "friction not (fully) compensated",
            "木板没有垫高或垫得太低，小车受到的摩擦力和纸带阻力没有被重力沿斜面的分力抵消，"
            "细线拉力有一部分用来克服阻力。",
            "The board was not raised, or not raised enough, so rolling friction and tape drag are not "
            "cancelled by the component of gravity along the board; part of the pull goes into overcoming them.",
            "x_offset", ("intercept_neg",),
            linear=(LinearParam("f_unbal", -0.05, 0.015, "未被抵消的阻力（负值）", "N"),),
            fix_zh="不挂槽码，把木板不带滑轮的一端垫高，直到轻推后拖着纸带的小车做匀速运动（纸带点迹间隔相等）。",
            fix_en="Without the hanging mass, raise the far end of the board until the cart (pulling its "
            "tape) moves at constant speed after a nudge.",
        ),
        # 推导：垫得太高时 F_res = Mg sinθ − f > 0 → a = (mg + F_res)/M，不挂槽码小车也会加速。
        Cause(
            "newton2.overbalanced", "平衡摩擦力过度", "friction over-compensated",
            "木板垫得太高，重力沿斜面的分力超过了阻力，即使不挂槽码，小车也会沿木板加速下滑。",
            "The board was raised too much: the component of gravity along it exceeds friction, so the "
            "cart speeds up down the board even with no hanging mass.",
            "x_offset", ("intercept_pos",),
            linear=(LinearParam("f_over", 0.05, 0.015, "多出来的下滑力", "N"),),
            fix_zh="把垫块往外移一点，减小倾角，直到轻推小车后纸带点迹间隔相等。",
            fix_en="Lower the raised end until the nudged cart moves at constant speed.",
        ),
        # 推导：小车 M a = T，槽码 m a = mg − T → a = mg/(M + m)，T = mg·M/(M + m) < mg。
        # 相对偏差 1 − a/(F/M) = m/(M + m)，随 m 增大而增大：图线斜率偏小，m 接近 M/5 以上时末端明显下弯。
        Cause(
            "newton2.heavy_hanging", "槽码质量没有远小于小车质量", "hanging mass not much less than cart mass",
            "槽码和小车一起加速，槽码自身也要一部分重力来获得加速度，细线拉力比槽码的重力小；"
            "槽码越重，拉力和槽码重力差得越多。",
            "The hanging mass accelerates together with the cart, so part of its weight goes into "
            "accelerating itself; the string tension is less than its weight, more so for heavier loads.",
            "bend", ("slope_low", "curve_down"),
            fix_zh="在小车上加砝码增大 M，或减小槽码质量，使 m 不超过 M 的十分之一；也可改用力传感器直接测细线拉力。",
            fix_en="Add mass to the cart or use lighter hanging masses (m ≤ M/10), or measure the tension "
            "directly with a force sensor.",
        ),
        # 推导：细线与板成 φ 角，沿板拉力 T cosφ；垂直分量 T sinφ 使正压力减小，阻力少 μT sinφ
        # （μ ≈ 0.01 的滚动阻力，二阶小量，略去）。T ≈ mg → a = mg·cosφ/M：斜率乘 cosφ。
        Cause(
            "newton2.string_angle", "细线与木板不平行", "string not parallel to the board",
            "定滑轮的高度没有调好，细线斜着拉小车，只有沿木板方向的分力使小车加速。",
            "The pulley height was not adjusted, so the string pulls the cart at an angle and only its "
            "component along the board accelerates the cart.",
            "x_scale", ("slope_low",),
            magnitude=Prior.uniform(8.0, 25.0, 18),
            magnitude_label_zh="细线与木板的夹角", magnitude_unit="°",
            fix_zh="调节定滑轮的高度，使细线与木板平行。",
            fix_en="Adjust the pulley height so that the string is parallel to the board.",
        ),
        # 推导：滑轮两侧拉力差带动滑轮转动并克服轴摩擦：(T1 − T2) = (I/r²)·a + f_p；
        # 车轮转动同理并入等效质量 J = Σ I_i/r_i²。mg − T1 ≈ 0（理想近似 T1 ≈ mg），T2 = M a
        # → a = (mg − f_p)/(M + J)：斜率按 M/(M + J) 变小，且拉力要先克服 f_p。
        Cause(
            "newton2.pulley_drag", "滑轮转动惯量和轴摩擦", "pulley inertia and axle friction",
            "定滑轮较重或轴不灵活：细线要带动滑轮（以及小车车轮）转起来，还要克服滑轮轴的摩擦；"
            "平衡摩擦力时细线没有挂在滑轮上，这部分阻力没有被补偿。",
            "The pulley is heavy or its axle is stiff: the string must spin up the pulley (and the cart's "
            "wheels) and overcome axle friction, which the friction-compensation step never included.",
            "x_offset", ("intercept_neg",),
            magnitude=Prior.uniform(5.0, 25.0, 16),
            magnitude_label_zh="滑轮和车轮的等效质量 I/r²", magnitude_unit="g",
            linear=(LinearParam("f_pulley", 0.02, 0.006, "滑轮轴摩擦力", "N"),),
            fix_zh="换用轻质、转动灵活的滑轮（如带轴承的铝滑轮），并检查滑轮轴是否卡滞。",
            fix_en="Use a light, free-running pulley (e.g. with a ball bearing) and check the axle.",
        ),
    )

    protocols = {
        "single": Protocol("single", "单次测量", "single run", (("main", 30.0),)),
        "standard": Protocol(
            "standard", "课本标准方案（5 组槽码）", "textbook worksheet (5 hanging masses)",
            tuple(("main", m) for m in (10.0, 20.0, 30.0, 40.0, 50.0)),
        ),
    }

    menu = (
        MenuItem("balance_check", "取下细线，轻推小车，用纸带测小车自己的加速度",
                 "unhook the string, nudge the cart and measure its own acceleration", 3.0,
                 (("balance_check", 0.0),)),
        MenuItem("string_height", "用刻度尺量细线两端离木板的高度差", "measure how far the string is from parallel",
                 1.0, (("string_height", 0.0),)),
        MenuItem("repeat_10", "在 10 g 处重复测一次", "repeat the 10 g run", 2.0, (("main", 10.0),)),
        MenuItem("repeat_50", "在 50 g 处重复测一次", "repeat the 50 g run", 2.0, (("main", 50.0),)),
    )

    def __init__(self, cart_mass_g: float = 500.0, timer: str = "ticker"):
        if timer not in TIMERS:
            raise ValueError(f"timer must be one of {tuple(TIMERS)}")
        self.cart_mass_g = float(cart_mass_g)
        self.timer = timer
        self.obs_kinds = _obs_kinds(timer)

    @property
    def M(self) -> float:
        return self.cart_mass_g / 1000.0

    def relevant_nuisance(self, cause_id, kinds):
        return ()

    def predict(self, kind, settings, order, params, cause_id):
        P = params["mag"].shape[0]
        n = settings.size
        L = len(self.linear_params(cause_id))
        U = np.zeros((P, n, 1, L))
        c = np.zeros((P, n))
        M = self.M
        if kind == "main":
            m = settings[None, :] / 1000.0  # kg
            F = m * G  # (1, n)
            if cause_id in ("newton2.unbalanced", "newton2.overbalanced"):
                c[:] = F / M
                U[:, :, 0, 0] = 1.0 / M
            elif cause_id == "newton2.heavy_hanging":
                c[:] = F / (M + m)
            elif cause_id == "newton2.string_angle":
                c[:] = np.cos(np.radians(params["mag"])) * F / M
            elif cause_id == "newton2.pulley_drag":
                Mj = M + params["mag"] / 1000.0  # (P, 1)
                c[:] = F / Mj
                U[:, :, 0, 0] = np.broadcast_to(-1.0 / Mj, (P, n))
            else:
                c[:] = F / M
            return c[:, :, None], U
        if kind == "balance_check":
            # 不挂细线：a0 = F_res/M；滑轮不参与，其他原因下为 0
            if cause_id in ("newton2.unbalanced", "newton2.overbalanced"):
                U[:, :, 0, 0] = 1.0 / M
            return c[:, :, None], U
        if kind == "string_height":
            if cause_id == "newton2.string_angle":
                c[:] = STRING_SPAN_CM * np.tan(np.radians(params["mag"]))
            return c[:, :, None], U
        raise KeyError(kind)

    def linearise(self, ds: Dataset) -> LinearData:
        idx = [i for i, o in enumerate(ds.observations) if o.kind == "main"]
        m = np.array([ds.observations[i].setting for i in idx])
        x = m / 1000.0 * G
        y = np.array([ds.observations[i].values[0] for i in idx])
        r = self.obs_kinds["main"].readings[0]
        sy = np.full(y.shape, float(np.sqrt(r.sigma**2 + r.resolution**2 / 12)))
        return LinearData(x, y, sy, np.array(idx), "拉力 F = mg (N)", "加速度 a (m/s²)", 1.0 / self.M, 0.0)
