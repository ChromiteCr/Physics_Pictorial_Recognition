"""杠杆平衡条件：一侧挂钩码、另一侧用弹簧测力计竖直向下拉（留出实验，远迁移）。

学生的做法（人教版八年级下册"探究杠杆的平衡条件"，默认方案）：
1. 调节杠杆两端的平衡螺母，使杠杆不挂钩码时在水平位置平衡；
2. 在支点左侧 L2 = 10 cm 处挂 4 个 50 g 钩码（阻力 F2 = 1.96 N）；
3. 弹簧测力计（先在竖直向下的使用方向上调零）挂在支点右侧 L1 = 5、10、15、20、25 cm 处，
   竖直向下拉，使杠杆在水平位置平衡，读示数 F1；
4. 比较 F1·L1 与 F2·L2。理想情况 F1 = F2·L2 / L1。

面板坐标：横轴 = 1/L1（1/m，L1 由杠杆上的刻度读出，视为准确），纵轴 = 测力计示数 F1（N）。
理想直线 F1 = (F2·L2)·(1/L1)：斜率已知 = F2·L2（N·m），截距 0。

为什么要"在水平位置"平衡：力竖直时两侧力臂都按 cosφ 缩短，倾角 φ 对平衡没有影响；
水平只是让力臂能直接从杠杆刻度上读出，并让调平衡时消掉杠杆自重的力矩。
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

MASS_EACH_G = 50.0  # 每个钩码 50 g
RIDER_G = 20.0  # 检查是否调平用的小钩码 20 g
L_REF = 10.0  # cm：支点摩擦"死区"的参考力臂（标准方案各 L1 的 1/L1 均方根约对应 9.2 cm）
F_DZ0 = 0.03  # N：正常杠杆在 L_REF 处的摩擦死区半宽（已含在读数随机误差里）
SIGMA_F = 0.04  # N：测力计读数随机误差（指针估读 + 外壳内摩擦 + 正常的支点摩擦）
SCALE_MAX = 5.0  # N：测力计量程


def _obs_kinds(division: float) -> dict[str, ObsKind]:
    f1 = Reading("force", "测力计示数", "spring-scale reading", "N", resolution=division, sigma=SIGMA_F)
    return {
        "main": ObsKind(
            "main", "测力计挂在右侧、竖直向下拉到杠杆水平，读示数", "pull the scale down until level, read it",
            (f1,), setting_label_zh="动力臂 L1", setting_label_en="scale arm L1", setting_unit="cm",
        ),
        "main_far": ObsKind(
            "main_far", "把钩码移到左侧 20 cm 处，再拉到水平读示数", "load moved to 20 cm, read the scale",
            (Reading("force_far", "测力计示数（钩码在 20 cm）", "scale reading (load at 20 cm)", "N",
                     resolution=division, sigma=SIGMA_F),),
            setting_label_zh="动力臂 L1", setting_label_en="scale arm L1", setting_unit="cm",
        ),
        "balance": ObsKind(
            "balance", "取下钩码和测力计，用 20 g 小钩码找使杠杆水平的位置",
            "remove loads; find where a 20 g rider levels the bare lever",
            (Reading("rider_pos", "小钩码到支点的距离（右正左负）", "rider position (right +)", "cm",
                     resolution=0.5, sigma=1.0),),
        ),
        "angle": ObsKind(
            "angle", "用量角器量测力计与竖直方向的夹角", "measure the scale's tilt from vertical",
            (Reading("tilt", "测力计偏离竖直的角度（偏向支点为正）", "tilt from vertical (toward pivot +)", "°",
                     resolution=1.0, sigma=2.0),),
        ),
        "arm": ObsKind(
            "arm", "用刻度尺从转轴中心量到右侧（测力计一侧）15 cm 孔，与记录的力臂比较",
            "measure from the axle centre to the 15 cm hole on the scale side",
            (Reading("arm_diff", "实际距离 − 记录的力臂", "true − recorded arm", "cm",
                     resolution=0.1, sigma=0.1),),
        ),
        "zero": ObsKind(
            "zero", "测力计挂钩朝上竖直拿着、不受拉力，读指针（零刻度线以上记负）",
            "hold the scale hook-up as used, unloaded, read the pointer",
            (Reading("zero_reading", "不受力时的示数", "unloaded reading", "N",
                     resolution=division, sigma=0.02),),
        ),
        "two_way": ObsKind(
            "two_way", "在 L1 = 10 cm 处，从小往大、从大往小各拉到刚好水平，读两次之差",
            "at 10 cm approach level from below and from above; record the difference",
            (Reading("dead_zone", "两次示数之差（大 − 小）", "upper − lower reading", "N",
                     resolution=division, sigma=SIGMA_F * math.sqrt(2.0)),),
        ),
    }


class Lever(Experiment):
    id = "lever"
    name_zh = "探究杠杆的平衡条件"
    name_en = "Lever balance condition"
    held_out = True
    transfer = "far"

    obs_kinds = _obs_kinds(0.1)
    nuisance = {}

    # 钩码实际力矩 / 标称力矩：钩码质量公差和挂孔位置公差合计约 1%
    linear = (LinearParam("load_factor", 1.0, 0.01, "实际阻力矩/标称阻力矩", ""),)

    causes = (
        Cause(
            "lever.random_only", "只有随机误差", "random error only",
            "步骤正确，数据偏离只来自测力计读数和支点处正常的小摩擦带来的随机误差。",
            "The procedure is correct; deviations come only from random reading error and the "
            "normal small friction at the pivot.",
            "none", ("none",),
            fix_zh="多测几组、取平均即可。", fix_en="Take more readings and average.",
        ),
        # 推导：杠杆重心偏离转轴，自重产生恒定力矩 τ0（偏向钩码一侧为正）。
        # 水平平衡：F1·L1 = F2·L2 + τ0 → F1 = (F2·L2 + τ0)/L1。
        # τ0 与 L1 无关，F1–1/L1 图仍是过原点直线，只是斜率由 F2·L2 变成 F2·L2 + τ0。
        Cause(
            "lever.unbalanced", "实验前杠杆没有调平衡", "lever not balanced beforehand",
            "挂钩码之前没有调节平衡螺母让杠杆在水平位置平衡，杠杆自身的重心不在转轴正上方或正下方，"
            "它的重力一直对转轴产生一个固定的转动作用。",
            "The balance nuts were not adjusted before loading, so the lever's own centre of mass is "
            "off the axle and its weight keeps exerting a fixed turning effect.",
            "y_scale", ("slope_high", "slope_low"),
            linear=(LinearParam("offset_torque", 0.0, 0.012, "杠杆自重的不平衡力矩", "N·m"),),
            fix_zh="取下钩码和测力计，先调节平衡螺母使杠杆在水平位置平衡，再重新实验。",
            fix_en="Remove the loads, level the bare lever with the balance nuts, then repeat.",
        ),
        # 推导：参考点相对转轴向测力计一侧偏了 p，记录 L1 = L1真 − p、L2 = L2真 + p。
        # 平衡：F1·(L1 + p) = F2·(L2 − p) → F1 = F2·(L2 − p)/(L1 + p)。
        # p > 0：斜率小于 F2·L2 且 L1 越小偏得越多（图线向下弯）；p < 0 相反。
        Cause(
            "lever.pivot_offset", "力臂没有从转轴中心量起", "arms measured from the wrong point",
            "量力臂时用的起点不是转轴中心：刻度尺零刻度线对着支架边缘，或者转轴并不在杠杆印的零刻度上，"
            "两侧的力臂都是从同一个错误的点量起的。",
            "The arms were measured from a point that is not the axle centre (ruler zero at the edge of "
            "the support, or the axle not at the lever's printed zero), the same wrong point for both sides.",
            # 次要特征（200 次按先验抽样）：p > 0 时约 60% 截距为正，p < 0 时约 70% 向上弯
            "x_offset", ("slope_low", "slope_high", "intercept_pos", "curve_up"),
            magnitude=Prior.sym_uniform(0.3, 1.5, 40),
            magnitude_label_zh="起点相对转轴的偏移（偏向测力计一侧为正）", magnitude_unit="cm",
            fix_zh="力臂一律从转轴中心量到力的作用线（挂点）；用杠杆刻度时先确认零刻度正对转轴。",
            fix_en="Measure every arm from the axle centre to the point where the force acts.",
        ),
        # 推导：拉力与竖直方向成 α，力臂 = 转轴到力的作用线的距离 = L1·cosα。
        # 平衡：F1·L1·cosα = F2·L2 → F1 = F2·L2/(L1·cosα)，各点同乘 1/cosα > 1。
        # （挂点与转轴等高，水平分力的作用线过转轴附近，力矩可忽略。）
        Cause(
            "lever.scale_angle", "测力计没有竖直向下拉", "scale pulled at an angle",
            "拉测力计时手偏向一侧，拉力方向和竖直方向有夹角，这时转轴到拉力作用线的距离"
            "比挂点到转轴的距离短。",
            "The scale was pulled at a slant, so the perpendicular distance from the axle to the line "
            "of the pull is shorter than the distance to the hook point.",
            "x_scale", ("slope_high",),
            magnitude=Prior.sym_uniform(8.0, 35.0, 28),
            magnitude_label_zh="拉力偏离竖直的角度（偏向支点为正）", magnitude_unit="°",
            fix_zh="测力计保持竖直向下拉，可以对照铅垂线或请同学从侧面检查。",
            fix_en="Keep the scale vertical; check against a plumb line or from the side.",
        ),
        # 推导：转轴摩擦力矩上限 τf，|F1·L1 − F2·L2| ≤ τf 时杠杆都能静止在水平位置，
        # 读数落在 F2·L2/L1 ± τf/L1 内的任意位置 → 近似均匀分布，标准差 = 半宽/√3。
        # 严格说半宽 ∝ 1/L1；引擎的散布不能随控制量变化，取 L1 = 10 cm 处的半宽作代表。
        Cause(
            "lever.pivot_friction", "支点摩擦太大", "too much friction at the pivot",
            "转轴生锈、弯曲或孔太紧，转动时摩擦较大：拉力在一个范围内变化时杠杆都能停在水平位置不动，"
            "每次停下时的拉力在这个范围里的位置都不一样。",
            "The axle is rusty, bent or tight, so the lever stays level over a whole range of pull; "
            "each time it stops at a different point within that range.",
            "scatter", ("scatter_excess",),
            magnitude=Prior.uniform(0.1, 0.35, 26),
            magnitude_label_zh="L1 = 10 cm 处摩擦死区的半宽", magnitude_unit="N",
            fix_zh="清洁或更换转轴、给转轴加一点润滑；读数前轻敲杠杆，让它自己停在水平位置。",
            fix_en="Clean or replace the axle; tap the lever gently before reading.",
        ),
        # 推导：测力计按正常悬挂（挂钩朝下）标定，示数 R = T − m杆g（T 为弹簧拉力，m杆 为挂钩和拉杆质量）。
        # 挂钩朝上倒过来向下拉杠杆时，杠杆受力 F1 = T + m杆g → R = F1 − 2m杆g；
        # 再加上指针本身的零点偏差 z，示数恒差 z0 = z − 2m杆g，与 L1 无关（图线整体平移）。
        Cause(
            "lever.scale_zero", "测力计没有在使用方向上调零", "scale not zeroed in the direction of use",
            "测力计是挂钩朝上倒过来向下拉的，但调零是在正常竖直悬挂（或平放）时做的，或者根本没有调零；"
            "倒过来以后挂钩和拉杆自身的重力不再由弹簧承担，指针不在零刻度线上。",
            "The scale is used upside down (hook up, pulled downward) but was zeroed hanging normally "
            "or lying flat, or not at all; once inverted the hook and rod no longer hang on the spring.",
            "y_offset", ("intercept_neg", "intercept_pos"),
            linear=(LinearParam("zero_shift", 0.0, 0.2, "测力计零点偏差", "N"),),
            fix_zh="测力计按实验时的方向（挂钩朝上、竖直）拿好，不受拉力时把指针调到零刻度线。",
            fix_en="Zero the scale held exactly as it is used (hook up, vertical), with no load.",
        ),
    )

    protocols = {
        "single": Protocol("single", "单次读数", "single reading", (("main", 20.0),)),
        "standard": Protocol(
            "standard", "课本标准方案（L1 = 5~25 cm 共 5 组）", "textbook worksheet (5 arms)",
            tuple(("main", L) for L in (5.0, 10.0, 15.0, 20.0, 25.0)),
        ),
    }

    menu = (
        MenuItem("balance_check", "取下钩码和测力计，用 20 g 小钩码检查杠杆是否调平",
                 "check the bare lever's balance with a 20 g rider", 3.0, (("balance", 0.0),)),
        MenuItem("move_load", "钩码移到左侧 20 cm 处，在 L1 = 10、20 cm 各读一次",
                 "move the load to 20 cm and read at L1 = 10, 20 cm", 3.0,
                 (("main_far", 10.0), ("main_far", 20.0))),
        MenuItem("angle_check", "量一下测力计与竖直方向的夹角", "measure the scale's tilt", 1.0, (("angle", 0.0),)),
        MenuItem("arm_check", "用刻度尺从转轴中心量到右侧（测力计一侧）15 cm 孔",
                 "measure the scale-side arm from the axle centre",
                 2.0, (("arm", 15.0),)),
        MenuItem("zero_check", "按使用方向拿着测力计，检查零点", "check the scale zero as used", 1.0, (("zero", 0.0),)),
        MenuItem("two_way", "在 10 cm 处从两个方向拉到水平，读两次之差",
                 "approach level from both directions at 10 cm", 2.0, (("two_way", 10.0),)),
    )

    def __init__(self, scale_division: float = 0.1, n_masses: int = 4, load_arm_cm: float = 10.0):
        """scale_division：测力计分度值（0.1 或 0.2 N）；n_masses：50 g 钩码个数；load_arm_cm：阻力臂。"""
        self.scale_division = float(scale_division)
        self.n_masses = int(n_masses)
        self.load_arm_cm = float(load_arm_cm)
        self.obs_kinds = _obs_kinds(self.scale_division)  # 实例自己的字典，不改类属性
        # 标准方案和"移动钩码"里最大的拉力都是 F2·L2/(5 cm)（竖直拉时）
        f_max = self.load_weight * self.load_arm_cm / 5.0
        if f_max > SCALE_MAX:
            raise ValueError(f"最大拉力约 {f_max:.1f} N，超过测力计量程 {SCALE_MAX} N")

    @property
    def load_weight(self) -> float:
        """阻力 F2 = n·m·g（N）。"""
        return self.n_masses * MASS_EACH_G / 1000.0 * G

    @property
    def ideal_torque(self) -> float:
        """标称阻力矩 F2·L2（N·m）。"""
        return self.load_weight * self.load_arm_cm / 100.0

    def _force_columns(self, cause_id, L1, L2, params, P, n):
        """测力计示数的 (c, U)：F1 = κ·F2·(L2 − p)/((L1 + p)cosα) + τ0/L1 + z0。"""
        L = len(self.linear_params(cause_id))
        U = np.zeros((P, n, 1, L))
        p = params["mag"] if cause_id == "lever.pivot_offset" else 0.0
        cos_a = np.cos(np.radians(params["mag"])) if cause_id == "lever.scale_angle" else 1.0
        L1t = L1 + p  # 真实动力臂（cm）
        L2t = L2 - p
        U[:, :, 0, 0] = np.broadcast_to(self.load_weight * L2t / (L1t * cos_a), (P, n))
        if cause_id == "lever.unbalanced":
            U[:, :, 0, 1] = np.broadcast_to(100.0 / L1, (P, n))  # τ0(N·m) / L1(m)
        if cause_id == "lever.scale_zero":
            U[:, :, 0, 1] = 1.0
        return np.zeros((P, n, 1)), U

    def predict(self, kind, settings, order, params, cause_id):
        P = params["mag"].shape[0]
        n = settings.size
        L = len(self.linear_params(cause_id))
        U = np.zeros((P, n, 1, L))
        c = np.zeros((P, n, 1))
        if kind == "main":
            return self._force_columns(cause_id, settings[None, :], self.load_arm_cm, params, P, n)
        if kind == "main_far":
            return self._force_columns(cause_id, settings[None, :], 2.0 * self.load_arm_cm, params, P, n)
        if kind == "balance":
            # 小钩码平衡位置 d = τ0 /(m_r·g)，换成 cm
            if cause_id == "lever.unbalanced":
                U[:, :, 0, 1] = 100.0 / (RIDER_G / 1000.0 * G)
            return c, U
        if kind == "angle":
            if cause_id == "lever.scale_angle":
                c = np.broadcast_to(params["mag"][:, :, None], (P, n, 1))
            return c, U
        if kind == "arm":
            if cause_id == "lever.pivot_offset":
                c = np.broadcast_to(params["mag"][:, :, None], (P, n, 1))
            return c, U
        if kind == "zero":
            if cause_id == "lever.scale_zero":
                U[:, :, 0, 1] = 1.0
            return c, U
        if kind == "two_way":
            # 从两侧逼近：读数差 = 死区全宽 = 2·半宽（L1 = 10 cm 时正好是幅度的定义处）
            half = params["mag"] if cause_id == "lever.pivot_friction" else np.full((P, 1), F_DZ0)
            c = (2.0 * half * (L_REF / settings[None, :]))[:, :, None]
            return c, U
        raise KeyError(kind)

    def reading_sigma(self, kind, params, cause_id):
        base = super().reading_sigma(kind, params, cause_id)
        if cause_id != "lever.pivot_friction":
            return base
        half = params["mag"][:, :, None]  # (P,1,1)，L_REF 处的死区半宽（N）
        if kind in ("main", "main_far"):
            return np.sqrt(base**2 + half**2 / 3.0)
        if kind == "balance":
            # 死区换成小钩码位置：τf/(m_r·g) = half·L_REF/(m_r·g)
            dz_cm = half * L_REF / (RIDER_G / 1000.0 * G)
            return np.sqrt(base**2 + dz_cm**2 / 3.0)
        return base

    def linearise(self, ds: Dataset) -> LinearData:
        idx = [i for i, o in enumerate(ds.observations) if o.kind == "main"]
        L1 = np.array([ds.observations[i].setting for i in idx])
        x = 100.0 / L1  # 1/m
        y = np.array([ds.observations[i].values[0] for i in idx])
        d = self.scale_division
        sy = np.full(y.shape, float(np.sqrt(SIGMA_F**2 + d**2 / 12)))
        return LinearData(x, y, sy, np.array(idx), "1/L1 (1/m)", "测力计示数 F1 (N)", self.ideal_torque, 0.0)
