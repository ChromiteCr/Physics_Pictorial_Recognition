"""探究影响滑动摩擦力大小的因素（初中，八年级下册"摩擦力"）。

学生的做法（人教版默认，"standard" 方案）：木块放在水平长木板上，用弹簧测力计水平拉木块，
使它沿木板做匀速直线运动；由二力平衡，测力计示数 F 等于滑动摩擦力 f。木块重 2 N，
依次在木块上加 0~4 个 1 N 的钩码，压力 N = 2、3、4、5、6 N，各读一次示数。
"single" 方案：只在 N = 4 N 读一次。
改进方案（菜单项 fixed_scale）：把测力计水平固定住、拉住木块，抽动下面的木板；木块相对地面静止，
示数稳定，而且不要求木板匀速（滑动摩擦力与相对速度无关）。

面板坐标：横轴 = 压力 N（N，由木块和钩码的重力算出，视为准确），纵轴 = 测力计示数 F（N）。
理想直线 F = μ·N，截距 0，斜率 μ 自由（线性参数）。

仪器：弹簧测力计分度值默认 0.1 N（2.5 N 量程；构造参数 scale_division 可改为 0.2 N），
初中读到分度值。手拉时指针随木块的"走走停停"抖动，读数随机误差取 0.05 N；固定测力计时指针稳定，取 0.02 N。

几个原因都是"纯比例"的（斜拉、加速拉、木板不水平）：斜率自由时它们只是让 μ 看起来偏大或偏小，
标准方案的数据里看不出来；要靠 fixed_scale（去掉拉动方式的影响）或 reverse_pull（掉头拉，暴露木板倾斜）来区分。
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

SIGMA_HAND = 0.05  # 手拉时指针抖动（N）
SIGMA_FIXED = 0.02  # 固定测力计时（N）
SIGMA_ZERO = 0.02  # 不挂物体时读零点（N）


def _obs_kinds(division: float) -> dict[str, ObsKind]:
    def force(key, label, sigma):
        return Reading(key, label, key, "N", resolution=division, sigma=sigma)

    return {
        "main": ObsKind(
            "main", "手拉木块匀速运动，读测力计示数", "pull the block steadily by hand, read the scale",
            (force("pull_force", "拉力（测力计示数）", SIGMA_HAND),),
            setting_label_zh="压力（木块与钩码总重）", setting_label_en="normal force", setting_unit="N",
        ),
        "reverse": ObsKind(
            "reverse", "木块掉头，沿木板反方向匀速拉动，读示数", "pull the block the other way along the board",
            (force("pull_force_rev", "反向拉力（测力计示数）", SIGMA_HAND),),
            setting_label_zh="压力（木块与钩码总重）", setting_label_en="normal force", setting_unit="N",
        ),
        "fixed": ObsKind(
            "fixed", "测力计水平固定、抽动木板，读示数", "fix the scale, pull the board out, read the scale",
            (force("fixed_force", "测力计示数（木块静止）", SIGMA_FIXED),),
            setting_label_zh="压力（木块与钩码总重）", setting_label_en="normal force", setting_unit="N",
        ),
        "zero_h": ObsKind(
            "zero_h", "测力计水平放置、不挂物体，读示数", "lay the scale horizontally with no load, read it",
            (force("zero_h", "水平空载示数", SIGMA_ZERO),),
        ),
    }


class Friction(Experiment):
    id = "friction"
    name_zh = "探究影响滑动摩擦力大小的因素"
    name_en = "Sliding friction vs normal force"

    obs_kinds = _obs_kinds(0.1)

    # 拉角原因里读数对 μ 非线性，μ 改放网格（只对该原因生效，见 relevant_nuisance）
    nuisance = {"mu_pull": Prior.normal(0.30, 0.07, 141)}

    # 木块与木板间的动摩擦因数：0.09~0.51（±3σ）
    linear = (LinearParam("mu", 0.30, 0.07, "动摩擦因数 μ", ""),)

    causes = (
        Cause(
            "friction.random_only", "只有随机误差", "random error only",
            "测量步骤正确，数据偏离只来自读数的随机误差。",
            "The procedure is correct; deviations come only from random reading error.",
            "none", ("none",),
            fix_zh="多测几组、取平均即可。", fix_en="Take more readings and average.",
        ),
        # 竖直调零：挂钩、拉杆、指针和弹簧下半段（约一半弹簧质量）的重力 w 已把弹簧拉长 w/k，
        # 指针在这时被调到 0。横放后这部分重力不再沿弹簧轴向，弹簧缩回 w/k，
        # 于是拉力为 F 时示数 = F − w（水平空载时指针在零刻度线上方，读作 −w）。
        # 每个读数都小 w → 截距 −w；w 是读数里的常数项，做线性参数。
        Cause(
            "friction.zero_vertical", "竖直调零后水平使用", "zeroed vertically, used horizontally",
            "测力计是竖直挂着调的零，实验时却水平拉；竖直时挂钩、拉杆和弹簧自身的重力把弹簧拉长了一点，"
            "这一点在调零时被当成了零，横过来以后这部分重力不再拉弹簧。",
            "The spring scale was zeroed hanging vertically but used horizontally; when vertical, the weight "
            "of the hook, pull rod and spring itself stretches the spring a little, and that stretch was "
            "taken as zero.",
            "y_offset", ("intercept_neg",),
            linear=(LinearParam("hook_weight", 0.15, 0.05, "挂钩、拉杆等的有效重力", "N"),),
            fix_zh="在水平方向上给测力计调零（或先水平放置检查指针是否指零）再拉木块。",
            fix_en="Zero the scale in the horizontal position before pulling.",
        ),
        # 斜向上拉（与水平成 α）：木块匀速，F cos α = μ(N − F sin α) → F = μN / (cos α + μ sin α)。
        # 与 N 成正比、过原点，只是比例变了：斜率自由时与 μ 完全混淆。
        # cos α + μ sin α = √(1+μ²)·cos(α − arctan μ)：0 < α < 2·arctan μ 时示数偏小，
        # α = arctan μ 时最小，最多小 1 − 1/√(1+μ²)（μ = 0.3 时约 4%）；α 更大时反而偏大（45° 时约 +9%）。
        # 只有 fixed_scale（绳子水平、测力计固定）测出的才是真正的 μN，角度大时能看出差别。
        Cause(
            "friction.pull_angle", "拉力斜向上", "pulled at an upward angle",
            "拉测力计的手比木块高，拉力斜向上而不是沿水平方向；拉力的一部分向上提木块，"
            "木块对木板的压力变小，同时只有水平分力在克服摩擦。",
            "The hand was higher than the block, so the pull pointed upward instead of horizontally; part "
            "of the pull lifts the block (less normal force) and only the horizontal part overcomes friction.",
            "y_scale", ("none",),
            magnitude=Prior.uniform(5.0, 45.0, 9),
            magnitude_label_zh="拉力与水平方向的夹角", magnitude_unit="°",
            fix_zh="让测力计和细线与木板平行，沿水平方向拉。",
            fix_en="Keep the scale and string parallel to the board and pull horizontally.",
        ),
        # 没有匀速（读数时木块在加速）：F − μN = m·a，m = N/g → F = N(μ + a/g)。
        # a 每次差不多时仍是纯比例（μ 看起来偏大 a/g），标准数据看不出；
        # 固定测力计的方法里木块静止，示数 = μN，与木板快慢无关，由此可以分开。
        Cause(
            "friction.accelerating", "没有匀速拉动", "not pulled at constant speed",
            "读数时木块还在越拉越快，拉力除了克服摩擦，还要让木块加速。",
            "The block was still speeding up when the scale was read, so the pull both overcame friction "
            "and accelerated the block.",
            "y_scale", ("none",),
            magnitude=Prior.uniform(0.2, 1.0, 9),
            magnitude_label_zh="读数时木块的加速度", magnitude_unit="m/s²",
            fix_zh="等木块匀速运动后再读数；更好的办法是固定测力计、抽动木板。",
            fix_en="Read only once the block moves steadily, or fix the scale and pull the board instead.",
        ),
        # 木板不水平（倾角 β，β > 0 表示向高处拉）：F = μ W cos β + W sin β（W = 记录的"压力"，其实是重力）。
        # 掉头反向拉：F' = μ W cos β − W sin β。两个方向都与 W 成正比，各自都是纯比例；
        # 两次之差 = 2W sin β，只有 reverse_pull 能暴露。固定测力计的方法里木块相对木板的运动方向不变，仍有 W sin β。
        Cause(
            "friction.board_tilt", "木板没有放水平", "board not level",
            "长木板（或桌面）有一点倾斜，木块是沿着一个很缓的斜坡被拉动的；重力沿木板方向有一个分力，"
            "向高处拉时要多用力，向低处拉时少用力。",
            "The board (or table) is slightly tilted, so the block is pulled along a gentle slope; a part "
            "of its weight acts along the board, adding to or subtracting from the pull.",
            "y_scale", ("none",),
            magnitude=Prior.sym_uniform(1.0, 6.0, 12),
            magnitude_label_zh="木板倾角（正 = 向高处拉）", magnitude_unit="°",
            fix_zh="用水平仪或小球检查木板是否水平；或沿两个方向各拉一次取平均。",
            fix_en="Level the board (spirit level or a marble), or pull both ways and average.",
        ),
        Cause(
            "friction.reading_scatter", "指针还在晃动时读数", "read while the pointer was moving",
            "手拉时木块走走停停，测力计指针一直在晃，学生在指针晃动时随手读了一个数。",
            "The block moved in jerks and the pointer kept swinging; the reading was taken at a random "
            "moment of the swing.",
            "scatter", ("scatter_excess",),
            magnitude=Prior.uniform(0.10, 0.35, 10),
            magnitude_label_zh="额外读数误差", magnitude_unit="N",
            fix_zh="尽量匀速拉，指针稳定后再读；或改用固定测力计、抽动木板的方法。",
            fix_en="Pull steadily and read when the pointer is still, or fix the scale and pull the board.",
        ),
    )

    protocols = {
        "single": Protocol("single", "单次读数（压力 4 N）", "single reading (4 N)", (("main", 4.0),)),
        "standard": Protocol(
            "standard", "课本标准方案（加 0~4 个钩码，5 组压力）", "textbook worksheet (5 normal forces)",
            tuple(("main", w) for w in (2.0, 3.0, 4.0, 5.0, 6.0)),
        ),
        "fixed_scale": Protocol(
            "fixed_scale", "改进方案：固定测力计、抽动木板（5 组压力）", "fixed scale, pull the board (5 forces)",
            tuple(("fixed", w) for w in (2.0, 3.0, 4.0, 5.0, 6.0)),
        ),
    }

    menu = (
        MenuItem("fixed_scale", "改用固定测力计、抽动木板的方法，在 2 N、4 N、6 N 压力下各读一次",
                 "fix the scale and pull the board out at 2, 4, 6 N", 6.0,
                 (("fixed", 2.0), ("fixed", 4.0), ("fixed", 6.0))),
        MenuItem("reverse_pull", "木块掉头，沿木板反方向匀速拉动，在 2 N、4 N、6 N 压力下各读一次",
                 "pull the block the other way along the board at 2, 4, 6 N", 4.0,
                 (("reverse", 2.0), ("reverse", 4.0), ("reverse", 6.0))),
        MenuItem("zero_h", "把测力计水平放置、不挂物体，读示数", "read the scale lying horizontally with no load",
                 1.0, (("zero_h", 0.0),)),
        MenuItem("repeat_4", "压力 4 N 时再拉两次", "repeat the 4 N pull twice", 2.0,
                 (("main", 4.0), ("main", 4.0))),
    )

    def __init__(self, scale_division: float = 0.1):
        self.scale_division = float(scale_division)
        if self.scale_division != 0.1:
            self.obs_kinds = _obs_kinds(self.scale_division)

    def draw_truth(self, rng, cause_id, magnitude=None, nuisance=None, linear=None):
        t = super().draw_truth(rng, cause_id, magnitude, nuisance, linear)
        # 网格上的 μ（拉角原因用）与线性参数 μ 取同一个真值
        if nuisance and "mu_pull" in nuisance:
            t["linear"]["mu"] = t["nuisance"]["mu_pull"]
        else:
            t["nuisance"]["mu_pull"] = t["linear"]["mu"]
        return t

    def relevant_nuisance(self, cause_id, kinds):
        return ("mu_pull",) if cause_id == "friction.pull_angle" else ()

    def predict(self, kind, settings, order, params, cause_id):
        P = params["mag"].shape[0]
        n = settings.size
        L = len(self.linear_params(cause_id))
        U = np.zeros((P, n, 1, L))
        c = np.zeros((P, n, 1))
        name = cause_id.split(".", 1)[1]
        if kind == "zero_h":
            if name == "zero_vertical":
                U[:, :, 0, 1] = -1.0
            return c, U
        if kind not in ("main", "reverse", "fixed"):
            raise KeyError(kind)
        W = np.broadcast_to(settings[None, :], (P, n))  # 记录的压力（N）
        hand = kind != "fixed"  # 手拉（含反向）才受拉法影响
        if name == "pull_angle":
            mu = params["mu_pull"]
            if hand:
                a = np.deg2rad(params["mag"])
                c[:, :, 0] = mu * W / (np.cos(a) + mu * np.sin(a))
            else:
                c[:, :, 0] = mu * W
            return c, U
        u = W
        if name == "board_tilt":
            b = np.deg2rad(params["mag"])
            u = W * np.cos(b)
            c[:, :, 0] = (-1.0 if kind == "reverse" else 1.0) * W * np.sin(b)
        elif name == "accelerating" and hand:
            c[:, :, 0] = W * params["mag"] / G
        U[:, :, 0, 0] = u
        if name == "zero_vertical":
            U[:, :, 0, 1] = -1.0  # 所有水平读数都小 w
        return c, U

    def reading_sigma(self, kind, params, cause_id):
        base = super().reading_sigma(kind, params, cause_id)
        if kind in ("main", "reverse") and cause_id == "friction.reading_scatter":
            return np.sqrt(base**2 + params["mag"][:, :, None] ** 2)
        return base

    def linearise(self, ds: Dataset) -> LinearData:
        idx = [i for i, o in enumerate(ds.observations) if o.kind == "main"]
        x = np.array([ds.observations[i].setting for i in idx])
        y = np.array([ds.observations[i].values[0] for i in idx])
        r = self.obs_kinds["main"].readings[0]
        sy = np.full(y.shape, float(np.sqrt(r.sigma**2 + r.resolution**2 / 12)))
        return LinearData(x, y, sy, np.array(idx), "压力 N (N)", "拉力 F (N)", None, 0.0)
