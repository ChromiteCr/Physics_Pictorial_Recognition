"""胡克定律：探究弹簧弹力与形变量的关系（样板实验，其余实验照此实现）。

学生的做法（人教版默认）：弹簧竖直悬挂，先读原长 L0；挂钩上依次加 50~250 g 钩码，
读现长 L，记伸长量 x = L − L0，拉力 F = m·g。理想情况 F 与 x 成正比、过原点。

面板坐标：横轴 = 记录的拉力 F（N，由钩码质量算出，视为准确），纵轴 = 记录的伸长量 x（cm，有读数误差）。
理想直线 x = (100/k)·F，截距 0，斜率自由（k 未知）。
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

F_REF = 0.25 * G  # 250 g 钩码的拉力，作为"超出弹性限度"弯曲的参考力

EXT = Reading("extension", "伸长量", "extension", "cm", resolution=0.1, sigma=0.05)


class Hooke(Experiment):
    id = "hooke"
    name_zh = "探究弹簧弹力与形变量的关系（胡克定律）"
    name_en = "Hooke's law: spring force vs extension"

    obs_kinds = {
        "main": ObsKind(
            "main", "挂钩码读伸长量", "hang masses, read extension", (EXT,),
            setting_label_zh="钩码质量", setting_label_en="slotted mass", setting_unit="g",
        ),
        "hanger_mass": ObsKind(
            "hanger_mass", "用天平称挂钩质量", "weigh the hanger on a balance",
            (Reading("hanger_mass", "挂钩质量", "hanger mass", "g", resolution=0.1, sigma=0.1),),
        ),
        "zero_check": ObsKind(
            "zero_check", "取下钩码后重读原长，与原记录比较", "re-read the unloaded length",
            (Reading("zero_diff", "原长原记录 − 重读值", "recorded L0 − re-read L0", "cm",
                     resolution=0.1, sigma=0.05),),
        ),
    }

    nuisance = {"hanger_g": Prior.uniform(5.0, 50.0, 181)}

    # 1/k（m/N）：k 大约在 12~65 N/m（±3σ）
    linear = (LinearParam("inv_k", 0.05, 0.0115, "1/k", "m/N"),)

    causes = (
        Cause(
            "hooke.random_only", "只有随机误差", "random error only",
            "测量步骤正确，数据偏离只来自读数的随机误差。",
            "The procedure is correct; deviations come only from random reading error.",
            "none", ("none",),
            fix_zh="多测几组、取平均即可。", fix_en="Take more readings and average.",
        ),
        Cause(
            "hooke.hanger_omitted", "漏算挂钩质量", "hanger mass left out",
            "钩码挂在一个有质量的挂钩上；原长是在挂钩挂上之前量的，算拉力时却只用了钩码的质量，没把挂钩的质量算进去。",
            "The masses hang on a hanger that has its own mass; the unloaded length was read "
            "before the hanger was attached, but the force was computed from the slotted masses only.",
            "x_offset", ("intercept_pos",),
            magnitude=Prior.point(1.0),
            fix_zh="称出挂钩质量，拉力按（钩码 + 挂钩）的总质量计算；或挂上挂钩后再量原长。",
            fix_en="Weigh the hanger and include it in the force, or read the unloaded length "
            "with the hanger attached.",
        ),
        Cause(
            "hooke.zero_offset", "原长读数有偏差", "unloaded length misread",
            "读弹簧原长时视线没有与刻度平齐，或看错了刻度线，之后每个伸长量都用这个有偏差的原长去减。",
            "The unloaded length was misread (parallax or wrong mark), and every extension was "
            "computed from that wrong value.",
            "y_offset", ("intercept_pos", "intercept_neg"),
            linear=(LinearParam("zero_offset", 0.0, 0.4, "原长读数偏差", "cm"),),
            fix_zh="取下钩码重读原长，眼睛与弹簧下端指针平齐。",
            fix_en="Re-read the unloaded length with your eye level with the pointer.",
        ),
        Cause(
            "hooke.total_length", "记录了总长而不是伸长量", "total length recorded",
            "表格里记的是弹簧挂重物后的总长度，没有减去原长。",
            "The table records the total length of the stretched spring instead of the "
            "extension (the unloaded length was never subtracted).",
            "y_offset", ("intercept_pos",),
            linear=(LinearParam("unloaded_length", 12.0, 2.0, "弹簧原长", "cm"),),
            fix_zh="伸长量 = 挂钩码后的长度 − 原长。",
            fix_en="Extension = stretched length − unloaded length.",
        ),
        Cause(
            "hooke.beyond_limit", "超出弹性限度", "beyond the elastic limit",
            "所挂钩码太重，弹簧被拉过了弹性限度，大拉力时弹簧变得更容易被拉长。",
            "The load was too large; past its elastic limit the spring stretches more easily "
            "at large forces.",
            "bend", ("curve_up",),
            magnitude=Prior.uniform(0.05, 0.5, 46),
            magnitude_label_zh="大拉力处的额外伸长比例", magnitude_unit="",
            fix_zh="减小钩码总质量，只用弹性限度内的数据作图。",
            fix_en="Use smaller loads and plot only data within the elastic limit.",
        ),
        Cause(
            "hooke.reading_scatter", "读数时弹簧没有静止", "read while still oscillating",
            "弹簧还在上下振动时就读数，或者每次读数时眼睛的高度都不一样。",
            "Readings were taken while the spring was still bouncing, or from a different eye "
            "height each time.",
            "scatter", ("scatter_excess",),
            magnitude=Prior.uniform(0.15, 0.6, 46),
            magnitude_label_zh="额外读数误差", magnitude_unit="cm",
            fix_zh="等弹簧完全静止后再读数，视线与刻度平齐。",
            fix_en="Wait until the spring is still and keep your eye level with the scale.",
        ),
    )

    protocols = {
        "single": Protocol("single", "单次读数", "single reading", (("main", 200.0),)),
        "standard": Protocol(
            "standard", "课本标准方案（5 组钩码）", "textbook worksheet (5 loads)",
            tuple(("main", m) for m in (50.0, 100.0, 150.0, 200.0, 250.0)),
        ),
    }

    menu = (
        MenuItem("weigh_hanger", "用天平称挂钩质量", "weigh the hanger", 2.0, (("hanger_mass", 0.0),)),
        MenuItem("zero_check", "取下钩码，重读原长", "re-read the unloaded length", 2.0, (("zero_check", 0.0),)),
        MenuItem("zero_load", "只挂挂钩、不挂钩码，读伸长量", "read the extension with the hanger only",
                 1.0, (("main", 0.0),)),
        MenuItem("heavy_load", "再挂一组更重的钩码（350 g）", "add a heavier load (350 g)", 2.0, (("main", 350.0),)),
        MenuItem("repeat_150", "在 150 g 处重复读一次", "repeat the 150 g reading", 1.0, (("main", 150.0),)),
    )

    def relevant_nuisance(self, cause_id, kinds):
        if cause_id == "hooke.hanger_omitted" or "hanger_mass" in kinds:
            return ("hanger_g",)
        return ()

    def predict(self, kind, settings, order, params, cause_id):
        P = params["mag"].shape[0]
        n = settings.size
        L = len(self.linear_params(cause_id))
        U = np.zeros((P, n, 1, L))
        if kind == "main":
            m = settings[None, :]  # g
            F = m / 1000.0 * G
            if cause_id == "hooke.hanger_omitted":
                F_true = (m + params["hanger_g"]) / 1000.0 * G
            else:
                F_true = np.broadcast_to(F, (P, n))
            u = 100.0 * F_true  # cm·N/m：伸长量(cm) = (1/k) × 100 F
            if cause_id == "hooke.beyond_limit":
                u = u * (1.0 + params["mag"] * (F / F_REF) ** 2)
            U[:, :, 0, 0] = np.broadcast_to(u, (P, n))
            if cause_id == "hooke.zero_offset":
                U[:, :, 0, 1] = -1.0  # 原长记大了 δ → 伸长量都小 δ
            if cause_id == "hooke.total_length":
                U[:, :, 0, 1] = 1.0
            return np.zeros((P, n, 1)), U
        if kind == "hanger_mass":
            return np.broadcast_to(params["hanger_g"][:, :, None], (P, n, 1)), U
        if kind == "zero_check":
            if cause_id == "hooke.zero_offset":
                U[:, :, 0, 1] = 1.0
            return np.zeros((P, n, 1)), U
        raise KeyError(kind)

    def reading_sigma(self, kind, params, cause_id):
        base = super().reading_sigma(kind, params, cause_id)
        if kind == "main" and cause_id == "hooke.reading_scatter":
            return np.sqrt(base**2 + params["mag"][:, :, None] ** 2)
        return base

    def linearise(self, ds: Dataset) -> LinearData:
        idx = [i for i, o in enumerate(ds.observations) if o.kind == "main"]
        m = np.array([ds.observations[i].setting for i in idx])
        x = m / 1000.0 * G
        y = np.array([ds.observations[i].values[0] for i in idx])
        sy = np.full(y.shape, float(np.sqrt(EXT.sigma**2 + EXT.resolution**2 / 12)))
        return LinearData(x, y, sy, np.array(idx), "拉力 F (N)", "伸长量 x (cm)", None, 0.0)
