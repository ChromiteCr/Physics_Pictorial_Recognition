"""通用关系族 + 通用故障算子："其他实验"通道和程序化任务族。

GenericExperiment 由一份类似实验卡的说明（spec）构造，不需要手写孪生：
  读数 y（纵轴）与控制量 x（横轴）满足  y = a·X + b，X = x^q（关系族决定 q）：
    proportional / linear: q = 1；inverse: q = −1；square: q = 2；sqrt: q = 1/2；power: q = spec["exponent"]
  linear 族的截距 b 默认未知；其他族默认 b = 0（spec 里可改）。
  面板坐标：横轴 X = x^q，纵轴 y，理想斜率 = spec["ideal_slope"]（None = 斜率自由），
  理想截距 = b（未知时用 NaN，面板就不判截距）。

通用原因（描述按变量名和仪器名模板生成，只写物理过程）：
  random_only、x_offset（x 的起点/零点错了，网格参数）、y_offset（y 仪器没归零，线性参数）、
  scale（y 仪器比例不准，线性参数；只在斜率已知时可辨识）、bend（大 x 时偏离理想规律，线性参数）、
  scatter（读数散布偏大，网格参数）、drift（条件随测量顺序漂移，线性参数）。
  标准方案下数据分不出来的原因（例如截距未知时 y_offset）直接不列入，并记在 ``omitted`` 里。

先验按本实例的噪声定标（不同定律的噪声/量程比差几个数量级）：
  网格下界 ≈ 2 个"可检出标准差"，线性参数的先验标准差 = 4 个标准误；"大故障"另要求面板 |z| ≥ 6；
  "可检出"指把共享线性参数（斜率未知时的斜率、截距未知时的截距）能吸收的部分投影掉以后剩下的偏离。

标准方案："去程测奇数点、回程测偶数点"的往返顺序，使测量顺序和 x 不共线，漂移才看得出来。

procedural_task_family(rng, n)：从一组真实的中学/大学预科定律（不含 12 个命名实验及其器材）里随机抽定律、
变量叫法、量程、仪器分辨率和措辞，返回 (GenericExperiment, 抽中的原因 id)，用于训练迁移。
"""

from __future__ import annotations

import math
import warnings
from typing import Iterator, Mapping, Sequence

import numpy as np

from .core import (
    Cause,
    Dataset,
    Experiment,
    LinearData,
    LinearParam,
    MenuItem,
    Observation,
    ObsKind,
    Prior,
    Protocol,
    Reading,
)

FAMILY_EXPONENT: dict[str, float | None] = {
    "proportional": 1.0,
    "linear": 1.0,
    "inverse": -1.0,
    "square": 2.0,
    "sqrt": 0.5,
    "power": None,
}

SLOPE_REL_SD_KNOWN = 0.01  # 斜率已知时，实际器材与理论值的相对差（器材参数公差）
SLOPE_REL_SD_FREE = 0.25  # 斜率未知时，先验相对标准差（围绕估计值）
Z_LOW = 2.0  # 网格下界：可检出偏离 ≈ 2σ
LIN_SD_Z = 4.0  # 线性参数先验标准差 = 4 个标准误
LARGE_LIN = 2.5  # "大故障" ≥ 2.5 个先验标准差（约 7.5σ）
Z_PANEL = 6.0  # "大故障"同时要求面板上 |z| ≥ 6

# ── 措辞模板（{x}/{y} 变量名，{xt}/{yt} 仪器名）──────────────────────────
# 每个故障：(名称 zh, 名称 en, [描述 zh…], [描述 en…], 修正 zh, 修正 en)
_TEMPLATES: dict[str, tuple] = {
    "random_only": (
        "只有随机误差", "random error only",
        ["操作没有问题，{y}与理想关系的偏差全部来自每次读数的随机起伏。",
         "各个环节都按要求完成了，{y}的偏离只是读数时不可避免的随机波动。",
         "装置和读数方法都正常，偏差来自{yt}读数时偶然的上下波动。"],
        ["Nothing went wrong; deviations of {y} come only from random reading-to-reading fluctuation.",
         "Every step was done properly; the scatter in {y} is the unavoidable random part of reading the {yt}."],
        "保持现有做法，增加测量次数求平均可以进一步减小随机误差。",
        "Keep the method; more repeats and averaging shrink the random error further.",
    ),
    "x_offset": (
        "{x}的起点有偏差", "{x} measured from a wrong origin",
        ["{x}不是从真正的起点量起的：{xt}的零点和实际起点没有对齐，所以每一个记下的{x}都和真实值差着同样大小的一段。",
         "确定{x}时参考位置选错了，每次设定的{x}都整体偏了一个固定量，偏差不随{x}的大小改变。",
         "{xt}自身有零点误差，读出的{x}总比实际多（或少）同一个数。"],
        ["{x} was not measured from the true origin: the zero of the {xt} was misaligned, so every recorded "
         "{x} differs from the real value by the same amount.",
         "A wrong reference point was used when setting {x}; every setting is shifted by one fixed amount."],
        "重新确定{x}真正的起点（零点），按实际值重新计算每一组的{x}。",
        "Find the true origin of {x} and recompute every setting.",
    ),
    "y_offset": (
        "{yt}没有归零", "{yt} not zeroed",
        ["{yt}在测量前没有归零，不加任何输入时它就已经显示一个不为零的数，之后每个{y}读数都带着这个数。",
         "读{y}时总是从一个错误的基准线开始读，所有读数整体抬高或压低了同样的量。",
         "{yt}的初始示数不在零上，记录的{y}都多出（或少了）一个固定值。"],
        ["The {yt} was not set to zero before measuring; it already showed a nonzero value with no input, "
         "and every {y} reading carries it.",
         "{y} was always read from a wrong baseline, shifting every reading by the same amount."],
        "测量前让{yt}在没有输入时归零，或者测出这个初始示数后从每个读数里减掉。",
        "Zero the {yt} with no input before measuring, or subtract its initial reading.",
    ),
    "scale": (
        "{yt}比例不准", "{yt} reads in the wrong proportion",
        ["{yt}的刻度不准（标定有误），读出的{y}都按同一个比例偏大或偏小。",
         "{yt}用错了量程或倍率，每个{y}读数都被乘上了同一个不等于 1 的系数。",
         "{yt}老化后灵敏度变了，同样大小的{y}显示出来按固定比例偏多或偏少。"],
        ["The {yt} is miscalibrated, so every {y} reading is too large or too small by the same ratio.",
         "The wrong range or multiplier was used on the {yt}; every {y} reading is multiplied by the same factor."],
        "用已知的标准量校准{yt}（或者换一台），按校准结果修正读数。",
        "Calibrate the {yt} against a known standard (or swap it) and correct the readings.",
    ),
    "bend": (
        "大{x}时偏离理想规律", "law breaks down at large {x}",
        ["{x}取得较大时，理想规律所依赖的近似条件不再成立，{y}偏离理想关系，而且{x}越大偏得越快。",
         "器材在{x}较大时超出了正常工作范围，出现了理想规律没有考虑的效应，偏差随{x}增大而迅速变大。",
         "某个在{x}小时可以忽略的因素，在{x}大时变得重要起来，使{y}与理想关系的偏差越来越大。"],
        ["At large {x} the approximation behind the ideal law no longer holds, and {y} departs from it "
         "faster and faster as {x} grows.",
         "At large {x} the apparatus leaves its normal working range, and an effect the ideal law ignores grows quickly."],
        "只在近似成立的范围内取数据，或者缩小{x}的取值范围后重新测量。",
        "Keep to the range where the approximation holds, or narrow the range of {x} and remeasure.",
    ),
    "scatter": (
        "{y}读数忽高忽低", "{y} readings unsteady",
        ["每次读{y}时操作都不一致：读数时机、视线角度或者装置是否稳定都在变，读数忽高忽低。",
         "{yt}在读数时还没有稳定下来（示数在跳动），每次读到的是不同时刻的值。",
         "实验环境有干扰（振动、气流等），{y}每次都被随机地推高或拉低一些。"],
        ["The {y} readings were taken inconsistently (timing, viewing angle, stability), so they jump up and down.",
         "The {yt} had not settled when it was read, so each value comes from a different moment."],
        "等示数稳定后再读，视线和读数方法保持一致，每组多读几次取平均。",
        "Wait for the display to settle, read the same way every time, and average repeats.",
    ),
    "drift": (
        "条件随时间漂移", "conditions drift during the run",
        ["实验过程中条件在慢慢变化（例如器材发热、电源电压逐渐下降），越往后测的{y}偏得越多。",
         "{yt}在使用过程中逐渐走偏，同样的输入，后测的读数和先测的不一样，偏差随测量顺序累积。",
         "环境（温度、湿度等）在实验过程中缓慢改变，{y}随时间一点点偏离，与{x}取多大无关。"],
        ["Conditions change slowly during the run (apparatus warming up, supply voltage sagging), so later "
         "{y} readings are shifted more.",
         "The {yt} slowly creeps while in use; the same input reads differently later, accumulating with order."],
        "缩短实验时间、让器材预热后再测；打乱{x}的测量顺序，最后重测第一组看是否回到原值。",
        "Warm the apparatus up first, shorten the run, shuffle the order and re-measure the first setting at the end.",
    ),
}

_MECH = {"random_only": "none", "x_offset": "x_offset", "y_offset": "y_offset", "scale": "y_scale",
         "bend": "bend", "scatter": "scatter", "drift": "drift"}


def _var(v, unit: str, res: float, symbol: str, default_tool: str, default_tool_en: str) -> dict:
    """变量说明统一成 dict：name, name_en, symbol, unit, tool, tool_en, resolution。"""
    if isinstance(v, str):
        v = {"name": v}
    out = {"name": v["name"], "name_en": v.get("name_en", v["name"]), "symbol": v.get("symbol", symbol),
           "unit": v.get("unit", unit), "tool": v.get("tool", default_tool),
           "tool_en": v.get("tool_en", default_tool_en), "resolution": float(v.get("resolution", res))}
    return out


def zigzag_order(n: int) -> list[int]:
    """往返顺序：去程测 0,2,4…，回程测 …5,3,1。"""
    return list(range(0, n, 2)) + list(range(n - 1 if (n - 1) % 2 == 1 else n - 2, 0, -2))


class GenericExperiment(Experiment):
    """由 spec 构造的通用实验。注册表只实例化类属性 id 非空的类，所以这里 id 留空（实例上再设）。

    spec 字段：
      id（默认 "generic"）, name, name_en,
      x_var / y_var：名字字符串，或 {name, name_en, symbol, unit, tool, tool_en, resolution}
      units: {"x": .., "y": ..}, resolutions: {"x": .., "y": ..}, sigma: {"x": .., "y": ..}（默认 = 分辨率/2）
      relation_family, exponent（power 族）, ideal_slope（None = 未知）, slope_guess（未知时的估计）,
      ideal_intercept（None = 未知；linear 族默认未知，其余默认 0）, intercept_guess, intercept_sd,
      settings（控制量取值）或 x_range + n_points（默认 6）, variants（可选：{原因后缀: 措辞序号}）,
      hints（可选：{原因后缀: 文字 或 (文字, ±1)}，带方向时该原因的先验只取这一侧）,
      exclude（可选：该实验物理上不会出现的原因后缀）
    构造后可查：omitted（未列入的原因及理由）、large（各原因"大故障"幅度）、fault_signs、signature_by_sign。
    """

    id = ""

    def __init__(self, spec: Mapping, rng: np.random.Generator | None = None):
        spec = dict(spec)
        self.spec = spec
        self.id = spec.get("id", "generic")
        self.name_zh = spec.get("name", "其他实验")
        self.name_en = spec.get("name_en", "custom experiment")
        fam = spec.get("relation_family", "proportional")
        if fam not in FAMILY_EXPONENT:
            raise ValueError(f"unknown relation_family {fam}")
        self.family = fam
        q = FAMILY_EXPONENT[fam]
        if q is None and "exponent" not in spec:
            raise ValueError("power 关系族需要给出 exponent")
        self.q = float(spec["exponent"]) if q is None else q
        units = spec.get("units", {})
        res = spec.get("resolutions", {})
        self.xv = _var(spec.get("x_var", "x"), units.get("x", ""), res.get("x", 0.1), "x", "量具", "instrument")
        self.yv = _var(spec.get("y_var", "y"), units.get("y", ""), res.get("y", 0.1), "y", "测量仪器", "meter")
        sig = spec.get("sigma", {})
        self.sigma_x = float(sig.get("x", self.xv["resolution"] / 2))
        self.sigma_y = float(sig.get("y", self.yv["resolution"] / 2))
        if "settings" in spec:
            xs = np.sort(np.asarray(spec["settings"], dtype=float))
        else:
            lo, hi = spec.get("x_range", (1.0, 6.0))
            xs = np.linspace(float(lo), float(hi), int(spec.get("n_points", 6)))
        if np.any(xs <= 0) and (self.q < 0 or self.q != int(self.q)):
            raise ValueError("该关系族要求控制量为正")
        self.xs = xs
        self.X = xs**self.q
        self.X_max = float(np.max(np.abs(self.X)))
        self.ideal_slope = spec.get("ideal_slope")
        self.slope_nom = float(self.ideal_slope if self.ideal_slope is not None else spec.get("slope_guess", 1.0))
        b_default = None if fam == "linear" else 0.0
        self.ideal_intercept = spec.get("ideal_intercept", b_default)
        self.b_nom = float(self.ideal_intercept if self.ideal_intercept is not None
                           else spec.get("intercept_guess", 0.0))
        self.sigma_tot = math.sqrt(self.sigma_y**2 + self.yv["resolution"] ** 2 / 12)

        # 共享线性参数：斜率（及未知截距）
        a_sd = abs(self.slope_nom) * (SLOPE_REL_SD_KNOWN if self.ideal_slope is not None else SLOPE_REL_SD_FREE)
        lin = [LinearParam("slope", self.slope_nom, a_sd, "斜率", "")]
        if self.ideal_intercept is None:
            y_span = abs(self.slope_nom) * float(np.ptp(self.X)) or 1.0
            lin.append(LinearParam("intercept", self.b_nom, float(spec.get("intercept_sd", y_span)), "截距", ""))
        self.linear = tuple(lin)
        self.nuisance = {}

        n = xs.size
        self.std_order = zigzag_order(n)
        self.protocols = {
            "single": Protocol("single", "单次读数", "single reading", (("main", float(xs[n // 2])),)),
            "standard": Protocol("standard", f"往返测 {n} 组", f"{n} settings, out-and-back order",
                                 tuple(("main", float(xs[i])) for i in self.std_order)),
        }
        self._build_obs_kinds()
        self._build_causes(rng)
        self._build_menu()

    # ---- 构造 ----

    def _fmt(self, s: str) -> str:
        return s.format(x=self.xv["name"], y=self.yv["name"], xt=self.xv["tool"], yt=self.yv["tool"])

    def _fmt_en(self, s: str) -> str:
        return s.format(x=self.xv["name_en"], y=self.yv["name_en"], xt=self.xv["tool_en"], yt=self.yv["tool_en"])

    def _build_obs_kinds(self):
        xu, yu = self.xv["unit"], self.yv["unit"]
        y_main = Reading("y", self.yv["name"], self.yv["name_en"], yu, self.yv["resolution"], self.sigma_y)
        self.obs_kinds = {
            "main": ObsKind("main", f"设定{self.xv['name']}，读{self.yv['name']}",
                            f"set {self.xv['name_en']}, read {self.yv['name_en']}", (y_main,),
                            setting_label_zh=self.xv["name"], setting_label_en=self.xv["name_en"], setting_unit=xu),
            "x_zero": ObsKind("x_zero", f"重新核对{self.xv['name']}的起点（实际值 − 记录值）",
                              f"re-check the origin of {self.xv['name_en']}",
                              (Reading("x_diff", f"{self.xv['name']}实际值 − 记录值", "true − recorded", xu,
                                       self.xv["resolution"], self.sigma_x),)),
            "y_zero": ObsKind("y_zero", f"不加输入时读{self.yv['tool']}的示数",
                              f"read the {self.yv['tool_en']} with no input",
                              (Reading("y_zero", f"{self.yv['tool']}空载示数", "no-input reading", yu,
                                       self.yv["resolution"], self.sigma_y),)),
            "y_calib": ObsKind("y_calib", f"用{self.yv['tool']}测一个已知的标准量",
                               f"measure a known standard with the {self.yv['tool_en']}",
                               (Reading("y_calib", "标准量的读数", "reading of the standard", yu,
                                        self.yv["resolution"], self.sigma_y),),
                               setting_label_zh="标准量的已知值", setting_label_en="known value", setting_unit=yu),
        }

    def _nuisance_cols(self) -> np.ndarray:
        """标准方案下共享线性参数里"自由"的列（斜率未知 → X 列；截距未知 → 常数列）。"""
        cols = []
        X = self.X[self.std_order]
        if self.ideal_slope is None:
            cols.append(X)
        if self.ideal_intercept is None:
            cols.append(np.ones_like(X))
        return np.column_stack(cols) if cols else np.zeros((X.size, 0))

    def _resid_norm(self, d: np.ndarray) -> float:
        """偏离 d（按标准方案顺序）去掉共享参数能吸收的部分后的范数。"""
        N = self._nuisance_cols()
        if N.shape[1]:
            d = d - N @ np.linalg.lstsq(N, d, rcond=None)[0]
        return float(np.linalg.norm(d))

    def _panel_z(self, ymean: np.ndarray) -> float:
        """无噪声均值（标准方案顺序）在面板上最显著的 |z|（只看面板能判的量：已知截距、已知斜率、曲率）。"""
        from ..panel import compute_panel

        obs = [Observation("main", float(x), (float(y),)) for x, y in zip(self.xs[self.std_order], ymean)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 无噪声数据残差全为 0，秩相关会报常数输入
            p = compute_panel(self.linearise(Dataset(self.id, obs)))
        zs = [abs(z) for z in (p.intercept_z, p.slope_ratio_z, p.curvature_z) if np.isfinite(z)]
        return max(zs) if zs else 0.0

    def _build_causes(self, rng):
        xs, X, order = self.xs[self.std_order], self.X[self.std_order], np.arange(len(self.std_order), dtype=float)
        a, s = self.slope_nom, self.sigma_tot
        variants = dict(self.spec.get("variants", {}))
        self.omitted: dict[str, str] = {}
        self.large: dict[str, dict] = {}  # 原因 → 大故障的幅度（正数；符号见 fault_signs）
        self.fault_signs: dict[str, tuple[int, ...]] = {}
        exclude = set(self.spec.get("exclude", ()))
        causes = []

        clean = a * X + self.b_nom

        def lin_sd(u):
            r = self._resid_norm(u)
            return None if r < 1e-9 * max(1.0, float(np.linalg.norm(u))) else LIN_SD_Z * s / r

        def large_lin(u, sd):
            """大故障：至少 2.5 个先验标准差，并且面板上 |z| ≥ 6（面板总是自由拟合斜率和截距）。"""
            z1 = self._panel_z(clean + sd * u) / sd
            return max(LARGE_LIN * sd, Z_PANEL / z1) if z1 > 0 else LARGE_LIN * sd

        for key in ("random_only", "x_offset", "y_offset", "scale", "bend", "scatter", "drift"):
            cid = f"{self.id}.{key}"
            mag, lin = Prior.point(0.0), ()
            if key in exclude:
                self.omitted[key] = "说明里排除：该实验的物理过程不会产生这类误差"
                continue
            # 物理提示可以带方向：(文字, +1/−1)。带方向时先验只取这一侧，保证"描述 → 特征"一致
            hint = self.spec.get("hints", {}).get(key)
            hint, sign = (hint, 0) if hint is None or isinstance(hint, str) else (hint[0], int(hint[1]))
            if key == "x_offset":
                eps = 1e-4 * float(np.ptp(self.xs))
                z1 = self._resid_norm(a * ((xs + eps) ** self.q - X)) / (s * eps)
                cap = 0.5 * float(np.ptp(self.xs))
                if self.q < 0 or self.q != int(self.q):
                    cap = min(cap, 0.6 * float(np.min(self.xs)))
                lo = Z_LOW / z1 if z1 > 0 else np.inf
                zp = self._panel_z(a * (xs + eps) ** self.q + self.b_nom) / eps
                hi = min(max(6 * lo, Z_PANEL / zp if zp > 0 else 0.0), cap)
                if not lo < 0.5 * hi:
                    self.omitted[key] = "共享参数（截距/斜率）能完全吸收，或在合理范围内太小，标准数据看不出来"
                    continue
                mag = (Prior.sym_uniform(lo, hi, 30) if sign == 0 else
                       Prior.uniform(lo, hi, 15) if sign > 0 else Prior.uniform(-hi, -lo, 15))
                self.large[cid] = {"magnitude": hi}
            elif key == "y_offset":
                sd = lin_sd(np.ones_like(X))
                if sd is None:
                    self.omitted[key] = "截距未知时，整体平移被截距完全吸收"
                    continue
                lin = (LinearParam("y_shift", 0.0, sd, "零点偏差", self.yv["unit"]),)
                big = large_lin(np.ones_like(X), sd)
            elif key == "scale":
                sd = lin_sd(a * X) if self.ideal_slope is not None else None
                if sd is None:
                    self.omitted[key] = "斜率未知时，比例误差被斜率完全吸收"
                    continue
                sd = max(sd, 3 * SLOPE_REL_SD_KNOWN)
                lin = (LinearParam("scale_ratio", 0.0, sd, "比例偏差", ""),)
                big = large_lin(a * X, sd)
            elif key == "bend":
                sd = lin_sd(a * X**2 / self.X_max)
                if sd is None:
                    self.omitted[key] = "点数太少，弯曲看不出来"
                    continue
                lin = (LinearParam("bend", 0.0, sd, "最大处的相对弯曲", ""),)
                big = large_lin(a * X**2 / self.X_max, sd)
            elif key == "scatter":
                mag = Prior.uniform(1.5 * s, 5.0 * s, 24)
                self.large[cid] = {"magnitude": 5.0 * s}
            elif key == "drift":
                sd = lin_sd(order)
                if sd is None:
                    self.omitted[key] = "测量顺序与控制量共线，漂移被斜率吸收"
                    continue
                lin = (LinearParam("drift", 0.0, sd, "每次测量的漂移", self.yv["unit"]),)
                big = LARGE_LIN * sd
            if lin:
                if sign:  # 单侧：均值放在 1.5 个标准差处（反号的抽样只占约 7%，而且幅度很小）
                    lin = (LinearParam(lin[0].name, sign * 1.5 * lin[0].sd, lin[0].sd, lin[0].label_zh, lin[0].unit),)
                self.large[cid] = {"linear": {lin[0].name: big}}
            self.fault_signs[cid] = (1, -1) if sign == 0 or key == "scatter" else (sign,)
            if key == "scatter":
                self.fault_signs[cid] = (1,)
            name_zh, name_en, dz, de, fz, fe = _TEMPLATES[key]
            if key in variants:
                vz = vez = int(variants[key])
            elif rng is not None:
                vz, vez = int(rng.integers(len(dz))), int(rng.integers(len(de)))
            else:
                vz = vez = 0
            desc_zh = hint if hint else self._fmt(dz[vz % len(dz)])
            causes.append(Cause(
                cid, self._fmt(name_zh), self._fmt_en(name_en), desc_zh, self._fmt_en(de[vez % len(de)]),
                _MECH[key], ("none",), magnitude=mag, fix_zh=self._fmt(fz), fix_en=self._fmt_en(fe),
                source="assumption", linear=lin,
            ))
        self.causes = tuple(causes)
        self.signature_by_sign: dict[str, dict[int, tuple[str, ...]]] = {}
        self.causes = tuple(self._with_signatures(c) for c in self.causes)

    def _with_signatures(self, cause: Cause) -> Cause:
        """预期特征：大故障（允许的每个符号）的无噪声均值算面板，按 z 取稳健特征。"""
        from dataclasses import replace

        from ..panel import compute_panel

        key = self.suffix(cause.id)
        if key == "random_only":
            return cause
        if key == "scatter":
            self.signature_by_sign[cause.id] = {1: ("scatter_excess",)}
            return replace(cause, signatures=("scatter_excess",))
        by_sign: dict[int, tuple[str, ...]] = {}
        for sign in self.fault_signs[cause.id]:
            kw = self.large[cause.id]
            truth = {"cause": cause.id, "nuisance": {}, "magnitude": sign * kw.get("magnitude", 0.0),
                     "linear": {"slope": self.slope_nom, "intercept": self.b_nom}}
            for k, v in kw.get("linear", {}).items():
                truth["linear"][k] = sign * v
            obs = []
            for i, (kind, x) in enumerate(self.protocols["standard"].plan):
                mean, _ = self._mean_and_sigma(truth, kind, np.array([x]), np.array([i]))
                obs.append(Observation(kind, x, (float(mean[0, 0]),)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = compute_panel(self.linearise(Dataset(self.id, obs)))
            sigs = []
            # 截距、斜率：无噪声 |z| ≥ 3 才算稳健特征；曲率 |z| ≥ 0.5 就列入——面板一旦判出弯曲就不再报截距，
            # 带噪声时弯曲有相当概率越过 2，会把截距特征"顶掉"
            if np.isfinite(p.intercept_z) and abs(p.intercept_z) >= 3:
                sigs.append("intercept_pos" if p.intercept_z > 0 else "intercept_neg")
            if np.isfinite(p.slope_ratio_z) and abs(p.slope_ratio_z) >= 3:
                sigs.append("slope_high" if p.slope_ratio > 1 else "slope_low")
            if np.isfinite(p.curvature_z) and abs(p.curvature_z) >= 0.5:
                sigs.append("curve_up" if p.curvature_z > 0 else "curve_down")
            # 漂移、残差偏大来自均值本身的形状；无噪声时"散布偏小"是假象，不算
            sigs += [t for t in p.signatures if t in ("drift", "scatter_excess")]
            by_sign[sign] = tuple(sigs) or ("none",)
        self.signature_by_sign[cause.id] = by_sign
        union = tuple(dict.fromkeys(t for v in by_sign.values() for t in v if t != "none")) or ("none",)
        return replace(cause, signatures=union)

    def _build_menu(self):
        n = self.xs.size
        mid = float(self.xs[n // 2])
        first = float(self.xs[self.std_order[0]])
        x_ext = float(self.xs[-1] * 1.3) if self.q > 0 else float(self.xs[0] * 0.75)
        y_ref = float(self.slope_nom * self.X[n // 2] + self.b_nom)
        d = self.yv["resolution"]
        y_ref = round(y_ref / d) * d if d > 0 else y_ref
        x, y, yt = self.xv["name"], self.yv["name"], self.yv["tool"]
        self.menu = (
            MenuItem("x_zero_check", f"重新核对{x}的起点", f"re-check the origin of {self.xv['name_en']}",
                     2.0, (("x_zero", 0.0),)),
            MenuItem("y_zero_check", f"不加输入时读一次{yt}", f"read the {self.yv['tool_en']} with no input",
                     1.0, (("y_zero", 0.0),)),
            MenuItem("y_calibrate", f"用{yt}测一个已知标准量", f"measure a known standard", 3.0,
                     (("y_calib", y_ref),)),
            MenuItem("repeat_mid", f"在中间的{x}处重复测 3 次", "repeat the middle setting 3 times", 3.0,
                     tuple(("main", mid) for _ in range(3))),
            MenuItem("repeat_first", "最后再重测第一组", "re-measure the first setting at the end", 1.0,
                     (("main", first),)),
            MenuItem("extend_range", f"把{x}再往外扩一组", "add one setting beyond the range", 2.0,
                     (("main", x_ext),)),
        )

    # ---- 引擎接口 ----

    def suffix(self, cause_id: str) -> str:
        return cause_id[len(self.id) + 1:]

    def _idx(self, cause_id: str, name: str) -> int:
        return [p.name for p in self.linear_params(cause_id)].index(name)

    def predict(self, kind, settings, order, params, cause_id):
        P = params["mag"].shape[0]
        n = settings.size
        key = self.suffix(cause_id)
        L = len(self.linear_params(cause_id))
        U = np.zeros((P, n, 1, L))
        c = np.zeros((P, n, 1))
        if kind == "main":
            x = np.broadcast_to(settings[None, :], (P, n))
            if key == "x_offset":
                x = x + params["mag"]  # 真实值 = 记录值 + δ
            X = np.power(x, self.q)
            U[:, :, 0, 0] = X
            if self.ideal_intercept is None:
                U[:, :, 0, 1] = 1.0
            else:
                c[:] = self.b_nom
            if key == "y_offset":
                U[:, :, 0, self._idx(cause_id, "y_shift")] = 1.0
            elif key == "scale":
                U[:, :, 0, self._idx(cause_id, "scale_ratio")] = self.slope_nom * X  # y·(1+r) ≈ y + r·a·X
            elif key == "bend":
                U[:, :, 0, self._idx(cause_id, "bend")] = self.slope_nom * X**2 / self.X_max
            elif key == "drift":
                U[:, :, 0, self._idx(cause_id, "drift")] = np.broadcast_to(order[None, :], (P, n))
            return c, U
        if kind == "x_zero":
            if key == "x_offset":
                c = np.broadcast_to(params["mag"][:, :, None], (P, n, 1)).copy()
            return c, U
        if kind == "y_zero":
            if key == "y_offset":
                U[:, :, 0, self._idx(cause_id, "y_shift")] = 1.0
            return c, U
        if kind == "y_calib":
            ref = np.broadcast_to(settings[None, :], (P, n))
            c = ref[:, :, None].copy()
            if key == "y_offset":
                U[:, :, 0, self._idx(cause_id, "y_shift")] = 1.0
            elif key == "scale":
                U[:, :, 0, self._idx(cause_id, "scale_ratio")] = ref
            return c, U
        raise KeyError(kind)

    def reading_sigma(self, kind, params, cause_id):
        base = super().reading_sigma(kind, params, cause_id)
        if kind == "main" and self.suffix(cause_id) == "scatter":
            return np.sqrt(base**2 + params["mag"][:, :, None] ** 2)
        return base

    def linearise(self, ds: Dataset) -> LinearData:
        idx = [i for i, o in enumerate(ds.observations) if o.kind == "main"]
        x = np.array([ds.observations[i].setting for i in idx], dtype=float)
        y = np.array([ds.observations[i].values[0] for i in idx], dtype=float)
        sy = np.full(y.shape, self.sigma_tot)
        xl = f"{self.xv['symbol']}^{self.q:g}" if self.q != 1 else self.xv["symbol"]
        unit = f"{self.xv['unit']}^{self.q:g}" if self.q != 1 else self.xv["unit"]
        b = float("nan") if self.ideal_intercept is None else float(self.ideal_intercept)
        return LinearData(np.power(x, self.q), y, sy, np.array(idx), f"{xl} ({unit})",
                          f"{self.yv['name']} {self.yv['symbol']} ({self.yv['unit']})", self.ideal_slope, b)


def spec_from_card(card: Mapping, **overrides) -> dict:
    """实验卡 → GenericExperiment 的 spec（取第一个 control 变量当 x、第一个 measured 变量当 y）。

    id 默认加前缀 other_，避免原因 id 和手写孪生（如 pendulum.x_offset）撞名。
    """
    ctrl = next((v for v in card["variables"] if v["role"] == "control"), None)
    meas = next((v for v in card["variables"] if v["role"] == "measured"), None)
    if ctrl is None or meas is None:
        raise ValueError(f"实验卡 {card.get('id')} 缺少 control 或 measured 变量")
    spec = {
        "id": f"other_{card['id']}", "name": card["name_zh"], "name_en": card.get("name_en", card["id"]),
        "x_var": {"name": ctrl["name"], "symbol": ctrl["symbol"], "unit": ctrl["unit"]},
        "y_var": {"name": meas["name"], "symbol": meas["symbol"], "unit": meas["unit"]},
        "relation_family": card["relation_family"], "ideal_slope": None,
    }
    if "exponent" in card:  # power 族的指数（卡片可选字段）
        spec["exponent"] = card["exponent"]
    spec.update(overrides)
    return spec


# ── 程序化任务族 ──────────────────────────────────────────────────────────
# 每条定律：key, 名称, x/y（叫法候选、符号、单位、仪器、分辨率候选、仪器噪声）, 关系族, 指数,
# 斜率范围（真实物理量算出）, 斜率已知的概率, 截距（None = 未知，给出估计范围）, x 取值范围, 物理提示（可选）。
# 不含 12 个命名实验，也不含弹簧测力计/钩码/下落/摆动/抛体/斜面类定律（避免和留出实验的器材与措辞重叠）。

_LAWS: tuple[dict, ...] = (
    dict(key="density", name=("探究同种物质的质量与体积的关系", "mass vs volume of one substance"),
         x=(["体积", "样品体积"], ["volume", "sample volume"], "V", "cm³", ("量筒", "measuring cylinder"), (1.0, 2.0), 0.5),
         y=(["质量", "样品质量"], ["mass", "sample mass"], "m", "g", ("电子天平", "electronic balance"), (0.1, 0.2), 0.05),
         family="proportional", slope=[0.79, 1.0, 2.7, 7.9, 8.9], p_known=0.6, xr=(10.0, 60.0),
         exclude=("bend",),
         hints={"x_offset": ("读量筒时把液面以下的样品托架体积也算进去了，每次记录的体积都多出同样的一小块。", -1)}),
    dict(key="liquid_pressure", name=("探究液体内部压强与深度的关系", "liquid pressure vs depth"),
         x=(["探头所在深度", "金属盒的深度"], ["probe depth", "depth"], "h", "cm", ("刻度尺", "ruler"), (0.1, 0.5), 0.1),
         y=(["U 形管液面高度差", "压强计液面差"], ["manometer height difference", "U-tube difference"], "Δh", "cm",
            ("U 形管压强计", "U-tube manometer"), (0.1, 0.5), 0.1),
         family="proportional", slope=(0.85, 1.25), p_known=0.5, xr=(2.0, 15.0),
         hints={"x_offset": ("深度是从容器口量下去的，而不是从液面量起，每个深度都多算了液面到容器口的那一段。", -1),
                "scale": "U 形管里装的液体不是水，密度不同，液面高度差与压强之间的换算比例变了。"}),
    dict(key="ohm", name=("探究电流与电压的关系", "current vs voltage for a resistor"),
         x=(["电阻两端电压", "电压"], ["voltage across the resistor", "voltage"], "U", "V", ("电压表", "voltmeter"), (0.1,), 0.03),
         y=(["通过电阻的电流", "电流"], ["current", "current through the resistor"], "I", "A", ("电流表", "ammeter"), (0.02, 0.01), 0.005),
         family="proportional", slope=[0.2, 0.1, 1 / 15, 0.05], p_known=0.5, xr=(0.5, 3.0),
         hints={"drift": ("电阻通电后慢慢发热，阻值随温度升高而变大，越往后测电流越偏小。", -1)}),
    dict(key="wire_length", name=("探究导体电阻与长度的关系", "resistance vs wire length"),
         x=(["接入电路的合金丝长度", "导线长度"], ["wire length in circuit", "wire length"], "L", "cm", ("刻度尺", "ruler"), (0.1, 0.5), 0.1),
         y=(["合金丝电阻", "电阻"], ["wire resistance", "resistance"], "R", "Ω", ("欧姆表", "ohmmeter"), (0.1, 0.01), 0.03),
         family="proportional", slope=(0.02, 0.1), p_known=0.0, xr=(20.0, 100.0), exclude=("bend",),
         hints={"x_offset": "夹子夹在合金丝上的位置不是刻度尺的零点，每段接入电路的长度都多算或少算了同一截。"}),
    dict(key="wire_area", name=("探究导体电阻与横截面积的关系", "resistance vs cross-section"),
         x=(["合金丝横截面积", "横截面积"], ["cross-sectional area", "wire cross-section"], "S", "mm²", ("螺旋测微器", "micrometer"), (0.01,), 0.005),
         y=(["合金丝电阻", "电阻"], ["wire resistance", "resistance"], "R", "Ω", ("欧姆表", "ohmmeter"), (0.1, 0.01), 0.03),
         family="inverse", slope=(0.5, 3.0), p_known=0.0, xr=(0.1, 0.8), exclude=("bend",)),
    dict(key="expansion", name=("测量金属杆的线膨胀", "linear thermal expansion of a rod"),
         x=(["温度升高量", "升温"], ["temperature rise", "heating above room temperature"], "Δt", "°C", ("温度计", "thermometer"), (1.0, 0.5), 0.3),
         y=(["金属杆伸长量", "杆的伸长"], ["rod elongation", "extension of the rod"], "ΔL", "mm", ("千分表", "dial gauge"), (0.01, 0.001), 0.003),
         family="proportional", slope=[0.006, 0.0085, 0.0115], p_known=0.5, xr=(10.0, 70.0),
         hints={"drift": ("支架本身也在缓慢受热伸长，千分表的读数随加热时间额外多累积了一点。", 1),
                "bend": ("温度较高时金属的膨胀系数略有增大，伸长量比正比关系增长得更快。", 1)}),
    dict(key="heating", name=("探究水升高的温度与加热时间的关系", "water temperature vs heating time"),
         x=(["加热时间", "通电时间"], ["heating time", "time switched on"], "t", "min", ("计时器", "timer"), (0.1,), 0.02),
         y=(["水温", "水的温度"], ["water temperature", "temperature of the water"], "T", "°C", ("温度计", "thermometer"), (1.0, 0.5), 0.3),
         family="linear", slope=(1.5, 6.0), p_known=0.0, intercept=(12.0, 28.0), xr=(1.0, 6.0),
         hints={"bend": ("水温升高后向周围散热越来越快，后面每分钟升高的温度变少。", -1)}),
    dict(key="joule", name=("探究电流产生的热量与电流的关系", "heating vs current (Joule)"),
         x=(["电流", "通过电阻丝的电流"], ["current", "heater current"], "I", "A", ("电流表", "ammeter"), (0.02,), 0.005),
         y=(["煤油升高的温度", "温升"], ["kerosene temperature rise", "temperature rise"], "Δt", "°C", ("温度计", "thermometer"), (0.5, 1.0), 0.2),
         family="square", slope=(3.0, 12.0), p_known=0.0, xr=(0.3, 1.2)),
    dict(key="centripetal_w", name=("探究向心力与角速度的关系", "centripetal force vs angular speed"),
         x=(["角速度", "转动角速度"], ["angular speed", "rotation rate"], "ω", "rad/s", ("转速计", "tachometer"), (0.1,), 0.03),
         y=(["向心力", "所需向心力"], ["centripetal force", "required inward force"], "F", "N", ("力传感器", "force sensor"), (0.01,), 0.01),
         family="square", slope=(0.005, 0.04), p_known=0.5, xr=(2.0, 8.0),
         hints={"bend": ("角速度大时物块被甩得向外挪了一点，实际转动半径变大，所需的力比理想值增长得快。", 1)}),
    dict(key="string", name=("探究弦振动频率与弦长的关系", "string frequency vs length"),
         x=(["弦的振动长度", "振动部分的弦长"], ["vibrating length", "string length"], "L", "cm", ("刻度尺", "ruler"), (0.1, 0.5), 0.1),
         y=(["振动频率", "频率"], ["frequency", "vibration frequency"], "f", "Hz", ("频率计", "frequency meter"), (1.0,), 1.0),
         family="inverse", slope=(5000.0, 20000.0), p_known=0.0, xr=(20.0, 80.0),
         hints={"x_offset": "弦长是从琴码边缘量的，而弦真正的振动端点在琴码顶部的接触点，每个长度都差了同一小段。"}),
    dict(key="illuminance", name=("探究照度与到光源距离的关系", "illuminance vs distance"),
         x=(["到灯的距离", "照度计与光源的距离"], ["distance to lamp", "sensor distance"], "d", "cm", ("刻度尺", "ruler"), (0.5, 0.1), 0.2),
         y=(["照度", "光照强度"], ["illuminance", "light level"], "E", "lx", ("照度计", "lux meter"), (1.0,), 2.0),
         family="power", exponent=-2.0, slope=(5e5, 3e6), p_known=0.0, xr=(25.0, 80.0),
         hints={"x_offset": ("距离量到了照度计外壳前沿，而感光元件在外壳里面，每个距离都少算了同样一段。", 1),
                "y_offset": ("房间里还有窗外的杂散光，没有扣掉，每个照度读数都多了差不多相同的一份。", 1),
                "bend": ("距离很近时灯不能再看成点光源，近处的照度比平方反比算出的偏小。", -1)}),
    dict(key="boyle", name=("探究气体等温变化时压强与体积的关系", "gas pressure vs volume at constant temperature"),
         x=(["注射器内气体体积", "气柱体积"], ["gas volume in the syringe", "air column volume"], "V", "mL", ("注射器刻度", "syringe scale"), (0.5, 1.0), 0.2),
         y=(["气体压强", "压强"], ["gas pressure", "pressure"], "p", "kPa", ("压强传感器", "pressure sensor"), (0.1,), 0.3),
         family="inverse", slope=(800.0, 2000.0), p_known=0.0, xr=(8.0, 20.0),
         hints={"x_offset": ("注射器与压强传感器之间的连接管里也有气体，这段体积没有算进去。", 1),
                "drift": ("推活塞太快、手握着注射器，气体温度慢慢升高，压强越往后越偏大。", 1)}),
    dict(key="resonance", name=("用共鸣管测声速：空气柱长度与频率", "resonance tube length vs frequency"),
         x=(["音叉频率", "声源频率"], ["tuning-fork frequency", "source frequency"], "f", "Hz", ("音叉标称值", "fork label"), (1.0,), 0.5),
         y=(["第一次共鸣的空气柱长度", "共鸣空气柱长度"], ["first resonance length", "air-column length"], "L", "cm", ("刻度尺", "ruler"), (0.1, 0.5), 0.3),
         family="inverse", slope=(8400.0, 8700.0), p_known=0.5, intercept=(-2.5, -0.5), xr=(256.0, 512.0),
         exclude=("x_offset", "bend")),
)


def _sample_law_spec(rng: np.random.Generator, law: dict, uid: str) -> dict:
    xn_zh, xn_en, xsym, xunit, (xt, xt_en), xres_c, xsig = law["x"]
    yn_zh, yn_en, ysym, yunit, (yt, yt_en), yres_c, ysig = law["y"]
    i, j = int(rng.integers(len(xn_zh))), int(rng.integers(len(yn_zh)))
    xres, yres = float(rng.choice(xres_c)), float(rng.choice(yres_c))
    lo, hi = law["xr"]
    span = hi - lo
    x_lo = lo + rng.uniform(0.0, 0.3) * span
    x_hi = hi - rng.uniform(0.0, 0.3) * span
    n = int(rng.integers(6, 9))
    xs = np.linspace(x_lo, x_hi, n)
    xs = np.unique(np.round(xs / xres) * xres)
    sl = law["slope"]
    slope = float(rng.choice(sl)) if isinstance(sl, list) else float(rng.uniform(*sl))
    fam = law["family"]
    q = law.get("exponent", FAMILY_EXPONENT[fam])
    # x 的读数误差折算到 y：σ_y² = σ_仪器² + (dy/dx·σ_x)²（取中点斜率）
    x_mid = float(np.median(xs))
    dydx = abs(slope * q * x_mid ** (q - 1))
    sig_y = math.sqrt(max(ysig, yres / 4) ** 2 + (dydx * max(xsig, xres / 4)) ** 2)
    known = rng.random() < law["p_known"]
    spec = {
        "id": uid, "name": law["name"][0], "name_en": law["name"][1],
        "x_var": {"name": xn_zh[i], "name_en": xn_en[i], "symbol": xsym, "unit": xunit, "tool": xt,
                  "tool_en": xt_en, "resolution": xres},
        "y_var": {"name": yn_zh[j], "name_en": yn_en[j], "symbol": ysym, "unit": yunit, "tool": yt,
                  "tool_en": yt_en, "resolution": yres},
        "relation_family": fam, "settings": xs.tolist(), "sigma": {"x": xsig, "y": sig_y},
        "ideal_slope": slope if known else None,
        "slope_guess": slope * float(rng.uniform(0.8, 1.25)),
    }
    if q is not None and FAMILY_EXPONENT[fam] is None:
        spec["exponent"] = q
    if "intercept" in law:
        b = float(rng.uniform(*law["intercept"]))
        spec["ideal_intercept"] = None
        spec["intercept_guess"] = b
        spec["intercept_sd"] = 0.5 * (law["intercept"][1] - law["intercept"][0])
    spec["exclude"] = list(law.get("exclude", ()))
    # 一半任务用定律专属的物理提示替换通用措辞
    spec["hints"] = {k: v for k, v in law.get("hints", {}).items() if rng.random() < 0.5}
    return spec


def procedural_task_family(rng: np.random.Generator, n: int) -> Iterator[tuple[GenericExperiment, str]]:
    """逐个产生 (GenericExperiment, 抽中的原因 id)。原因在该实例可辨识的原因里均匀抽取（含 random_only）。"""
    for k in range(n):
        law = _LAWS[int(rng.integers(len(_LAWS)))]
        spec = _sample_law_spec(rng, law, f"proc_{law['key']}_{k:05d}")
        exp = GenericExperiment(spec, rng=rng)
        ids = exp.cause_ids()
        yield exp, ids[int(rng.integers(len(ids)))]


def law_keys() -> tuple[str, ...]:
    return tuple(l["key"] for l in _LAWS)
