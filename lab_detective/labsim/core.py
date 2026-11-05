"""labsim 公共接口：实验、读数、原因、先验、观测、数据集。

所有实验（课内 6 个、留出 6 个、程序化任务族）都按这里的约定实现，引擎（engine.py）、
特征面板（panel.py）、EIG 顾问（eig.py）和评测框架只依赖这里的接口。

建模约定（重要，实现新实验前先读）：

1. 学生记录的每个读数 = 均值 + 噪声，再按仪器分辨率取整。
   - 噪声：高斯，标准差由 ``Experiment.reading_sigma`` 给出（默认是 ``Reading.sigma``；
     "读数散布偏大"这类原因会加大它）。
   - 取整：按 ``Reading.resolution`` 四舍五入。引擎用 sqrt(sigma² + d²/12) 近似。
2. 均值写成 ``c + U @ θ``：
   - θ 是"线性参数"（如 1/k、摩擦系数、偏移量），先验是高斯，引擎对它解析积分；
   - 其他参数（干扰量、故障幅度）放在网格上求和。
   线性参数让引擎又快又准，能写成线性的量尽量写成线性参数。
3. 每个实验第一个原因必须是"只有随机误差"（id = ``<实验 id>.random_only``）。
4. 数据特征（signature）只用面板能观测到的量；"x 偏移 / y 偏移"只是原因的机制标签，
   斜率自由的直线上两者无法区分。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

G = 9.8  # 人教版课本取 g = 9.8 m/s²

# 面板能观测到的数据特征（封闭词表）
SIGNATURES: tuple[str, ...] = (
    "none",
    "intercept_pos",
    "intercept_neg",
    "slope_high",
    "slope_low",
    "curve_up",
    "curve_down",
    "scatter_excess",
    "scatter_deficit",
    "drift",
)

# 原因层面的机制标签（相对面板坐标：x = 控制量/横轴，y = 读数/纵轴）
MECHANISMS: tuple[str, ...] = (
    "none",
    "x_offset",
    "y_offset",
    "x_scale",
    "y_scale",
    "bend",
    "scatter",
    "drift",
    "aux_only",
)

SOURCES: tuple[str, ...] = ("assumption", "literature")


# ── 先验 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Prior:
    """一维先验。网格用于引擎求和，``draw`` 用于模拟器抽真值（连续分布，不落在网格点上）。

    kind: point | uniform | loguniform | normal | sym_uniform（±[lo, hi] 两侧均匀）
    """

    kind: str
    a: float
    b: float = 0.0
    n: int = 1

    @staticmethod
    def point(v: float) -> "Prior":
        return Prior("point", float(v), float(v), 1)

    @staticmethod
    def uniform(lo: float, hi: float, n: int = 41) -> "Prior":
        return Prior("uniform", float(lo), float(hi), int(n))

    @staticmethod
    def loguniform(lo: float, hi: float, n: int = 41) -> "Prior":
        return Prior("loguniform", float(lo), float(hi), int(n))

    @staticmethod
    def normal(mu: float, sd: float, n: int = 41) -> "Prior":
        return Prior("normal", float(mu), float(sd), int(n))

    @staticmethod
    def sym_uniform(lo: float, hi: float, n: int = 40) -> "Prior":
        """幅度在 [lo, hi]，正负号各一半（例如零点偏移可正可负）。n 是两侧合计的网格点数。"""
        return Prior("sym_uniform", float(lo), float(hi), int(n))

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (网格点, 归一化的 log 权重)。均匀类先验用中点规则。"""
        if self.kind == "point":
            return np.array([self.a]), np.array([0.0])
        if self.kind == "uniform":
            edges = np.linspace(self.a, self.b, self.n + 1)
            pts = 0.5 * (edges[:-1] + edges[1:])
            return pts, np.full(self.n, -math.log(self.n))
        if self.kind == "loguniform":
            edges = np.geomspace(self.a, self.b, self.n + 1)
            pts = np.sqrt(edges[:-1] * edges[1:])
            return pts, np.full(self.n, -math.log(self.n))
        if self.kind == "sym_uniform":
            half = self.n // 2
            edges = np.linspace(self.a, self.b, half + 1)
            pos = 0.5 * (edges[:-1] + edges[1:])
            pts = np.concatenate([-pos[::-1], pos])
            return pts, np.full(pts.size, -math.log(pts.size))
        if self.kind == "normal":
            pts = np.linspace(self.a - 4 * self.b, self.a + 4 * self.b, self.n)
            lw = -0.5 * ((pts - self.a) / self.b) ** 2
            lw -= np.logaddexp.reduce(lw)
            return pts, lw
        raise ValueError(f"unknown prior kind {self.kind}")

    def draw(self, rng: np.random.Generator, size: int | None = None):
        if self.kind == "point":
            return self.a if size is None else np.full(size, self.a)
        if self.kind == "uniform":
            return rng.uniform(self.a, self.b, size)
        if self.kind == "loguniform":
            return np.exp(rng.uniform(math.log(self.a), math.log(self.b), size))
        if self.kind == "sym_uniform":
            mag = rng.uniform(self.a, self.b, size)
            sign = rng.choice([-1.0, 1.0], size=size)
            return mag * sign
        if self.kind == "normal":
            return rng.normal(self.a, self.b, size)
        raise ValueError(f"unknown prior kind {self.kind}")


@dataclass(frozen=True)
class LinearParam:
    """线性参数：均值里以 ``U @ θ`` 出现，先验为高斯 N(mean, sd²)，引擎解析积分。

    选 mean、sd 时保证 mean ± 3·sd 仍在物理上合理（例如 1/k > 0），模拟器直接从这个高斯里抽。
    """

    name: str
    mean: float
    sd: float
    label_zh: str = ""
    unit: str = ""

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mean, self.sd))


# ── 读数、观测种类、原因、测量方案 ──────────────────────────────────────


@dataclass(frozen=True)
class Reading:
    """学生从仪器上读的一个量。resolution = 记录分辨率（取整单位），sigma = 读数随机误差。"""

    key: str
    label_zh: str
    label_en: str
    unit: str
    resolution: float
    sigma: float = 0.0


@dataclass(frozen=True)
class ObsKind:
    """一种观测：主测量（每次改控制量读一组数）或附加测量（称挂钩、重读零点……）。"""

    key: str
    label_zh: str
    label_en: str
    readings: tuple[Reading, ...]
    setting_label_zh: str = ""
    setting_label_en: str = ""
    setting_unit: str = ""


@dataclass(frozen=True)
class Cause:
    """一个候选原因。description 只写物理过程，不写数据特征（特征要学生/模型自己推）。"""

    id: str
    name_zh: str
    name_en: str
    description_zh: str
    description_en: str
    mechanism: str
    signatures: tuple[str, ...]  # 标准方案下预期的面板特征（只给测试和物理卡用，不进模型提示）
    magnitude: Prior = Prior.point(0.0)
    magnitude_label_zh: str = ""
    magnitude_unit: str = ""
    fix_zh: str = ""
    fix_en: str = ""
    source: str = "assumption"
    linear: tuple[LinearParam, ...] = ()  # 本原因额外的线性参数（如偏移量）

    def __post_init__(self):
        assert self.mechanism in MECHANISMS, self.mechanism
        assert all(s in SIGNATURES for s in self.signatures), self.signatures
        assert self.source in SOURCES, self.source


@dataclass(frozen=True)
class Protocol:
    id: str
    label_zh: str
    label_en: str
    plan: tuple[tuple[str, float], ...]  # (观测种类, 控制量)，按试次顺序


@dataclass(frozen=True)
class MenuItem:
    id: str
    label_zh: str
    label_en: str
    minutes: float
    plan: tuple[tuple[str, float], ...]


# ── 观测和数据集 ────────────────────────────────────────────────────────


@dataclass
class Observation:
    kind: str
    setting: float
    values: tuple[float, ...]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "setting": self.setting, "values": list(self.values)}

    @staticmethod
    def from_dict(d: dict) -> "Observation":
        return Observation(d["kind"], float(d["setting"]), tuple(float(v) for v in d["values"]))


@dataclass
class Dataset:
    experiment_id: str
    observations: list[Observation]
    truth: dict | None = None  # 生成时的真值：cause、magnitude、nuisance、linear、seed、world、noise_scale
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "observations": [o.to_dict() for o in self.observations],
            "truth": self.truth,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "Dataset":
        return Dataset(
            d["experiment_id"],
            [Observation.from_dict(o) for o in d["observations"]],
            d.get("truth"),
            d.get("meta", {}),
        )

    def kinds(self) -> set[str]:
        return {o.kind for o in self.observations}


@dataclass
class LinearData:
    """面板用的线性化数据：y = a·x + b，理想时 b = ideal_intercept，a = ideal_slope（None 表示斜率自由）。"""

    x: np.ndarray
    y: np.ndarray
    sy: np.ndarray
    order: np.ndarray
    x_label: str
    y_label: str
    ideal_slope: float | None = None
    ideal_intercept: float | None = 0.0  # None（或 NaN）= 截距自由，面板不检验截距


# ── 实验基类 ────────────────────────────────────────────────────────────


class Experiment:
    """所有实验的基类。子类填类属性并实现 ``predict`` 和 ``linearise``。

    子类必须给出：
      id, name_zh, name_en, obs_kinds, nuisance, causes, protocols, menu
      linear: 所有原因共用的线性参数（如 1/k），可为空元组
    可选：held_out（留出实验）, transfer（"near" / "far"）
    """

    id: str = ""
    name_zh: str = ""
    name_en: str = ""
    held_out: bool = False
    transfer: str | None = None
    obs_kinds: Mapping[str, ObsKind] = {}
    nuisance: Mapping[str, Prior] = {}
    linear: tuple[LinearParam, ...] = ()
    causes: tuple[Cause, ...] = ()
    protocols: Mapping[str, Protocol] = {}
    menu: tuple[MenuItem, ...] = ()

    # ---- 子类实现 ----

    def predict(
        self,
        kind: str,
        settings: np.ndarray,
        order: np.ndarray,
        params: Mapping[str, np.ndarray],
        cause_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 (c, U)：读数均值 = c + U @ θ。

        settings, order: 形状 (n,)，order 是这次观测在数据集里的试次序号。
        params: 名字 -> 形状 (P, 1) 的数组，包含所有相关干扰量和 'mag'（本原因幅度）。
        c: 可广播到 (P, n, r)；U: 可广播到 (P, n, r, L)，L = len(self.linear_params(cause_id))。
        """
        raise NotImplementedError

    def linearise(self, ds: Dataset) -> LinearData:
        raise NotImplementedError

    def reading_sigma(self, kind: str, params: Mapping[str, np.ndarray], cause_id: str) -> np.ndarray:
        """取整之前的高斯噪声标准差，可广播到 (P, 1, r)。散布类原因在子类里覆盖。"""
        return np.array([r.sigma for r in self.obs_kinds[kind].readings], dtype=float)

    def reading_sigma_at(self, kind: str, settings: np.ndarray, params: Mapping[str, np.ndarray],
                         cause_id: str) -> np.ndarray:
        """噪声随控制量变化时覆盖这个（可广播到 (P, n, r)）；默认与控制量无关，调用 reading_sigma。"""
        s = np.asarray(self.reading_sigma(kind, params, cause_id), dtype=float)
        return s if s.ndim == 3 else s.reshape((1,) * (3 - s.ndim) + s.shape)

    def relevant_nuisance(self, cause_id: str, kinds: set[str]) -> tuple[str, ...]:
        """给定原因和已观测的种类，哪些干扰量会影响似然。默认全部；子类可收窄以减小网格。"""
        return tuple(self.nuisance)

    # ---- 通用工具 ----

    @property
    def random_only_id(self) -> str:
        return f"{self.id}.random_only"

    def cause(self, cause_id: str) -> Cause:
        for c in self.causes:
            if c.id == cause_id:
                return c
        raise KeyError(cause_id)

    def cause_ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.causes)

    def linear_params(self, cause_id: str) -> tuple[LinearParam, ...]:
        return tuple(self.linear) + tuple(self.cause(cause_id).linear)

    def resolutions(self, kind: str) -> np.ndarray:
        return np.array([r.resolution for r in self.obs_kinds[kind].readings], dtype=float)

    def total_sd(self, kind: str, params: Mapping[str, np.ndarray], cause_id: str) -> np.ndarray:
        s = np.asarray(self.reading_sigma(kind, params, cause_id), dtype=float)
        d = self.resolutions(kind)
        return np.sqrt(s**2 + d**2 / 12.0)

    def total_sd_at(self, kind: str, settings: np.ndarray, params: Mapping[str, np.ndarray],
                    cause_id: str) -> np.ndarray:
        """引擎用：随控制量变化的总标准差，可广播到 (P, n, r)。"""
        s = self.reading_sigma_at(kind, settings, params, cause_id)
        d = self.resolutions(kind)
        return np.sqrt(s**2 + d**2 / 12.0)

    def menu_item(self, item_id: str) -> MenuItem:
        for m in self.menu:
            if m.id == item_id:
                return m
        raise KeyError(item_id)

    def draw_truth(
        self,
        rng: np.random.Generator,
        cause_id: str,
        magnitude: float | None = None,
        nuisance: Mapping[str, float] | None = None,
        linear: Mapping[str, float] | None = None,
    ) -> dict:
        cause = self.cause(cause_id)
        nuis = {k: float(p.draw(rng)) for k, p in self.nuisance.items()}
        if nuisance:
            nuis.update({k: float(v) for k, v in nuisance.items()})
        lin = {p.name: p.draw(rng) for p in self.linear_params(cause_id)}
        if linear:
            lin.update({k: float(v) for k, v in linear.items()})
        mag = float(cause.magnitude.draw(rng)) if magnitude is None else float(magnitude)
        return {"cause": cause_id, "magnitude": mag, "nuisance": nuis, "linear": lin}

    def _mean_and_sigma(self, truth: dict, kind: str, settings: np.ndarray, order: np.ndarray):
        params = {k: np.array([[v]]) for k, v in truth["nuisance"].items()}
        params["mag"] = np.array([[truth["magnitude"]]])
        c, U = self.predict(kind, settings, order, params, truth["cause"])
        r = len(self.obs_kinds[kind].readings)
        n = settings.size
        lin = self.linear_params(truth["cause"])
        c = np.broadcast_to(c, (1, n, r))
        mean = np.array(c, dtype=float)
        if lin:
            U = np.broadcast_to(U, (1, n, r, len(lin)))
            theta = np.array([truth["linear"][p.name] for p in lin])
            mean = mean + U @ theta
        sig = np.broadcast_to(self.reading_sigma_at(kind, settings, params, truth["cause"]), (1, n, r))
        return mean[0], sig[0, 0]

    def observe(
        self,
        rng: np.random.Generator,
        truth: dict,
        plan: Sequence[tuple[str, float]],
        start_order: int = 0,
        noise_scale: float = 1.0,
    ) -> list[Observation]:
        """按 plan 生成观测（加噪、按分辨率取整）。truth 来自 draw_truth。"""
        out: list[Observation] = []
        for i, (kind, setting) in enumerate(plan):
            settings = np.array([float(setting)])
            order = np.array([start_order + i])
            mean, sig = self._mean_and_sigma(truth, kind, settings, order)
            vals = mean[0] + rng.normal(0.0, 1.0, mean.shape[1]) * sig * noise_scale
            d = self.resolutions(kind)
            vals = np.where(d > 0, np.round(vals / np.where(d > 0, d, 1.0)) * d, vals)
            out.append(Observation(kind, float(setting), tuple(float(v) for v in vals)))
        return out

    def simulate(
        self,
        rng: np.random.Generator,
        cause_id: str,
        protocol: str | Sequence[tuple[str, float]] = "standard",
        magnitude: float | None = None,
        nuisance: Mapping[str, float] | None = None,
        linear: Mapping[str, float] | None = None,
        noise_scale: float = 1.0,
        world: str = "labsim",
        seed: int | None = None,
    ) -> Dataset:
        plan = self.protocols[protocol].plan if isinstance(protocol, str) else tuple(protocol)
        truth = self.draw_truth(rng, cause_id, magnitude, nuisance, linear)
        obs = self.observe(rng, truth, plan, 0, noise_scale)
        truth = dict(truth, seed=seed, world=world, noise_scale=noise_scale)
        meta = {"protocol": protocol if isinstance(protocol, str) else "custom"}
        return Dataset(self.id, obs, truth, meta)

    def extend(
        self,
        rng: np.random.Generator,
        ds: Dataset,
        plan: Sequence[tuple[str, float]],
    ) -> Dataset:
        """用数据集里记下的真值继续测量（闭环实验用）。"""
        if ds.truth is None:
            raise ValueError("dataset has no truth; cannot simulate a continuation")
        new = self.observe(
            rng, ds.truth, plan, len(ds.observations), ds.truth.get("noise_scale", 1.0)
        )
        return Dataset(ds.experiment_id, ds.observations + new, ds.truth, dict(ds.meta))
