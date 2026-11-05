"""物理组件层的基础类型：误差机制 Effect、计算上下文 Ctx、组件 Component、附加测量、装置 Setup。

为什么要这一层：精选的课内/留出实验（labsim/experiments/）只是起点。照片里认出一套没见过的器材组合时，
从组件库里取出"物体、连接件、轨道/斜面、测量仪器"，配上一条规律模板，就能组合成一个新的
labsim Experiment（compose.py），引擎、面板、EIG 顾问、可辨识性图谱和事实单不用改。

四层结构：
1. 组件（components.py）：物体、连接件、表面。带自己的物理参数和误差机制（Effect），
   例如滑轮带"转动惯量不可忽略"，小车带"车轮转动动能"，导轨带"摩擦没平衡"。
2. 仪器（instruments.py）：把真实物理量变成学生记下的读数（尽量是仿射映射，线性参数才能留下来），
   自带分辨率、随机误差和仪器误差（调零、遮光片宽度、反应时间、数错次数……）。
3. 规律模板（templates.py）：用组件算出"真实的物理观测量"（伸长量、加速度、速度、周期……），
   并给出面板用的线性化。
4. 装置（Setup）= 一条模板 + 若干组件和仪器 + 控制量 + 测量方案；compose.py 把它变成 Experiment。

单故障约定（与 labsim/core.py 一致）：一个原因 = 一个 Effect；只有当前原因的 Effect 是"开"的，
"只有随机误差"时全部关闭。例外：Setup.exact_in_faults 里的 Effect 表示"课本近似不成立"，
在任何故障原因下都打开（例如牛顿第二定律里槽码的惯性，见 catalog 的镜像说明）。

线性参数：均值写成 c + U·θ。组件和模板用 Affine（下面）表示"关于线性参数 θ 的仿射量"，
可以与普通数组加减乘除，但两个都含 θ 的量不能相乘、也不能进开方等非线性函数——这时那个量
只能放到网格（干扰量或幅度）上。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..labsim.core import LinearParam, MenuItem, Prior, Reading

# ── 仿射量 ────────────────────────────────────────────────────────────


class Affine:
    """c + Σ_k u_k·θ_k：c、u_k 是可广播到 (P, n) 的数组，θ_k 是按名字索引的线性参数。"""

    __slots__ = ("c", "u")
    __array_ufunc__ = None  # 让 ndarray * Affine 交给 Affine.__rmul__，不变成对象数组

    def __init__(self, c: Any = 0.0, u: Mapping[str, Any] | None = None):
        self.c = c
        self.u = dict(u or {})

    @staticmethod
    def lin(name: str, coef: Any = 1.0) -> "Affine":
        return Affine(0.0, {name: coef})

    @property
    def is_const(self) -> bool:
        return not self.u

    def _add(self, o, sign: float) -> "Affine":
        o = as_affine(o)
        u = dict(self.u)
        for k, v in o.u.items():
            u[k] = u[k] + sign * v if k in u else sign * v
        return Affine(self.c + sign * o.c, u)

    def __add__(self, o):
        return self._add(o, 1.0)

    def __radd__(self, o):
        return self._add(o, 1.0)

    def __sub__(self, o):
        return self._add(o, -1.0)

    def __rsub__(self, o):
        return as_affine(o)._add(self, -1.0)

    def __neg__(self):
        return Affine(-self.c, {k: -v for k, v in self.u.items()})

    def __mul__(self, k):
        if isinstance(k, Affine):
            if k.u and self.u:
                raise TypeError("两个含线性参数的量不能相乘（把其中一个放到网格上）")
            if not k.u:
                k = k.c
            else:
                return k * self
        return Affine(self.c * k, {n: v * k for n, v in self.u.items()})

    __rmul__ = __mul__

    def __truediv__(self, k):
        if isinstance(k, Affine):
            if k.u:
                raise TypeError("不能除以含线性参数的量")
            k = k.c
        return Affine(self.c / k, {n: v / k for n, v in self.u.items()})

    def __rtruediv__(self, k):
        if self.u:
            raise TypeError("不能除以含线性参数的量")
        return as_affine(k) / self.c

    def names(self) -> set[str]:
        return set(self.u)

    def to_cU(self, names: Sequence[str], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """→ c (P, n)、U (P, n, L)，L 按 names 排序。用到 names 以外的线性参数时报错（多半是原因开关写错了）。"""
        extra = set(self.u) - set(names)
        if extra:
            raise KeyError(f"本原因没有这些线性参数：{sorted(extra)}")
        c = np.broadcast_to(np.asarray(self.c, dtype=float), shape).copy()
        U = np.zeros(shape + (len(names),))
        for j, n in enumerate(names):
            if n in self.u:
                U[:, :, j] = np.broadcast_to(np.asarray(self.u[n], dtype=float), shape)
        return c, U

    def value(self, theta: Mapping[str, float]) -> np.ndarray:
        """给定线性参数的数值，算出普通数组（测试和孪生用）。"""
        out = np.asarray(self.c, dtype=float)
        for n, v in self.u.items():
            out = out + np.asarray(v, dtype=float) * float(theta[n])
        return out


def as_affine(x) -> Affine:
    return x if isinstance(x, Affine) else Affine(x)


# ── 误差机制 ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Effect:
    """一种可能的系统误差机制，挂在某个组件或仪器上；compose 时变成一个 Cause。

    local：组件内名字；在装置里的完整 key = "<组件名>_<local>"（见 Component.key）。
    magnitude：非线性幅度的网格先验；偏移量这类线性进入的量放在 linear 里（名字要带组件前缀）。
    uses：本原因成立时会用到的干扰量（决定引擎网格里放哪些维度，越少越快）。
    scatter=True：只加大读数散布（幅度 = 额外标准差，单位与主读数相同），不改均值。
    description 只写物理过程，不写数据特征（特征要学生/模型自己推）。
    """

    local: str
    name_zh: str
    name_en: str
    description_zh: str
    description_en: str
    mechanism: str
    signatures: tuple[str, ...]
    magnitude: Prior = Prior.point(0.0)
    magnitude_label_zh: str = ""
    magnitude_unit: str = ""
    fix_zh: str = ""
    fix_en: str = ""
    linear: tuple[LinearParam, ...] = ()
    uses: tuple[str, ...] = ()
    scatter: bool = False
    source: str = "assumption"


# ── 附加测量与主测量变体 ───────────────────────────────────────────────


@dataclass(frozen=True)
class AuxMeasurement:
    """附加测量：直接读出某个参数（称挂钩、卡尺量遮光片、量斜面高度……），用来把混淆的原因分开。

    value(ctx) 返回读数均值（数组或 Affine，单位 = reading.unit），可依赖参数和当前原因。
    """

    kind: str
    label_zh: str
    label_en: str
    reading: Reading
    value: Callable[["Ctx"], Any]
    uses: tuple[str, ...] = ()
    minutes: float = 2.0
    setting: float = 0.0
    setting_label_zh: str = ""
    setting_unit: str = ""
    menu_label_zh: str = ""  # 空 = 用 label_zh
    menu_id: str = ""  # 空 = 用 kind


@dataclass(frozen=True)
class Variant:
    """主测量的变体：同一规律、同一仪器，只改一个条件（固定测力计、掉头拉、加配重、轻轻释放……）。

    flags 交给组件判断哪些效应在这个变体里不起作用；n_count 给计数类计时器（只数 10 次）；
    reading 非空时换读数规格（例如固定测力计后指针更稳）。menu：(id, 中文, 英文, 分钟, 控制量序列)。
    """

    kind: str
    label_zh: str
    label_en: str
    flags: frozenset = frozenset()
    reading: Reading | None = None
    n_count: int | None = None
    menu: tuple[tuple[str, str, str, float, tuple[float, ...]], ...] = ()


@dataclass(frozen=True)
class OpenQuestion:
    """看图建模后还需要学生回答的已知量（质量、角度、遮光片宽度……）。"""

    key: str
    text_zh: str
    unit: str
    default: float | None = None


# ── 控制量 ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Control:
    """主测量的控制量（横轴设定值）。quantity 决定哪些组件效应会改动它（记录值 → 真实值）。

    quantity：mass（悬挂质量）、force（压力等）、height（竖直高度）、length（摆长等）、distance（沿轨道距离）。
    to_si：记录单位 → SI 的倍数（g → kg 为 1e-3，cm → m 为 1e-2）。
    """

    quantity: str
    label_zh: str
    label_en: str
    unit: str
    to_si: float = 1.0


# ── 计算上下文 ────────────────────────────────────────────────────────


class Ctx:
    """一次 predict 调用的上下文：组件、模板和仪器通过它取参数、判断误差机制开关。

    数组形状都可广播到 (P, n)：P 是网格点数，n 是这一批观测的个数。
    """

    def __init__(self, setup: "Setup", kind: str, settings: np.ndarray, params: Mapping[str, np.ndarray],
                 active: str | None, flags: frozenset = frozenset(), n_count: int | None = None):
        self.setup = setup
        self.kind = kind
        self.rec = np.asarray(settings, dtype=float)[None, :]  # 记录的控制量（原单位）
        self.params = params
        self.active = active  # 当前原因对应的 Effect key；只有随机误差时为 None
        self.flags = flags
        self.n_count = n_count
        self.P = int(np.asarray(params["mag"]).shape[0])
        self.n = int(self.rec.shape[1])
        self._x = None

    # 误差机制
    def on(self, key: str) -> bool:
        if self.active == key:
            return True
        return self.active is not None and key in self.setup.exact_in_faults

    @property
    def mag(self) -> np.ndarray:
        return self.params["mag"]

    def flag(self, name: str) -> bool:
        return name in self.flags

    # 参数
    def p(self, name: str) -> np.ndarray:
        """干扰量 (P, 1)。"""
        return self.params[name]

    @staticmethod
    def lin(name: str, coef: Any = 1.0) -> Affine:
        return Affine.lin(name, coef)

    def part(self, name: str) -> "Component":
        return self.setup.part(name)

    # 控制量：记录值 → SI → 各组件效应修正后的真实值
    @property
    def x(self):
        if self._x is None:
            x = self.rec * self.setup.control.to_si
            for part in self.setup.parts:
                x = part.control(self, x, self.setup.control.quantity)
            self._x = x
        return self._x

    @property
    def x_rec(self) -> np.ndarray:
        return self.rec * self.setup.control.to_si


# ── 组件基类 ──────────────────────────────────────────────────────────


@dataclass
class Component:
    """组件基类。name 是装置里的实例名（"cart"、"pulley"…），参数名一律加前缀 name + "."，
    Effect 的完整 key = name + "_" + local。

    enabled：只启用这些误差机制（None = 全部默认机制），装置可以按已知量和仪器筛掉不适用的原因。
    """

    name: str
    enabled: tuple[str, ...] | None = None

    # 子类覆盖
    label_zh: str = ""

    def all_effects(self) -> list[Effect]:
        return []

    def nuisance(self) -> dict[str, Prior]:
        return {}

    def linear(self) -> tuple[LinearParam, ...]:
        """所有原因共用的线性参数（例如动摩擦因数、遮光片真实宽度）。"""
        return ()

    def aux(self, setup: "Setup") -> list[AuxMeasurement]:
        return []

    def variants(self, setup: "Setup") -> list[Variant]:
        return []

    def questions(self) -> list[OpenQuestion]:
        return []

    def control(self, ctx: Ctx, x, quantity: str):
        """记录的控制量 → 真实控制量（例如漏算挂钩质量、摆长量到球顶、把斜面长度当高度）。默认不改。"""
        return x

    def extra_sigma(self, ctx: Ctx):
        """散布类效应：额外标准差（读数单位）；None = 不适用。默认：本组件的散布效应开、且变体不排除时用幅度。"""
        return None

    # 工具
    def key(self, local: str) -> str:
        return f"{self.name}_{local}"

    def pname(self, local: str) -> str:
        return f"{self.name}.{local}"

    def effects(self) -> list[Effect]:
        effs = self.all_effects()
        if self.enabled is None:
            return effs
        unknown = set(self.enabled) - {e.local for e in effs}
        if unknown:
            raise KeyError(f"{self.name} 没有这些误差机制：{sorted(unknown)}")
        return [e for e in effs if e.local in self.enabled]

    def has(self, local: str) -> bool:
        return any(e.local == local for e in self.effects())


# ── 装置 ──────────────────────────────────────────────────────────────


@dataclass
class Setup:
    """一台装置 = 规律模板 + 组件和仪器 + 主测量仪器 + 控制量 + 测量方案。

    standard / single：标准方案和单次方案的控制量；extra_menu：在新控制量处再做一次主测量的菜单项。
    cause_alias / kind_alias：镜像精选实验时沿用原来的原因后缀和观测种类名（只改前缀），方便对照。
    sync：{干扰量名: 线性参数名}——同一个物理量既在分子里线性出现、又在分母里非线性出现时，
      分母那份放在小网格上，模拟和孪生时取线性参数的真值（见摩擦力斜拉）。
    exact_in_faults：表示"课本近似不成立"的 Effect key，在任何故障原因下都打开（见文件头）。
    """

    id: str
    name_zh: str
    name_en: str
    template: Any
    parts: list[Component]
    instrument: str
    control: Control
    standard: tuple[float, ...]
    single: float
    main_label_zh: str = ""
    main_label_en: str = ""
    random_zh: str = "装置和步骤都正确，数据偏离只来自读数的随机误差。"
    random_en: str = "The setup and procedure are correct; deviations come only from random reading error."
    extra_menu: tuple[MenuItem, ...] = ()
    extra_protocols: Mapping[str, tuple[str, str, tuple[tuple[str, float], ...]]] = field(default_factory=dict)
    held_out: bool = True
    transfer: str | None = "near"
    exclude: tuple[str, ...] = ()
    exact_in_faults: tuple[str, ...] = ()
    cause_alias: Mapping[str, str] = field(default_factory=dict)
    kind_alias: Mapping[str, str] = field(default_factory=dict)
    sync: Mapping[str, str] = field(default_factory=dict)
    knowledge_points: tuple[str, ...] = ()

    def __post_init__(self):
        names = [p.name for p in self.parts]
        if len(set(names)) != len(names):
            raise ValueError(f"组件名重复：{names}")
        if self.instrument not in names:
            raise ValueError(f"主测量仪器 {self.instrument} 不在组件里")
        if "." in self.id or not self.id:
            raise ValueError("装置 id 不能含点号")

    def part(self, name: str) -> Component:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(name)

    def find(self, cls) -> list[Component]:
        return [p for p in self.parts if isinstance(p, cls)]

    @property
    def main_instrument(self):
        return self.part(self.instrument)

    def questions(self) -> list[OpenQuestion]:
        out: list[OpenQuestion] = []
        for p in self.parts:
            out.extend(p.questions())
        return out
