"""训练集划分：训练实验 / 留出实验、每个训练实验扣下的一个原因、随机种子命名空间、训练/验证划分规则。

- 训练实验 / 留出实验：完全按 labsim 注册表的 held_out 标记（新实验只要在文件里标好，这里不用改）。
  留出实验的任何东西（id、名称、原因）都不进训练数据，只在 bench 里评分。
- 每个训练实验扣下一个原因（WITHHELD_CAUSES）：训练数据里既不用它模拟数据，也不把它列进候选清单、
  引擎原因集、"提出原因"的目标和知识问答，用来构成"已知实验里的新原因"这一级测试。
  选择标准：不是"只有随机误差"；标准方案下能和"只有随机误差"分开（图谱判为 separable 或 partial）；
  中文名在训练实验里唯一（防止按名字查泄漏时误报 / 漏报）；物理上值得考。
- 随机种子：每条样本的种子由 (基础种子, 任务, 划分, 序号) 经 SeedSequence 派生，互不重叠；
  验证集 = 独立的种子命名空间，条数取总数的 5%（四舍五入，至少 1 条）。
  bench（冻结测试集）必须用 SPLIT_CODES["bench"] 或另一个基础种子，这里的生成器从不使用它。
"""

from __future__ import annotations

import math

import numpy as np

from ..labsim.core import Experiment
from ..labsim.registry import EXPERIMENTS, get_experiment

# ── 实验划分 ──────────────────────────────────────────────────────────

TRAIN_EXPERIMENTS: tuple[str, ...] = tuple(sorted(k for k, e in EXPERIMENTS.items() if not e.held_out))
HELD_OUT_EXPERIMENTS: tuple[str, ...] = tuple(sorted(k for k, e in EXPERIMENTS.items() if e.held_out))

# 每个训练实验扣下的原因 → 选它的理由（物理上考什么）
WITHHELD_CAUSES: dict[str, str] = {
    "average_speed": "average_speed.cart_length",
    "energy": "energy.wheel_inertia",
    "friction": "friction.zero_vertical",
    "hooke": "hooke.total_length",
    "momentum": "momentum.friction",
    "newton2": "newton2.heavy_hanging",
}

WITHHELD_REASONS_ZH: dict[str, str] = {
    "average_speed.cart_length": "路程从车尾量起：考的是'哪个量被整体平移'，偏移落在路程上而不是时间上，图线既平移又弯曲。",
    "energy.wheel_inertia": "车轮转动分走动能：考转动动能，与同样让斜率偏小的轨道摩擦区分要靠追加测量。",
    "friction.zero_vertical": "竖直调零后水平使用：考测力计自身挂钩重力的影响，读数整体偏小一个常量。",
    "hooke.total_length": "记录了总长而不是伸长量：'错误'而非'误差'的典型，读数整体多出原长，量级远大于读数偏差。",
    "momentum.friction": "两门之间有摩擦：只会让小车减速的单侧效应，与可正可负的导轨倾斜区分要看方向。",
    "newton2.heavy_hanging": "槽码质量没有远小于小车质量：考 a = mg/(M+m) 的近似条件，图线随拉力增大而向下弯。",
}

# ── 种子命名空间 ──────────────────────────────────────────────────────

BASE_SEED = 20261106
TASK_CODES: dict[str, int] = {"diagnose": 1, "design": 2, "explain": 3, "knowledge": 4}
SPLIT_CODES: dict[str, int] = {"train": 1, "val": 2, "bench": 9}  # bench 只给冻结测试集用，生成器不碰
VAL_FRACTION = 0.05


def check_withheld() -> list[str]:
    """静态检查扣下的原因（不跑模拟）：存在、不是随机误差、名称在训练实验里唯一、每个训练实验恰好一个。"""
    probs: list[str] = []
    names: dict[str, list[str]] = {}
    for eid in TRAIN_EXPERIMENTS:
        for c in EXPERIMENTS[eid].causes:
            names.setdefault(c.name_zh, []).append(c.id)
    for eid in TRAIN_EXPERIMENTS:
        cid = WITHHELD_CAUSES.get(eid)
        if cid is None:
            probs.append(f"训练实验 {eid} 没有扣下的原因")
            continue
        exp = EXPERIMENTS[eid]
        if cid not in exp.cause_ids():
            probs.append(f"{cid} 不在 {eid} 的原因库里")
            continue
        if cid == exp.random_only_id:
            probs.append(f"{eid} 扣下的是'只有随机误差'")
        if len(names[exp.cause(cid).name_zh]) > 1:
            probs.append(f"{cid} 的名称与其他训练原因重名：{names[exp.cause(cid).name_zh]}")
        if cid not in WITHHELD_REASONS_ZH:
            probs.append(f"{cid} 缺少选择理由")
    for eid in WITHHELD_CAUSES:
        if eid not in TRAIN_EXPERIMENTS:
            probs.append(f"WITHHELD_CAUSES 里的 {eid} 不是训练实验")
    return probs


def withheld_ids() -> frozenset[str]:
    return frozenset(WITHHELD_CAUSES.values())


def allowed_causes(exp: Experiment) -> tuple[str, ...]:
    """训练数据里这个实验能用的原因（去掉扣下的那个）。程序化实例没有扣下的原因。留出实验报错。"""
    if exp.held_out or exp.id in HELD_OUT_EXPERIMENTS:
        raise ValueError(f"{exp.id} 是留出实验，不能进训练数据")
    w = WITHHELD_CAUSES.get(exp.id)
    return tuple(c for c in exp.cause_ids() if c != w)


def train_experiments() -> list[Experiment]:
    return [get_experiment(k) for k in TRAIN_EXPERIMENTS]


# ── 种子与条数 ────────────────────────────────────────────────────────


def example_seed(task: str, split: str, index: int, base: int = BASE_SEED) -> int:
    """第 index 条样本的种子（63 位整数）。同样的 (base, task, split, index) 永远得到同一个种子。"""
    if split not in ("train", "val"):
        raise ValueError(f"生成器只产出 train / val，不产出 {split}")
    ss = np.random.SeedSequence([int(base), TASK_CODES[task], SPLIT_CODES[split], int(index)])
    return int(ss.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def example_rng(task: str, split: str, index: int, base: int = BASE_SEED) -> np.random.Generator:
    return np.random.default_rng(example_seed(task, split, index, base))


def split_counts(n_total: int, val_fraction: float = VAL_FRACTION) -> tuple[int, int]:
    """总条数 → (训练条数, 验证条数)。验证 = 5%（四舍五入，n ≥ 2 时至少 1 条）。"""
    if n_total <= 0:
        return 0, 0
    n_val = int(math.floor(n_total * val_fraction + 0.5))
    if n_total >= 2:
        n_val = max(n_val, 1)
    return n_total - n_val, n_val
