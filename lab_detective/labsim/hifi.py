"""错配的"高保真"世界（只给 bench 用）：在任何 labsim 实验外面加上引擎从来不建模的效应。

引擎假设：读数噪声是高斯的、各原因单独作用、学生抄数不出错、规律本身是理想的。这里故意破坏这些假设，
看诊断在模型外的数据上会不会乱报。所有幅度都是写死的常数（下面 HifiConfig 的默认值），可在报告里引用：

  student_t_df   读数噪声改成 Student-t（自由度 4，方差缩放到与原高斯相同）—— 偶尔出现离群读数
  timing_tau     计时读数（单位 s）加 ex-Gaussian 的指数尾：平均多 0.8σ 的反应延迟，只往偏大一侧
  drag_rel       有小车 / 滑轮 / 下落物体的实验：主读数在最大控制量处相对减小 1.5%，随控制量线性增大
                 （滑轮轴摩擦 + 空气阻力的唯象近似，不是逐个实验推导的力学模型）
  string_rel     有细线和滑轮的实验：线的质量让主读数再相对偏 0.5%（同样随控制量线性）
  slip_p         每个读数有 2% 的概率抄错一位小数（×10 或 ÷10）
另外两种用法：
  noise_shift    只把噪声放大到 noise_scale 倍并改成 Student-t（不加其他效应）
  two causes     两个原因同时作用：均值 = 原因 A 的均值 + (原因 B 的均值 − 只有随机误差的均值)，
                 共用同一组干扰量和线性参数（小效应下的一阶叠加）
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .core import Dataset, Experiment, Observation

# 有运动物体（拖曳）、有细线滑轮（线的质量）的实验 id 前缀；程序化实例、组合实验按 id 前缀匹配
DRAG_EXPS = ("newton2", "energy", "momentum", "average_speed", "atwood", "free_fall", "incline_friction",
             "projectile", "incline_newton2", "system_energy", "photogate_free_fall", "incline_slide")
STRING_EXPS = ("newton2", "atwood", "incline_newton2", "system_energy")


@dataclass(frozen=True)
class HifiConfig:
    student_t_df: float | None = 4.0
    noise_scale: float = 1.0
    timing_tau: float = 0.8  # 以该读数的 σ 为单位
    drag_rel: float = 0.015
    string_rel: float = 0.005
    slip_p: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)


HIFI = HifiConfig()
NOISE_SHIFT = HifiConfig(student_t_df=4.0, noise_scale=2.0, timing_tau=0.0, drag_rel=0.0, string_rel=0.0, slip_p=0.0)
GAUSS = HifiConfig(student_t_df=None, timing_tau=0.0, drag_rel=0.0, string_rel=0.0, slip_p=0.0)


def _matches(exp_id: str, names) -> bool:
    return any(exp_id == n or exp_id.startswith(n + "_") for n in names)


def _noise(rng: np.random.Generator, n: int, df: float | None) -> np.ndarray:
    if df is None:
        return rng.normal(0.0, 1.0, n)
    return rng.standard_t(df, n) * np.sqrt((df - 2.0) / df)  # 方差 1


def _mean_sigma(exp: Experiment, truth: dict, kind: str, setting: float, order: int):
    mean, sig = exp._mean_and_sigma(truth, kind, np.array([float(setting)]), np.array([order]))
    return np.asarray(mean[0], float), np.asarray(sig, float).reshape(-1)


def simulate(exp: Experiment, rng: np.random.Generator, cause_id: str, plan, cfg: HifiConfig = HIFI,
             second_cause: str | None = None, world: str = "hifi", protocol: str = "custom",
             slip_force: bool = False) -> Dataset:
    """按 plan（(种类, 控制量) 序列）生成一份错配世界的数据。slip_force=True 时保证恰有一个读数抄错。"""
    plan = tuple(plan)
    truth = exp.draw_truth(rng, cause_id)
    truth_b = truth_ro = None
    if second_cause is not None:
        truth_b = exp.draw_truth(rng, second_cause, nuisance=truth["nuisance"], linear=truth["linear"])
        truth_ro = exp.draw_truth(rng, exp.random_only_id, magnitude=0.0, nuisance=truth["nuisance"],
                                  linear=truth["linear"])
    main = plan[0][0] if plan else None
    s_max = max((abs(s) for k, s in plan if k == main), default=1.0) or 1.0
    bend = (cfg.drag_rel if _matches(exp.id, DRAG_EXPS) else 0.0) + (
        cfg.string_rel if _matches(exp.id, STRING_EXPS) else 0.0)
    slip_at = int(rng.integers(len(plan))) if slip_force and plan else -1
    obs: list[Observation] = []
    slips = []
    for i, (kind, setting) in enumerate(plan):
        mean, sig = _mean_sigma(exp, truth, kind, setting, i)
        if truth_b is not None:
            mb, sb = _mean_sigma(exp, truth_b, kind, setting, i)
            m0, _ = _mean_sigma(exp, truth_ro, kind, setting, i)
            mean = mean + (mb - m0)
            sig = np.maximum(sig, sb)
        sig = np.broadcast_to(sig, mean.shape)
        vals = mean + _noise(rng, mean.size, cfg.student_t_df) * sig * cfg.noise_scale
        units = [r.unit for r in exp.obs_kinds[kind].readings]
        for j, u in enumerate(units):
            if cfg.timing_tau > 0 and u == "s":
                vals[j] += rng.exponential(cfg.timing_tau * sig[j])
        if bend and kind == main:
            vals = vals * (1.0 - bend * abs(setting) / s_max)
        for j in range(vals.size):
            if (slip_force and i == slip_at and j == 0) or (not slip_force and rng.random() < cfg.slip_p):
                f = 10.0 if rng.random() < 0.5 else 0.1
                vals[j] *= f
                slips.append({"index": i, "reading": j, "factor": f})
        d = exp.resolutions(kind)
        vals = np.where(d > 0, np.round(vals / np.where(d > 0, d, 1.0)) * d, vals)
        obs.append(Observation(kind, float(setting), tuple(float(v) for v in vals)))
    effects = {"config": cfg.to_dict(), "bend_rel": bend, "slips": slips}
    if second_cause is not None:
        effects["second_cause"] = second_cause
        effects["second_magnitude"] = truth_b["magnitude"]
    truth = dict(truth, world=world, noise_scale=cfg.noise_scale, effects=effects)
    return Dataset(exp.id, obs, truth, {"protocol": protocol})
