"""实验注册表。新实验在这里登记一行。"""
from __future__ import annotations

from .core import Experiment
from .experiments.hooke import Hooke

EXPERIMENTS: dict[str, Experiment] = {e.id: e for e in (Hooke(),)}


def get_experiment(exp_id: str) -> Experiment:
    return EXPERIMENTS[exp_id]
