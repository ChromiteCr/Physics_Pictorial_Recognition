"""实验注册表：自动发现 experiments/ 下所有 Experiment 子类（新实验只要新建文件，不用改这里）。"""
from __future__ import annotations

import importlib
import inspect
import pkgutil

from . import experiments as _pkg
from .core import Experiment

EXPERIMENTS: dict[str, Experiment] = {}


def _discover() -> None:
    for info in pkgutil.iter_modules(_pkg.__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{_pkg.__name__}.{info.name}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, Experiment) and obj is not Experiment and obj.__module__ == mod.__name__ and obj.id:
                EXPERIMENTS[obj.id] = obj()


_discover()


def get_experiment(exp_id: str) -> Experiment:
    return EXPERIMENTS[exp_id]


def training_experiments() -> list[Experiment]:
    return [e for e in EXPERIMENTS.values() if not e.held_out]


def held_out_experiments() -> list[Experiment]:
    return [e for e in EXPERIMENTS.values() if e.held_out]
