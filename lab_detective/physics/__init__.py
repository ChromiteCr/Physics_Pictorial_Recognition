"""物理组件层：把精选实验拆成可复用的物理组件（物体、连接件、表面、仪器）和规律模板，
再组合成符合 labsim 接口的新实验孪生（见 base.py 文件头）。精选实验只是起点，其他装置由组件组合而成。"""

from .base import Affine, AuxMeasurement, Component, Control, Ctx, Effect, OpenQuestion, Setup, Variant
from .compose import ComposedExperiment, compose

__all__ = ["Affine", "AuxMeasurement", "Component", "Control", "Ctx", "Effect", "OpenQuestion", "Setup", "Variant",
           "ComposedExperiment", "compose"]
