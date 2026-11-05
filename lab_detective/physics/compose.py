"""Setup → ComposedExperiment：按 labsim/core.py 的接口把组件、仪器和规律模板组合成一个 Experiment。

- 原因 = "只有随机误差" + 每个组件/仪器的每个 Effect（Setup.exclude 里的除外）；
- 干扰量 = 模板 + 各组件的网格参数；共用线性参数 = 模板 + 各组件的线性参数；
- 观测种类 = 主测量 "main" + 各组件的主测量变体 + 各组件的附加测量；
- 方案 standard / single（+ Setup.extra_protocols），菜单 = 附加测量 + 变体菜单 + Setup.extra_menu；
- predict 返回 (c, U)，线性化交给模板，所以引擎、面板、EIG、可辨识性图谱、孪生和事实单都不用改。

组合出来的实验不放进 labsim 注册表（不会改变训练/留出实验清单），默认 held_out=True。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..labsim.core import Cause, Dataset, Experiment, LinearData, MenuItem, ObsKind, Protocol
from .base import Affine, Ctx, Setup, as_affine


class ComposedExperiment(Experiment):
    """由 Setup 组合出来的实验孪生。"""

    def __init__(self, setup: Setup):
        self.setup = setup
        self.id = setup.id
        self.name_zh = setup.name_zh
        self.name_en = setup.name_en
        self.held_out = setup.held_out
        self.transfer = setup.transfer
        self.knowledge_points = tuple(setup.knowledge_points)
        tmpl, inst = setup.template, setup.main_instrument

        # 原因
        self._effect_of: dict[str, tuple] = {}  # cause_id -> (Effect key, Effect, 组件)
        causes = [Cause(
            self.random_only_id, "只有随机误差", "random error only", setup.random_zh, setup.random_en,
            "none", ("none",), fix_zh="多测几组、取平均即可。", fix_en="Take more readings and average.",
        )]
        for part in setup.parts:
            for e in part.effects():
                key = part.key(e.local)
                if key in setup.exclude:
                    continue
                cid = f"{self.id}.{setup.cause_alias.get(key, key)}"
                if cid in self._effect_of or cid == self.random_only_id:
                    raise ValueError(f"原因 id 重复：{cid}")
                self._effect_of[cid] = (key, e, part)
                causes.append(Cause(
                    cid, e.name_zh, e.name_en, e.description_zh, e.description_en, e.mechanism, e.signatures,
                    magnitude=e.magnitude, magnitude_label_zh=e.magnitude_label_zh, magnitude_unit=e.magnitude_unit,
                    fix_zh=e.fix_zh, fix_en=e.fix_en, source=e.source, linear=tuple(e.linear),
                ))
        self.causes = tuple(causes)
        # cause_alias 可以列出按选项（计时器、是否记录倾角）不存在的机制，这里不检查
        bad =set(setup.exact_in_faults) - {k for k, _, _ in self._effect_of.values()}
        if bad:
            raise KeyError(f"exact_in_faults 里有不存在的机制：{sorted(bad)}")

        # 干扰量、线性参数
        nuis = dict(tmpl.nuisance(setup))
        lin = list(tmpl.linear(setup))
        for part in setup.parts:
            nuis.update(part.nuisance())
            lin.extend(part.linear())
        self.nuisance = nuis
        self.linear = tuple(lin)
        names = [p.name for p in self.linear] + [p.name for c in self.causes for p in c.linear]
        if len(set(names)) != len(names):
            raise ValueError(f"线性参数重名：{names}")

        # 观测种类：内部名 → ("main"|"variant"|"aux", 规格)；对外名可按 kind_alias 改
        ctl = setup.control
        self._spec: dict[str, tuple[str, object]] = {"main": ("main", None)}
        kinds: dict[str, ObsKind] = {"main": ObsKind(
            "main", setup.main_label_zh or tmpl.main_label_zh, setup.main_label_en or tmpl.main_label_en,
            (inst.reading(setup),), setting_label_zh=ctl.label_zh, setting_label_en=ctl.label_en,
            setting_unit=ctl.unit,
        )}
        menu: list[MenuItem] = []
        for part in setup.parts:
            for v in part.variants(setup):
                k = self._ext(v.kind)
                rd = v.reading or inst.reading(setup, v.flags, v.n_count)
                if v.reading is None and v.n_count is None:
                    base = inst.reading(setup)
                    if rd == base:
                        rd = replace(rd, key=f"{rd.key}_{v.kind}")
                self._add_kind(kinds, k, ("variant", v), ObsKind(
                    k, v.label_zh, v.label_en, (rd,), setting_label_zh=ctl.label_zh,
                    setting_label_en=ctl.label_en, setting_unit=ctl.unit,
                ))
                for mid, mzh, men, minutes, settings in v.menu:
                    menu.append(MenuItem(mid, mzh, men, minutes, tuple((k, float(s)) for s in settings)))
        for part in setup.parts:
            for a in part.aux(setup):
                k = self._ext(a.kind)
                self._add_kind(kinds, k, ("aux", a), ObsKind(
                    k, a.label_zh, a.label_en, (a.reading,), setting_label_zh=a.setting_label_zh,
                    setting_unit=a.setting_unit,
                ))
                menu.append(MenuItem(a.menu_id or a.kind, a.menu_label_zh or a.label_zh, a.label_en, a.minutes,
                                     ((k, float(a.setting)),)))
        menu.extend(setup.extra_menu)
        ids = [m.id for m in menu]
        if len(set(ids)) != len(ids):
            raise ValueError(f"菜单 id 重复：{ids}")
        self.obs_kinds = kinds
        self.menu = tuple(menu)

        protocols = {
            "single": Protocol("single", f"单次测量（{setup.single:g} {ctl.unit}）", "single run",
                               (("main", float(setup.single)),)),
            "standard": Protocol("standard", f"标准方案（{len(setup.standard)} 组）",
                                 f"standard worksheet ({len(setup.standard)} settings)",
                                 tuple(("main", float(s)) for s in setup.standard)),
        }
        for pid, (zh, en, plan) in setup.extra_protocols.items():
            protocols[pid] = Protocol(pid, zh, en, tuple((self._ext(k), float(s)) for k, s in plan))
        self.protocols = protocols

    # ---- 内部工具 ----

    def _ext(self, kind: str) -> str:
        return self.setup.kind_alias.get(kind, kind)

    def _add_kind(self, kinds, k, spec, ok):
        if k in kinds:
            raise ValueError(f"观测种类重名：{k}")
        kinds[k] = ok
        self._spec[k] = spec

    def effect_key(self, cause_id: str) -> str | None:
        if cause_id == self.random_only_id:
            return None
        return self._effect_of[cause_id][0]

    def ctx(self, kind: str, settings, params, cause_id: str) -> Ctx:
        typ, spec = self._spec[kind]
        flags = spec.flags if typ == "variant" else frozenset()
        n_count = spec.n_count if typ == "variant" else None
        internal = spec.kind if typ in ("variant", "aux") else "main"
        return Ctx(self.setup, internal, np.asarray(settings, dtype=float), params, self.effect_key(cause_id),
                   flags, n_count)

    def mean_value(self, kind: str, settings, params, cause_id: str):
        """读数均值（数组或 Affine），形状可广播到 (P, n)。"""
        ctx = self.ctx(kind, settings, params, cause_id)
        typ, spec = self._spec[kind]
        if typ == "aux":
            return spec.value(ctx)
        return self.setup.main_instrument.read(ctx, self.setup.template)

    # ---- labsim 接口 ----

    def predict(self, kind, settings, order, params, cause_id):
        settings = np.asarray(settings, dtype=float)
        P, n = int(np.asarray(params["mag"]).shape[0]), settings.size
        val = as_affine(self.mean_value(kind, settings, params, cause_id))
        names = [p.name for p in self.linear_params(cause_id)]
        c, U = val.to_cU(names, (P, n))
        return c[:, :, None], U[:, :, None, :]

    def reading_sigma(self, kind, params, cause_id):
        base = super().reading_sigma(kind, params, cause_id)
        key = self.effect_key(cause_id)
        if key is None:
            return base
        _, e, part = self._effect_of[cause_id]
        if not e.scatter:
            return base
        extra = part.extra_sigma(self.ctx(kind, np.zeros(1), params, cause_id))
        if extra is None:
            return base
        return np.sqrt(base**2 + np.asarray(extra, dtype=float)[:, :, None] ** 2)

    def relevant_nuisance(self, cause_id, kinds):
        need = set(self.setup.template.always_uses(self.setup))
        if cause_id != self.random_only_id:
            need |= set(self._effect_of[cause_id][1].uses)
        for k in kinds:
            typ, spec = self._spec.get(k, (None, None))
            if typ == "aux":
                need |= set(spec.uses)
        return tuple(n for n in self.nuisance if n in need)

    def linearise(self, ds: Dataset) -> LinearData:
        return self.setup.template.linearise(self, ds)

    # 同一物理量既是线性参数又在网格上（Setup.sync）：模拟和孪生时取线性参数的真值

    def _sync(self, truth: dict) -> dict:
        if not self.setup.sync:
            return truth
        nuis = dict(truth["nuisance"])
        for nk, lk in self.setup.sync.items():
            if lk in truth.get("linear", {}):
                nuis[nk] = truth["linear"][lk]
        return dict(truth, nuisance=nuis)

    def draw_truth(self, rng, cause_id, magnitude=None, nuisance=None, linear=None):
        return self._sync(super().draw_truth(rng, cause_id, magnitude, nuisance, linear))

    def _mean_and_sigma(self, truth, kind, settings, order):
        return super()._mean_and_sigma(self._sync(truth), kind, settings, order)


def compose(setup: Setup) -> ComposedExperiment:
    return ComposedExperiment(setup)


__all__ = ["ComposedExperiment", "compose", "Affine"]
