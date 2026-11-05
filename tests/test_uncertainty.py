"""不确定度预算与线性传播：用加权最小二乘的解析结果做对照。"""
import json
import math

import numpy as np
import pytest

from lab_detective.labsim import get_experiment
from lab_detective.labsim.core import (
    Cause, Experiment, LinearData, LinearParam, ObsKind, Protocol, Reading,
)
from lab_detective.uncertainty import budget, propagate, slope_unit

EXP = get_experiment("hooke")
G = 9.8


def _wls_oracle(x, sy):
    """等权直线拟合的解析标准差：sd(a) = σ/√Sxx，sd(b) = σ·√(1/n + x̄²/Sxx)。"""
    sxx = np.sum((x - x.mean()) ** 2)
    return sy / math.sqrt(sxx), sy * math.sqrt(1 / x.size + x.mean() ** 2 / sxx)


def test_hooke_budget_matches_analytic_wls():
    b = budget(EXP, "standard", n=400, rng=np.random.default_rng(0))
    json.dumps(b, allow_nan=False, ensure_ascii=False)
    F = np.array([50, 100, 150, 200, 250]) / 1000 * G
    sy = math.sqrt(0.05**2 + 0.1**2 / 12)  # σ 与分辨率 d/√12 合成
    sa, sb = _wls_oracle(F, sy)
    s, i = b["slope"], b["intercept"]
    assert s["unit"] == "cm/N" and i["unit"] == "cm"
    assert s["nominal"] == pytest.approx(100 * 0.05, rel=1e-9)  # 名义 1/k = 0.05 m/N → 5 cm/N
    assert s["sd_formula"] == pytest.approx(sa, rel=1e-6)
    assert s["sd"] == pytest.approx(sa, rel=0.12)
    assert i["sd"] == pytest.approx(sb, rel=0.12)
    # 方差份额：σ² : d²/12 = 0.0025 : 0.00083 = 75% : 25%
    for q in (s, i):
        shares = {it["type"]: it["share"] for it in q["sources"]}
        assert sum(shares.values()) == pytest.approx(1.0)
        assert shares["random"] == pytest.approx(0.75, abs=0.1)
        assert shares["resolution"] == pytest.approx(0.25, abs=0.1)
    assert "伸长量" in s["sources"][0]["label_zh"]
    assert "斜率" in b["summary_zh"]


R = Reading("y", "位移", "displacement", "cm", resolution=0.1, sigma=0.0)


class Toy(Experiment):
    """只有分辨率、没有随机噪声的读数：取整若不抖动，预算会错算成 0。"""

    id = "toy"
    name_zh = "玩具实验"
    obs_kinds = {"main": ObsKind("main", "读位移", "read", (R,), setting_unit="s")}
    linear = (LinearParam("a", 1.234, 0.1, "速度", "cm/s"),)
    causes = (Cause("toy.random_only", "只有随机误差", "random only", "", "", "none", ("none",)),)
    protocols = {"standard": Protocol("standard", "五点", "five", tuple(("main", float(s)) for s in range(1, 6)))}

    def predict(self, kind, settings, order, params, cause_id):
        P, n = params["mag"].shape[0], settings.size
        U = np.zeros((P, n, 1, 1))
        U[:, :, 0, 0] = settings[None, :]
        return np.zeros((P, n, 1)), U

    def linearise(self, ds):
        x = np.array([o.setting for o in ds.observations])
        y = np.array([o.values[0] for o in ds.observations])
        return LinearData(x, y, np.full(x.shape, 0.1 / math.sqrt(12)), np.arange(x.size),
                          "时间 t (s)", "位移 y (cm)", None, 0.0)


def test_resolution_only_reading_is_not_zero():
    b = budget(Toy(), n=400, rng=np.random.default_rng(1))
    sa, _ = _wls_oracle(np.arange(1.0, 6.0), 0.1 / math.sqrt(12))
    assert b["slope"]["sd"] == pytest.approx(sa, rel=0.12)
    assert b["slope"]["unit"] == "cm/s"
    assert [it["type"] for it in b["slope"]["sources"]] == ["resolution"]
    assert b["slope"]["sources"][0]["share"] == pytest.approx(1.0)


def test_single_reading_protocol_gives_note_not_nan():
    b = budget(EXP, "single", n=50)
    json.dumps(b, allow_nan=False, ensure_ascii=False)
    assert b["slope"] is None and "无法" in b["summary_zh"]


def test_propagate_oracles():
    r = propagate(lambda s: 100 / s, [4.0], [0.04])
    assert r["value"] == pytest.approx(25.0)
    assert r["sd"] == pytest.approx(0.25, rel=1e-5)
    r = propagate(lambda a, b: a * b, {"a": 2.0, "b": 3.0}, {"a": 0.1, "b": 0.2})
    assert r["sd"] == pytest.approx(0.5, rel=1e-6)
    shares = {t["name"]: t["share"] for t in r["terms"]}
    assert shares["a"] == pytest.approx(0.36, rel=1e-5) and shares["b"] == pytest.approx(0.64, rel=1e-5)
    json.dumps(r, allow_nan=False)


def test_slope_units():
    assert slope_unit("拉力 F (N)", "伸长量 x (cm)") == "cm/N"
    assert slope_unit("1/L1 (1/m)", "拉力 F (N)") == "N·m"
    assert slope_unit("理论 (J/kg)", "实测 (J/kg)") == ""
    assert slope_unit("拉力 F (N)", "加速度 a (m/s²)") == "(m/s²)/N"


R3 = tuple(Reading(f"t{i}", f"时间第 {i} 次", f"time #{i}", "s", resolution=0.0, sigma=0.05) for i in (1, 2, 3))


class Toy3(Toy):
    """同一位置测 3 次、取平均：三个读数地位完全相同，份额应各占 1/3。"""

    id = "toy3"
    causes = (Cause("toy3.random_only", "只有随机误差", "random only", "", "", "none", ("none",)),)
    obs_kinds = {"main": ObsKind("main", "计时 3 次", "time x3", R3, setting_unit="m")}

    def predict(self, kind, settings, order, params, cause_id):
        P, n = params["mag"].shape[0], settings.size
        U = np.zeros((P, n, 3, 1))
        U[:, :, :, 0] = settings[None, :, None]
        return np.zeros((P, n, 3)), U

    def linearise(self, ds):
        x = np.array([o.setting for o in ds.observations])
        y = np.array([np.mean(o.values) for o in ds.observations])
        return LinearData(x, y, np.full(x.shape, 0.05 / math.sqrt(3)), np.arange(x.size),
                          "距离 s (m)", "时间 t (s)", None, 0.0)


def test_identical_readings_get_equal_shares():
    b = budget(Toy3(), n=400, rng=np.random.default_rng(2))
    sa, _ = _wls_oracle(np.arange(1.0, 6.0), 0.05 / math.sqrt(3))
    assert b["slope"]["sd"] == pytest.approx(sa, rel=0.12)
    shares = [it["share"] for it in b["slope"]["sources"]]
    assert len(shares) == 3 and all(it["type"] == "random" for it in b["slope"]["sources"])
    assert all(abs(s - 1 / 3) < 0.06 for s in shares), shares
