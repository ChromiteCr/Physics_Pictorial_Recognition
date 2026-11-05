"""事实单：形状与 ARCHITECTURE.md §2.1 一致、严格 JSON、文字里的每个数字都登记在 numbers 里。

并行模块（eig、knowledge）一律用 sys.modules 注入的假模块，不依赖它们的真实实现。
"""
import json
import re
import sys
import types
from dataclasses import dataclass, field

import numpy as np
import pytest

from lab_detective.facts import Numbers, band_of, build_fact_sheet, dec_for_unc, to_markdown_zh
from lab_detective.labsim import Dataset, Observation, get_experiment

EXP = get_experiment("hooke")
KEYS = {"experiment_id", "experiment_name", "protocol", "data", "panel", "agreement", "diagnosis",
        "next_measurement", "fix", "knowledge_points", "numbers"}


@dataclass
class FakeScore:
    item_id: str
    label_zh: str
    minutes: float
    eig_bits: float
    eig_per_min: float
    predicted: dict = field(default_factory=dict)
    separates: list = field(default_factory=list)


def _install(monkeypatch, heavy_first=False):
    eig = types.ModuleType("lab_detective.eig")
    weigh = FakeScore("weigh_hanger", "用天平称挂钩质量", 2.0, 0.8, 0.4,
                      {"hooke.hanger_omitted": {"mean": 23.04, "sd": 0.1, "unit": "g"},
                       "hooke.zero_offset": {"mean": 27.5, "sd": 13.0, "unit": "g"}},
                      ["hooke.hanger_omitted", "hooke.zero_offset"])
    heavy = FakeScore("heavy_load", "再挂一组更重的钩码（350 g）", 2.0, 0.3, 0.15,
                      {"hooke.beyond_limit": {"mean": 21.137, "sd": 1.0, "unit": "cm"}}, [])
    eig.recommend = lambda exp, ds, n_samples, rng: [heavy, weigh] if heavy_first else [weigh, heavy]
    kn = types.ModuleType("lab_detective.knowledge")
    kn.load_card = lambda exp_id: {"knowledge_points": ["kp.hooke_law", "kp.systematic_error"]}
    monkeypatch.setitem(sys.modules, "lab_detective.eig", eig)
    monkeypatch.setitem(sys.modules, "lab_detective.knowledge", kn)


@pytest.fixture
def fake(monkeypatch):
    _install(monkeypatch)


def _sim(cause, seed, **kw):
    return EXP.simulate(np.random.default_rng(seed), cause, **kw)


def test_shape_and_strict_json(fake):
    ds = _sim("hooke.total_length", 0)
    fs = build_fact_sheet(EXP, ds)
    assert set(fs) == KEYS
    json.dumps(fs, allow_nan=False, ensure_ascii=False)
    assert fs["panel"]["slope_ratio"] is None  # 胡克定律斜率自由：NaN → null
    assert fs["data"]["n_points"] == 5 and fs["data"]["table"][0] == {"kind": "main", "setting": 50.0,
                                                                       "values": list(ds.observations[0].values)}
    a = fs["agreement"]
    assert a["quantity"] == "截距" and a["unit"] == "cm" and a["band"] == "inconsistent"
    dg = fs["diagnosis"]
    assert len(dg["ranked"]) == 4 and dg["ranked"][0]["cause_id"] == "hooke.total_length"
    assert "截距" in dg["ranked"][0]["evidence"] and dg["not_in_list"] is False
    assert sum(r["prob"] for r in dg["ranked"]) <= 1.0 + 1e-9
    nx = fs["next_measurement"]
    assert nx["id"] == "weigh_hanger" and nx["eig_bits"] == 0.8
    assert nx["predicted"][0] == {"cause_id": "hooke.hanger_omitted", "name": "漏算挂钩质量", "value": 23.0, "unit": "g"}
    assert fs["fix"]["cause_id"] == "hooke.total_length" and "预计截距" in fs["fix"]["predicted_rerun"]
    assert fs["knowledge_points"] == ["kp.hooke_law", "kp.systematic_error"]
    assert all(set(n) == {"name", "value", "unit"} for n in fs["numbers"])


def _texts(fs):
    out = [r["evidence"] for r in fs["diagnosis"]["ranked"]]
    out += [fs["fix"]["text"], fs["fix"]["predicted_rerun"], to_markdown_zh(fs)]
    if fs.get("next_measurement"):
        out.append(fs["next_measurement"]["label"])
    return out


CASES = [
    ("hooke.total_length", {}),
    ("hooke.beyond_limit", {"magnitude": 0.5}),
    ("hooke.reading_scatter", {"magnitude": 0.6}),
    ("hooke.random_only", {}),
    ("hooke.hanger_omitted", {"nuisance": {"hanger_g": 40.0}, "linear": {"inv_k": 0.04}}),
    ("hooke.zero_offset", {"linear": {"zero_offset": -0.8}}),
]


@pytest.mark.parametrize("cause,kw", CASES)
@pytest.mark.parametrize("heavy_first", [False, True])
def test_every_displayed_number_is_registered(monkeypatch, cause, kw, heavy_first):
    """校验器只认 numbers：解释里可能出现的每个数字（含百分数、z、p、菜单名里的 350 g）都要在表里。"""
    _install(monkeypatch, heavy_first)
    fs = build_fact_sheet(EXP, _sim(cause, 7, **kw))
    values = [n["value"] for n in fs["numbers"]]
    for text in _texts(fs):
        for tok in re.findall(r"[-+]?\d+(?:\.\d+)?", text):
            v = float(tok)
            assert any(abs(v - u) <= 1e-9 * max(1.0, abs(u)) for u in values), (tok, text)


def test_rounding_follows_uncertainty():
    assert dec_for_unc(0.0605) == 2  # 0.06
    assert dec_for_unc(0.15) == 2  # 首位是 1：保留两位有效数字
    assert dec_for_unc(0.012) == 3
    assert dec_for_unc(23.0) == 0
    n = Numbers()
    assert n.add("intercept", 1.5234, "cm", dec=2, signed=True) == "+1.52"
    assert n.add("rerun", 1e-17, "cm", dec=2, signed=True) == "0.00"
    assert n.items[0] == {"name": "intercept", "value": 1.52, "unit": "cm"}
    assert n.p("p", 3e-7) == "<0.0001" and n.p("q", 0.04321) == "0.043"


def test_band_thresholds():
    assert band_of(1.9) == "consistent" and band_of(-2.5) == "tension"
    assert band_of(3.0) == "tension" and band_of(-3.1) == "inconsistent"
    assert band_of(None) is None and band_of(float("nan")) is None


def test_parallel_modules_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "lab_detective.eig", None)
    monkeypatch.setitem(sys.modules, "lab_detective.knowledge", None)
    fs = build_fact_sheet(EXP, _sim("hooke.random_only", 1))
    assert "next_measurement" not in fs and set(fs) == KEYS - {"next_measurement"}
    assert fs["knowledge_points"] == []
    assert fs["fix"]["predicted_rerun"] == "" or fs["diagnosis"]["ranked"][0]["cause_id"] != "hooke.random_only"
    assert "嫌疑人看板" in to_markdown_zh(fs)


def test_not_in_list_on_wild_data(fake):
    vals = [1.0, 9.0, 3.0, 11.0, 5.0]  # 锯齿：任何候选原因都解释不了
    ds = Dataset("hooke", [Observation("main", m, (v,)) for m, v in zip((50.0, 100.0, 150.0, 200.0, 250.0), vals)],
                 None, {"protocol": "standard"})
    fs = build_fact_sheet(EXP, ds)
    assert fs["diagnosis"]["not_in_list"] is True and fs["diagnosis"]["misfit_p"] < 0.01
    assert "清单以外" in to_markdown_zh(fs)


def test_experiment_defined_agreement_is_used(fake, monkeypatch):
    def agreement(ds, panel):  # 例如：k = 100/斜率，按线性传播给不确定度
        k = 100.0 / panel.slope
        return {"quantity": "k", "value": k, "uncertainty": k * panel.slope_se / panel.slope,
                "unit": "N/m", "z": 2.4}

    monkeypatch.setattr(EXP, "agreement", agreement, raising=False)
    fs = build_fact_sheet(EXP, _sim("hooke.random_only", 3, linear={"inv_k": 0.04}))
    a = fs["agreement"]
    assert a["quantity"] == "k" and a["unit"] == "N/m" and a["band"] == "tension"
    assert a["value"] == pytest.approx(25.0, abs=1.0)
    assert {"name": "agreement.value", "value": a["value"], "unit": "N/m"} in fs["numbers"]


def test_single_reading_hides_band(fake):
    fs = build_fact_sheet(EXP, _sim("hooke.random_only", 4, protocol="single"))
    json.dumps(fs, allow_nan=False, ensure_ascii=False)
    assert fs["agreement"]["band"] is None
    assert all("少于 3 个" in r["evidence"] for r in fs["diagnosis"]["ranked"])


def test_cause_subset_and_aux_rows(fake):
    rng = np.random.default_rng(5)
    ds = EXP.simulate(rng, "hooke.hanger_omitted", nuisance={"hanger_g": 20.0})
    ds = EXP.extend(rng, ds, EXP.menu_item("weigh_hanger").plan)
    fs = build_fact_sheet(EXP, ds, cause_ids=("hooke.random_only", "hooke.hanger_omitted", "hooke.zero_offset"))
    assert [r["cause_id"] for r in fs["diagnosis"]["ranked"]][0] == "hooke.hanger_omitted"
    assert len(fs["diagnosis"]["ranked"]) == 3
    aux = fs["data"]["table"][-1]
    assert aux["kind"] == "hanger_mass"
    assert {"name": "data[5].hanger_mass", "value": aux["values"][0], "unit": "g"} in fs["numbers"]
