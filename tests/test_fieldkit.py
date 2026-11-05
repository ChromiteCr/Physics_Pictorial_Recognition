"""真实故障注入工具包：随机表、密封卡、记录表、中途推荐、冻结和预先规定分析（全部用模拟数据）。"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from lab_detective.fieldkit import analysis, catalog, ingest, live, schedule
from lab_detective.fieldkit.cards import render_cards
from lab_detective.fieldkit.catalog import EXCLUDED, FIELD_EXPERIMENTS, TWIN_PREFIX, build_engines
from lab_detective.fieldkit.sheet import PENDING, parse_sheet, read_sheet, render_sheet, write_sheet

pytestmark = pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")

FAST = {"eig_n_samples": 32}


def _build(seed=7, experiments=("hooke", "friction", "pendulum"), repeats=2, atlas=False, **kw):
    return schedule.build_schedule(experiments, repeats=repeats, sessions=2, seed=seed, atlas=atlas,
                                   analysis=FAST, **kw)


# ── 随机表 ───────────────────────────────────────────────────────────────


def test_schedule_is_deterministic_and_seed_dependent():
    a_pub, a_sea = _build(seed=11)
    b_pub, b_sea = _build(seed=11)
    assert a_pub == b_pub and a_sea == b_sea
    assert a_pub["commitment"] == schedule.sha256_obj(a_sea)
    c_pub, c_sea = _build(seed=12)
    assert c_pub["commitment"] != a_pub["commitment"]
    assert [r["code"] for r in c_pub["runs"]] != [r["code"] for r in a_pub["runs"]]


def test_schedule_is_balanced():
    pub, sea = _build(seed=3, repeats=2)
    assert pub["design"]["n_runs"] == 3 * 4 * 2 == len(pub["runs"]) == len(sea["key"])
    cells = Counter((k["experiment"], k["condition"]) for k in sea["key"])
    assert set(cells.values()) == {2}
    for s in (1, 2):  # 每次实验课每个单元恰好一次
        per = Counter((k["experiment"], k["condition"]) for k, r in zip(sea["key"], pub["runs"]) if r["session"] == s)
        assert set(per.values()) == {1} and len(per) == 12
    codes = [r["code"] for r in pub["runs"]]
    assert len(set(codes)) == len(codes) and all(len(c) == 4 for c in codes)
    for e in ("hooke", "friction", "pendulum"):  # 每个槽位在每个位置的次数最多差 1
        orders = [r["slots"] for r in pub["runs"] if r["experiment"] == e]
        assert all(sorted(o) == sorted(catalog.DEFAULT_SLOTS) for o in orders)
        for slot in catalog.DEFAULT_SLOTS:
            pos = Counter(o.index(slot) for o in orders)
            counts = [pos.get(i, 0) for i in range(3)]
            assert max(counts) - min(counts) <= 1
    for k in sea["key"]:  # 幅度只来自配方选项
        rec = next(r for r in FIELD_EXPERIMENTS[k["experiment"]].recipes if r.key == k["condition"])
        if rec.magnitudes:
            assert k["magnitude"]["value"] in [m.value for m in rec.magnitudes]
        else:
            assert k["magnitude"] is None


def test_public_schedule_hides_assignment():
    pub, sea = _build(seed=5)
    text = json.dumps(pub["runs"], ensure_ascii=False)
    for k in sea["key"]:
        assert k["condition"] not in text
    assert "seed" not in pub and "salt" not in json.dumps(pub)


def test_commitment_detects_tampering():
    pub, sea = _build(seed=9)
    assert schedule.verify_commitment(pub, sea)["ok"]
    assert schedule.regenerate_check(pub, sea)["ok"]
    bad = json.loads(json.dumps(sea))
    bad["key"][0]["condition"] = bad["key"][1]["condition"] if bad["key"][1]["condition"] != bad["key"][0]["condition"] \
        else bad["key"][2]["condition"]
    assert not schedule.verify_commitment(pub, bad)["ok"]
    bad_pub = json.loads(json.dumps(pub))
    bad_pub["runs"][0]["slots"] = list(reversed(bad_pub["runs"][0]["slots"]))
    assert not schedule.verify_commitment(bad_pub, sea)["ok"]


def test_atlas_check_records_pair_status():
    pub, _ = schedule.build_schedule(("friction", "pendulum"), repeats=1, sessions=1, seed=1,
                                     atlas={"n_per_cause": 40}, analysis=FAST)
    for e in ("friction", "pendulum"):
        pairs = pub["experiments"][e]["atlas_pairs"]
        assert len(pairs) == 6
        assert {p["status"] for p in pairs} <= {"confused", "partial", "separable", "same_class"}
    # 通用通道斜率未知：两种比例类故障与不设故障在引擎里是同一个原因
    same = [p for p in pub["experiments"]["pendulum"]["atlas_pairs"] if p["status"] == "same_class"]
    assert same and all(p["cause_a"] == p["cause_b"] for p in same)


def test_no_destructive_fault_and_build_refuses_one(monkeypatch):
    for fexp in FIELD_EXPERIMENTS.values():
        for rec in fexp.recipes:
            for suffixes in rec.truth.values():
                assert not any(f"{fexp.key}.{s}" in EXCLUDED for s in suffixes)
    bad = catalog.Recipe("x", {"twin": ("beyond_limit",)}, "adult", ("做点什么",))
    hooke = FIELD_EXPERIMENTS["hooke"]
    monkeypatch.setitem(FIELD_EXPERIMENTS, "hooke",
                        catalog.FieldExperiment(**{**hooke.__dict__, "recipes": hooke.recipes + (bad,)}))
    with pytest.raises(ValueError):
        schedule.build_schedule(("hooke",), repeats=1, sessions=1, seed=1, atlas=False)


def test_unsafe_menu_item_is_never_offered():
    pub, _ = _build(seed=2, experiments=("hooke",))
    info = pub["experiments"]["hooke"]
    assert "heavy_load" not in info["allowed_items"] and info["best_fixed"] in info["allowed_items"]
    pub2, _ = schedule.build_schedule(("hooke",), repeats=1, sessions=1, seed=2, atlas=False,
                                      config={"hooke": {"max_load_g": 400}}, analysis=FAST)
    assert "heavy_load" in pub2["experiments"]["hooke"]["allowed_items"]


def test_pendulum_generic_spec_is_frozen_and_stable():
    pub, _ = _build(seed=4, experiments=("pendulum",))
    info = pub["experiments"]["pendulum"]
    assert info["primary"] == "generic" and info["spec"]["id"] == "other_pendulum"
    e1 = build_engines("pendulum", info["config"], info["spec"])["generic"]
    e2 = build_engines("pendulum", info["config"], json.loads(json.dumps(info["spec"])))["generic"]
    assert e1.cause_ids() == e2.cause_ids()
    for c1, c2 in zip(e1.causes, e2.causes):
        assert c1.magnitude == c2.magnitude and c1.linear == c2.linear and c1.description_zh == c2.description_zh
    assert [m.plan for m in e1.menu] == [m.plan for m in e2.menu]


# ── 密封卡 ───────────────────────────────────────────────────────────────


def _forbidden_words(pub):
    words = {"干净", "clean", "random_only", "对照", "不设故障"}
    for e in pub["design"]["experiments"]:
        info = pub["experiments"][e]
        for eng in build_engines(e, info["config"], info.get("spec")).values():
            for c in eng.causes:
                words |= {c.id, c.name_zh, c.description_zh, c.id.split(".", 1)[1]}
        words |= {r.key for r in FIELD_EXPERIMENTS[e].recipes}
    return {w for w in words if len(w) >= 3 or any("一" <= ch <= "鿿" for ch in w)}


def test_cards_contain_codes_but_no_condition_names():
    pub, sea = _build(seed=21, experiments=("hooke", "friction", "newton2", "pendulum"), repeats=1)
    html = render_cards(pub, sea)
    for k in sea["key"]:
        assert k["code"] in html
    low = html.lower()
    hits = [w for w in _forbidden_words(pub) if w.lower() in low]
    assert not hits, hits
    assert "dashed" in html and "✂" in html  # 剪切线
    assert not re.search(r"(https?:)?//[a-z0-9]", html)  # 离线：不引用外部文件
    assert "<script" not in html


# ── 记录表 ───────────────────────────────────────────────────────────────


def test_sheet_roundtrip_and_strict_columns(tmp_path):
    pub, _ = _build(seed=8, experiments=("friction",), repeats=1)
    sh = schedule.run_sheet(pub, pub["runs"][0], "friction")
    text = render_sheet(sh)
    back = parse_sheet(text)
    assert back.meta == sh.meta and [r.as_list() for r in back.rows] == [r.as_list() for r in sh.rows]
    with pytest.raises(ValueError):
        parse_sheet(text.replace("step,stage,item,kind,setting,value,unit", "step,stage,item,kind,setting,value,unit,姓名"))
    with pytest.raises(ValueError):
        parse_sheet("#@ 姓名: 张三\n" + text)


def _gen_truth(engines, key, rng):
    """模拟用的真值：数据由留出孪生 / 手写孪生生成（物理上的故障）；通用通道独有的读数用通用引擎补。"""
    tw = engines["twin"]
    t_twin = tw.draw_truth(rng, key["truth"]["twin"]["cause_id"])
    t_gen = None
    if "generic" in engines:
        g = engines["generic"]
        cid = key["truth"]["generic"]["cause_id"]
        t_gen = g.draw_truth(rng, cid)
        if cid.endswith(".x_offset"):
            t_gen["magnitude"] = t_twin["nuisance"]["r_cm"]
    return t_twin, t_gen


def _observe(engines, truths, kind, setting, order, rng):
    k = kind[len(TWIN_PREFIX):] if kind.startswith(TWIN_PREFIX) else kind
    eng, t = (engines["twin"], truths[0]) if k in engines["twin"].obs_kinds else (engines["generic"], truths[1])
    return eng.observe(rng, t, [(k, float(setting))], start_order=order)[0].values[0]


def simulate_run(path: Path, pub: dict, key: dict, seed: int) -> None:
    """模拟学生做一次运行：基础方案 → 运行 live → 追加测量。"""
    rng = np.random.default_rng(seed)
    sh = read_sheet(path)
    engines = live.sheet_engines(pub, sh)
    truths = _gen_truth(engines, key, rng)
    for i, r in enumerate(sh.base_rows()):
        r.value = f"{_observe(engines, truths, r.kind, r.setting, i, rng):.6g}"
    write_sheet(sh, path)
    live.run_live(path, pub, now="2026-11-10T10:00:00+08:00")
    sh = read_sheet(path)
    for i, r in enumerate(sh.rows):
        if r.stages != ("base",):
            r.value = f"{_observe(engines, truths, r.kind, r.setting, i, rng):.6g}"
    write_sheet(sh, path)


def _lab(tmp_path, experiments=("hooke", "friction", "pendulum"), repeats=1, seed=31, n=None):
    pub, sea = schedule.build_schedule(experiments, repeats=repeats, sessions=1, seed=seed,
                                       atlas={"n_per_cause": 20}, analysis=FAST)
    out = schedule.write_outputs(pub, sea, tmp_path / "kit")
    data = tmp_path / "data"
    data.mkdir()
    keys = {k["code"]: k for k in sea["key"]}
    for i, r in enumerate(pub["runs"][: n or len(pub["runs"])]):
        src = tmp_path / "kit" / "public" / "runs" / f"{r['run_id']}_{r['code']}.csv"
        dst = data / src.name
        dst.write_bytes(src.read_bytes())
        simulate_run(dst, pub, keys[r["code"]], seed=1000 + i)
    return pub, sea, data, out


def test_write_outputs_and_templates(tmp_path):
    pub, sea = _build(seed=13, experiments=("hooke", "newton2"), repeats=1)
    out = schedule.write_outputs(pub, sea, tmp_path)
    for e in ("hooke", "newton2"):
        t = read_sheet(tmp_path / "public" / "templates" / f"{e}.csv")
        assert t.meta["experiment"] == e and len(t.base_rows()) == len(pub["experiments"][e]["base_plan"])
        assert any(r.item == PENDING for r in t.rows)
    assert len(list((tmp_path / "public" / "runs").glob("*.csv"))) == 8
    loaded = schedule.load_json(out["public"])
    assert loaded["commitment"] == pub["commitment"]
    assert schedule.verify_commitment(loaded, schedule.load_json(out["sealed"]))["ok"]


def test_live_fills_eig_without_revealing_and_merges_duplicates(tmp_path):
    pub, sea, data, _ = _lab(tmp_path, experiments=("hooke",), n=4)
    for p in sorted(data.glob("*.csv")):
        sh = read_sheet(p)
        rc = json.loads(p.with_suffix(".eig.json").read_text(encoding="utf-8"))
        chosen = rc["slots"]["eig"]["chosen"]
        assert chosen in pub["experiments"]["hooke"]["allowed_items"]
        assert "eig" in {s for r in sh.followup_rows() for s in r.stages}
        assert not any(r.item == PENDING for r in sh.rows)
        assert {r.item for r in sh.followup_rows() if "eig" in r.stages} == {chosen}
        if chosen == pub["experiments"]["hooke"]["best_fixed"]:  # 相同只做一次
            assert any(set(r.stages) == {"best_fixed", "eig"} for r in sh.rows)
        dump = json.dumps(rc, ensure_ascii=False)
        assert "hooke." not in dump and "p_true" not in dump  # 回执里没有原因和后验
        assert "ranking" not in dump and "eig_bits" not in dump  # 也没有排名（会透露引擎的偏向）
        lines = live.todo_lines(pub, sh)
        assert not any("hooke." in ln for ln in lines)


def test_live_refuses_after_followups_started(tmp_path):
    pub, sea = _build(seed=17, experiments=("friction",), repeats=1)
    schedule.write_outputs(pub, sea, tmp_path / "kit")
    run = pub["runs"][0]
    p = tmp_path / "kit" / "public" / "runs" / f"{run['run_id']}_{run['code']}.csv"
    sh = read_sheet(p)
    for r in sh.base_rows():
        r.value = "1.2"
    next(r for r in sh.followup_rows() if r.item != PENDING).value = "1.0"
    write_sheet(sh, p)
    with pytest.raises(ValueError):
        live.run_live(p, pub)


# ── 冻结和数据集 ─────────────────────────────────────────────────────────


def test_ingest_roundtrip_freeze_and_tamper(tmp_path):
    pub, sea, data, out = _lab(tmp_path, experiments=("friction", "pendulum"), n=None)
    res = ingest.check_folder(data, pub)
    assert res["errors"] == {} and res["missing_runs"] == []
    m = ingest.freeze(data, pub, out["public"])
    assert ingest.verify(data)["ok"] and len(m["files"]) == 2 * len(pub["runs"])
    assert "frozen_by" not in json.dumps(m)  # 不记录人
    for p in sorted(data.glob("*.csv")):
        sh = read_sheet(p)
        ds = ingest.run_datasets(sh, pub)
        info = pub["experiments"][sh.meta["experiment"]]
        prim = ds[info["primary"]]
        vals = [float(r.value) for r in sh.base_rows()]
        assert [o.values[0] for o in prim["base"].observations] == vals
        assert [o.setting for o in prim["base"].observations] == [float(r.setting) for r in sh.base_rows()]
        run = next(r for r in pub["runs"] if r["code"] == sh.meta["code"])
        for slot in run["slots"]:
            if slot in prim["single"]:
                extra = [r for r in sh.followup_rows() if slot in r.stages]
                assert len(prim["single"][slot].observations) == len(vals) + len(extra)
        if sh.meta["experiment"] == "pendulum":  # 留出孪生只拿到共享的 main 行
            tw = ds["twin"]
            assert all(o.kind in tw["engine"].obs_kinds for o in tw["all"].observations)
            assert all(o.kind == "main" for o in tw["all"].observations)
    first = sorted(data.glob("*.csv"))[0]
    first.write_text(first.read_text(encoding="utf-8-sig").replace(",base,", ",base,", 1) + "\n", encoding="utf-8")
    assert not ingest.verify(data)["ok"]


def test_ingest_reports_bad_sheets(tmp_path):
    pub, sea, data, _ = _lab(tmp_path, experiments=("hooke",), n=2)
    paths = sorted(data.glob("*.csv"))
    sh = read_sheet(paths[0])
    sh.base_rows()[0].value = ""
    write_sheet(sh, paths[0])
    sh = read_sheet(paths[1])
    sh.base_rows()[1].value = "12.3"  # 推荐之后改了基础读数
    sh.rows[-1].unit = "mm"
    write_sheet(sh, paths[1])
    res = ingest.check_folder(data, pub)
    assert paths[0].name in res["errors"] and paths[1].name in res["errors"]
    joined = " ".join(res["errors"][paths[1].name])
    assert "指纹" in joined and "单位" in joined
    with pytest.raises(ValueError):
        ingest.freeze(data, pub)
    assert len(res["missing_runs"]) == len(pub["runs"]) - 2


# ── 分析 ─────────────────────────────────────────────────────────────────


def test_analysis_requires_student_and_frozen_data(tmp_path):
    pub, sea, data, out = _lab(tmp_path, experiments=("friction",), n=2)
    with pytest.raises(PermissionError):
        analysis.run_analysis(pub, sea, data, tmp_path / "res", confirm="")
    with pytest.raises(ValueError):  # 还没冻结
        analysis.run_analysis(pub, sea, data, tmp_path / "res", confirm=analysis.CONFIRM_PHRASE)


def test_analysis_end_to_end(tmp_path):
    pub, sea, data, out = _lab(tmp_path, experiments=("hooke", "friction", "pendulum"), seed=41)
    ingest.freeze(data, pub, out["public"])
    res = analysis.run_analysis(pub, sea, data, tmp_path / "res", confirm=analysis.CONFIRM_PHRASE)
    assert len(res["runs"]) == 12
    s = res["summary"]
    prim = s["contrasts"]["eig − best_fixed"]
    assert prim["role"] == "primary" and prim["pooled"]["experiments"] == ["hooke", "friction"]
    assert prim["pooled"]["n_pairs"] == 8
    for k, by in s["accuracy"].items():
        a = by["base"]
        assert a["n"] >= 1 and (a["lo"] is None or 0.0 <= a["lo"] <= a["acc"] <= a["hi"] <= 1.0)
    assert all(x["reproduced"] for v in s["eig_reproduced"].values() for x in v)
    for r in res["runs"]:
        for m in r["engines"].values():
            assert 0.0 <= m["base"]["p_true"] <= 1.0
    assert "pendulum/twin" in s["accuracy"]  # 单摆同时用留出孪生分析
    n_scale = sum(1 for k in sea["key"] if k["experiment"] == "pendulum" and k["condition"] in ("wide_swing", "count_slip"))
    eq = s["equivalent"]["pendulum/generic"]
    assert eq["n_equivalent"] == n_scale == 2 and set(eq["conditions"]) == {"wide_swing", "count_slip"}
    assert eq["non_equivalent"]["base"]["n"] == 4 - n_scale
    assert s["equivalent"]["hooke/twin"]["n_equivalent"] == 0 and s["equivalent"]["pendulum/twin"]["n_equivalent"] == 0
    assert "等价" in (tmp_path / "res" / "analysis_tables.md").read_text(encoding="utf-8")
    assert (tmp_path / "res" / "analysis_tables.md").read_text(encoding="utf-8").startswith("# 真实实验分析结果")
    json.dumps(res)  # 没有 NaN、numpy 类型


def test_stale_truth_ids_are_reported():
    pub, sea = _build(seed=19, experiments=("friction",), repeats=1)
    assert analysis._stale_truths(pub, sea) == []
    bad = json.loads(json.dumps(sea))
    bad["key"][0]["truth"]["twin"]["cause_id"] = "friction.renamed_cause"
    assert analysis._stale_truths(pub, bad) == [f"{bad['key'][0]['run_id']} twin: friction.renamed_cause"]


def test_analysis_spec_and_atlas_overrides_are_committed():
    base_pub, _ = _build(seed=23, experiments=("friction",), repeats=1)
    kv = schedule.parse_kv(["wilcoxon_alternative=greater", "alpha=0.01", 'primary_contrast=["eig","repeat"]'])
    assert kv == {"wilcoxon_alternative": "greater", "alpha": 0.01, "primary_contrast": ["eig", "repeat"]}
    pub, sea = schedule.build_schedule(("friction",), repeats=1, sessions=2, seed=23, analysis=dict(FAST, **kv),
                                       atlas={"confused_below": 0.7, "n_per_cause": 20})
    assert pub["analysis_spec"]["wilcoxon_alternative"] == "greater" and pub["analysis_spec"]["alpha"] == 0.01
    assert pub["atlas_check"]["confused_below"] == 0.7
    assert pub["commitment"] != base_pub["commitment"] and schedule.verify_commitment(pub, sea)["ok"]
    for bad in ({"wilcoxon_zero_method": "drop"}, {"no_such_key": 1}, {"primary_contrast": ["eig", "magic"]}):
        with pytest.raises(ValueError):
            schedule.build_schedule(("friction",), repeats=1, sessions=1, seed=1, atlas=False, analysis=bad)
    with pytest.raises(ValueError):
        schedule.build_schedule(("friction",), repeats=1, sessions=1, seed=1, atlas={"confused_below": 0.95})


def test_sealed_key_tamper_blocks_analysis(tmp_path):
    pub, sea, data, out = _lab(tmp_path, experiments=("friction",), n=2)
    ingest.freeze(data, pub, out["public"])
    bad = json.loads(json.dumps(sea))
    bad["key"][0]["truth"]["twin"]["cause_id"] = "friction.random_only"
    bad["key"][0]["condition"] = "clean" if bad["key"][0]["condition"] != "clean" else "board_shim"
    with pytest.raises(ValueError):
        analysis.run_analysis(pub, bad, data, tmp_path / "res", confirm=analysis.CONFIRM_PHRASE)


def test_sheet_tolerates_spreadsheet_resave():
    pub, _ = _build(seed=8, experiments=("pendulum",), repeats=1)
    text = render_sheet(schedule.run_sheet(pub, pub["runs"][0], "pendulum"))
    resaved = "\n".join(ln + ",,,,,," if ln.startswith("#") else ln for ln in text.splitlines())
    a, b = parse_sheet(text), parse_sheet(resaved)
    assert a.meta == b.meta and a.config == b.config
    assert b.config["lengths_cm"] == [40.0, 55.0, 70.0, 85.0, 100.0]
    assert "," not in next(ln for ln in text.splitlines() if "lengths_cm" in ln)


def test_atlas_thresholds_match_atlas_module():
    import inspect

    from lab_detective.atlas import atlas_table

    sig = inspect.signature(atlas_table).parameters
    assert schedule.ATLAS_CHECK_DEFAULT["confused_below"] == sig["confused_below"].default
    assert schedule.ATLAS_CHECK_DEFAULT["separable_at"] == sig["separable_at"].default


# ── 给学生的文档 ─────────────────────────────────────────────────────────

DOCS = Path(__file__).resolve().parents[1] / "docs" / "student"


def test_derivation_sheet_has_every_training_cause_and_no_answers():
    from lab_detective.fieldkit.student_sheet import render
    from lab_detective.labsim.core import SIGNATURES
    from lab_detective.labsim.registry import held_out_experiments, training_experiments

    text = render()
    for exp in training_experiments():
        for c in exp.causes:
            assert c.id in text and c.name_zh in text
            if c.id != exp.random_only_id:
                assert c.description_zh in text
            for leak in (c.fix_zh, c.magnitude_label_zh):
                if leak and leak not in c.description_zh:  # 很短的标签（如"车身长"）可能本来就在描述里
                    assert leak not in text, (c.id, leak)
    descs = " ".join(c.description_zh for e in training_experiments() for c in e.causes)
    for exp in training_experiments():  # 不列产品的追加测量菜单（有些菜单名直接点明了分开它的测量）
        for m in exp.menu:
            if m.label_zh not in descs:
                assert m.label_zh not in text, (exp.id, m.label_zh)
    for exp in held_out_experiments():
        assert f"`{exp.id}" not in text and exp.name_zh not in text
    assert not any(f"`{s}`" in text or f" {s} " in text for s in SIGNATURES if s != "none")
    committed = (DOCS / "01_手推清单.md").read_text(encoding="utf-8")
    for exp in training_experiments():  # 提交的文档没有落后于代码（原因 id 都在）
        for c in exp.causes:
            assert c.id in committed, f"{c.id}：请重新生成 01_手推清单.md"


def test_student_docs_exist_and_avoid_fixed_experiment_counts():
    names = ["01_手推清单.md", "02_你要做的决定.md", "03_GPT审查包.md", "04_Hy4使用说明.md", "05_明天的5090操作.md",
             "06_真实实验工具包.md"]
    for n in names:
        t = (DOCS / n).read_text(encoding="utf-8")
        assert not re.search(r"\d+\s*个(中学力学|力学|课内|留出)?实验", t), n
        assert "http" not in t or n == "05_明天的5090操作.md"
