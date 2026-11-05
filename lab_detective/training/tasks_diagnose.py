"""诊断任务的训练样本：模拟数据 → 提示（prompts.build_diagnose_messages）→ 引擎给出的目标 JSON。

数据来源：训练实验（去掉扣下的原因，见 splits.py）+ labsim/generic.py 的程序化任务族。
- 方案配比：单次读数 15% / 标准方案 60% / 标准方案 + 1 项追加测量 25%；真实原因约 25% 是"只有随机误差"。
- 20% 是"自己提出原因"变体（不给候选清单），目标是引擎排名前 6 的原因（中文名 + 机制 + 预期特征 + 概率）。
- 8% 是"真实原因不在清单里"变体（只在真实原因不是随机误差、剩下候选 ≥ 3 个时）：目标照样来自引擎，
  教模型在对不上时说 not_in_list。
- 候选原因和追加测量的编号每条样本独立打乱（C1…、M1…）。

目标（全部由确定性代码给出，模型不算物理）：
  signatures   面板判出的数据特征
  evidence     事实单里最可能原因的证据句
  ranked       全部候选原因（按未取整的后验从大到小），prob 取整到 0.05，expect = 原因在标准方案下的预期特征
  not_in_list  引擎的失配判据
  next_measurement  EIG 最大的追加测量（n_samples≈64），predicted = 后验前两名原因下的预期读数
  agreement    一致性色带（consistent / tension / inconsistent / null）
  explanation  ≤ 80 字，用 tutor/templates.py 的讲解块拼成，数字与上面一致并尽量通过校验器
meta.causes = [{"id": "C3", "name": 中文名}]（打乱标签对照按它置换），meta.id_map 记显示编号 → 真实原因 id。
"""

from __future__ import annotations

import copy
import json
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..eig import protocol_plan
from ..engine import posterior
from ..facts import build_fact_sheet
from ..labsim.core import Dataset, Experiment
from ..labsim.generic import procedural_task_family
from ..labsim.registry import get_experiment
from ..tutor import templates as tpl
from ..tutor.verifier import verify
from . import prompts
from .format import Example
from .splits import TRAIN_EXPERIMENTS, allowed_causes, example_rng, example_seed

PROTOCOL_MIX: tuple[tuple[str, float], ...] = (("single", 0.15), ("standard", 0.60), ("extra", 0.25))
P_RANDOM_ONLY = 0.25
P_PROPOSE = 0.20
P_OMIT_TRUE = 0.08
GENERIC_FRACTION = 0.5
EIG_SAMPLES = 64
PROB_STEP = 0.05


@dataclass
class Case:
    """一份模拟出来的学生数据及其来历。"""

    exp: Experiment
    family: str  # 训练实验 id，或程序化任务族的 "proc_<定律>"
    generic: bool
    true_cause: str
    protocol: str
    ds: Dataset
    allowed: tuple[str, ...]
    meta: dict = field(default_factory=dict)


def _choice(rng: np.random.Generator, items):
    return items[int(rng.integers(len(items)))]


def sample_experiment(rng: np.random.Generator, generic_fraction: float = GENERIC_FRACTION):
    """→ (实验, family, 是否程序化)。"""
    if rng.random() < generic_fraction:
        exp, _ = next(procedural_task_family(rng, 1))
        law = exp.id.rsplit("_", 1)[0]  # proc_<定律>_00000 → proc_<定律>
        return exp, law, True
    exp = get_experiment(_choice(rng, TRAIN_EXPERIMENTS))
    return exp, exp.id, False


def sample_cause(rng: np.random.Generator, exp: Experiment, allowed, p_random: float = P_RANDOM_ONLY) -> str:
    ro = exp.random_only_id
    others = [c for c in allowed if c != ro]
    if ro in allowed and (not others or rng.random() < p_random):
        return ro
    return _choice(rng, others)


def sample_protocol(rng: np.random.Generator, exp: Experiment, mix=PROTOCOL_MIX) -> str:
    names = [m for m, _ in mix]
    w = np.array([p for _, p in mix], float)
    pick = names[int(rng.choice(len(names), p=w / w.sum()))]
    if pick == "single":
        return "single" if "single" in exp.protocols else "standard"
    if pick == "extra" and exp.menu:
        return "standard+" + _choice(rng, [m.id for m in exp.menu])
    return "standard"


def simulate_case(rng: np.random.Generator, generic_fraction: float = GENERIC_FRACTION,
                  mix=PROTOCOL_MIX, p_random: float = P_RANDOM_ONLY) -> Case:
    exp, family, generic = sample_experiment(rng, generic_fraction)
    allowed = allowed_causes(exp)
    cid = sample_cause(rng, exp, allowed, p_random)
    protocol = sample_protocol(rng, exp, mix)
    ds = exp.simulate(rng, cid, protocol_plan(exp, protocol))
    ds.meta["protocol"] = protocol
    return Case(exp, family, generic, cid, protocol, ds, allowed)


def fact_sheet(case: Case, cause_ids, rng: np.random.Generator, n_samples: int = EIG_SAMPLES) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 单点 / 无噪声数据时秩相关的常数输入警告
        return build_fact_sheet(case.exp, case.ds, rng, n_samples=n_samples, cause_ids=tuple(cause_ids))


# ── 取整、编号 ────────────────────────────────────────────────────────


def round_prob(p: float, step: float = PROB_STEP) -> float:
    return round(round(float(p) / step) * step, 2)


def shuffled_ids(rng: np.random.Generator, ids, prefix: str) -> dict[str, str]:
    """真实 id → 显示编号（C1…/M1…），显示顺序随机。返回的字典按显示顺序排列。"""
    perm = rng.permutation(len(ids))
    return {ids[int(j)]: f"{prefix}{i + 1}" for i, j in enumerate(perm)}


# ── 短讲解（≤ 80 字）────────────────────────────────────────────────

_NONE_CLAUSE = re.compile(r"[^，；。]*=\s*None[^，；。]*[，；]?")


def clean_evidence(ev: str) -> str:
    """去掉证据句里没有数值的分句（如截距没有理论值时的"截距 z=None"）。"""
    return _NONE_CLAUSE.sub("", ev or "").strip("，； ")



def short_explanation(fs: dict, probs: dict[str, float], top2: list[str],
                      limit: int = prompts.EXPLAIN_LIMIT, propose: bool = False) -> tuple[str, bool]:
    """用 templates 的讲解块和小工具拼一段 ≤ limit 字的讲解。返回 (文字, 是否通过校验器)。

    概率改成目标里取整到 0.05 的值、下一次测量只留前两名原因的预测值，再用这份事实单校验，
    保证讲解里的数字和 JSON 里的数字一致。从长到短试几种拼法，取第一个不超长且通过校验的。
    propose=True（不给清单）时把"清单里的原因"说成"常见原因"。
    """
    f2 = copy.deepcopy(fs)
    dg = f2.get("diagnosis") or {}
    for r in dg.get("ranked") or []:
        if r.get("cause_id") in probs:
            r["prob"] = probs[r["cause_id"]]
    nm = f2.get("next_measurement")
    if nm:
        nm.pop("eig_bits", None)
        order = {c: i for i, c in enumerate(top2)}
        nm["predicted"] = sorted((p for p in nm.get("predicted") or [] if p.get("cause_id") in order),
                                 key=lambda p: order[p["cause_id"]])
    ranked = dg.get("ranked") or [{}]
    top = ranked[0]
    name = top.get("name") or top.get("cause_id", "")
    pct = tpl.pct(top.get("prob") or 0.0)
    if dg.get("not_in_list"):
        mp = dg.get("misfit_p")
        mps = f"（失配 p = {tpl.fmt(mp, 2)}）" if mp is not None else ""
        word = "常见原因" if propose else "清单里的原因"
        c_long = c_short = f"数据和{word}都对不上{mps}，最接近的是「{name}」。"
    else:
        c_long = tpl.block_cause(f2)
        c_short = (f"数据在随机误差范围内，最可能是「{name}」（{pct}%）。"
                   if str(top.get("cause_id", "")).endswith(".random_only") else f"最可能是「{name}」（{pct}%）。")
    ev = clean_evidence(top.get("evidence") or "").rstrip("。")
    e_full = f"证据：{ev}。" if ev else ""
    e_one = f"证据：{ev.split('；')[0]}。" if ev else ""
    n_two = tpl.block_next(f2)
    if nm and nm.get("label"):
        head = f"下一步：{nm['label'].rstrip('。')}。"
        pr = nm.get("predicted") or []
        n_one = (head[:-1] + f"，若是「{pr[0].get('name', '')}」预计 {tpl.with_unit(pr[0]['value'], pr[0].get('unit', ''))}。"
                 if pr and pr[0].get("value") is not None else head)
    else:
        head = n_one = n_two
    options = [c_long + e_full + n_two, c_long + e_one + n_two, c_short + e_one + n_two, c_short + e_one + n_one,
               c_short + e_one + head, c_short + n_one, c_long + e_one + head, c_short + head]
    fitting = list(dict.fromkeys(t for t in options if tpl.nchars(t) <= limit))
    for t in fitting:
        if verify(t, f2)["ok"]:
            return t, True
    if fitting:
        return fitting[0], False
    return tpl.compose([c_short, head], limit), False


# ── 一条样本 ──────────────────────────────────────────────────────────


def _candidates(exp: Experiment, disp: dict[str, str]) -> list[dict]:
    return [{"id": d, "name": exp.cause(c).name_zh, "description": exp.cause(c).description_zh}
            for c, d in disp.items()]


def build_example(index: int, split: str, base_seed: int | None = None, generic_fraction: float = GENERIC_FRACTION,
                  p_propose: float = P_PROPOSE, p_omit: float = P_OMIT_TRUE, n_samples: int = EIG_SAMPLES) -> Example:
    kw = {} if base_seed is None else {"base": base_seed}
    seed = example_seed("diagnose", split, index, **kw)
    rng = example_rng("diagnose", split, index, **kw)
    case = simulate_case(rng, generic_fraction)
    exp = case.exp
    u = rng.random()
    if u < p_propose:
        variant = "propose"
    elif (u < p_propose + p_omit and case.true_cause != exp.random_only_id and len(case.allowed) >= 4):
        variant = "omit_true"
    else:
        variant = "list"
    cand = [c for c in case.allowed if not (variant == "omit_true" and c == case.true_cause)]
    eig_rng = np.random.default_rng(int(rng.integers(0, 2**62)))
    fs = fact_sheet(case, cand, eig_rng, n_samples)
    post = posterior(exp, case.ds, cand)
    order = sorted(range(len(cand)), key=lambda j: (-float(post.probs[j]), j))
    ranked_ids = [cand[j] for j in order]
    probs = {cand[j]: round_prob(post.probs[j]) for j in range(len(cand))}
    top2 = ranked_ids[:2]
    disp = shuffled_ids(rng, cand, "C")
    menu_disp = shuffled_ids(rng, [m.id for m in exp.menu], "M")
    menu = [{"id": d, "item_id": m} for m, d in menu_disp.items()]

    panel = fs.get("panel") or {}
    sigs = list(panel.get("signatures") or ["none"])
    dg = fs.get("diagnosis") or {}
    evidence = clean_evidence((dg.get("ranked") or [{}])[0].get("evidence", ""))
    nm = fs.get("next_measurement")
    pred_by = {p["cause_id"]: p for p in (nm or {}).get("predicted") or []}
    explanation, verified = short_explanation(fs, probs, top2, propose=variant == "propose")

    if variant == "propose":
        messages = prompts.build_diagnose_messages(exp, case.ds, None, menu)
        names = {c: exp.cause(c).name_zh for c in cand}
        target = {
            "signatures": sigs, "evidence": evidence,
            "ranked": [{"name": names[c], "mechanism": exp.cause(c).mechanism,
                        "expect": list(exp.cause(c).signatures), "prob": probs[c]}
                       for c in ranked_ids[:prompts.MAX_PROPOSE]],
            "next_measurement": None if not nm else {
                "id": menu_disp[nm["id"]],
                "predicted": [{"name": names[c], "value": pred_by[c]["value"], "unit": pred_by[c]["unit"]}
                              for c in top2 if c in pred_by]},
            "agreement": (fs.get("agreement") or {}).get("band"),
            "explanation": explanation,
        }
        schema = "propose"
    else:
        messages = prompts.build_diagnose_messages(exp, case.ds, _candidates(exp, disp), menu)
        target = {
            "signatures": sigs, "evidence": evidence,
            "ranked": [{"cause_id": disp[c], "prob": probs[c], "expect": list(exp.cause(c).signatures)}
                       for c in ranked_ids],
            "not_in_list": bool(dg.get("not_in_list")),
            "next_measurement": None if not nm else {
                "id": menu_disp[nm["id"]],
                "predicted": [{"cause_id": disp[c], "value": pred_by[c]["value"], "unit": pred_by[c]["unit"]}
                              for c in top2 if c in pred_by]},
            "agreement": (fs.get("agreement") or {}).get("band"),
            "explanation": explanation,
        }
        schema = "diagnose"
    text = json.dumps(target, ensure_ascii=False)
    meta = {
        "experiment_id": case.family, "split": split, "seed": seed, "variant": variant, "schema": schema,
        "generic": case.generic, "protocol": case.protocol, "truth_cause": case.true_cause,
        "truth_magnitude": case.ds.truth.get("magnitude") if case.ds.truth else None,
        "causes": [{"id": d, "name": exp.cause(c).name_zh} for c, d in disp.items()],
        "id_map": {d: c for c, d in disp.items()}, "menu_map": {d: m for m, d in menu_disp.items()},
        "top_cause": ranked_ids[0], "explanation_verified": verified, "prompt_version": prompts.PROMPT_VERSION,
    }
    if case.generic:
        meta["instance_id"] = exp.id
    ex = Example(f"diag-{case.family}-{split}-{index:06d}", "diagnose", "engine",
                 messages + [{"role": "assistant", "content": text}], meta)
    ex.validate()
    return ex
