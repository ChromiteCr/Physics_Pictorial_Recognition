"""讲解任务的训练样本：事实单 → 不超过 120 字的讲解（模板，每个槽位 3 种措辞），必须通过 tutor/verifier.py。

提示与推理完全相同：prompts.build_explain_messages = build_context(..., phase="explain")。
目标按 SYSTEM_PROMPT_ZH 第 4 条和 explain 阶段任务：最可能的原因（照抄名称）、至少一个数值证据、
下一次测量（照抄名称）及前两名原因的预测值、修正方法；总字数 ≤ 120。
槽位：原因、证据、下一次测量、修正；每个槽位 3 种措辞，按样本种子随机挑。超长时依次去掉修正、
第二个预测值、证据（只要预测值还在，数值证据就还在）。校验不过就换回第 0 种措辞，再不过就用
templates 的讲解块拼一段；仍不过就丢掉这条样本（generate 会统计丢弃数）。
"""

from __future__ import annotations

import json

import numpy as np

from ..tutor import templates as tpl
from ..tutor.verifier import verify
from . import prompts
from .format import Example
from .splits import example_rng, example_seed
from .tasks_diagnose import GENERIC_FRACTION, clean_evidence, fact_sheet, simulate_case

LIMIT = 120
# 讲解发生在诊断之后：不用单次读数（没有证据可讲）
EXPLAIN_MIX: tuple[tuple[str, float], ...] = (("standard", 0.7), ("extra", 0.3))

CAUSE_VARIANTS = (
    "最可能的原因是「{name}」（可能性 {p}%）。",
    "引擎判断最可能是「{name}」，可能性约 {p}%。",
    "嫌疑最大的是「{name}」（{p}%）。",
)
CAUSE_RANDOM_VARIANTS = (
    "数据和孪生预测相符，最可能是「{name}」（{p}%），没有发现系统误差。",
    "没有发现系统误差：最可能是「{name}」，可能性 {p}%。",
    "数据在随机误差范围内，最可能是「{name}」（{p}%）。",
)
CAUSE_MISFIT_VARIANTS = (
    "数据和清单里的原因都对不上，可能有清单之外的原因；最接近的是「{name}」。",
    "清单里的原因都解释不了这组数据，最接近的是「{name}」，也可能是清单之外的原因。",
    "这组数据和所有候选原因都对不上；清单里最接近的是「{name}」。",
)
EVIDENCE_VARIANTS = ("证据：{ev}。", "依据是{ev}。", "从数据看，{ev}。")
NEXT_VARIANTS = (
    "下一步：{label}；{preds}。",
    "建议接着{label}：{preds}。",
    "最能区分它们的测量是{label}，{preds}。",
)
PRED_VARIANTS = ("若是「{name}」，预计 {v}", "「{name}」对应约 {v}", "如果是「{name}」，应得到 {v}")
NEXT_NONE = ("现有的追加测量都分不开剩下的原因，可以按改进方法重做一遍来检验。",
             "菜单里没有能分开剩下原因的测量，改正后重做一遍最直接。",
             "追加测量已经帮不上忙，按修正方法重测来检验。")
FIX_VARIANTS = ("改进：{fix}。", "修正方法：{fix}。", "可以这样改进：{fix}。")


def _cause_slot(fs: dict, v: int) -> str:
    dg = fs.get("diagnosis") or {}
    top = (dg.get("ranked") or [{}])[0]
    name = top.get("name") or top.get("cause_id", "")
    p = tpl.pct(top.get("prob") or 0.0)
    if dg.get("not_in_list"):
        return CAUSE_MISFIT_VARIANTS[v].format(name=name)
    if str(top.get("cause_id", "")).endswith(".random_only"):
        return CAUSE_RANDOM_VARIANTS[v].format(name=name, p=p)
    return CAUSE_VARIANTS[v].format(name=name, p=p)


def _evidence_slot(fs: dict, v: int) -> str:
    ranked = (fs.get("diagnosis") or {}).get("ranked") or []
    ev = clean_evidence(ranked[0].get("evidence") or "").rstrip("。") if ranked else ""
    return EVIDENCE_VARIANTS[v].format(ev=ev) if ev else ""


def _preds(fs: dict, k: int, v: int) -> list[str]:
    nm = fs.get("next_measurement") or {}
    rank = {r.get("cause_id"): i for i, r in enumerate((fs.get("diagnosis") or {}).get("ranked") or [])}
    ps = sorted((p for p in nm.get("predicted") or [] if p.get("value") is not None and p.get("cause_id") in rank),
                key=lambda p: rank[p["cause_id"]])
    out = []
    for p in ps[:k]:
        name = p.get("name") or tpl.cause_name(fs, p.get("cause_id", ""))
        out.append(PRED_VARIANTS[v].format(name=name, v=tpl.with_unit(p["value"], p.get("unit", ""))))
    return out


def _next_slot(fs: dict, v: int, k: int) -> str:
    nm = fs.get("next_measurement")
    if not nm or not nm.get("label"):
        return NEXT_NONE[v]
    preds = _preds(fs, k, v)
    if not preds:
        return f"下一步：{nm['label']}。"
    return NEXT_VARIANTS[v].format(label=nm["label"].rstrip("。"), preds="，".join(preds))


def _fix_slot(fs: dict, v: int) -> str:
    fx = (fs.get("fix") or {}).get("text") or ""
    return FIX_VARIANTS[v].format(fix=fx.rstrip("。")) if fx else ""


def compose_explanation(fs: dict, choice: dict[str, int], limit: int = LIMIT, skip: int = 0) -> str | None:
    """按槽位措辞拼讲解；超长时依次去掉修正、第二个预测值、证据。拼不进 limit 返回 None。

    skip：跳过前几种拼法（校验不过时换下一种）。"""
    c = _cause_slot(fs, choice["cause"])
    e = _evidence_slot(fs, choice["evidence"])
    f = _fix_slot(fs, choice["fix"])
    # 最后两种不带预测值：预测值的单位校验器不认时（如 °C），靠证据里的数字过关
    tries = [(e, 2, f), (e, 2, ""), (e, 1, ""), ("", 2, ""), ("", 1, ""), (e, 0, f), (e, 0, "")]
    for ev, k, fx in tries[skip:]:
        text = c + ev + _next_slot(fs, choice["next"], k) + fx
        if tpl.nchars(text) <= limit:
            return text
    return None


def explanation_target(fs: dict, rng: np.random.Generator, limit: int = LIMIT) -> tuple[str | None, str, dict]:
    """→ (讲解, 来源 paraphrase|plain|blocks|None, 校验结果)。"""
    choice = {k: int(rng.integers(3)) for k in ("cause", "evidence", "next", "fix")}
    for how, ch in (("paraphrase", choice), ("plain", {k: 0 for k in choice})):
        for skip in range(7):
            text = compose_explanation(fs, ch, limit, skip)
            if not text:
                break
            v = verify(text, fs)
            if v["ok"]:
                return text, how, v
    b = tpl.explanation_blocks(fs)
    text = tpl.compose([b["cause"], b["evidence"], b["next"]], limit)
    v = verify(text, fs)
    return (text, "blocks", v) if v["ok"] else (None, "none", v)


def explain_case(index: int, split: str, base_seed: int | None = None,
                 generic_fraction: float = GENERIC_FRACTION, n_samples: int = 64):
    """第 index 条讲解样本的 (模拟数据, 事实单, 余下的随机数流, 种子)；build_example 和测试共用。"""
    kw = {} if base_seed is None else {"base": base_seed}
    seed = example_seed("explain", split, index, **kw)
    rng = example_rng("explain", split, index, **kw)
    case = simulate_case(rng, generic_fraction, EXPLAIN_MIX)
    fs = fact_sheet(case, case.allowed, np.random.default_rng(int(rng.integers(0, 2**62))), n_samples)
    return case, fs, rng, seed


def build_example(index: int, split: str, base_seed: int | None = None,
                  generic_fraction: float = GENERIC_FRACTION, n_samples: int = 64) -> Example | None:
    """一条讲解样本；讲解无法通过校验器时返回 None。"""
    case, fs, rng, seed = explain_case(index, split, base_seed, generic_fraction, n_samples)
    text, how, v = explanation_target(fs, rng)
    if text is None:
        return None
    card = prompts._card(case.exp)
    messages = prompts.build_explain_messages(fs, card, case.exp.id, case.protocol,
                                              [o.to_dict() for o in case.ds.observations])
    meta = {"experiment_id": case.family, "split": split, "seed": seed, "generic": case.generic,
            "protocol": case.protocol, "truth_cause": case.true_cause,
            "top_cause": ((fs.get("diagnosis") or {}).get("ranked") or [{}])[0].get("cause_id"),
            "explanation_source": how, "coverage": v["coverage"], "prompt_version": prompts.PROMPT_VERSION,
            "fact_sheet_sha": _sha(fs)}
    ex = Example(f"expl-{case.family}-{split}-{index:06d}", "explain", "template",
                 messages + [{"role": "assistant", "content": text}], meta)
    ex.validate()
    return ex


def _sha(obj) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
