"""识别流水线：规则优先 → 模型兜底 → 仍分不开就问一个澄清问题 → 手动选择永远说了算。

    result = recognize("photo.jpg", model)        # model 可为 None（纯规则）
    if result.needs_question:                      # 展示 result.question，学生点一个选项
        result = answer(result, option_index)
    result = choose_manual(result, "D")            # 学生手动改选，流水线不再改动

model 是任何带 chat(messages, json_schema=...) 方法的对象（lab_detective.llm 的 ChatModel 接口；
图片以训练格式 {"type": "image", "image": 路径} 放在 user 消息里），返回 {"text": ...} 或字符串。模型调用失败或输出不合法时退回规则结果，不抛异常。

澄清问题：从固定题库里选"期望熵减"最大的一题。设待定集合 U 上的先验为 p，
选项 j 覆盖字母集 S_j（每个字母若出现在 k 个选项里，各以 1/k 的概率被答到；不在任何选项里
的字母落进"都不是"），增益 = H(p) − Σ_j P(j)·H(p | j)，即答案与实验身份的互信息（比特）。
U 里总带着 Z（其他实验）：规则没把握时"都不在表里"始终是一个可能。
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import replace
from pathlib import Path

from . import cues as _cues
from .letters import LETTERS, OTHER_LETTER, entry, experiment_for, letter_for, normalize_letter
from .prompt import build_messages
from .schema import MODEL_OUTPUT_SCHEMA, RecognitionParseError, RecognitionResult, parse_model_output
from .som import draw_marks

MODEL_MIN_CONFIDENCE = 0.6  # 模型自报把握 ≥ 此值才直接采纳
NONE_LABEL = "都不是 / 不确定"

# 题库：(id, 问题, [(选项, 字母)])；顺序也是同分时的优先级
QUESTION_BANK: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    ("q.goal_cart", "你要测什么？", (
        ("小车通过一段固定距离所用的时间", "C"),
        ("悬挂重物拉动小车时的加速度", "D"),
        ("小车滑到斜面底端时的速度", "F"),
        ("两辆小车碰撞前后的速度", "E"),
    )),
    ("q.dynamometer", "弹簧测力计在实验里是怎么用的？", (
        ("竖直挂着，下面挂钩码或物体", "A"),
        ("水平拉着木块或小车匀速运动", "B"),
        ("沿斜面拉着物块", "G"),
    )),
    ("q.carts", "轨道上一共有几辆小车（滑块）？", (
        ("一辆", "CDF"),
        ("两辆", "E"),
    )),
    ("q.incline", "物体是沿着明显倾斜的轨道或斜面运动的吗？（只为平衡摩擦稍微垫高的不算）", (
        ("是，明显倾斜", "CFGK"),
        ("否，基本水平", "BDE"),
    )),
    ("q.pulley", "有没有细线跨过滑轮挂着重物？", (
        ("有：一端连着桌上的小车或木块，另一端挂钩码", "D"),
        ("有：细线两端都挂着重物", "I"),
        ("没有滑轮", "ABCEFGHJKL"),
    )),
    ("q.ball", "实验里的小球（或重物）是怎么运动的？", (
        ("从高处竖直下落", "H"),
        ("从斜槽末端水平飞出", "K"),
        ("挂在细线上来回摆动", "J"),
        ("沿轨道滚下", "F"),
    )),
    ("q.lever", "装置里有没有一根架在支点上、两边挂钩码的杆（杠杆）？", (
        ("有", "L"),
    )),
    ("q.goal_general", "这个实验主要想研究什么？", (
        ("弹簧受力与伸长的关系", "A"),
        ("摩擦力", "BG"),
        ("物体运动的快慢（平均速度）", "C"),
        ("力和加速度的关系", "DI"),
        ("碰撞前后的动量", "E"),
        ("高度与速度（机械能）", "F"),
        ("下落或抛出物体的运动", "HK"),
        ("摆动的周期", "J"),
        ("杠杆平衡", "L"),
    )),
)


# ── 澄清问题：期望熵减 ─────────────────────────────────────────────────


def _entropy(p: dict[str, float]) -> float:
    tot = sum(p.values())
    return -sum(v / tot * math.log2(v / tot) for v in p.values() if v > 0) if tot > 0 else 0.0


def _shown_options(options, U) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """只保留覆盖待定集合的选项；返回 (展示的选项 [(文字, 字母)], 落进"都不是"的字母)。"""
    shown = [(label, [c for c in letters if c in U]) for label, letters in options]
    shown = [(label, ls) for label, ls in shown if ls]
    covered = {c for _, ls in shown for c in ls}
    return shown, [c for c in U if c not in covered]


def _answer_likelihood(shown, bucket, letter) -> list[float]:
    """字母 letter 对每个展示选项（最后一个是"都不是"）的作答概率。"""
    k = sum(letter in ls for _, ls in shown)
    lik = [(1.0 / k if letter in ls else 0.0) for _, ls in shown] if k else [0.0] * len(shown)
    return lik + [1.0 if letter in bucket else 0.0]


def expected_gain(prior: dict[str, float], options) -> float:
    """问这一题的期望熵减（比特）。"""
    U = [c for c, v in prior.items() if v > 0]
    shown, bucket = _shown_options(options, U)
    tot = sum(prior[c] for c in U)
    p = {c: prior[c] / tot for c in U}
    h_after = 0.0
    for j in range(len(shown) + 1):
        joint = {c: p[c] * _answer_likelihood(shown, bucket, c)[j] for c in U}
        pj = sum(joint.values())
        if pj > 0:
            h_after += pj * _entropy(joint)
    return max(0.0, _entropy(p) - h_after)


def choose_question(prior: dict[str, float], asked: tuple[str, ...] = ()) -> dict | None:
    """选期望熵减最大的一题；同分时优先"都不是"里剩的非 Z 字母少的、选项少的、题库靠前的。"""
    U = [c for c, v in prior.items() if v > 0]
    if len(U) < 2:
        return None
    best = None
    for idx, (qid, text, options) in enumerate(QUESTION_BANK):
        if qid in asked:
            continue
        gain = expected_gain(prior, options)
        if gain <= 1e-9:
            continue
        shown, bucket = _shown_options(options, U)
        key = (-round(gain, 9), sum(c != OTHER_LETTER for c in bucket), len(shown), idx)
        if best is None or key < best[0]:
            best = (key, qid, text, shown, bucket, gain)
    if best is None:
        return None
    _, qid, text, shown, bucket, gain = best
    tot = sum(prior[c] for c in U)
    return {
        "id": qid,
        "text": text,
        "options": [{"label": label, "letters": ls} for label, ls in shown]
        + [{"label": NONE_LABEL, "letters": bucket}],
        "gain_bits": round(gain, 4),
        "prior": {c: round(prior[c] / tot, 6) for c in U},
        "asked": [*asked, qid],
    }


# ── 先验 ───────────────────────────────────────────────────────────────


def _rule_prior(cands: list[dict], letters: list[str]) -> dict[str, float]:
    """规则分数 softmax（Z 用虚拟分数），限制在 letters 上。"""
    score = {c["letter"]: c["score"] for c in cands}
    score.setdefault(OTHER_LETTER, _cues.OTHER_SCORE)
    base = min(score.values()) if score else 0.0
    w = {c: math.exp((score.get(c, base) - max(score.values())) / _cues.SOFTMAX_TAU) for c in letters}
    tot = sum(w.values())
    return {c: v / tot for c, v in w.items()}


def _mix_model(prior: dict[str, float], letter: str, conf: float) -> dict[str, float]:
    """规则先验与模型意见各占一半：模型选的字母得 conf，其余均分 1 − conf。"""
    n = len(prior)
    model = {c: (conf if c == letter else (1 - conf) / max(1, n - 1)) for c in prior}
    mixed = {c: 0.5 * prior[c] + 0.5 * model[c] for c in prior}
    tot = sum(mixed.values())
    return {c: v / tot for c, v in mixed.items()}


# ── 结果构造 ───────────────────────────────────────────────────────────


def _default_name(letter: str) -> str:
    return "未识别（请手动选择实验）" if letter == OTHER_LETTER else entry(letter).name_zh


def _merge(a: list[str], b: list[str], limit: int) -> list[str]:
    out: list[str] = []
    for x in [*a, *b]:
        if x not in out:
            out.append(x)
    return out[:limit]


def _uncertain_result(prior, base: dict, source: str, warnings: list[str], asked=()) -> RecognitionResult:
    """待定集合上：选题；当前默认字母取先验最大者。"""
    letter = max(prior, key=lambda c: (prior[c], c != OTHER_LETTER))
    q = choose_question(prior, tuple(asked))
    if q is None and len([c for c, v in prior.items() if v > 0]) > 1:
        warnings = [*warnings, "题库里没有能区分剩余候选的问题，请手动确认实验"]
    return RecognitionResult(
        letter=letter, experiment_id=experiment_for(letter),
        open_name=base.get("open_name") or _default_name(letter),
        cues=base["cues"], apparatus=base["apparatus"],
        confidence=round(prior[letter], 3), needs_question=q is not None,
        source=source, candidates=base["candidates"], question=q, warnings=warnings,
    )


def _call_model(model, image, dets, work_dir) -> RecognitionResult:
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ld_som_"))
    som_path = work / "som.jpg"
    _, legend = draw_marks(image, dets, out_path=som_path)
    messages = build_messages(som_path, legend)
    # 图片已经以 {"type": "image"} 片段放在消息里（与训练样本同格式），不再另传 images，免得发两遍
    resp = model.chat(messages, json_schema=MODEL_OUTPUT_SCHEMA)
    text = resp.get("text") if isinstance(resp, dict) else resp
    return parse_model_output(text)


def recognize(
    image_path,
    model=None,
    *,
    detections: list[dict] | None = None,
    manual: str | None = None,
    work_dir=None,
    weights=None,
    conf: float = 0.25,
    model_min_confidence: float = MODEL_MIN_CONFIDENCE,
) -> RecognitionResult:
    """识别一张照片。

    detections：已有检测框（如学生在界面上改过的器材列表）时直接用，不再跑 YOLO。
    manual：学生手动选的字母或 experiment_id，给出就直接返回，流水线不做任何改动。
    """
    warnings: list[str] = []
    if detections is None and manual is None:
        try:
            detections, _ = _cues.run_detector(image_path, weights, conf)
        except Exception as e:  # 权重缺失、图片读不了等：退回"无检测"
            warnings.append(f"器材检测失败：{e}")
            detections = []
    a = _cues.analyze(detections or [])
    base = {"cues": a["cues"], "apparatus": a["apparatus"],
            "candidates": [{k: c[k] for k in ("letter", "experiment_id", "name_zh", "score", "reason")}
                           for c in a["candidates"]]}

    if manual is not None:
        return _manual(manual, base, warnings)

    if a["confident"]:
        top = a["candidates"][0]
        return RecognitionResult(
            letter=top["letter"], experiment_id=top["experiment_id"], open_name=top["name_zh"],
            cues=a["cues"], apparatus=a["apparatus"], confidence=a["confidence"],
            needs_question=False, source="rule", candidates=base["candidates"], warnings=warnings,
        )

    # 规则没把握：待定集合 = 规则平局集合 + Z；规则一个候选都没有时 = 全表
    U = [*a["tied"], OTHER_LETTER] if a["tied"] else list(LETTERS)
    prior = _rule_prior(a["candidates"], U) if a["tied"] else {c: 1.0 / len(U) for c in U}
    source = "rule"

    if model is not None:
        try:
            mr = _call_model(model, image_path, detections or [], work_dir)
        except Exception as e:  # 含 RecognitionParseError
            warnings.append(f"模型识别失败，已退回规则结果：{e}")
            source = "rule_fallback"
        else:
            warnings += mr.warnings
            base = {**base, "cues": _merge(mr.cues, a["cues"], 8),
                    "apparatus": _merge(mr.apparatus, a["apparatus"], 16), "open_name": mr.open_name}
            source = "model"
            if mr.confidence >= model_min_confidence:
                return RecognitionResult(
                    letter=mr.letter, experiment_id=mr.experiment_id, open_name=mr.open_name,
                    cues=base["cues"], apparatus=base["apparatus"], confidence=mr.confidence,
                    needs_question=False, source="model", candidates=base["candidates"], warnings=warnings,
                )
            if mr.letter not in prior:
                prior[mr.letter] = 0.0
            prior = _mix_model(prior, mr.letter, mr.confidence)

    return _uncertain_result(prior, base, source, warnings)


def answer(result: RecognitionResult, option_index: int) -> RecognitionResult:
    """学生回答澄清问题（选项下标，最后一个是"都不是"）→ 更新后的结果，可能再带下一题。"""
    q = result.question
    if not result.needs_question or not q:
        raise ValueError("这个结果没有待回答的问题")
    if result.source == "manual":
        return result
    options = q["options"]
    if not 0 <= option_index < len(options):
        raise IndexError(f"选项下标越界：{option_index}")
    prior = q["prior"]
    shown = [(o["label"], o["letters"]) for o in options[:-1]]
    bucket = options[-1]["letters"]
    post = {c: prior[c] * _answer_likelihood(shown, bucket, c)[option_index] for c in prior}
    post = {c: v for c, v in post.items() if v > 0}
    base = {"cues": result.cues, "apparatus": result.apparatus, "candidates": result.candidates}
    if not post:  # 选了"都不是"而且没有剩余字母：按其他实验处理
        post = {OTHER_LETTER: 1.0}
    tot = sum(post.values())
    post = {c: v / tot for c, v in post.items()}
    if len(post) == 1:
        letter = next(iter(post))
        return RecognitionResult(
            letter=letter, experiment_id=experiment_for(letter), open_name=_default_name(letter),
            cues=result.cues, apparatus=result.apparatus, confidence=1.0, needs_question=False,
            source="question", candidates=result.candidates, question=None, warnings=list(result.warnings),
        )
    return _uncertain_result(post, base, "question", list(result.warnings), asked=q.get("asked", ()))


def _manual(choice: str, base: dict, warnings: list[str]) -> RecognitionResult:
    letter = normalize_letter(choice) if len(choice.strip()) <= 3 else None
    if letter is None:
        letter = letter_for(choice)
        if letter == OTHER_LETTER and choice not in (OTHER_LETTER, "other"):
            warnings = [*warnings, f"手动选择 {choice!r} 不在字母表里，按其他实验处理"]
    return RecognitionResult(
        letter=letter, experiment_id=experiment_for(letter),
        open_name=base.get("open_name") or _default_name(letter),
        cues=base["cues"], apparatus=base["apparatus"], confidence=1.0, needs_question=False,
        source="manual", candidates=base["candidates"], question=None, warnings=warnings,
    )


def choose_manual(result: RecognitionResult, choice: str, open_name: str | None = None) -> RecognitionResult:
    """学生手动选择（字母或 experiment_id）：覆盖一切自动结果。"""
    base = {"cues": result.cues, "apparatus": result.apparatus, "candidates": result.candidates,
            "open_name": open_name}
    r = _manual(choice, base, list(result.warnings))
    return replace(r, open_name=open_name or r.open_name)
