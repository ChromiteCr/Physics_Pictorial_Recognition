"""提示词的唯一来源：训练、蒸馏、bench 和推理都从这里拿消息（改动必须升 PROMPT_VERSION）。

每个构造函数都返回 [system, user] 两条消息，system 一律是 tutor/context.py 的 SYSTEM_PROMPT_ZH
（训练时看到的指令和推理时一致，ARCHITECTURE §2.5）。user 里放任务专用的材料和输出要求。

诊断、设计两类提示只放"学生能看到 / 代码能算出的观测量"：实验卡要点、数据表、特征面板的数字、
一致性 z、候选原因的物理描述（编号随机）、追加测量菜单。绝不放：原因的预期特征标签、面板判出的特征、
引擎后验概率、EIG、一致性结论。特征词表和输出格式是所有样本共用的固定文字，不含任何样本信息。

讲解（explain）直接用 tutor.context.build_context(..., phase="explain")，与 policy.guarded_explanation
推理时的调用完全相同；它的输入本来就是事实单。
"""

from __future__ import annotations

import json
import math
import warnings
from typing import Mapping, Sequence

from ..facts import _agreement, _panel_numbers, Numbers, plain
from ..labsim.core import SIGNATURES, Dataset, Experiment, MECHANISMS
from ..panel import compute_panel
from ..tutor.context import SYSTEM_PROMPT_ZH, build_context
from ..tutor.session import LabSession
from ..uncertainty import slope_unit, unit_of

PROMPT_VERSION = "lab-v1"
EXPLAIN_LIMIT = 80  # 诊断 / 设计 JSON 里 explanation 的字数上限（去空白后的字符数）
MAX_PROPOSE = 6

ROLE_ZH = {"control": "控制量", "measured": "测量量", "derived": "导出量"}

# ── 固定文字：特征词表、机制词表、一致性规则 ────────────────────────────

SIGNATURE_VOCAB_ZH: dict[str, str] = {
    "none": "面板没有显著异常（点数少于 3 时只能写 none）",
    "intercept_pos": "截距 z > 2（且图线没有判为弯曲）",
    "intercept_neg": "截距 z < −2（且图线没有判为弯曲）",
    "slope_high": "斜率比的 z > 2（只在有理论斜率时判）",
    "slope_low": "斜率比的 z < −2（只在有理论斜率时判）",
    "curve_up": "曲率 z > 2（至少 4 个点）",
    "curve_down": "曲率 z < −2（至少 4 个点）",
    "scatter_excess": "残差散布偏大：偏大 p < 0.05",
    "scatter_deficit": "残差散布偏小：偏小 p < 0.05",
    "drift": "残差随试次顺序漂移：|ρ| ≥ 0.8 且 p < 0.05（至少 5 个点）",
}
assert tuple(SIGNATURE_VOCAB_ZH) == SIGNATURES

MECHANISM_VOCAB_ZH: dict[str, str] = {
    "none": "没有系统误差，只有随机误差",
    "x_offset": "横轴的量（控制量一侧）整体差了一个固定值",
    "y_offset": "纵轴的读数整体差了一个固定值",
    "x_scale": "横轴的量按固定比例偏大或偏小",
    "y_scale": "纵轴的读数按固定比例偏大或偏小",
    "bend": "物理关系本身偏离理想规律（图线弯曲或随大小变化）",
    "scatter": "读数的随机散布变大",
    "drift": "条件随测量顺序慢慢漂移",
    "aux_only": "只影响追加测量，不影响主测量",
}
assert tuple(MECHANISM_VOCAB_ZH) == MECHANISMS

SIGNATURE_BLOCK = "【特征词表】（signatures、expect 只能用这些词）\n" + "\n".join(
    f"{k}：{v}" for k, v in SIGNATURE_VOCAB_ZH.items())
MECHANISM_BLOCK = "【机制词表】（mechanism 只能用这些词）\n" + "\n".join(
    f"{k}：{v}" for k, v in MECHANISM_VOCAB_ZH.items())
AGREEMENT_RULE = "agreement 按一致性 z 判断：|z| < 2 写 consistent，2 ≤ |z| ≤ 3 写 tension，|z| > 3 写 inconsistent；没有 z 时写 null。"

# ── 输出格式（示例行是固定文字）与 JSON Schema ───────────────────────────

DIAGNOSE_EXAMPLE = ('{"signatures": ["none"], "evidence": "一句话证据", '
                    '"ranked": [{"cause_id": "C1", "prob": 0.6, "expect": ["none"]}, {"cause_id": "C2", "prob": 0.4, "expect": ["none"]}], '
                    '"not_in_list": false, '
                    '"next_measurement": {"id": "M1", "predicted": [{"cause_id": "C1", "value": 0.0, "unit": "单位"}, '
                    '{"cause_id": "C2", "value": 0.0, "unit": "单位"}]}, '
                    '"agreement": "consistent", "explanation": "不超过80字"}')
PROPOSE_EXAMPLE = ('{"signatures": ["none"], "evidence": "一句话证据", '
                   '"ranked": [{"name": "原因名称", "mechanism": "none", "expect": ["none"], "prob": 0.6}], '
                   '"next_measurement": {"id": "M1", "predicted": [{"name": "原因名称", "value": 0.0, "unit": "单位"}]}, '
                   '"agreement": "consistent", "explanation": "不超过80字"}')
DESIGN_EXAMPLE = ('{"expected": [{"setting": 0.0, "value": 0.0, "sd": 0.0, "unit": "单位"}], '
                  '"dominant_source": {"reading": "读数名称", "type": "random"}, '
                  '"recommend": {"n_points": 5, "range": [0.0, 0.0], "unit": "单位", "repeat_each": false}, '
                  '"unresolved": [{"pair": ["C1", "C2"], "status": "partial", "measurement": "M1"}], '
                  '"explanation": "不超过80字"}')
KNOWLEDGE_EXAMPLE = ('{"knowledge_points": [{"id": "kp.xxx", "name": "知识点名称"}], '
                     '"prerequisites": [{"id": "kp.yyy", "name": "前置知识点名称"}], "answer": "简短回答"}')

_SIG = {"enum": list(SIGNATURES)}
_NUM = {"type": "number"}
_STR = {"type": "string"}
_PROB = {"type": "number", "minimum": 0, "maximum": 1}
_EXPLAIN = {"type": "string", "minLength": 1}
_AGREE = {"enum": ["consistent", "tension", "inconsistent", None]}


def _pred_item(key: str, pattern: str | None) -> dict:
    k = {"type": "string", "pattern": pattern} if pattern else _STR
    return {"type": "object", "required": [key, "value", "unit"], "additionalProperties": False,
            "properties": {key: k, "value": _NUM, "unit": _STR}}


DIAGNOSE_SCHEMA: dict = {
    "type": "object", "additionalProperties": False,
    "required": ["signatures", "evidence", "ranked", "not_in_list", "next_measurement", "agreement", "explanation"],
    "properties": {
        "signatures": {"type": "array", "minItems": 1, "items": _SIG},
        "evidence": _STR,
        "ranked": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["cause_id", "prob", "expect"], "additionalProperties": False,
            "properties": {"cause_id": {"type": "string", "pattern": "^C[0-9]+$"}, "prob": _PROB,
                           "expect": {"type": "array", "minItems": 1, "items": _SIG}}}},
        "not_in_list": {"type": "boolean"},
        "next_measurement": {"anyOf": [{"type": "null"}, {
            "type": "object", "required": ["id", "predicted"], "additionalProperties": False,
            "properties": {"id": {"type": "string", "pattern": "^M[0-9]+$"},
                           "predicted": {"type": "array", "maxItems": 2, "items": _pred_item("cause_id", "^C[0-9]+$")}}}]},
        "agreement": _AGREE,
        "explanation": _EXPLAIN,
    },
}

PROPOSE_SCHEMA: dict = {
    "type": "object", "additionalProperties": False,
    "required": ["signatures", "evidence", "ranked", "next_measurement", "agreement", "explanation"],
    "properties": {
        "signatures": {"type": "array", "minItems": 1, "items": _SIG},
        "evidence": _STR,
        "ranked": {"type": "array", "minItems": 1, "maxItems": MAX_PROPOSE, "items": {
            "type": "object", "required": ["name", "mechanism", "expect", "prob"], "additionalProperties": False,
            "properties": {"name": {"type": "string", "minLength": 1}, "mechanism": {"enum": list(MECHANISMS)},
                           "expect": {"type": "array", "minItems": 1, "items": _SIG}, "prob": _PROB}}},
        "next_measurement": {"anyOf": [{"type": "null"}, {
            "type": "object", "required": ["id", "predicted"], "additionalProperties": False,
            "properties": {"id": {"type": "string", "pattern": "^M[0-9]+$"},
                           "predicted": {"type": "array", "maxItems": 2, "items": _pred_item("name", None)}}}]},
        "agreement": _AGREE,
        "explanation": _EXPLAIN,
    },
}

DESIGN_SCHEMA: dict = {
    "type": "object", "additionalProperties": False,
    "required": ["expected", "dominant_source", "recommend", "unresolved", "explanation"],
    "properties": {
        "expected": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["setting", "value", "sd", "unit"], "additionalProperties": False,
            "properties": {"setting": _NUM, "value": _NUM, "sd": _NUM, "unit": _STR}}},
        "dominant_source": {"type": "object", "required": ["reading", "type"], "additionalProperties": False,
                            "properties": {"reading": _STR, "type": {"enum": ["random", "resolution"]}}},
        "recommend": {"type": "object", "required": ["n_points", "range", "unit", "repeat_each"],
                      "additionalProperties": False,
                      "properties": {"n_points": {"type": "integer", "minimum": 1},
                                     "range": {"type": "array", "minItems": 2, "maxItems": 2, "items": _NUM},
                                     "unit": _STR, "repeat_each": {"type": "boolean"}}},
        "unresolved": {"type": "array", "items": {
            "type": "object", "required": ["pair", "status", "measurement"], "additionalProperties": False,
            "properties": {"pair": {"type": "array", "minItems": 2, "maxItems": 2,
                                    "items": {"type": "string", "pattern": "^C[0-9]+$"}},
                           "status": {"enum": ["confused", "partial"]},
                           "measurement": {"anyOf": [{"type": "null"}, {"type": "string", "pattern": "^M[0-9]+$"}]}}}},
        "explanation": _EXPLAIN,
    },
}

_KP_ITEM = {"type": "object", "required": ["id", "name"], "additionalProperties": False,
            "properties": {"id": {"type": "string", "pattern": "^kp\\."}, "name": _STR}}
KNOWLEDGE_SCHEMA: dict = {
    "type": "object", "additionalProperties": False,
    "required": ["knowledge_points", "prerequisites", "answer"],
    "properties": {"knowledge_points": {"type": "array", "minItems": 1, "items": _KP_ITEM},
                   "prerequisites": {"type": "array", "items": _KP_ITEM},
                   "answer": {"type": "string", "minLength": 1}},
}

SCHEMAS: dict[str, dict] = {"diagnose": DIAGNOSE_SCHEMA, "propose": PROPOSE_SCHEMA, "design": DESIGN_SCHEMA,
                            "knowledge": KNOWLEDGE_SCHEMA}


def validate_output(variant: str, obj) -> list[str]:
    """按 schema 检查一个输出对象（或 JSON 字符串），返回问题列表（空 = 合法）。

    variant: diagnose | propose | design | knowledge。诊断 / 设计还检查 explanation 的字数。
    """
    import jsonschema

    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError as e:
            return [f"不是合法 JSON：{e}"]
    v = jsonschema.Draft202012Validator(SCHEMAS[variant])
    errs = [f"{'/'.join(map(str, e.path)) or '(根)'}: {e.message}" for e in v.iter_errors(obj)]
    if not errs and variant in ("diagnose", "propose", "design"):
        n = nchars(obj["explanation"])
        if n > EXPLAIN_LIMIT:
            errs.append(f"explanation 有 {n} 字，超过 {EXPLAIN_LIMIT}")
    return errs


def nchars(text: str) -> int:
    """字数 = 去掉空白后的字符数（与 tutor/templates.nchars 相同的口径）。"""
    return len("".join(str(text).split()))


# ── 材料：实验卡要点 ──────────────────────────────────────────────────

_FAMILY_FORMULA = {
    "proportional": "{y} = a·{x}", "linear": "{y} = a·{x} + b", "inverse": "{y} = a/{x}",
    "square": "{y} = a·{x}²", "sqrt": "{y} = a·√{x}", "power": "{y} = a·{x}^{q}",
}


def _card(exp: Experiment) -> dict | None:
    try:
        from .. import knowledge

        return knowledge.load_card(exp.id)
    except Exception:  # noqa: BLE001  程序化实例、"其他实验"没有实验卡
        return None


def card_text(exp: Experiment) -> str:
    """实验卡要点：名称、目的、公式、变量。程序化实例按关系族和变量现场生成同样的几行。"""
    card = _card(exp)
    if card:
        lines = [f"实验：{card.get('name_zh', exp.name_zh)}"]
        if card.get("goal"):
            lines.append(f"目的：{card['goal']}")
        if card.get("formula"):
            lines.append(f"公式：{card['formula']}")
        vs = "；".join(f"{v.get('symbol', '')} {v.get('name', '')}（{v.get('unit', '') or '无单位'}，"
                      f"{ROLE_ZH.get(v.get('role', ''), v.get('role', ''))}）" for v in card.get("variables") or [])
        if vs:
            lines.append(f"变量：{vs}")
        return "\n".join(lines)
    xv, yv = getattr(exp, "xv", None), getattr(exp, "yv", None)
    lines = [f"实验：{exp.name_zh}"]
    if xv and yv:
        lines.append(f"目的：探究{yv['name']}与{xv['name']}的关系。")
        fam = getattr(exp, "family", "proportional")
        q = getattr(exp, "q", 1.0)
        f = _FAMILY_FORMULA.get(fam, "{y} = a·{x}").format(y=yv["symbol"], x=xv["symbol"], q=f"{q:g}")
        lines.append(f"公式：{f}" + ("（a 有理论值）" if getattr(exp, "ideal_slope", None) is not None else "（a 待测）"))
        lines.append(f"变量：{xv['symbol']} {xv['name']}（{xv['unit'] or '无单位'}，控制量，用{xv['tool']}）；"
                     f"{yv['symbol']} {yv['name']}（{yv['unit'] or '无单位'}，测量量，用{yv['tool']}）")
    return "\n".join(lines)


# ── 材料：方案与数据表 ────────────────────────────────────────────────


def protocol_label(exp: Experiment, protocol: str) -> str:
    """方案 id（可带 "+菜单项"）→ 中文说明。"""
    parts = str(protocol).split("+")
    base = exp.protocols[parts[0]].label_zh if parts[0] in exp.protocols else "自定义方案"
    extra = []
    for tok in parts[1:]:
        try:
            extra.append(exp.menu_item(tok).label_zh)
        except KeyError:
            extra.append(tok)
    return base + ("；之后追加：" + "；".join(extra) if extra else "")


def _reading_head(rd) -> str:
    return f"{rd.label_zh}（{rd.unit}）" if rd.unit else rd.label_zh


def data_text(exp: Experiment, ds: Dataset) -> str:
    """数据表：主测量逐行列出控制量和读数（照学生记录的样子），其他种类的观测单独列出。"""
    lines: list[str] = [f"方案：{protocol_label(exp, ds.meta.get('protocol', 'custom'))}"]
    by_kind: dict[str, list[tuple[int, object]]] = {}
    for i, o in enumerate(ds.observations):
        by_kind.setdefault(o.kind, []).append((i, o))
    for kind, rows in by_kind.items():
        ok = exp.obs_kinds[kind]
        has_setting = bool(ok.setting_label_zh)
        head = ([f"{ok.setting_label_zh}（{ok.setting_unit}）" if ok.setting_unit else ok.setting_label_zh]
                if has_setting else []) + [_reading_head(r) for r in ok.readings]
        lines.append(f"〔{ok.label_zh}〕")
        lines.append("试次｜" + "｜".join(head))
        for i, o in rows:
            cells = ([plain(o.setting)] if has_setting else []) + [plain(v) for v in o.values]
            lines.append(f"{i + 1}｜" + "｜".join(cells))
    return "\n".join(lines)


# ── 材料：特征面板（只有数字，不带判出的特征）────────────────────────────


def panel_material(exp: Experiment, ds: Dataset) -> dict:
    """面板的数字（与事实单同一套取整，讲解里引用的数字能在这里找到）和一致性的数值部分（不含结论）。"""
    lin = exp.linearise(ds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = compute_panel(lin)
    units = {"x": unit_of(lin.x_label), "y": unit_of(lin.y_label), "slope": slope_unit(lin.x_label, lin.y_label)}
    nums = Numbers()
    t = _panel_numbers(nums, p, units, lin.ideal_slope)
    ag = _agreement(exp, ds, p, lin, units, nums)
    ideal_b = lin.ideal_intercept
    return {"x_label": lin.x_label, "y_label": lin.y_label, "units": units, "ideal_slope": lin.ideal_slope,
            "ideal_intercept": ideal_b if ideal_b is not None and math.isfinite(ideal_b) else None,
            "n": p.n, "sufficient": p.sufficient, "text": t,
            "agreement": {k: ag[k] for k in ("quantity", "value", "uncertainty", "unit", "z")}}


def _u(text, unit: str) -> str:
    return f"{text} {unit}" if unit else str(text)


def panel_text(m: dict) -> str:
    t, u = m["text"], m["units"]
    lines = [f"作图：横轴 {m['x_label']}，纵轴 {m['y_label']}"]
    if m["ideal_slope"]:
        slope_txt = _u(f"{m['ideal_slope']:.4g}", u["slope"])
        lines.append(f"理论斜率 {slope_txt}；理论截距 "
                     + (_u(f"{m['ideal_intercept']:g}", u["y"]) if m["ideal_intercept"] is not None else "未知"))
    else:
        lines.append("斜率没有理论值（自由拟合）；理论截距 "
                     + (_u(f"{m['ideal_intercept']:g}", u["y"]) if m["ideal_intercept"] is not None else "未知"))
    lines.append(f"点数 n = {m['n']}")
    if t.get("slope") is None:
        lines.append("点数太少，无法拟合直线，面板不能给出斜率、截距和其他统计量。")
        return "\n".join(lines)
    lines.append("斜率 = " + _u(f"{t['slope']} ± {t['slope_se']}", u["slope"]))
    icp = "截距 = " + _u(f"{t['intercept']} ± {t['intercept_se']}", u["y"])
    if t.get("intercept_z") is not None and m["ideal_intercept"] is not None:
        icp += f"（相对理论截距 z = {t['intercept_z']}）"
    lines.append(icp)
    if t.get("slope_ratio") is not None:
        lines.append(f"斜率比（实测/理论）= {t['slope_ratio']}（z = {t['slope_ratio_z']}）")
    lines.append(f"曲率 z = {t['curvature_z']}" if t.get("curvature_z") is not None else "曲率：点数不足 4，无法检验")
    if t.get("chi2_dof") is not None:
        lines.append(f"χ²/自由度 = {t['chi2_dof']}（偏大 p = {t['scatter_p_high']}，偏小 p = {t['scatter_p_low']}）")
    else:
        lines.append("残差散布：自由度不足，无法检验")
    if t.get("drift_rho") is not None:
        lines.append(f"残差与试次顺序的秩相关 ρ = {t['drift_rho']}（p = {t['drift_p']}）")
    else:
        lines.append("漂移：点数不足 5，无法检验")
    if not m["sufficient"]:
        lines.append("（点数少于 3，面板不判任何特征）")
    return "\n".join(lines)


def agreement_text(m: dict) -> str:
    a = m["agreement"]
    if a.get("z") is None or not m["sufficient"]:
        return "一致性：点数不足，没有 z。"
    val = f"{a['value']} ± {a['uncertainty']}" if a.get("uncertainty") is not None else f"{a['value']}"
    return f"一致性：{a['quantity']} = {_u(val, a.get('unit', ''))}，z = {a['z']}"


# ── 材料：候选原因、追加测量 ──────────────────────────────────────────


def candidates_text(candidates: Sequence[Mapping]) -> str:
    """candidates: [{"id": "C1", "name": ..., "description": ...}]（顺序就是显示顺序）。"""
    return "【候选原因】（编号随机，与排列顺序无关）\n" + "\n".join(
        f"{c['id']} {c['name']}：{c['description']}" for c in candidates)


def menu_text(exp: Experiment, menu: Sequence[Mapping]) -> str:
    """menu: [{"id": "M1", "item_id": 原菜单 id}]。每项写名称、用时和会得到的读数。"""
    lines = ["【可做的追加测量】"]
    for m in menu:
        item = exp.menu_item(m["item_id"])
        kinds = list(dict.fromkeys(k for k, _ in item.plan))
        reads = "；".join("、".join(_reading_head(r) for r in exp.obs_kinds[k].readings) for k in kinds)
        lines.append(f"{m['id']} {item.label_zh}（约 {item.minutes:g} 分钟；读数：{reads}）")
    return "\n".join(lines)


def _messages(user: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT_ZH}, {"role": "user", "content": user}]


# ── 诊断 ──────────────────────────────────────────────────────────────

DIAGNOSE_TASK = ("【任务】诊断：学生做完了下面的测量。根据数据和特征面板，判断各候选原因的可能性，"
                 "并从追加测量里选出最能区分它们的下一次测量。")
DIAGNOSE_RULES = (
    "【要求】\n"
    "1. signatures：按特征面板的数字和特征词表写出这组数据的特征。\n"
    "2. evidence：一句话证据，只引用面板或数据里的数字。\n"
    "3. ranked：列出全部候选原因，按可能性从大到小；prob 是你的估计（取 0.05 的倍数）；"
    "expect 是这个原因在标准方案下会让面板出现的特征。\n"
    "4. not_in_list：数据和所有候选原因都对不上时写 true，否则写 false。\n"
    "5. next_measurement：选一项追加测量（写编号）；predicted 写排名前两位的原因下，这项测量第一个读数的预期值"
    "（你的估计）和单位。没有能区分它们的测量时写 null。\n"
    f"6. {AGREEMENT_RULE}\n"
    f"7. explanation：不超过 {EXPLAIN_LIMIT} 字的中文讲解：最可能的原因（照抄名称）、一个数值证据、下一次测量。\n"
    "8. 只输出一个 JSON 对象，不要输出其他文字，字段顺序固定：\n" + DIAGNOSE_EXAMPLE
)
PROPOSE_TASK = ("【任务】诊断（自己提出原因）：学生做完了下面的测量。本题不给候选原因清单，请你根据实验和数据"
                f"提出最多 {MAX_PROPOSE} 个可能的原因（包括“只有随机误差”），判断可能性，并选出下一次测量。")
PROPOSE_RULES = (
    "【要求】\n"
    "1. signatures：按特征面板的数字和特征词表写出这组数据的特征。\n"
    "2. evidence：一句话证据，只引用面板或数据里的数字。\n"
    f"3. ranked：最多 {MAX_PROPOSE} 个原因，按可能性从大到小；name 用简短的中文写出物理原因；mechanism 用机制词表；"
    "expect 是这个原因在标准方案下会让面板出现的特征；prob 是你的估计（取 0.05 的倍数）。\n"
    "4. next_measurement：选一项追加测量（写编号）；predicted 写排名前两位的原因（照抄你写的 name）下，"
    "这项测量第一个读数的预期值（你的估计）和单位。没有能区分它们的测量时写 null。\n"
    f"5. {AGREEMENT_RULE}\n"
    f"6. explanation：不超过 {EXPLAIN_LIMIT} 字的中文讲解：最可能的原因、一个数值证据、下一次测量。\n"
    "7. 只输出一个 JSON 对象，不要输出其他文字，字段顺序固定：\n" + PROPOSE_EXAMPLE
)


def build_diagnose_messages(exp: Experiment, ds: Dataset, candidates: Sequence[Mapping] | None,
                            menu: Sequence[Mapping]) -> list[dict]:
    """诊断提示。candidates=None 时是"自己提出原因"变体（不给清单）。

    candidates: [{"id": "C1", "name", "description"}]，menu: [{"id": "M1", "item_id"}]，编号由调用方打乱。
    这里从不接收后验、事实单或原因的特征标签。
    """
    m = panel_material(exp, ds)
    parts = [PROPOSE_TASK if candidates is None else DIAGNOSE_TASK,
             "【实验】\n" + card_text(exp),
             "【数据】\n" + data_text(exp, ds),
             "【特征面板】\n" + panel_text(m) + "\n" + agreement_text(m)]
    if candidates is not None:
        parts.append(candidates_text(candidates))
    parts.append(menu_text(exp, menu))
    parts.append(SIGNATURE_BLOCK)
    if candidates is None:
        parts.append(MECHANISM_BLOCK)
    parts.append(PROPOSE_RULES if candidates is None else DIAGNOSE_RULES)
    return _messages("\n\n".join(parts))


# ── 实验前设计 ────────────────────────────────────────────────────────

DESIGN_TASK = ("【任务】实验前规划：学生还没开始测量。根据实验、计划的方案、仪器和孪生预测，写出预期读数表、"
               "主要的不确定度来源、建议的测量点数和范围，以及这个方案分不开的原因和能把它们分开的追加测量。")
DESIGN_RULES = (
    "【要求】\n"
    "1. expected：照抄孪生预测，每行写控制量 setting、预期读数 value、标准差 sd 和单位。\n"
    "2. dominant_source：对斜率不确定度贡献最大的来源；reading 照抄读数名称，type 写 random（随机读数误差）"
    "或 resolution（读数分辨率）。\n"
    "3. recommend：建议的点数 n_points、控制量范围 range [最小, 最大] 和单位；计划只测 1 组时改用标准方案的点数和范围；"
    "标准方案下斜率的相对不确定度超过 5% 时点数加倍（最多 12）；repeat_each 写每个点重复读数取平均是否值得"
    "（随机读数误差为主时值得，分辨率为主时不值得）。\n"
    "4. unresolved：这个方案在典型噪声下分不开（confused）或只能部分分开（partial）的原因对，最多 6 对；"
    "measurement 写最能把这一对分开的追加测量编号，没有明显帮助就写 null。\n"
    f"5. explanation：不超过 {EXPLAIN_LIMIT} 字的中文说明。\n"
    "6. 只输出一个 JSON 对象，不要输出其他文字，字段顺序固定：\n" + DESIGN_EXAMPLE
)


def instruments_text(exp: Experiment, kinds: Sequence[str]) -> str:
    lines = ["【仪器】"]
    for k in kinds:
        for r in exp.obs_kinds[k].readings:
            lines.append(f"{r.label_zh}：分度值 {_u(f'{r.resolution:g}', r.unit)}，随机读数误差（标准差）约 "
                         f"{_u(f'{r.sigma:.2g}', r.unit)}")
    return "\n".join(lines)


def expected_text(rows: Sequence[Mapping], setting_label: str, setting_unit: str) -> str:
    """rows: [{"setting", "reading", "value", "sd", "unit"}]（引擎孪生算好、已取整）。"""
    head = f"{setting_label}（{setting_unit}）" if setting_unit else setting_label
    lines = ["【孪生预测】（引擎按这台装置的参数、假设只有随机误差算出）", f"序号｜{head}｜预期读数"]
    for i, r in enumerate(rows):
        val = _u(f"{plain(r['value'])} ± {plain(r['sd'])}", r["unit"])
        lines.append(f"{i + 1}｜{plain(r['setting'])}｜{r['reading']} = {val}")
    return "\n".join(lines)


def build_design_messages(exp: Experiment, protocol: str, expected_rows: Sequence[Mapping],
                          budget_line: str, candidates: Sequence[Mapping], menu: Sequence[Mapping]) -> list[dict]:
    """实验前设计提示。expected_rows 来自 twin（引擎），budget_line 是标准方案的斜率不确定度（引擎），
    candidates / menu 的编号由调用方打乱。不含不确定度来源的份额，也不含图谱结果。"""
    plan = exp.protocols[protocol].plan
    main = plan[0][0]
    ok = exp.obs_kinds[main]
    settings = "、".join(plain(s) for _, s in plan)
    parts = [DESIGN_TASK,
             "【实验】\n" + card_text(exp),
             f"【计划的方案】{protocol_label(exp, protocol)}：{ok.setting_label_zh} 依次取 {settings}"
             + (f" {ok.setting_unit}" if ok.setting_unit else ""),
             instruments_text(exp, [main]),
             expected_text(expected_rows, ok.setting_label_zh, ok.setting_unit) + "\n" + budget_line,
             candidates_text(candidates),
             menu_text(exp, menu),
             DESIGN_RULES]
    return _messages("\n\n".join(parts))


# ── 讲解（与推理时完全相同的上下文组装）──────────────────────────────────


def build_explain_messages(fact_sheet: dict, card: dict | None, experiment_id: str | None,
                           protocol: str = "standard", observations: Sequence[Mapping] = (),
                           budget_chars: int = 3000) -> list[dict]:
    """讲解提示 = policy.guarded_explanation 推理时的 build_context(..., phase="explain") 调用。"""
    s = LabSession(experiment_id=experiment_id, protocol=protocol, phase="explain", card=card,
                   observations=[dict(o) for o in observations], fact_sheet=fact_sheet)
    return build_context(s, fact_sheet, card, budget_chars, phase="explain")


# ── 知识问答 ──────────────────────────────────────────────────────────

KNOWLEDGE_RULES = (
    "【要求】knowledge_points 列出相关知识点（最相关的在前，最多 3 个），prerequisites 列出要先掌握的前置知识点"
    "（先学的在前，最多 5 个，没有就写空列表），id 和名称照抄知识点表的写法；answer 用不超过 80 字的中文简短回答。"
    "只输出一个 JSON 对象，字段顺序固定：\n" + KNOWLEDGE_EXAMPLE
)


def build_knowledge_messages(question: str) -> list[dict]:
    return _messages("【任务】知识点问答\n【问题】" + question.strip() + "\n\n" + KNOWLEDGE_RULES)


def fixed_blocks() -> tuple[str, ...]:
    """所有样本共用的固定文字（泄漏测试先把它们去掉，再查提示里有没有特征标签）。"""
    return (SIGNATURE_BLOCK, MECHANISM_BLOCK, DIAGNOSE_RULES, PROPOSE_RULES, DESIGN_RULES, KNOWLEDGE_RULES,
            DIAGNOSE_TASK, PROPOSE_TASK, DESIGN_TASK)
