"""按阶段组装给模型的上下文：只放这一阶段需要的信息，从不放完整对话历史。

输出 = [system（SYSTEM_PROMPT_ZH，训练和推理共用、保持不变）, user（本阶段任务 + 精简材料）]。
材料按优先级排：任务 > 学生刚说的话 > 事实单要点 > 学生预测 > 实验卡要点 > 数据 > 最近 2 轮摘要；
超出字数预算时从低优先级开始压缩。同样的输入一定得到同样的输出（训练样本可复现）。
事实单只给"可引用数字"和中文要点，不给面板原始浮点数，免得模型引用校验器不认的数字。
"""

from __future__ import annotations

import json

from .session import LabSession
from .templates import BAND_ZH, KP_MAX, SIGNATURE_ZH, kp_name, rank_kps

SYSTEM_PROMPT_ZH = (
    "你是“实验侦探”，陪中学生做力学实验的探究伙伴。\n"
    "规则：\n"
    "1. 你不做任何物理计算。数字只能照抄材料里给出的数字（事实单、可引用数字、学生的数据和预测），单位照抄。\n"
    "2. 原因名称和测量名称照抄事实单，不要编造清单之外的原因。\n"
    "3. 探究式引导：先让学生预测再测量；一次只问一个问题；不替学生写结论。\n"
    "4. 讲解不超过120字，必须包含：最可能的原因、至少一个数值证据、下一次测量。\n"
    "5. 偏大、偏小等方向词必须和数据的正负一致；拿不准就不说。\n"
    "6. 用简体中文，简洁友好；直接回答，不输出思考过程。"
)

PHASE_TASKS_ZH = {
    "identify": "确认学生在做哪个实验。不确定时只问一个最能区分的问题。",
    "plan": "帮学生明确要改变的量、要测的量和步骤。用一个问题引导，不要直接给出全部答案。",
    "predict": "请学生在测量前写下预测（数值或选项）。不要说出答案。",
    "measure": "告诉学生下一行要测什么、怎么读数。只说下一步。",
    "check": "根据这一行的核对结果，用一两句话告诉学生是否一致；不一致时提醒检查操作。",
    "diagnose": "先请学生说出他认为最可能的原因；学生已经选过时，对照事实单说明排名和证据，并介绍下一次测量及各原因的预测值。",
    "explain": "根据事实单写一段不超过120字的讲解：最可能的原因（照抄名称）、至少一个数值证据（只用可引用数字）、下一次测量（照抄名称）及各原因的预测值、修正方法。",
    "reflect": "请学生用自己的话写结论（结果、证据、原因、改进）。你只提问和点评，不替他写。",
}

# 各阶段需要哪些材料
_NEEDS = {
    "identify": {"card_min"},
    "plan": {"card"},
    "predict": {"card", "pending"},
    "measure": {"card_min", "data", "pending"},
    "check": {"card_min", "data", "check"},
    "diagnose": {"card_min", "facts"},
    "explain": {"card_min", "facts", "fix", "kp"},
    "reflect": {"card_min", "facts_min", "fix", "kp"},
}


ROLE_ZH = {"control": "控制量", "measured": "测量量", "derived": "导出量"}


def _r(v):
    """紧凑显示用：浮点数保留 4 位有效数字。"""
    if isinstance(v, float):
        return float(f"{v:.4g}")
    if isinstance(v, dict):
        return {k: _r(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_r(x) for x in v]
    return v


def _j(obj) -> str:
    return json.dumps(_r(obj), ensure_ascii=False, separators=(",", ":"))


def _card_variants(card: dict | None, full: bool) -> list[str]:
    if not card:
        return [""]
    name = card.get("name_zh", "")
    vars_ = "；".join(
        f"{v.get('symbol', '')} {v.get('name', '')}（{v.get('unit', '')}，{ROLE_ZH.get(v.get('role', ''), v.get('role', ''))}）"
        for v in card.get("variables") or []
    )
    lines_full = [f"实验：{name}"]
    if card.get("goal"):
        lines_full.append(f"目的：{card['goal']}")
    if card.get("formula"):
        lines_full.append(f"公式：{card['formula']}")
    if vars_:
        lines_full.append(f"变量：{vars_}")
    cols = (card.get("data_table") or {}).get("columns") or []
    if cols:
        lines_full.append("记录表：" + "｜".join(str(c) for c in cols))
    if card.get("steps"):
        lines_full.append("步骤：" + "；".join(str(s) for s in card["steps"][:6]))
    minimal = [f"实验：{name}"] + ([f"公式：{card['formula']}"] if card.get("formula") else [])
    out = []
    if full:
        out.append("\n".join(lines_full))
        out.append("\n".join(lines_full[:4]))
    out.append("\n".join(minimal))
    out.append(f"实验：{name}")
    return out


# 可引用数字里这些条目已在别处出现（嫌疑人、下一次测量、一致性）或只是文字里的数字，不再重复列出
_DUP_PREFIXES = ("prob.", "prob_pct.", "next.", "agreement.", "cause_name.", "experiment_name", "protocol_label")


def _citable(fs: dict, level: int) -> list[str]:
    """可引用数字，压成 "名字=数值 单位"。level 0 = 去重后全部，1 = 只留前 12 个，2 = 不列。"""
    if level >= 2:
        return []
    out = []
    for e in fs.get("numbers") or []:
        name = str(e.get("name", ""))
        if "[" in name or name.startswith(_DUP_PREFIXES) or e.get("value") is None:
            continue
        out.append(f"{name}={_r(e['value'])}" + (f" {e['unit']}" if e.get("unit") else ""))
    return out[:12] if level == 1 else out


def _fact_essentials(fs: dict, phase: str, needs: set[str], kp_names: dict | None, top_n: int,
                     num_level: int = 0) -> dict:
    out: dict = {}
    ag = fs.get("agreement")
    if ag:
        a = {k: ag[k] for k in ("quantity", "value", "uncertainty", "unit", "z") if ag.get(k) is not None}
        if ag.get("band"):
            a["结论"] = BAND_ZH.get(ag["band"], ag["band"])
        out["一致性"] = a
    dg = fs.get("diagnosis") or {}
    ranked = dg.get("ranked") or []
    if "facts_min" in needs:
        if ranked:
            out["最可能的原因"] = ranked[0].get("name")
    elif ranked:
        out["嫌疑人"] = [
            {k: v for k, v in (("原因", r.get("name")), ("概率", r.get("prob")), ("证据", r.get("evidence")))
             if v not in (None, "")}
            for r in ranked[:top_n]
        ]
        if dg.get("not_in_list"):
            out["不在清单"] = True
        sig = (fs.get("panel") or {}).get("signatures")
        if sig:
            out["数据特征"] = [SIGNATURE_ZH.get(s, s) for s in sig]
    nm = fs.get("next_measurement")
    if nm and "facts_min" not in needs:
        names = {r.get("cause_id"): r.get("name") for r in ranked}
        rank = {r.get("cause_id"): i for i, r in enumerate(ranked)}
        preds = sorted(nm.get("predicted") or [], key=lambda p: rank.get(p.get("cause_id"), len(rank)))
        out["下一次测量"] = {
            "名称": nm.get("label"),
            **({"信息增益bit": nm["eig_bits"]} if nm.get("eig_bits") is not None else {}),
            "各原因预测": [
                {"原因": p.get("name") or names.get(p.get("cause_id"), p.get("cause_id")), "值": p.get("value"),
                 "单位": p.get("unit", "")}
                for p in preds[:top_n]
            ],
        }
    elif nm is None and "facts_min" not in needs and ranked:
        out["下一次测量"] = "菜单里没有能分开剩下原因的测量"
    fx = fs.get("fix") or {}
    if "fix" in needs and fx.get("text"):
        out["修正"] = {k: v for k, v in (("方法", fx.get("text")), ("复测预期", fx.get("predicted_rerun"))) if v}
    if "kp" in needs and fs.get("knowledge_points"):
        out["知识点"] = [kp_name(k, kp_names) for k in rank_kps(fs, fs["knowledge_points"])[:KP_MAX]]
    cite = _citable(fs, num_level) if "facts_min" not in needs else []
    if cite:
        out["可引用数字"] = cite
    return out


def _fmt_obs(o: dict, i: int) -> str:
    vals = ",".join(f"{float(v):.6g}" for v in o.get("values", []))
    kind = "" if o.get("kind") == "main" else f"[{o.get('kind')}]"
    return f"{i + 1}{kind}: {float(o.get('setting', 0)):.6g} → {vals}"


def build_context(
    session: LabSession,
    fact_sheet: dict | None = None,
    card: dict | None = None,
    budget_chars: int = 3000,
    phase: str | None = None,
    student_text: str = "",
    kp_names: dict | None = None,
) -> list[dict]:
    """组装 [system, user] 两条消息。budget_chars 管两条消息内容的总字数（system 提示固定不压缩）。"""
    phase = phase or session.phase
    fs = fact_sheet if fact_sheet is not None else (session.fact_sheet or {})
    card = card if card is not None else session.card
    needs = _NEEDS[phase]

    # 每个段落：(优先级, [从完整到最简的几种写法])；优先级低的先压缩
    sections: list[tuple[int, str, list[str]]] = []
    sections.append((100, "task", [f"【阶段】{phase}\n【任务】{PHASE_TASKS_ZH[phase]}"]))
    if student_text:
        st = " ".join(student_text.split())
        sections.append((95, "student", [f"【学生刚才说】{st}", f"【学生刚才说】{st[:200]}…"]))
    if needs & {"facts", "facts_min"} and fs:
        variants = [f"【事实单】{_j(_fact_essentials(fs, phase, needs, kp_names, n, lv))}"
                    for n, lv in ((3, 0), (2, 0), (2, 1), (1, 1), (1, 2))]
        sections.append((90, "facts", variants))
    if "check" in needs and session.checks:
        c = session.checks[-1]
        keep = {k: c[k] for k in ("index", "value", "predicted", "uncertainty", "unit", "z", "band") if k in c}
        if "band" in keep:
            keep["band"] = BAND_ZH.get(keep["band"], keep["band"])
        sections.append((85, "check", [f"【本行核对】{_j(keep)}"]))
    if "pending" in needs and session.pending_measurement:
        pm = session.pending_measurement
        txt = f"【待做的追加测量】{pm.get('label', pm.get('id', ''))}"
        nm = (fs or {}).get("next_measurement") or {}
        if nm.get("predicted") and nm.get("id") == pm.get("id"):
            names = {r.get("cause_id"): r.get("name") for r in (fs.get("diagnosis") or {}).get("ranked") or []}
            txt += "；各原因预测：" + "；".join(
                f"{names.get(p.get('cause_id'), p.get('cause_id'))} {_r(p.get('value'))} {p.get('unit', '')}".strip()
                for p in nm["predicted"][:3])
        sections.append((80, "pending", [txt]))
    if session.student_predictions:
        preds = [
            f"{p['target']}: {_r(p['value'])}{(' ' + p['unit']) if p.get('unit') else ''}"
            for p in session.student_predictions
        ]
        sections.append((60, "preds", [f"【学生的预测】{'；'.join(preds[-k:])}" for k in (5, 2)] + [""]))
    if needs & {"card", "card_min"}:
        sections.append((50, "card", _card_variants(card, "card" in needs)))
    if "data" in needs and session.observations:
        obs = session.observations
        rows = [_fmt_obs(o, i) for i, o in enumerate(obs)]
        sections.append((40, "data", [f"【已测数据】共 {len(rows)} 行\n" + "\n".join(rows[-k:])
                                      for k in (8, 3)] + [f"【已测数据】共 {len(rows)} 行"]))
    turns = session.last_turns(2)
    if turns:
        def tl(ts):
            return "【最近对话】\n" + "\n".join(
                f"[{t['phase']}] 老师：{t['tutor']}" + (f" / 学生：{t['student']}" if t.get("student") else "")
                for t in ts)
        sections.append((30, "turns", [tl(turns), tl(turns[-1:]), ""]))

    order = [name for _, name, _ in sections]  # 输出顺序 = 加入顺序
    choice = {name: 0 for name in order}
    by_name = {name: (pri, vs) for pri, name, vs in sections}

    def render() -> str:
        return "\n\n".join(v for v in (by_name[n][1][choice[n]] for n in order) if v)

    user = render()
    limit = budget_chars - len(SYSTEM_PROMPT_ZH)
    while len(user) > limit:
        cands = [n for n in order if choice[n] < len(by_name[n][1]) - 1]
        if not cands:
            user = user[: max(limit - 1, 0)] + "…"
            break
        n = min(cands, key=lambda x: (by_name[x][0], order.index(x)))
        choice[n] += 1
        user = render()
    return [{"role": "system", "content": SYSTEM_PROMPT_ZH}, {"role": "user", "content": user}]
