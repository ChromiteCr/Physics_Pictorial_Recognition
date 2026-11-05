"""中文模板：各阶段的引导语，以及根据事实单写的完整讲解（脚本模式，不用模型）。

讲解分五块：原因、证据、下一次测量（附各嫌疑原因的预测值）、修正方法、知识点；每块不超过 120 字。
模板里的数字全部原样取自事实单（只做四位有效数字的显示取整），所以一定能通过 verifier。
"""

from __future__ import annotations

import math
import re

BLOCK_LIMIT = 120
BLOCK_KEYS = ("cause", "evidence", "next", "fix", "kp")
KP_MAX = 5  # 讲解里最多列几个知识点（先按与最可能原因的相关性排序，见 rank_kps）

SIGNATURE_ZH = {
    "none": "没有明显异常",
    "intercept_pos": "直线不过原点，截距为正",
    "intercept_neg": "直线不过原点，截距为负",
    "slope_high": "斜率比理论值大",
    "slope_low": "斜率比理论值小",
    "curve_up": "图线向上弯",
    "curve_down": "图线向下弯",
    "scatter_excess": "数据点比仪器误差散得开",
    "scatter_deficit": "数据点比仪器误差还整齐（可能抄写或凑数）",
    "drift": "读数随试次顺序漂移",
}

BAND_ZH = {"consistent": "与预测一致", "tension": "与预测有些偏离", "inconsistent": "与预测明显不符"}

PHASE_ZH = {
    "identify": "认实验", "plan": "定方案", "predict": "先预测", "measure": "测数据",
    "check": "对一对", "diagnose": "找原因", "explain": "讲明白", "reflect": "写结论",
}

# 没有知识点模块时的兜底名称
_KP_FALLBACK = {
    "kp.hooke_law": "胡克定律",
    "kp.systematic_error": "系统误差",
    "kp.random_error": "随机误差",
    "kp.elastic_limit": "弹性限度",
    "kp.newton_second_law": "牛顿第二定律",
    "kp.friction": "滑动摩擦力",
    "kp.free_fall": "自由落体运动",
    "kp.energy_conservation": "机械能守恒",
    "kp.momentum_conservation": "动量守恒",
    "kp.pendulum": "单摆周期",
    "kp.uncertainty": "测量不确定度",
}


# ── 小工具 ──────────────────────────────────────────────────────────────


def fmt(v, sig: int = 4) -> str:
    """数字显示：最多 sig 位有效数字，去掉多余的 0，不用科学计数法（极小/极大值除外）。"""
    v = float(v)
    if v == 0 or not math.isfinite(v):
        return "0" if v == 0 else str(v)
    mag = math.floor(math.log10(abs(v)))
    if mag >= 6 or mag <= -5:
        m = f"{v / 10**mag:.{sig - 1}f}".rstrip("0").rstrip(".")
        return f"{m}×10^{mag}"
    dec = max(sig - 1 - mag, 0)
    s = f"{round(v, dec):.{dec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def pct(p: float) -> str:
    """概率 → 百分数字符串（不带 %），保证至少两位有效数字，避免 '3%' 这种被校验器判成粗略取整。"""
    x = float(p) * 100
    r = round(x)
    if r == 100 and x < 100:  # 0.997 → "99.7"，不写成 "100"
        sig = 3
        while fmt(x, sig) == "100" and sig < 8:
            sig += 1
        return fmt(x, sig)
    if r >= 10:
        return str(int(r))
    return fmt(x, 2)


def with_unit(v, unit: str) -> str:
    return f"{fmt(v)} {unit}".strip() if unit else fmt(v)


def nchars(text: str) -> int:
    """字数：去掉空白后的字符数（比只数汉字更严）。"""
    return len(re.sub(r"\s+", "", text))


def compose(parts: list[str], limit: int = BLOCK_LIMIT) -> str:
    """按顺序拼接各部分，超过字数就丢掉后面的部分；第一部分本身超长时在标点处截断。"""
    out = ""
    for i, p in enumerate(p for p in parts if p):
        if nchars(out + p) <= limit:
            out += p
        elif i == 0:
            cut = p
            while nchars(cut) > limit:
                k = max(cut.rfind(ch, 0, len(cut) - 1) for ch in "，。；、")
                cut = cut[:k] if k > 0 else cut[: limit - 1]
            out = cut.rstrip("，；、") + "。"
            break
    return out


def kp_name(kp, kp_names: dict | None = None) -> str:
    """知识点 id → 中文名：先查传入的表，再查知识点模块，最后用兜底表。"""
    if isinstance(kp, dict):
        return kp.get("name_zh") or kp_name(kp.get("id", ""), kp_names)
    if kp_names and kp in kp_names:
        return kp_names[kp]
    try:
        from .. import knowledge  # 懒加载：知识点模块在并行开发中

        k = knowledge.kp(kp)
        name = k.get("name_zh") if isinstance(k, dict) else getattr(k, "name_zh", None)
        if name:
            return name
    except Exception:  # noqa: BLE001
        pass
    return _KP_FALLBACK.get(kp, str(kp).removeprefix("kp.").replace("_", " "))


def cause_name(fs: dict, cause_id: str) -> str:
    for r in (fs.get("diagnosis") or {}).get("ranked") or []:
        if r.get("cause_id") == cause_id:
            return r.get("name") or cause_id
    return cause_id


def signatures_zh(fs: dict) -> str:
    sig = (fs.get("panel") or {}).get("signatures") or ["none"]
    return "；".join(SIGNATURE_ZH.get(s, s) for s in sig)


# ── 讲解五块 ────────────────────────────────────────────────────────────


def block_cause(fs: dict) -> str:
    dg = fs.get("diagnosis") or {}
    ranked = dg.get("ranked") or []
    if not ranked:
        return "数据还不够，暂时判断不了误差来源。"
    top = ranked[0]
    p = f"（可能性 {pct(top['prob'])}%）" if top.get("prob") is not None and top["prob"] >= 0.005 else ""
    if dg.get("not_in_list"):
        mp = dg.get("misfit_p")
        mp_s = f"（失配 p = {fmt(mp, 2)}）" if mp is not None else ""
        return compose([f"数据和清单里的原因都对不上{mp_s}，可能有清单之外的原因。",
                        f"清单里最接近的是「{top['name']}」。"])
    if str(top.get("cause_id", "")).endswith(".random_only"):
        return compose([f"数据和孪生预测相符，最可能是「{top['name']}」{p}，没有发现系统误差。"])
    parts = [f"最可能的原因是「{top['name']}」{p}"]
    if len(ranked) > 1 and ranked[1].get("prob", 0) >= 0.05:
        parts.append(f"，其次是「{ranked[1]['name']}」（{pct(ranked[1]['prob'])}%）")
    return compose(parts + ["。"])


def block_evidence(fs: dict) -> str:
    ranked = (fs.get("diagnosis") or {}).get("ranked") or []
    ag = fs.get("agreement") or {}
    parts = []
    ev = ranked[0].get("evidence") if ranked else ""
    if ev:
        parts.append(f"证据：{ev}。")
    if ag.get("value") is not None:
        q, unit = ag.get("quantity", "测量值"), ag.get("unit", "")
        s = f"测得 {q} = {fmt(ag['value'])}" if str(q).isascii() else f"测得{q} = {fmt(ag['value'])}"
        if ag.get("uncertainty") is not None:
            s += f" ± {fmt(ag['uncertainty'])}"
        s += f" {unit}".rstrip()
        if ag.get("z") is not None:
            z = float(ag["z"])
            band = BAND_ZH.get(ag.get("band", ""), "")
            zs = fmt(z, 2 if abs(z) < 10 else 3)
            s += f"（z = {zs}，{band}）" if band else f"（z = {zs}）"
        parts.append(("" if parts else "证据：") + s + "。")
    if not parts:
        parts.append(f"证据：数据特征是{signatures_zh(fs)}。")
    return compose(parts)


def block_next(fs: dict) -> str:
    nm = fs.get("next_measurement")
    if not nm or not nm.get("label"):
        return "现有的追加测量都分不开剩下的原因；可以按改进方法重做一遍来检验。"
    head = f"下一步：{nm['label']}"
    if nm.get("eig_bits") is not None:
        head += f"（信息增益 {fmt(nm['eig_bits'], 2)} bit）"
    parts = [head + "。"]
    preds = [p for p in nm.get("predicted") or [] if p.get("value") is not None]
    rank = {r.get("cause_id"): i for i, r in enumerate((fs.get("diagnosis") or {}).get("ranked") or [])}
    preds.sort(key=lambda p: rank.get(p.get("cause_id"), len(rank)))  # 先列排名靠前的嫌疑原因（稳定排序）
    for i, p in enumerate(preds[:3]):
        sep = "；" if i else ""
        name = p.get("name") or cause_name(fs, p.get("cause_id", ""))
        parts.append(f"{sep}若是「{name}」，预计 {with_unit(p['value'], p.get('unit', ''))}")
    out = compose(parts, BLOCK_LIMIT - 1)  # 留一个字给句号
    return out if out.endswith("。") else out + "。"


def block_fix(fs: dict) -> str:
    fx = fs.get("fix") or {}
    if not fx.get("text"):
        return "改进：按实验卡的步骤操作，多测几组取平均。"
    parts = [f"改进：{fx['text'].rstrip('。')}。"]
    rerun = (fx.get("predicted_rerun") or "").rstrip("。")
    if rerun:
        parts.append(f"{rerun}。" if "预计" in rerun else f"改正后复测，预计{rerun}。")
    return compose(parts)


def rank_kps(fs: dict, kps: list) -> list:
    """按与最可能原因的相关性给事实单里的知识点重新排序（只排序、不增删）；知识点模块不可用时保持原顺序。"""
    ranked = (fs.get("diagnosis") or {}).get("ranked") or []
    if not ranked or not fs.get("experiment_id") or (fs.get("diagnosis") or {}).get("not_in_list"):
        return list(kps)
    try:
        from .. import knowledge  # 懒加载

        order = knowledge.related_kps(fs["experiment_id"], ranked[0].get("cause_id"))
    except Exception:  # noqa: BLE001
        return list(kps)
    pos = {k: i for i, k in enumerate(order)}
    key = [k if isinstance(k, str) else (k or {}).get("id") for k in kps]
    idx = sorted(range(len(kps)), key=lambda i: (pos.get(key[i], len(pos)), i))
    return [kps[i] for i in idx]


def block_kp(fs: dict, kp_names: dict | None = None) -> str:
    kps = rank_kps(fs, fs.get("knowledge_points") or [])
    if not kps:
        return ""
    names = []
    for k in kps:
        n = kp_name(k, kp_names)
        if n not in names:
            names.append(n)
    names = names[:KP_MAX]
    return compose(["关联知识点：" + "、".join(names) + "。"])


def explanation_blocks(fs: dict, kp_names: dict | None = None) -> dict[str, str]:
    """事实单 → 五块讲解（脚本模式）。每块不超过 120 字。"""
    return {
        "cause": block_cause(fs),
        "evidence": block_evidence(fs),
        "next": block_next(fs),
        "fix": block_fix(fs),
        "kp": block_kp(fs, kp_names),
    }


def explanation_text(fs: dict, kp_names: dict | None = None, keys=BLOCK_KEYS) -> str:
    b = explanation_blocks(fs, kp_names)
    return "\n".join(b[k] for k in keys if b.get(k))


# ── 各阶段的引导语 ───────────────────────────────────────────────────────

T = {
    "identify_known": "这看起来是「{name}」。对吗？",
    "identify_unknown": "请拍一张实验装置的照片，或者告诉我你在做哪个实验。",
    "plan": "我们来做「{name}」。目的：{goal}\n先想一想：这个实验里，你要主动改变的量是哪一个？",
    "plan_steps": "课本方案：{steps}。准备好就开始。",
    "predict_first": "动手之前先预测：{setting_label}为 {setting} 时，{reading_label}大约是多少{unit_hint}？写下你的估计，测完再对照。",
    "predict_next": "下一步要做：{label}。不同的原因会给出不同的结果：{options}。你猜会是哪一种？",
    "measure_row": "请测第 {i} 组：{setting_label} = {setting}，读出{reading_label}{unit_hint}。",
    "measure_extra": "请做：{label}，把读数填进表格。",
    "measure_done": "这一轮的数据测完了。我们一起看看数据说明了什么。",
    "check_ok": "第 {i} 组：你读到 {value}，孪生预测 {pred}（z = {z}），{band}。继续下一组。",
    "check_warn": "第 {i} 组：你读到 {value}，孪生预测 {pred}（z = {z}），{band}。先检查一下：仪器静止了吗？视线和刻度平齐了吗？",
    "check_recorded": "第 {i} 组已记录：{value}。",
    "diagnose_ask": "看看数据：{signatures}。你觉得最可能的原因是哪一个？",
    "diagnose_none": "数据和孪生预测相符，没有发现系统误差。你觉得还有哪一步可能出问题吗？",
    "reflect": "请用自己的话写结论：① 你测得的结果是多少；② 最可能的误差来源是什么，证据是哪个数据；③ 下次怎样改进。",
    "reflect_pred": "你实验前预测的是 {pred}，实际测到 {actual}。想一想差别从哪里来。",
}

HINTS = {
    "plan": [
        "提示：自变量是你主动去改变的量，因变量是随之变化、需要读数的量。",
        "提示：看看实验卡里的变量表，哪个标着“控制量”？",
        "提示：这个实验里要主动改变的是「{control}」。",
    ],
    "predict": [
        "提示：回忆一下{kp}的公式，先估算一个数量级就好。",
        "提示：公式是 {formula}，代入你估计的数值试试。",
        "提示：先写下一个大概的数，猜错也没关系，测完对照才是关键。",
    ],
    "diagnose": [
        "提示：看图线的特征——{signatures}。哪种操作会造成这样的特征？",
        "提示：{evidence}",
        "提示：引擎认为最可能的是「{top}」。你同意吗？为什么？",
    ],
    "reflect": [
        "提示：结论要有数据支撑，引用你测到的具体数字。",
        "提示：说清楚误差来源时，要把它和图线特征联系起来。",
        "提示：可以按“结果—证据—原因—改进”四句话来写。",
    ],
}


def hint(phase: str, level: int, **kw) -> str:
    """第 level 级提示（1..3），没有就返回空串。缺少字段时退回上一级。"""
    ladder = HINTS.get(phase) or []
    for lv in range(min(level, len(ladder)), 0, -1):
        try:
            return ladder[lv - 1].format(**kw)
        except (KeyError, IndexError):
            continue
    return ""
