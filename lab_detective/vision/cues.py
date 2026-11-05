"""YOLO 线索：检测框 → 器材清单 + 装置构型线索 + 规则候选实验（带分数）。

规则表改编自旧版 intent_classifier.classify_intent()（那个文件会 import explain_api，这里不能引用），
差异（2026-11-05）：
- 实验 id 换成 labsim 的 id（hooke_law→hooke 等），并带上字母；
- 动量守恒不再要求检测到细线：同一轨道上两辆小车本身就是强特征，细线只算可选器材；
- 新增测力计朝向加分（竖直 → 胡克，水平 → 摩擦），以及单摆的弱规则（细线下挂钢球）。
  留出实验的规则分数 < 1，永远不会单独让规则"有把握"，只用来进候选、交给模型或澄清问题。
检测器只在 run_detector() 里懒加载，导入本模块不会加载 ultralytics。
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from collections import Counter
from pathlib import Path

from .letters import entry, letter_for

# YOLO 8 类（与 detect.py / dataset.yaml 编号一致）
YOLO_CLASSES: dict[int, str] = {
    0: "cart", 1: "track", 2: "spring", 3: "string",
    4: "ruler", 5: "dynamometer", 6: "wooden_block", 7: "iron_block",
}
YOLO_CLASS_ZH: dict[str, str] = {
    "cart": "小车", "track": "轨道", "spring": "弹簧", "string": "细线",
    "ruler": "刻度尺", "dynamometer": "弹簧测力计", "wooden_block": "木块", "iron_block": "钢球/铁块",
}
# detect.py 在推理时丢掉这两类（ruler 与 track 严重混淆、spring 已停用），规则输入与之保持一致
EXCLUDED_FROM_DETECT = frozenset({"ruler", "spring"})

CONFIDENT_SCORE_GAP = 0.3  # 第一、二名分差 ≥ 此值才算规则"分得出高下"（沿用旧值）
CONFIDENT_MIN_SCORE = 1.0  # 第一名至少满足一个课内实验的必需器材
OTHER_SCORE = 0.8  # 置信度换算时"其他实验"的虚拟分数
SOFTMAX_TAU = 0.2

# 课内 6 个实验的器材规则（复制自 physics.EXPERIMENT_TYPES，见文件头差异说明）
RULES: dict[str, dict] = {
    "friction": {"required": {"dynamometer"}, "optional": {"cart", "wooden_block", "track", "string"}},
    "average_speed": {"required": {"cart", "track"}, "optional": set()},
    "hooke": {"required": {"dynamometer"}, "optional": set()},
    "newton2": {"required": {"cart", "track"}, "optional": set()},
    "momentum": {"required": {"cart", "track"}, "optional": {"string"}},
    "energy": {"required": {"cart", "track"}, "optional": set()},
}


# ── 几何小工具 ───────────────────────────────────────────────────────


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _size(b):
    return max(abs(b[2] - b[0]), abs(b[3] - b[1]))


def _wh(b):
    return abs(b[2] - b[0]), abs(b[3] - b[1])


def _boxes(dets, name):
    return [d["bbox"] for d in dets if d["class_name"] == name and d.get("bbox")]


def _near(dets, a, b, max_ratio: float = 2.0) -> bool:
    """a、b 两类中任意一对框的中心距 ≤ 平均尺寸的 max_ratio 倍（旧版 _proximity_bonus 的判据）。"""
    for ba in _boxes(dets, a):
        for bb in _boxes(dets, b):
            (xa, ya), (xb, yb) = _center(ba), _center(bb)
            avg = (_size(ba) + _size(bb)) / 2 or 1.0
            if math.hypot(xa - xb, ya - yb) / avg <= max_ratio:
                return True
    return False


def _orientation(b) -> str:
    w, h = _wh(b)
    if w >= 1.5 * h:
        return "horizontal"
    if h >= 1.5 * w:
        return "vertical"
    return "unclear"


def _hanging_ball(dets) -> bool:
    """有细线框在钢球框上方、水平位置重叠：像细线下挂着小球。"""
    for s in _boxes(dets, "string"):
        for ball in _boxes(dets, "iron_block"):
            sx, sy = _center(s)
            bx, by = _center(ball)
            if sy < by and ball[0] - _wh(ball)[0] <= sx <= ball[2] + _wh(ball)[0]:
                return True
    return False


def _cart_on_track(dets) -> bool:
    for c in _boxes(dets, "cart"):
        cx, cy = _center(c)
        for t in _boxes(dets, "track"):
            pad = 0.25 * _size(t)
            if t[0] - pad <= cx <= t[2] + pad and t[1] - pad <= cy <= t[3] + pad:
                return True
    return False


# ── 对外接口 ─────────────────────────────────────────────────────────


def usable(dets: list[dict]) -> list[dict]:
    """去掉推理时不输出的类别（与 detect.py 一致）。"""
    return [d for d in dets if d.get("class_name") not in EXCLUDED_FROM_DETECT]


def run_detector(image_path, weights=None, conf: float = 0.25) -> tuple[list[dict], tuple[int, int]]:
    """调用 detect.detect_components()（懒加载 YOLO），返回 (检测框, (h, w))。"""
    os.environ.setdefault("YOLO_OFFLINE", "true")  # 关掉 ultralytics 的联网检查与统计上报
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    det_mod = importlib.import_module("detect")
    out = det_mod.detect_components(image_path, weights, conf)
    dets = [{k: d[k] for k in ("class_id", "class_name", "confidence", "bbox")} for d in out["detections"]]
    return dets, tuple(int(v) for v in out["image_shape"])


def apparatus_from_detections(dets: list[dict]) -> list[str]:
    """检测到的器材 id（去重，按 YOLO 类编号排序）。"""
    order = {n: i for i, n in YOLO_CLASSES.items()}
    names = {d["class_name"] for d in usable(dets) if d.get("class_name") in order}
    return sorted(names, key=order.__getitem__)


def configuration_cues(dets: list[dict]) -> list[str]:
    """由检测框几何关系得到的构型线索（中文短句；只说检测框能支持的事）。"""
    dets = usable(dets)
    n = Counter(d["class_name"] for d in dets)
    cues: list[str] = []
    if n["cart"] >= 2:
        cues.append(f"检测到 {n['cart']} 辆小车")
    if _cart_on_track(dets):
        cues.append("小车在轨道上")
    for b in _boxes(dets, "dynamometer")[:1]:
        o = _orientation(b)
        if o == "horizontal":
            cues.append("测力计大致水平放置")
        elif o == "vertical":
            cues.append("测力计大致竖直放置")
    if _near(dets, "dynamometer", "cart") or _near(dets, "dynamometer", "wooden_block"):
        cues.append("测力计挨着小车或木块")
    if _near(dets, "string", "cart"):
        cues.append("细线连着小车")
    if _hanging_ball(dets):
        cues.append("细线下方挂着小球")
    if n["wooden_block"]:
        cues.append("有木块")
    if n["iron_block"]:
        cues.append("有钢球或铁块")
    return cues


def rule_candidates(dets: list[dict]) -> list[dict]:
    """按器材组合 + 相对位置给实验打分，降序返回 [{letter, experiment_id, name_zh, score, reason}]。"""
    dets = usable(dets)
    counts = Counter(d["class_name"] for d in dets)
    names = set(counts)
    dyn_orient = {_orientation(b) for b in _boxes(dets, "dynamometer")}
    out = []

    for exp_id, rule in RULES.items():
        required = rule["required"]
        if not required <= names:
            continue
        if exp_id == "momentum" and counts["cart"] < 2:
            continue
        # 摩擦力：必须还有被拉的物体（小车或木块），否则单个测力计会和胡克定律撞车
        if exp_id == "friction" and not ({"cart", "wooden_block"} & names):
            continue
        score = 1.0
        reasons = [f"检测到必需器材：{'、'.join(YOLO_CLASS_ZH[x] for x in sorted(required))}"]
        if exp_id == "momentum":
            score += 0.5
            reasons.append("检测到两辆小车，是动量守恒的强特征")
        opt = rule["optional"] & names
        if opt:
            score += 0.2 * len(opt)
            reasons.append(f"匹配可选器材：{'、'.join(YOLO_CLASS_ZH[x] for x in sorted(opt))}")
        if exp_id == "friction":
            if _near(dets, "dynamometer", "cart") or _near(dets, "dynamometer", "wooden_block"):
                score += 0.2
                reasons.append("测力计挨着被拉物体，符合摩擦力实验的摆法")
            if "horizontal" in dyn_orient:
                score += 0.2
                reasons.append("测力计水平")
        if exp_id == "hooke" and "vertical" in dyn_orient:
            score += 0.2
            reasons.append("测力计竖直")
        out.append(_cand(exp_id, score, reasons))

    # 留出实验的弱规则（分数 < 1）
    if _hanging_ball(dets):
        out.append(_cand("pendulum", 0.6, ["细线下方挂着小球，可能是单摆"]))

    out.sort(key=lambda c: (-c["score"], c["letter"]))
    return out


def _cand(exp_id: str, score: float, reasons: list[str]) -> dict:
    letter = letter_for(exp_id)
    return {"letter": letter, "experiment_id": exp_id, "name_zh": entry(letter).name_zh,
            "score": round(score, 3), "reason": "；".join(reasons)}


def is_confident(cands: list[dict]) -> bool:
    """规则是否有把握：第一名满足课内实验必需器材，且唯一或领先第二名 ≥ 0.3。"""
    if not cands or cands[0]["score"] < CONFIDENT_MIN_SCORE:
        return False
    return len(cands) == 1 or cands[0]["score"] - cands[1]["score"] >= CONFIDENT_SCORE_GAP - 1e-9


def tied_letters(cands: list[dict]) -> list[str]:
    """与第一名分差 < 0.3 的候选字母（规则分不开的集合）。"""
    if not cands:
        return []
    top = cands[0]["score"]
    return [c["letter"] for c in cands if top - c["score"] < CONFIDENT_SCORE_GAP - 1e-9]


def rule_confidence(cands: list[dict]) -> float:
    """启发式置信度（未校准）：候选分数加一个"其他"虚拟项做 softmax，取第一名的概率。"""
    if not cands:
        return 0.0
    scores = [c["score"] for c in cands] + [OTHER_SCORE]
    m = max(scores)
    w = [math.exp((s - m) / SOFTMAX_TAU) for s in scores]
    return round(w[0] / sum(w), 3)


def analyze(dets: list[dict]) -> dict:
    """一次算齐：器材、构型线索、规则候选、是否有把握、平局集合、置信度。"""
    cands = rule_candidates(dets)
    return {
        "apparatus": apparatus_from_detections(dets),
        "cues": configuration_cues(dets),
        "candidates": cands,
        "confident": is_confident(cands),
        "tied": tied_letters(cands),
        "confidence": rule_confidence(cands),
    }
