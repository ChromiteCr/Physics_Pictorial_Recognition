"""
实验意图识别：从 Part 1 确认后的器材列表判断"这是什么实验"。

策略（2026-07-11 确认）：器材组合 + 相对位置规则优先判断；规则拿不准（候选为空，
或多个候选分数太接近分不出高下）时调 GLM API 兜底；无论哪种方式，最终结果都只是
"建议"，网页上必须始终允许用户在下拉框里手动改成 physics.EXPERIMENT_TYPES 里的
任意实验类型——这里的分数不是为了追求自动化正确率，是为了减少用户手动选择的次数。

入口：
    classify_intent_with_fallback(detections, image_path=None) -> dict
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

from physics import EXPERIMENT_TYPES
from explain_api import classify_intent_via_api, GLMAPIError

# 规则表打分给到的分数差距，大于这个阈值才认为规则表"分得出高下"，
# 不用再麻烦 GLM API。分数差距小说明规则表本身也很含糊，兜底更靠谱。
CONFIDENT_SCORE_GAP = 0.3


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _bbox_size(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(abs(x2 - x1), abs(y2 - y1))


def _proximity_bonus(
    detections: list[dict], class_a: str, class_b: str, max_dist_ratio: float = 2.0
) -> float:
    """
    class_a、class_b 任意一对实例的 bbox 中心距离，如果小于二者平均尺寸的
    max_dist_ratio 倍，认为"位置上挨得近"，给一点加分（相对位置规则）。
    没有 bbox 信息（比如 Part 1 网页上被人工加的行没填坐标）时直接跳过，不报错。
    """
    a_boxes = [d["bbox"] for d in detections if d["class_name"] == class_a and d.get("bbox")]
    b_boxes = [d["bbox"] for d in detections if d["class_name"] == class_b and d.get("bbox")]
    if not a_boxes or not b_boxes:
        return 0.0

    best = None
    for ba in a_boxes:
        for bb in b_boxes:
            ca, cb = _bbox_center(ba), _bbox_center(bb)
            dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
            avg_size = (_bbox_size(ba) + _bbox_size(bb)) / 2 or 1.0
            ratio = dist / avg_size
            if best is None or ratio < best:
                best = ratio
    if best is not None and best <= max_dist_ratio:
        return 0.2
    return 0.0


def classify_intent(detections: list[dict]) -> list[dict]:
    """
    按器材组合(+相对位置)规则给每个满足"必需器材"的实验类型打分，降序返回。

    Args:
        detections: [{"class_name": str, "bbox": [x1,y1,x2,y2] (可选)}, ...]
                    通常是 Part 1 网页上用户确认/修正后的器材列表
    Returns:
        [{"experiment_id", "name", "level", "score", "reason"}, ...]，按 score 降序
    """
    name_counts = Counter(d["class_name"] for d in detections)
    name_set = set(name_counts)

    candidates = []
    for exp_id, exp in EXPERIMENT_TYPES.items():
        required = exp["required_components"]
        if not required.issubset(name_set):
            continue

        # 动量守恒需要两辆小车，只有 cart 类别名字不够，得看数量
        if exp_id == "momentum_conservation" and name_counts.get("cart", 0) < 2:
            continue

        score = 1.0
        reasons = [f"检测到必需器材：{', '.join(sorted(required))}"]

        # 两辆车是动量守恒实验的强特征信号，明显优先于只需单车的实验类型
        # （单车+导轨规则本身分不出 average_speed/newton_second_law/energy_conservation，
        # 但只要出现第二辆车，几乎不可能是那几个单车实验）
        if exp_id == "momentum_conservation":
            score += 0.5
            reasons.append("检测到 2 辆小车，是动量守恒实验的强特征")

        matched_optional = exp["optional_components"] & name_set
        if matched_optional:
            score += 0.2 * len(matched_optional)
            reasons.append(f"额外匹配可选器材：{', '.join(sorted(matched_optional))}")

        if exp_id == "friction":
            bonus = _proximity_bonus(detections, "dynamometer", "cart")
            if bonus:
                score += bonus
                reasons.append("测力计与小车位置接近，符合摩擦力实验典型摆法")

        candidates.append({
            "experiment_id": exp_id,
            "name": exp["name"],
            "level": exp["level"],
            "score": round(score, 3),
            "reason": "；".join(reasons),
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def classify_intent_with_fallback(
    detections: list[dict], image_path: Optional[str | Path] = None
) -> dict:
    """
    Part 2 的意图识别入口，规则优先、API 兜底。

    Returns:
        {
          "picked": {...} | None,      # 最终建议的实验类型（网页默认选中项）
          "candidates": [...],          # 规则表算出的全部候选，供下拉框参考
          "source": "rule" | "api" | "rule_fallback" | "none",
          "warning": str (可选，异常情况下的提示文字),
        }
    """
    candidates = classify_intent(detections)

    if candidates:
        if len(candidates) == 1:
            return {"picked": candidates[0], "candidates": candidates, "source": "rule"}
        gap = candidates[0]["score"] - candidates[1]["score"]
        if gap >= CONFIDENT_SCORE_GAP:
            return {"picked": candidates[0], "candidates": candidates, "source": "rule"}

    # 规则表没有候选，或多个候选分数太接近 → 调 GLM API 兜底
    component_names = sorted({d["class_name"] for d in detections})
    try:
        api_result = classify_intent_via_api(
            candidate_names=[c["name"] for c in candidates],
            detected_components=component_names,
            image_path=image_path,
        )
    except GLMAPIError as e:
        if candidates:
            return {
                "picked": candidates[0], "candidates": candidates, "source": "rule_fallback",
                "warning": f"候选分数接近本该调 GLM API 二次判断，但 API 不可用（{e}），"
                           f"已退而求其次选规则表最高分候选，建议人工核对。",
            }
        return {
            "picked": None, "candidates": [], "source": "none",
            "warning": f"规则表无匹配，GLM API 也不可用（{e}），请在下拉框手动选择实验类型。",
        }

    picked_name = api_result.get("experiment_name")
    matched = next((c for c in candidates if c["name"] == picked_name), None)
    if matched is None:
        # GLM 给了候选之外的名字：构造一个不含 experiment_id 的自由结果，
        # 网页需要处理"没有 experiment_id"的情况——提示用户在下拉框里手动挑一个
        # physics.EXPERIMENT_TYPES 里最接近的类型，因为 compute() 需要 id 才能算。
        matched = {
            "experiment_id": None, "name": picked_name, "level": None,
            "score": None, "reason": api_result.get("reason", "GLM 判断，不在规则候选内"),
        }
    return {
        "picked": matched, "candidates": candidates, "source": "api",
        "api_reason": api_result.get("reason"),
    }


# ── 简单测试（纯规则表，不调 API，不需要网络/照片） ──────────────────
if __name__ == "__main__":
    import json

    # 场景1：cart+track+dynamometer+string，应该唯一匹配摩擦力
    dets1 = [
        {"class_name": "cart", "bbox": [100, 100, 150, 130]},
        {"class_name": "track", "bbox": [0, 90, 400, 140]},
        {"class_name": "dynamometer", "bbox": [160, 95, 220, 125]},
        {"class_name": "string", "bbox": [148, 108, 162, 118]},
    ]
    print("场景1(摩擦力):", json.dumps(classify_intent(dets1), ensure_ascii=False, indent=2))

    # 场景2：两辆 cart + track + string，应该匹配动量守恒（且不该匹配到只需1辆车的类型）
    dets2 = [
        {"class_name": "cart", "bbox": [50, 100, 100, 130]},
        {"class_name": "cart", "bbox": [300, 100, 350, 130]},
        {"class_name": "track", "bbox": [0, 90, 400, 140]},
        {"class_name": "string", "bbox": [120, 108, 140, 118]},
    ]
    print("场景2(动量守恒):", json.dumps(classify_intent(dets2), ensure_ascii=False, indent=2))

    # 场景3：只有 dynamometer，应该只匹配胡克定律
    dets3 = [{"class_name": "dynamometer", "bbox": [100, 100, 150, 200]}]
    print("场景3(胡克定律):", json.dumps(classify_intent(dets3), ensure_ascii=False, indent=2))

    # 场景4：cart+track 单车，会同时匹配 average_speed / newton_second_law /
    # energy_conservation_input 三个（都只要求 cart+track），分数应该打平，
    # 走 classify_intent_with_fallback 应该触发 API 兜底路径（这里没配 Key，
    # 预期抛出 warning 并退回规则表最高分）
    dets4 = [
        {"class_name": "cart", "bbox": [100, 100, 150, 130]},
        {"class_name": "track", "bbox": [0, 90, 400, 140]},
    ]
    print("场景4(单车+导轨，规则表应打平):",
          json.dumps(classify_intent(dets4), ensure_ascii=False, indent=2))
    print("场景4 走 fallback:",
          json.dumps(classify_intent_with_fallback(dets4), ensure_ascii=False, indent=2))
