"""看图建模：视觉识别结果（器材清单 + 构型线索，见 vision/schema.py）→ 排好序的候选装置 + 学生还要回答的问题。

做法（不调用任何模型，纯规则）：
1. 目录匹配：每个目录条目有必需器材组、可选器材、线索词和常见程度。
   分数 = 2 × 必需组满足比例 + 0.3 × 命中的可选器材 − 0.25 × 解释不了的器材 + 0.4 × 命中的线索词（最多 2 个）
         + 常见程度 + 1.5（识别结果的 experiment_id 正好是这个镜像的精选实验）。
   必需组一个都不满足的条目不列出。
2. 组合规则：由器材推构造选项——有光电门就用光电门、有打点计时器/纸带就用打点计时器、
   有秒表就按手按秒表计时；照片里有斜面或线索里有"斜"字时，斜面类装置加分、水平轨道的牛顿第二定律减分。
3. 每个候选给出未决问题：装置里各组件需要的已知量（质量、倾角、直径……），按组件顺序去重。

精选实验只是起点：目录里的新组合同样参与匹配；都匹配不上时返回空列表，交给"其他实验"通道（labsim/generic.py）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .base import OpenQuestion, Setup
from .catalog import CATALOG, CatalogEntry
from .compose import ComposedExperiment

# 这些器材几乎每个实验都可能出现在画面里，不算"解释不了"
NEUTRAL = frozenset({"ruler", "iron_stand", "balance", "caliper", "coordinate_board"})
INCLINE_WORDS = ("斜面", "倾斜", "倾角", "斜坡", "垫高")


@dataclass
class PhotoCandidate:
    entry_id: str
    name_zh: str
    score: float
    mirror_of: str | None
    held_out: bool
    options: dict
    reasons: list[str] = field(default_factory=list)
    open_questions: list[OpenQuestion] = field(default_factory=list)
    setup: Setup | None = None

    def experiment(self) -> ComposedExperiment:
        return ComposedExperiment(self.setup)

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id, "name_zh": self.name_zh, "score": round(self.score, 3),
            "mirror_of": self.mirror_of, "held_out": self.held_out, "options": dict(self.options),
            "reasons": list(self.reasons), "open_questions": [asdict(q) for q in self.open_questions],
        }


def _fields(rec: Any) -> tuple[list[str], list[str], str | None]:
    """RecognitionResult 或 dict → (器材 id, 线索, experiment_id)。器材名先按视觉词表归一化。"""
    if isinstance(rec, Mapping):
        app, cues, eid = rec.get("apparatus") or [], rec.get("cues") or [], rec.get("experiment_id")
    else:
        app, cues, eid = getattr(rec, "apparatus", []), getattr(rec, "cues", []), getattr(rec, "experiment_id", None)
    from ..vision.schema import normalize_apparatus  # 懒加载：只用词表，不在导入本模块时加载视觉流水线

    ids, _ = normalize_apparatus(list(app))
    return ids, [str(c) for c in cues], eid if isinstance(eid, str) else None


def _score(entry: CatalogEntry, app: set[str], cues: list[str], eid: str | None) -> tuple[float, list[str]] | None:
    hit_groups = [g for g in entry.requires if g & app]
    if not hit_groups:
        return None
    frac = len(hit_groups) / len(entry.requires)
    reasons = [f"必需器材满足 {len(hit_groups)}/{len(entry.requires)} 组"]
    score = 2.0 * frac + entry.prior
    explained = set().union(*entry.requires) | entry.optional | NEUTRAL
    opt = sorted(entry.optional & app)
    if opt:
        score += 0.3 * len(opt)
        reasons.append("匹配可选器材：" + "、".join(opt))
    extra = sorted(app - explained)
    if extra:
        score -= 0.25 * len(extra)
        reasons.append("这些器材在该装置里用不上：" + "、".join(extra))
    text = "；".join(cues)
    words = [w for w in entry.cue_words if w in text][:2]
    if words:
        score += 0.4 * len(words)
        reasons.append("构型线索：" + "、".join(words))
    inclined = "incline" in app or any(w in text for w in INCLINE_WORDS)
    if inclined:
        if entry.id in ("incline_newton2", "incline_slide", "energy_composed"):
            score += 0.5
            reasons.append("画面里有斜面")
        elif entry.id == "newton2_composed":
            score -= 0.3
            reasons.append("画面里有明显的斜面，不像阻力补偿法的水平导轨")
    if eid is not None and entry.mirror_of == eid:
        score += 1.5
        reasons.append(f"识别结果就是这个实验（{eid}）")
    return score, reasons


def _dedup(questions: list[OpenQuestion]) -> list[OpenQuestion]:
    seen, out = set(), []
    for q in questions:
        if q.key not in seen:
            seen.add(q.key)
            out.append(q)
    return out


def candidates_from_photo(rec: Any, top_k: int = 5) -> list[PhotoCandidate]:
    """识别结果 → 候选装置（分数从高到低）。rec 可以是 RecognitionResult 或同字段的 dict。"""
    ids, cues, eid = _fields(rec)
    app = set(ids)
    out: list[PhotoCandidate] = []
    for entry in CATALOG.values():
        res = _score(entry, app, cues, eid)
        if res is None:
            continue
        score, reasons = res
        opts = entry.options(app, cues)
        setup = entry.setup(**opts)
        if opts:
            reasons.append("按照片里的计时器选用：" + "、".join(f"{k}={v}" for k, v in opts.items()))
        out.append(PhotoCandidate(entry.id, entry.name_zh, score, entry.mirror_of, entry.held_out, opts,
                                  reasons, _dedup(setup.questions()), setup))
    out.sort(key=lambda c: (-c.score, c.entry_id))
    return out[:top_k]
