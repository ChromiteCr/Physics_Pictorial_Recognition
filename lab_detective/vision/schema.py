"""识别结果 RecognitionResult（ARCHITECTURE §2.3）+ JSON schema + 容错解析。

模型只输出 MODEL_OUTPUT_SCHEMA 里的 5 个字段（letter 放第一个，方便读 logprobs）；
experiment_id 由字母推出（字母说了算），needs_question / source / candidates / question
由流水线填。所有模型（本地、API）共用这一份 schema。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import jsonschema

from .letters import EXPERIMENT_IDS, LETTERS, OTHER_LETTER, entry, experiment_for, letter_for, normalize_letter

# 器材封闭词表：id → 中文名（YOLO 的 8 类在前；F1 评分按 id 算）
APPARATUS: dict[str, str] = {
    "cart": "小车",
    "track": "轨道/长木板",
    "spring": "弹簧",
    "string": "细线",
    "ruler": "刻度尺",
    "dynamometer": "弹簧测力计",
    "wooden_block": "木块",
    "iron_block": "钢球/铁块",
    "slotted_mass": "钩码",
    "pulley": "滑轮",
    "photogate": "光电门",
    "ticker_timer": "打点计时器",
    "paper_tape": "纸带",
    "stopwatch": "秒表",
    "iron_stand": "铁架台",
    "incline": "斜面",
    "ball": "小球",
    "chute": "斜槽",
    "lever": "杠杆",
    "balance": "天平",
    "heavy_weight": "重锤/重物",
    "metal_stop": "金属片/挡板",
    "protractor": "量角器",
    "caliper": "游标卡尺",
    "coordinate_board": "坐标纸/白板",
}

# 容错：中文名、常见别称 → id
_APPARATUS_SYNONYMS: dict[str, str] = {
    **{k: k for k in APPARATUS},
    **{v: k for k, v in APPARATUS.items()},
    "长木板": "track", "轨道": "track", "导轨": "track", "气垫导轨": "track", "木板": "track",
    "测力计": "dynamometer", "弹簧秤": "dynamometer", "钢球": "iron_block", "铁块": "iron_block",
    "砝码": "slotted_mass", "定滑轮": "pulley", "滑块": "cart", "绳": "string", "细绳": "string",
    "直尺": "ruler", "米尺": "ruler", "重锤": "heavy_weight", "重物": "heavy_weight",
    "摆球": "ball", "挡板": "metal_stop", "金属片": "metal_stop", "坐标纸": "coordinate_board",
    "spring_scale": "dynamometer", "mass": "slotted_mass", "weights": "slotted_mass",
}

MAX_CUES = 6
MAX_APPARATUS = 12

# 模型输出的 schema（受约束解码 / API 的 response_format 都用它）
MODEL_OUTPUT_SCHEMA: dict[str, Any] = {
    "title": "recognition",
    "type": "object",
    "properties": {
        "letter": {"type": "string", "enum": list(LETTERS)},
        "open_name": {"type": "string", "minLength": 1, "maxLength": 40},
        "cues": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": MAX_CUES},
        "apparatus": {"type": "array", "items": {"type": "string", "enum": list(APPARATUS)},
                      "maxItems": MAX_APPARATUS},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["letter", "open_name", "cues", "apparatus", "confidence"],
    "additionalProperties": False,
}

# 完整识别结果（流水线的输出）
RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "letter": {"type": "string", "enum": list(LETTERS)},
        "experiment_id": {"enum": [*EXPERIMENT_IDS, None]},
        "open_name": {"type": "string"},
        "cues": {"type": "array", "items": {"type": "string"}},
        "apparatus": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "needs_question": {"type": "boolean"},
        "source": {"enum": ["rule", "model", "manual", "question", "rule_fallback", "none"]},
        "candidates": {"type": "array", "items": {
            "type": "object",
            "properties": {"letter": {"enum": list(LETTERS)}, "score": {"type": "number"}},
            "required": ["letter", "score"],
        }},
        "question": {"type": ["object", "null"]},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["letter", "experiment_id", "open_name", "cues", "apparatus", "confidence", "needs_question"],
}


class RecognitionParseError(ValueError):
    """模型输出里找不到可用的识别 JSON。"""


@dataclass
class RecognitionResult:
    letter: str
    experiment_id: str | None
    open_name: str
    cues: list[str] = field(default_factory=list)
    apparatus: list[str] = field(default_factory=list)
    confidence: float = 0.0
    needs_question: bool = False
    # 以下由流水线填，不要求模型输出
    source: str = "none"
    candidates: list[dict] = field(default_factory=list)  # [{letter, experiment_id, name_zh, score}]
    question: dict | None = None  # 澄清问题 {id, text, options[{label, letters}], gain_bits}
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.letter not in LETTERS:
            raise ValueError(f"未知字母: {self.letter!r}")
        self.experiment_id = experiment_for(self.letter)  # 字母说了算

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def model_part(self) -> dict:
        """只含模型输出字段（训练样本的 assistant 内容用这个）。"""
        return {"letter": self.letter, "open_name": self.open_name, "cues": list(self.cues),
                "apparatus": list(self.apparatus), "confidence": round(float(self.confidence), 3)}

    @classmethod
    def from_dict(cls, d: dict) -> "RecognitionResult":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: d[k] for k in keys if k in d})


def validate(obj: Any, schema: dict = RESULT_SCHEMA) -> list[str]:
    """按 schema 校验，返回错误信息列表（空 = 合法）。"""
    v = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in v.iter_errors(obj)]


_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    """从模型文本里取出第一个 JSON 对象（容忍代码块围栏、前后多余文字、思考段落）。"""
    if not isinstance(text, str):
        raise RecognitionParseError("模型输出不是字符串")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    chunks = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    dec = json.JSONDecoder()
    for chunk in chunks:
        for i, ch in enumerate(chunk):
            if ch != "{":
                continue
            try:
                obj, _ = dec.raw_decode(chunk[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise RecognitionParseError("模型输出里没有 JSON 对象")


def _str_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = re.split(r"[，,；;、\n]+", raw)
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for x in raw:
        s = str(x).strip()
        if s and s not in out:
            out.append(s)
    return out


def normalize_apparatus(raw: Any) -> tuple[list[str], list[str]]:
    """器材名 → 词表 id（去重、保持顺序）；返回 (ids, 认不出的名字)。"""
    ids, unknown = [], []
    for s in _str_list(raw):
        k = _APPARATUS_SYNONYMS.get(s) or _APPARATUS_SYNONYMS.get(s.lower())
        if k is None:
            unknown.append(s)
        elif k not in ids:
            ids.append(k)
    return ids[:MAX_APPARATUS], unknown


def _confidence(raw: Any, warnings: list[str]) -> float:
    try:
        c = float(str(raw).strip().rstrip("%"))
    except (TypeError, ValueError):
        warnings.append("模型没给出可读的 confidence，按 0.5 处理")
        return 0.5
    if 1.0 < c <= 100.0:  # 写成了百分数
        c /= 100.0
    return min(1.0, max(0.0, c))


def parse_model_output(text_or_obj: str | dict) -> RecognitionResult:
    """容错解析模型输出 → RecognitionResult（source='model'）。

    字母认不出时用 experiment_id 反推；两者都没有就抛 RecognitionParseError。
    """
    obj = text_or_obj if isinstance(text_or_obj, dict) else extract_json(text_or_obj)
    warnings: list[str] = []
    letter = normalize_letter(obj.get("letter"))
    exp_id = obj.get("experiment_id")
    if letter is None:
        if isinstance(exp_id, str) and exp_id in EXPERIMENT_IDS:
            letter = letter_for(exp_id)
            warnings.append("字母缺失，按 experiment_id 反推")
        else:
            raise RecognitionParseError(f"认不出字母: {obj.get('letter')!r}")
    elif isinstance(exp_id, str) and exp_id and exp_id != experiment_for(letter):
        warnings.append(f"experiment_id={exp_id} 与字母 {letter} 不一致，以字母为准")

    open_name = str(obj.get("open_name") or "").strip()[:40]
    if not open_name:
        open_name = entry(letter).short_zh if letter != OTHER_LETTER else "未命名实验"
    apparatus, unknown = normalize_apparatus(obj.get("apparatus"))
    if unknown:
        warnings.append("词表外器材已忽略：" + "、".join(unknown))
    return RecognitionResult(
        letter=letter,
        experiment_id=experiment_for(letter),
        open_name=open_name,
        cues=_str_list(obj.get("cues"))[:MAX_CUES],
        apparatus=apparatus,
        confidence=_confidence(obj.get("confidence"), warnings),
        needs_question=bool(obj.get("needs_question", False)),
        source="model",
        warnings=warnings,
    )
