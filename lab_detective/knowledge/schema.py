"""实验卡、知识点、原因—知识点映射的 JSON schema 和校验函数。

字段约定见 docs/ARCHITECTURE.md §2.2。校验函数只返回问题清单（中文字符串），不抛异常，
方便 validate_all 一次汇总所有作者的卡片问题。
"""

from __future__ import annotations

from typing import Any

import jsonschema

# 12 个实验：id -> (是否留出, 迁移距离)
EXPERIMENT_SPLIT: dict[str, tuple[bool, str | None]] = {
    "hooke": (False, None),
    "friction": (False, None),
    "average_speed": (False, None),
    "newton2": (False, None),
    "momentum": (False, None),
    "energy": (False, None),
    "incline_friction": (True, "near"),
    "free_fall": (True, "near"),
    "atwood": (True, "near"),
    "pendulum": (True, "far"),
    "projectile": (True, "far"),
    "lever": (True, "far"),
}
EXPERIMENT_IDS: tuple[str, ...] = tuple(EXPERIMENT_SPLIT)

# 原因机制标签（与 labsim.core.MECHANISMS 一致；这里抄一份，避免加载数据时就导入 labsim）
MECHANISM_TAGS: tuple[str, ...] = (
    "none", "x_offset", "y_offset", "x_scale", "y_scale", "bend", "scatter", "drift", "aux_only",
)

KP_ID = r"^kp\.[a-z0-9_]+$"
MC_ID = r"^mc\.[a-z0-9_]+$"
CAUSE_ID = r"^[a-z0-9_]+\.[a-z0-9_]+$"

_STR = {"type": "string", "minLength": 1}
_STR_LIST = {"type": "array", "items": _STR}
_KP_LIST = {"type": "array", "items": {"type": "string", "pattern": KP_ID}, "uniqueItems": True}

# 实验卡。必填字段 = ARCHITECTURE §2.2；允许额外字段（并行作者可能加备注）。
# yolo_cues 写 data/dataset.yaml 里的 YOLO 类名（cart, track, spring, string, ruler,
# dynamometer, wooden_block, iron_block），这里不做枚举限制，由识别模块自己取交集。
# 可选 cause_kps：{cause_id: [kp_id]}，把某个误差原因额外关联到知识点。
CARD_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "id", "name_zh", "name_en", "grade", "goal", "principle", "formula", "variables",
        "apparatus", "steps", "data_table", "safety", "common_causes", "knowledge_points",
        "relation_family", "labsim_id", "yolo_cues",
    ],
    "properties": {
        "id": {"type": "string", "pattern": r"^[a-z0-9_]+$"},
        "name_zh": _STR,
        "name_en": _STR,
        "grade": {"enum": ["初中", "高中"]},
        "goal": _STR,
        "principle": _STR,
        "formula": {"type": "string"},
        "variables": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["symbol", "name", "unit", "role"],
                "properties": {
                    "symbol": _STR,
                    "name": _STR,
                    "unit": {"type": "string"},
                    "role": {"enum": ["control", "measured", "derived"]},
                },
            },
        },
        "apparatus": {**_STR_LIST, "minItems": 1},
        "steps": {**_STR_LIST, "minItems": 1},
        "data_table": {
            "type": "object",
            "required": ["columns"],
            "properties": {"columns": {**_STR_LIST, "minItems": 1}},
        },
        "safety": _STR_LIST,
        "common_causes": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "pattern": CAUSE_ID},
            "uniqueItems": True,
        },
        "knowledge_points": {**_KP_LIST, "minItems": 1},
        "relation_family": {"type": "string", "pattern": r"^[a-z0-9_]+$"},
        "labsim_id": {"type": "string", "pattern": r"^[a-z0-9_]+$"},
        "yolo_cues": _STR_LIST,
        "cause_kps": {
            "type": "object",
            "propertyNames": {"pattern": CAUSE_ID},
            "additionalProperties": _KP_LIST,
        },
    },
}

KP_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["id", "name_zh", "statement", "formula", "prerequisites", "misconceptions", "experiments"],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "pattern": KP_ID},
        "aliases": _KP_LIST,  # 可选：别名（其他作者可能用的近义 id），加载器会解析到本 id
        "name_zh": _STR,
        "statement": _STR,
        "formula": {"type": "string"},
        "prerequisites": _KP_LIST,
        "misconceptions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "text", "symptom"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "pattern": MC_ID},
                    "text": _STR,
                    "symptom": _STR,
                },
            },
        },
        "experiments": {
            "type": "array",
            "items": {"enum": list(EXPERIMENT_IDS)},
            "uniqueItems": True,
        },
    },
}

KP_LIST_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "items": KP_SCHEMA,
}

# cause_kp_map.json：mechanism 段按机制标签映射到测量科学知识点；cause 段给个别原因补充专属知识点
CAUSE_MAP_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["mechanism", "cause"],
    "additionalProperties": False,
    "properties": {
        "mechanism": {
            "type": "object",
            "propertyNames": {"enum": list(MECHANISM_TAGS)},
            "additionalProperties": _KP_LIST,
        },
        "cause": {
            "type": "object",
            "propertyNames": {"pattern": CAUSE_ID},
            "additionalProperties": _KP_LIST,
        },
    },
}


def _errors(schema: dict, obj: Any, where: str) -> list[str]:
    v = jsonschema.Draft7Validator(schema)
    out = []
    for e in sorted(v.iter_errors(obj), key=lambda e: list(e.absolute_path)):
        path = "/".join(str(p) for p in e.absolute_path) or "(根)"
        out.append(f"{where}: {path}: {e.message}")
    return out


def validate_card(card: Any, where: str = "card") -> list[str]:
    """实验卡的结构校验（不查知识点是否存在、原因是否在注册表里，那些在 validate_all 里查）。"""
    return _errors(CARD_SCHEMA, card, where)


def validate_kp(kp: Any, where: str = "kp") -> list[str]:
    return _errors(KP_SCHEMA, kp, where)


def validate_kp_list(kps: Any, where: str = "knowledge_points.json") -> list[str]:
    return _errors(KP_LIST_SCHEMA, kps, where)


def validate_cause_map(m: Any, where: str = "cause_kp_map.json") -> list[str]:
    return _errors(CAUSE_MAP_SCHEMA, m, where)
