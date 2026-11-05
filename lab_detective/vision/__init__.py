"""照片 → 构型 → 实验：YOLO 线索 + Set-of-Mark 提示 + 容错解析 + 澄清问题（不在导入时加载 YOLO）。"""

from .letters import LETTERS, OTHER_LETTER, TABLE, experiment_for, letter_for, name_zh
from .pipeline import answer, choose_manual, choose_question, recognize
from .schema import (
    MODEL_OUTPUT_SCHEMA,
    RESULT_SCHEMA,
    RecognitionParseError,
    RecognitionResult,
    parse_model_output,
    validate,
)

__all__ = [
    "LETTERS", "OTHER_LETTER", "TABLE", "experiment_for", "letter_for", "name_zh",
    "answer", "choose_manual", "choose_question", "recognize",
    "MODEL_OUTPUT_SCHEMA", "RESULT_SCHEMA", "RecognitionParseError", "RecognitionResult",
    "parse_model_output", "validate",
]
