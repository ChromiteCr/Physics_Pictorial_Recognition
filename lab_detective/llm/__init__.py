"""模型客户端：mlx_vlm.server / OpenAI 兼容 API（只用 urllib）和离线脚本化兜底。"""
from .client import (  # noqa: F401
    PROVIDERS,
    ChatModel,
    HTTPStatusError,
    LLMError,
    OpenAICompatClient,
    ReasoningLeakError,
    ScriptedModel,
    image_data_url,
    no_reasoning_check,
    parse_json,
    prepare_messages,
    schema_instruction,
)
