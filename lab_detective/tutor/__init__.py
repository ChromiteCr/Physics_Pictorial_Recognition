"""探究式引导：会话状态机、上下文组装、模板讲解、声明校验器、脚本化导师。

本包不在导入时加载 labsim 注册表或知识点模块（都是用到时才懒加载），并行开发时不会被别的模块拖垮。
"""
from .context import SYSTEM_PROMPT_ZH, build_context  # noqa: F401
from .policy import ScriptedTutor, guarded_explanation  # noqa: F401
from .session import PHASES, TRANSITIONS, IllegalTransition, LabSession  # noqa: F401
from .templates import explanation_blocks, explanation_text  # noqa: F401
from .verifier import verify  # noqa: F401
