"""识别提示词：训练和推理共用同一份（改动必须升 PROMPT_VERSION）。

消息格式与训练样本一致（ARCHITECTURE §2.5）：
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user", "content": [{"type": "image", "image": <图片路径>}, {"type": "text", "text": ...}]}]
训练时图片路径写相对数据包根目录的路径，推理时写临时的 Set-of-Mark 图路径。
to_openai_messages() 把图片换成 base64 data URI，供任意 OpenAI 兼容的视觉模型使用。
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from .letters import table_text
from .schema import APPARATUS, MAX_CUES, RecognitionResult
from .som import legend_text

PROMPT_VERSION = "rec-v1"

SYSTEM_PROMPT = (
    "你是中学力学实验室的助手。你会看到一张实验装置照片，照片上检测到的器材用带编号的彩色框标出。"
    "你的任务只有一个：判断这是哪一个实验，并说出你看到的装置特征。不要做任何物理计算。\n\n"
    "实验字母表（只能从中选一个字母）：\n"
    f"{table_text()}\n\n"
    "器材 id（apparatus 只能用这些 id）：\n"
    + "、".join(f"{k}={v}" for k, v in APPARATUS.items())
    + "\n\n"
    "规则：\n"
    "1. 编号框只是检测器的猜测，可能漏检或认错，以照片本身为准。\n"
    "2. 照片不像表中任何一个实验、或看不清时，选 Z，并在 open_name 里写你认为的实验名称（不知道就写\"未知\"）。\n"
    f"3. cues 写 1 到 {MAX_CUES} 条你在照片里看到的装置构型线索（中文短句），"
    "比如\"弹簧竖直悬挂\"\"细线跨过滑轮挂钩码\"\"轨道倾斜\"\"两车相对\"\"测力计水平拉木块\"。\n"
    "4. confidence 是你对所选字母的把握，0 到 1 之间。\n"
    "5. 只输出一个 JSON 对象，不要输出其他文字，字段顺序固定：\n"
    '{"letter": "A", "open_name": "实验名称", "cues": ["..."], "apparatus": ["..."], "confidence": 0.8}'
)

USER_TEMPLATE = (
    "照片中的编号框：\n{legend}\n\n"
    "这是哪个实验？按要求只输出 JSON。"
)


def user_text(legend: list[dict]) -> str:
    return USER_TEMPLATE.format(legend=legend_text(legend))


def build_messages(image_ref: str | Path, legend: list[dict]) -> list[dict]:
    """识别消息（system + user[图片, 文字]）。image_ref 是图片路径（训练时用相对路径）。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": str(image_ref)},
            {"type": "text", "text": user_text(legend)},
        ]},
    ]


def assistant_target(result: RecognitionResult) -> str:
    """训练样本里 assistant 的内容：只含模型字段的紧凑 JSON（字段顺序与提示一致）。"""
    return json.dumps(result.model_part(), ensure_ascii=False)


def prompt_hash() -> str:
    """提示词指纹：训练包和推理端各记一份，不一致就说明提示词漂移了。"""
    h = hashlib.sha1((PROMPT_VERSION + SYSTEM_PROMPT + USER_TEMPLATE).encode("utf-8"))
    return h.hexdigest()[:12]


def _data_uri(path: str | Path) -> str:
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def to_openai_messages(messages: list[dict], root: str | Path | None = None) -> list[dict]:
    """把 {"type": "image", "image": 路径} 换成 OpenAI 的 image_url（base64 data URI）。

    root：相对路径的根目录（训练包根目录）；绝对路径不受影响。
    """
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "image":
                    p = Path(part["image"])
                    if root is not None and not p.is_absolute():
                        p = Path(root) / p
                    parts.append({"type": "image_url", "image_url": {"url": _data_uri(p)}})
                else:
                    parts.append(part)
            content = parts
        out.append({"role": m["role"], "content": content})
    return out
