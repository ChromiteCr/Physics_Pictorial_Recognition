"""
GLM（智谱AI）多模态 API 封装。

两个用途：
  1. 意图识别兜底：intent_classifier.py 的规则表匹配不确定时，把照片+器材列表
     发给 GLM，问"这是初高中力学实验里的哪一种"
  2. 讲解生成：Part 2 算完物理量后，把 实验类型+用户输入+计算结果(+原图) 发给
     GLM，生成文字讲解 + 涉及公式 + 知识点清单

选型理由（2026-07-11 确认）：用户不在中国大陆，选 GLM 是出于 API 调用价格，
不是网络可达性。

依赖：zhipuai（官方 SDK），`pip install zhipuai`
API Key：环境变量 GLM_API_KEY 或 ZHIPUAI_API_KEY（不硬编码在代码里）。
没配置 Key 时，本模块的函数会抛出 GLMAPIError，调用方（app.py）负责捕获并在
网页上给出清晰提示，不应该让整个页面崩溃。
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional

VISION_MODEL = "glm-4v-plus"
TEXT_MODEL = "glm-4-plus"


class GLMAPIError(RuntimeError):
    """GLM API 调用失败（缺 Key / 网络错误 / 返回格式不对）时统一抛这个，
    方便 app.py 用一个 except 分支处理，给用户看得懂的提示而不是堆栈。"""


def _get_api_key() -> str:
    key = os.environ.get("GLM_API_KEY") or os.environ.get("ZHIPUAI_API_KEY")
    if not key:
        raise GLMAPIError(
            "没有配置 GLM API Key。请设置环境变量 GLM_API_KEY（或 ZHIPUAI_API_KEY）"
            "后再试——出于安全考虑，Key 不写在代码里。"
        )
    return key


def _get_client():
    try:
        from zhipuai import ZhipuAI
    except ImportError as e:
        raise GLMAPIError(
            "缺少 zhipuai 库，请先 `pip install zhipuai`（见 requirements.txt）。"
        ) from e
    return ZhipuAI(api_key=_get_api_key())


def _encode_image(image_path: str | Path) -> str:
    """图片转 base64（GLM-4V 的 image_url 支持直接传 base64 字符串）。"""
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _chat(prompt: str, image_path: Optional[str | Path] = None) -> str:
    """底层调用：有图用 glm-4v-plus，纯文本用 glm-4-plus。返回原始文本回复。"""
    client = _get_client()
    model = VISION_MODEL if image_path else TEXT_MODEL

    if image_path:
        content = [
            {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as e:  # zhipuai 的具体异常类型不确定，统一收窄成 GLMAPIError
        raise GLMAPIError(f"GLM API 调用失败: {e}") from e

    return response.choices[0].message.content


def _parse_json_response(raw: str) -> dict:
    """
    GLM 有时会在 JSON 前后加 ```json ... ``` 或额外说明文字，这里做宽松解析：
    先找第一个 '{' 到最后一个 '}' 之间的内容再 json.loads。
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise GLMAPIError(f"GLM 返回内容无法解析成 JSON: {raw[:200]}")
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        raise GLMAPIError(f"GLM 返回的 JSON 格式有误: {e}\n原始内容: {raw[:200]}") from e


# ── 用途 1：意图识别兜底 ─────────────────────────────────────────
def classify_intent_via_api(
    candidate_names: list[str],
    detected_components: list[str],
    image_path: Optional[str | Path] = None,
) -> dict:
    """
    规则表匹配不确定时的兜底：把候选实验类型 + 检测到的器材（+ 可选原图）
    发给 GLM，问该选哪一个，或者是否是候选之外的其他初高中力学实验。

    Returns:
        {"experiment_name": str, "reason": str, "is_known_candidate": bool}
    """
    candidates_str = "、".join(candidate_names) if candidate_names else "（无明确候选）"
    components_str = "、".join(detected_components) if detected_components else "（未检测到器材）"

    prompt = f"""你是一名物理教师。有一张物理实验照片，图中检测到的器材是：{components_str}。

候选的实验类型（按器材组合规则初步匹配到的）：{candidates_str}

请判断这最可能是初中或高中力学课程里的哪一个探究/验证实验。如果候选列表里有合适的，
优先选候选里的；如果都不合适，可以给出你认为更准确的实验名称（仍然只限初高中力学
范畴，不要给出候选之外的力学以外学科实验）。

严格按下面的 JSON 格式回复，不要有多余文字：
{{
  "experiment_name": "实验名称",
  "reason": "一句话说明判断依据",
  "is_known_candidate": true 或 false（是否是给定候选之一）
}}"""

    raw = _chat(prompt, image_path=image_path)
    return _parse_json_response(raw)


# ── 用途 2：讲解生成 ─────────────────────────────────────────────
def generate_explanation(
    experiment_name: str,
    level: str,
    user_inputs: dict,
    computed_result: dict,
    image_path: Optional[str | Path] = None,
) -> dict:
    """
    生成实验讲解文字 + 涉及公式 + 知识点清单。

    Args:
        experiment_name: 实验类型名称（如"探究摩擦力大小的影响因素"）
        level: "初中" / "高中"
        user_inputs: 用户填写的实测数据
        computed_result: physics.compute_experiment() 的计算结果
        image_path: 可选，初状态照片，让 GLM 结合实际画面讲解更具体

    Returns:
        {"explanation": str, "formulas": list[str], "knowledge_points": list[str]}
    """
    prompt = f"""你是一名{level}物理教师，正在给学生讲解一个刚完成的实验。

实验类型：{experiment_name}
学生实测输入：{json.dumps(user_inputs, ensure_ascii=False)}
计算结果：{json.dumps(computed_result, ensure_ascii=False)}

请据此生成：
1. 一段面向{level}学生的实验讲解（说清楚做了什么、结果说明了什么，语言通俗，
   200-400字）
2. 涉及的物理公式列表（LaTeX 或纯文本均可，只列这个实验真正用到的公式）
3. 一份知识点清单（3-6条，对应本地教材大纲里这个实验考察的知识点）

严格按下面的 JSON 格式回复，不要有多余文字：
{{
  "explanation": "讲解文字",
  "formulas": ["公式1", "公式2"],
  "knowledge_points": ["知识点1", "知识点2", "知识点3"]
}}"""

    raw = _chat(prompt, image_path=image_path)
    return _parse_json_response(raw)


# ── 简单测试（不实际调用 API，只验证没配置 Key 时报错清晰） ─────────
if __name__ == "__main__":
    try:
        classify_intent_via_api(["friction"], ["cart", "track", "dynamometer"])
    except GLMAPIError as e:
        print(f"预期内的报错（没配置 API Key）: {e}")
