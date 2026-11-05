"""模型模式：探测本机的 OpenAI 兼容服务（mlx_vlm.server），可用时给出客户端。不 import streamlit。

产品默认是脚本模式（模板讲解，不需要任何模型）。只有学生在侧栏打开"模型模式"时才探测一次本机地址，
超时 0.5 秒；探测不到就继续用脚本模式。这里只连用户填写的本机地址，从不连外部 API。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_BASE = "http://127.0.0.1:8080/v1"
PROBE_TIMEOUT_S = 0.5
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
LOG_PATH = Path(__file__).resolve().parents[1] / "results" / "logs" / "app_llm.jsonl"


def is_local(base_url: str) -> bool:
    try:
        host = urlparse(base_url).hostname or ""
    except ValueError:
        return False
    return host in LOCAL_HOSTS


def probe(base_url: str, timeout: float = PROBE_TIMEOUT_S, opener=None) -> dict:
    """GET {base}/models → {ok, models, error}。只允许本机地址（产品离线运行，不连外部服务）。"""
    base = (base_url or "").rstrip("/")
    if not is_local(base):
        return {"ok": False, "models": [], "error": "只支持本机地址（127.0.0.1 / localhost）"}
    open_ = opener or urllib.request.urlopen
    try:
        with open_(base + "/models", timeout=timeout) as r:  # noqa: S310  只连本机
            data = json.loads(r.read().decode("utf-8", errors="replace") or "{}")
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
        return {"ok": False, "models": [], "error": f"{type(e).__name__}: {e}"}
    models = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict) and m.get("id")]
    return {"ok": True, "models": models, "error": ""}


def make_client(base_url: str, model: str, timeout: float = 90.0):
    """本地模型客户端（关思考、不重试太多，请求与回复写进 JSONL 日志）。"""
    from lab_detective.llm import OpenAICompatClient

    return OpenAICompatClient(base_url, model or "local", provider="mlx", timeout=timeout, max_retries=1,
                              log_path=LOG_PATH)
