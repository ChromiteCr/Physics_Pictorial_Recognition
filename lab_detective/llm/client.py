"""模型客户端：OpenAI 兼容接口（mlx_vlm.server / DeepSeek / OpenAI），只用标准库 urllib，不用 SDK。

- 图片转成 base64 data URL；训练格式里的 {"type": "image", "image": 路径} 也会被转换，训练样本能原样重放；
- JSON 输出按服务商区分：mlx / openai 用 response_format=json_schema；deepseek 用 json_object，并把 schema 写进提示；
- 一律关闭思考：mlx（Qwen）传 chat_template_kwargs.enable_thinking=false，deepseek 传 thinking.type=disabled；
  回复里出现 <think> 或非空 reasoning_content 就报错（no_reasoning_check）；
- 429 / 5xx / 网络错误 / 要 JSON 却回空内容时退避重试，其他 4xx 不重试；
- 每次请求和回复（含失败）都追加写进 JSONL 日志：时间、模型、参数、完整提示（图片只记哈希和字节数）、回复。
transport 可注入，测试里不联网。
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, Sequence

PROVIDERS = ("mlx", "deepseek", "openai")

# transport(url, headers, body, timeout) -> (HTTP 状态码, 回复字节)
Transport = Callable[[str, dict, bytes, float], tuple[int, bytes]]


class ChatModel(Protocol):
    def chat(self, messages: list[dict], json_schema: dict | None = None, images: Sequence | None = None,
             max_tokens: int = 1024, temperature: float = 0.0) -> dict:
        """返回 {text, raw, usage, latency_s}。"""
        ...


class LLMError(RuntimeError):
    pass


class HTTPStatusError(LLMError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status


class ReasoningLeakError(LLMError):
    """思考没关掉：回复里带了推理文本。"""


def urllib_transport(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310  只连用户配置的地址
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# ── 图片与消息 ──────────────────────────────────────────────────────────


def _sniff(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


def image_data_url(img, root: str | Path | None = None) -> str:
    """路径 / 字节 / 已有的 data URL → data URL。"""
    if isinstance(img, str) and img.startswith(("data:", "http://", "https://")):
        return img
    if isinstance(img, (bytes, bytearray)):
        data = bytes(img)
        mime = _sniff(data)
    else:
        path = Path(img)
        if root is not None and not path.is_absolute():
            path = Path(root) / path
        data = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or _sniff(data)
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def prepare_messages(messages: list[dict], images: Sequence | None = None,
                     image_root: str | Path | None = None) -> list[dict]:
    """转成 OpenAI 格式：训练格式的 image 片段转 image_url；images 加到最后一条 user 消息的文字前面。"""
    out = copy.deepcopy(list(messages))
    for m in out:
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                if p.get("type") == "image":
                    parts.append({"type": "image_url", "image_url": {"url": image_data_url(p["image"], image_root)}})
                else:
                    parts.append(p)
            m["content"] = parts
    if images:
        idx = max((i for i, m in enumerate(out) if m.get("role") == "user"), default=None)
        if idx is None:
            out.append({"role": "user", "content": ""})
            idx = len(out) - 1
        c = out[idx]["content"]
        parts = c if isinstance(c, list) else ([{"type": "text", "text": c}] if c else [])
        imgs = [{"type": "image_url", "image_url": {"url": image_data_url(i, image_root)}} for i in images]
        out[idx]["content"] = imgs + parts
    return out


def _mentions_json(messages: list[dict]) -> bool:
    for m in messages:
        c = m.get("content")
        texts = [c] if isinstance(c, str) else [p.get("text", "") for p in c or [] if isinstance(p, dict)]
        if any("json" in (t or "").lower() for t in texts):
            return True
    return False


def schema_instruction(schema: dict) -> str:
    """DeepSeek 的 JSON 模式要求提示里出现 json 并给出格式。评测时可以把这句放进所有对比组的共享提示。"""
    return "请只输出一个 JSON 对象（json），格式必须符合下面的 JSON schema：\n" + json.dumps(
        schema, ensure_ascii=False, separators=(",", ":"))


def _append_text(msg: dict, text: str) -> None:
    c = msg.get("content")
    if isinstance(c, list):
        msg["content"] = c + [{"type": "text", "text": text}]
    else:
        msg["content"] = (c + "\n\n" if c else "") + text


_DATA_URL = re.compile(r"data:([\w/+.-]+);base64,([A-Za-z0-9+/=]+)")


def _redact(obj):
    """日志里把 base64 图片换成哈希和字节数。"""
    if isinstance(obj, str):
        def rep(m):
            raw = base64.b64decode(m.group(2))
            return f"data:{m.group(1)};sha256={hashlib.sha256(raw).hexdigest()};bytes={len(raw)}"
        return _DATA_URL.sub(rep, obj)
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _redact(v) for k, v in obj.items()}
    return obj


def no_reasoning_check(text: str, message: dict | None = None) -> None:
    """回复里有 <think> 标签或非空的 reasoning_content / reasoning 字段就报错。"""
    msg = message or {}
    for k in ("reasoning_content", "reasoning"):
        r = msg.get(k)
        if isinstance(r, str) and r.strip():
            raise ReasoningLeakError(f"回复带有 {k}（思考没有关闭）")
    if re.search(r"</?think>", text or "", re.I):
        raise ReasoningLeakError("回复里有 <think> 标签（思考没有关闭）")


def parse_json(text: str):
    """从回复里取出 JSON（容忍 ```json 代码块和前后多余文字）。"""
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        s, e = t.find("{"), t.rfind("}")
        if s >= 0 and e > s:
            return json.loads(t[s:e + 1])
        raise


# ── 客户端 ──────────────────────────────────────────────────────────────


class OpenAICompatClient:
    """OpenAI 兼容的对话接口客户端。

    base_url 例：mlx_vlm.server → "http://127.0.0.1:8080/v1"；deepseek → "https://api.deepseek.com"。
    extra_body 会合并进每个请求（服务端字段名和这里的默认不一致时用它覆盖）。
    """

    def __init__(self, base_url: str, model: str, api_key_env: str | None = None, provider: str = "mlx",
                 transport: Transport | None = None, log_path: str | Path | None = None,
                 timeout: float = 120.0, max_retries: int = 3, backoff_s: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep, extra_body: dict | None = None,
                 image_root: str | Path | None = None):
        if provider not in PROVIDERS:
            raise ValueError(f"provider 必须是 {PROVIDERS} 之一")
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url if self.base_url.endswith("/chat/completions") else self.base_url + "/chat/completions"
        self.model = model
        self.api_key_env = api_key_env
        self.provider = provider
        self.transport = transport or urllib_transport
        self.log_path = Path(log_path) if log_path else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.sleep = sleep
        self.extra_body = dict(extra_body or {})
        self.image_root = image_root

    # ---- 请求体 ----

    def build_payload(self, messages: list[dict], json_schema: dict | None = None, images: Sequence | None = None,
                      max_tokens: int = 1024, temperature: float = 0.0) -> dict:
        msgs = prepare_messages(messages, images, self.image_root)
        payload: dict = {"model": self.model, "messages": msgs, "max_tokens": int(max_tokens),
                         "temperature": float(temperature), "stream": False}
        if self.provider == "mlx":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif self.provider == "deepseek":
            payload["thinking"] = {"type": "disabled"}
        if json_schema is not None:
            if self.provider == "deepseek":
                payload["response_format"] = {"type": "json_object"}
                if not _mentions_json(msgs):
                    users = [m for m in msgs if m.get("role") == "user"]
                    target = users[-1] if users else msgs[-1]
                    _append_text(target, schema_instruction(json_schema))
            else:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": str(json_schema.get("title", "output")), "schema": json_schema,
                                    "strict": True},
                }
        payload.update(copy.deepcopy(self.extra_body))
        return payload

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise LLMError(f"环境变量 {self.api_key_env} 没有设置")
            h["Authorization"] = f"Bearer {key}"
        return h

    def _log(self, rec: dict) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_redact(rec), ensure_ascii=False) + "\n")

    # ---- 主入口 ----

    def chat(self, messages: list[dict], json_schema: dict | None = None, images: Sequence | None = None,
             max_tokens: int = 1024, temperature: float = 0.0) -> dict:
        payload = self.build_payload(messages, json_schema, images, max_tokens, temperature)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = self._headers()
        params = {k: v for k, v in payload.items() if k not in ("messages", "model")}
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            rec = {"ts": datetime.now(timezone.utc).isoformat(), "provider": self.provider, "url": self.url,
                   "model": self.model, "attempt": attempt, "params": params, "request": payload}
            t0 = time.perf_counter()
            retry = False
            try:
                status, raw = self.transport(self.url, headers, body, self.timeout)
            except (OSError, TimeoutError) as e:  # 网络错误、超时
                rec.update(error=f"{type(e).__name__}: {e}", latency_s=time.perf_counter() - t0)
                self._log(rec)
                last_err, retry = LLMError(rec["error"]), True
            else:
                latency = time.perf_counter() - t0
                text_body = raw.decode("utf-8", errors="replace")
                rec.update(status=status, latency_s=latency)
                if status == 429 or status >= 500:
                    rec["error"] = text_body[:2000]
                    self._log(rec)
                    last_err, retry = HTTPStatusError(status, text_body), True
                elif status >= 400:
                    rec["error"] = text_body[:2000]
                    self._log(rec)
                    raise HTTPStatusError(status, text_body)
                else:
                    try:
                        data = json.loads(text_body)
                        msg = (data.get("choices") or [{}])[0].get("message") or {}
                    except (json.JSONDecodeError, AttributeError, IndexError) as e:
                        rec["error"] = f"回复不是合法 JSON: {e}"
                        rec["response"] = text_body[:2000]
                        self._log(rec)
                        last_err, retry = LLMError(rec["error"]), True
                    else:
                        content = msg.get("content")
                        if isinstance(content, list):
                            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                        text = content or ""
                        rec.update(response=data, response_model=data.get("model"), text=text,
                                   usage=data.get("usage") or {})
                        try:
                            no_reasoning_check(text, msg)
                        except ReasoningLeakError as e:
                            rec["error"] = str(e)
                            self._log(rec)
                            raise
                        if json_schema is not None and not text.strip():
                            rec["error"] = "要求 JSON 但回复内容为空"
                            self._log(rec)
                            last_err, retry = LLMError(rec["error"]), True
                        else:
                            self._log(rec)
                            return {"text": text, "raw": data, "usage": data.get("usage") or {},
                                    "latency_s": latency, "model": data.get("model", self.model)}
            if retry and attempt < self.max_retries:
                self.sleep(self.backoff_s * (2**attempt))
        raise LLMError(f"重试 {self.max_retries} 次后仍失败：{last_err}") from last_err


class ScriptedModel:
    """离线兜底 / 测试用的"模型"：按顺序吐出预设回复，或调用 responder(messages)。"""

    def __init__(self, responses: Sequence | None = None, responder: Callable[[list[dict]], str] | None = None,
                 default: str = "", model: str = "scripted"):
        self.responses = list(responses or [])
        self.responder = responder
        self.default = default
        self.model = model
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], json_schema: dict | None = None, images: Sequence | None = None,
             max_tokens: int = 1024, temperature: float = 0.0) -> dict:
        self.calls.append({"messages": copy.deepcopy(messages), "json_schema": json_schema,
                           "n_images": len(images or []), "max_tokens": max_tokens, "temperature": temperature})
        if self.responder is not None:
            text = self.responder(messages)
        elif self.responses:
            text = self.responses.pop(0)
        else:
            text = self.default
        if isinstance(text, Exception):
            raise text
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        return {"text": text, "raw": {"scripted": True}, "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "latency_s": 0.0, "model": self.model}
