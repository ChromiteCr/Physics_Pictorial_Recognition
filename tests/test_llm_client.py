"""模型客户端：请求体（关思考、JSON、图片）、重试退避、推理文本检查、JSONL 日志、脚本化模型。全部用注入的假 transport，不联网。"""
import base64
import json

import pytest

from lab_detective.llm import (
    HTTPStatusError,
    LLMError,
    OpenAICompatClient,
    ReasoningLeakError,
    ScriptedModel,
    image_data_url,
    parse_json,
)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
SCHEMA = {"title": "diag", "type": "object", "properties": {"cause": {"type": "string"}}, "required": ["cause"]}


def ok_body(content="好的", model="qwen-4b", **msg_extra):
    msg = {"role": "assistant", "content": content, **msg_extra}
    return json.dumps({"model": model, "choices": [{"message": msg}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 3}}).encode()


class FakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": json.loads(body.decode()), "timeout": timeout})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def client(replies, provider="mlx", tmp_path=None, **kw):
    t = FakeTransport(replies)
    sleeps = []
    c = OpenAICompatClient("http://127.0.0.1:8080/v1", "qwen-4b", provider=provider, transport=t,
                           log_path=(tmp_path / "log.jsonl") if tmp_path else None, sleep=sleeps.append, **kw)
    return c, t, sleeps


MSGS = [{"role": "system", "content": "你是实验侦探。"}, {"role": "user", "content": "看看这张图。"}]


def test_mlx_payload_thinking_off_schema_and_images(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    c, t, _ = client([(200, ok_body('{"cause": "x"}'))], tmp_path=tmp_path)
    r = c.chat(MSGS, json_schema=SCHEMA, images=[str(img)], max_tokens=1024)
    p = t.calls[0]["payload"]
    assert t.calls[0]["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["temperature"] == 0.0 and p["max_tokens"] == 1024 and p["stream"] is False
    assert p["response_format"]["type"] == "json_schema"
    assert p["response_format"]["json_schema"]["schema"] == SCHEMA
    parts = p["messages"][-1]["content"]
    assert parts[0]["type"] == "image_url" and parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[-1] == {"type": "text", "text": "看看这张图。"}
    assert "Authorization" not in t.calls[0]["headers"]
    assert r["text"] == '{"cause": "x"}' and r["usage"]["prompt_tokens"] == 10 and r["latency_s"] >= 0
    assert parse_json(r["text"]) == {"cause": "x"}
    assert MSGS[-1]["content"] == "看看这张图。"  # 不改调用方的消息


def test_training_format_image_parts_are_converted(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "x.png").write_bytes(PNG)
    msgs = [{"role": "user", "content": [{"type": "image", "image": "images/x.png"}, {"type": "text", "text": "这是什么实验？"}]}]
    c, t, _ = client([(200, ok_body())], image_root=tmp_path)
    c.chat(msgs)
    part = t.calls[0]["payload"]["messages"][0]["content"][0]
    assert part["type"] == "image_url" and part["image_url"]["url"] == image_data_url(PNG)


def test_deepseek_payload_and_key_never_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("DS_KEY_TEST", "sk-secret-123")
    c, t, _ = client([(200, ok_body('{"cause": "x"}', model="deepseek-flash"))], provider="deepseek",
                     tmp_path=tmp_path, api_key_env="DS_KEY_TEST")
    r = c.chat(MSGS, json_schema=SCHEMA)
    p = t.calls[0]["payload"]
    assert p["thinking"] == {"type": "disabled"} and "chat_template_kwargs" not in p
    assert p["response_format"] == {"type": "json_object"}
    assert "json" in p["messages"][-1]["content"].lower() and '"cause"' in p["messages"][-1]["content"]
    assert t.calls[0]["headers"]["Authorization"] == "Bearer sk-secret-123"
    log = (tmp_path / "log.jsonl").read_text(encoding="utf-8")
    assert "sk-secret-123" not in log and "Bearer" not in log
    rec = json.loads(log.splitlines()[0])
    assert rec["response_model"] == "deepseek-flash" and rec["model"] == "qwen-4b"
    assert rec["params"]["thinking"] == {"type": "disabled"} and rec["params"]["temperature"] == 0.0
    assert "json" in rec["request"]["messages"][-1]["content"].lower()  # 记的是真正发出去的提示
    assert r["model"] == "deepseek-flash"
    # 提示里已经提到 JSON 时不再追加（便于各对比组共用同一份提示）
    c2, t2, _ = client([(200, ok_body("{}"))], provider="deepseek", api_key_env="DS_KEY_TEST")
    c2.chat([{"role": "user", "content": "请输出 JSON。"}], json_schema=SCHEMA)
    assert t2.calls[0]["payload"]["messages"][-1]["content"] == "请输出 JSON。"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("NO_SUCH_KEY_X", raising=False)
    c, t, _ = client([], provider="openai", api_key_env="NO_SUCH_KEY_X")
    with pytest.raises(LLMError):
        c.chat(MSGS)
    assert t.calls == []


def test_retry_with_backoff_then_success(tmp_path):
    c, t, sleeps = client([(503, b"busy"), OSError("连接被拒绝"), (200, ok_body())], tmp_path=tmp_path)
    r = c.chat(MSGS)
    assert r["text"] == "好的" and len(t.calls) == 3 and sleeps == [1.0, 2.0]
    lines = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["attempt"] for x in lines] == [0, 1, 2]
    assert lines[0]["status"] == 503 and "error" in lines[1] and lines[2]["text"] == "好的"
    assert all("ts" in x and x["model"] == "qwen-4b" for x in lines)


def test_gives_up_after_max_retries():
    c, t, sleeps = client([(429, b"slow down")] * 4, max_retries=3)
    with pytest.raises(LLMError):
        c.chat(MSGS)
    assert len(t.calls) == 4 and sleeps == [1.0, 2.0, 4.0]


def test_client_error_is_not_retried():
    c, t, sleeps = client([(400, b"bad request"), (200, ok_body())])
    with pytest.raises(HTTPStatusError) as e:
        c.chat(MSGS)
    assert e.value.status == 400 and len(t.calls) == 1 and sleeps == []


def test_empty_json_reply_is_retried():
    c, t, _ = client([(200, ok_body("")), (200, ok_body('{"cause": "y"}'))], provider="deepseek")
    assert c.chat(MSGS, json_schema=SCHEMA)["text"] == '{"cause": "y"}' and len(t.calls) == 2


@pytest.mark.parametrize("body", [
    ok_body("<think>先想想</think>答案"),
    ok_body("答案", reasoning_content="我先推理一下"),
    ok_body("答案", reasoning="hidden"),
])
def test_reasoning_text_is_rejected(body, tmp_path):
    c, t, _ = client([(200, body)], tmp_path=tmp_path)
    with pytest.raises(ReasoningLeakError):
        c.chat(MSGS)
    assert len(t.calls) == 1
    assert "error" in json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])


def test_null_reasoning_content_passes():
    c, _, _ = client([(200, ok_body("答案", reasoning_content=None))])
    assert c.chat(MSGS)["text"] == "答案"


def test_images_are_hashed_in_log(tmp_path):
    c, _, _ = client([(200, ok_body())], tmp_path=tmp_path)
    c.chat(MSGS, images=[PNG])
    log = (tmp_path / "log.jsonl").read_text(encoding="utf-8")
    assert base64.b64encode(PNG).decode() not in log and "sha256=" in log and f"bytes={len(PNG)}" in log


def test_extra_body_overrides_provider_defaults():
    c, t, _ = client([(200, ok_body())], extra_body={"chat_template_kwargs": {"enable_thinking": False, "x": 1}})
    c.chat(MSGS)
    assert t.calls[0]["payload"]["chat_template_kwargs"] == {"enable_thinking": False, "x": 1}


def test_scripted_model():
    m = ScriptedModel(["一", {"cause": "x"}], default="兜底")
    assert m.chat(MSGS)["text"] == "一"
    assert json.loads(m.chat(MSGS)["text"]) == {"cause": "x"}
    assert m.chat(MSGS)["text"] == "兜底"
    assert len(m.calls) == 3
    echo = ScriptedModel(responder=lambda msgs: msgs[-1]["content"][::-1])
    assert echo.chat(MSGS)["text"] == "。图张这看看"
    with pytest.raises(RuntimeError):
        ScriptedModel([RuntimeError("坏了")]).chat(MSGS)


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('结果：{"a": [1, 2]} 完') == {"a": [1, 2]}
