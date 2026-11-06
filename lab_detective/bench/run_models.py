"""在物理 bench 上跑模型 → 预测 JSONL（每行 {"id", "text", "model", "split", "params", "date", ...}）。

从不自动运行，由使用者手动启动。所有模型用同一份提示（bench 文件里 assistant 之前的消息）、同样的 max_tokens 和温度。
  --model engine   引擎参照：直接输出引擎目标（bench 的标准答案本来就是引擎算的，所以与标准答案一致率 = 1）
  --model rule     规则参照：约 20 行的面板匹配器——按"原因的预期特征"与面板判出特征的重合度排序，
                   下一次测量固定选 M1，不判"不在清单里"
  其他             OpenAI 兼容接口（lab_detective/llm/client.py）：本机 mlx_vlm.server、服务器 vLLM、DeepSeek 等
DeepSeek 默认关闭思考；--thinking high 打开思考作为上限参照（思考内容在回复里被剥离、只记长度，不当作答案）。
例：
  python -m lab_detective.bench.run_models --model engine --splits in_distribution held_out
  python -m lab_detective.bench.run_models --model deepseek-chat --provider deepseek \\
      --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --splits held_out --limit 50
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ..training import prompts
from ..training.format import read_jsonl
from .physics_bench import OUT, SPLITS, read_gold

MAX_TOKENS = 1024


def rule_predict(gold: dict) -> str:
    """规则参照：特征重合度（Jaccard）排序；"只有随机误差"的预期特征是 none。"""
    seen = set(gold.get("signatures") or ["none"])
    score = {}
    for cid, exp in (gold.get("expect") or {}).items():
        e = set(exp)
        score[cid] = len(e & seen) / max(len(e | seen), 1)
    order = sorted(score, key=lambda c: (-score[c], int(c[1:])))
    w = [score[c] + 0.05 for c in order]
    s = sum(w)
    probs = [round(round(x / s / 0.05) * 0.05, 2) for x in w]
    menu = sorted(gold.get("menu_map") or {}, key=lambda m: int(m[1:]))
    out = {"signatures": sorted(seen), "evidence": "按面板特征匹配。",
           "ranked": [{"cause_id": c, "prob": p, "expect": list(gold["expect"][c])} for c, p in zip(order, probs)],
           "not_in_list": False,
           "next_measurement": {"id": menu[0], "predicted": []} if menu else None,
           "agreement": None, "explanation": "规则匹配给出的排序。"}
    return json.dumps(out, ensure_ascii=False)


def _bench_rows(bench_dir: Path, splits, limit: int | None):
    for s in splits:
        p = bench_dir / f"bench_{s}.jsonl"
        if not p.exists():
            continue
        gold = read_gold(bench_dir / f"bench_{s}.gold.jsonl")
        for i, ex in enumerate(read_jsonl(p)):
            if limit is not None and i >= limit:
                break
            yield s, ex, gold.get(ex.id, {})


def _strip_reasoning(transport):
    """思考开启时 DeepSeek 回复带 reasoning_content；客户端会拒收，这里先剥离（只留长度）再交给客户端。"""
    stats = {"reasoning_chars": 0}

    def t(url, headers, body, timeout):
        status, raw = transport(url, headers, body, timeout)
        try:
            data = json.loads(raw.decode("utf-8"))
            for ch in data.get("choices") or []:
                msg = ch.get("message") or {}
                r = msg.pop("reasoning_content", None) or msg.pop("reasoning", None)
                if isinstance(r, str):
                    stats["reasoning_chars"] += len(r)
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        except (ValueError, AttributeError):
            pass
        return status, raw

    return t, stats


def make_client(a):
    from ..llm.client import OpenAICompatClient, urllib_transport

    extra = {}
    transport = urllib_transport
    rstats = None
    if a.thinking == "high":
        if a.provider != "deepseek":
            raise SystemExit("--thinking high 只用于 DeepSeek 上限参照")
        extra = {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
        transport, rstats = _strip_reasoning(urllib_transport)
    log = Path(a.out).with_suffix(".calls.jsonl") if a.log_calls else None
    c = OpenAICompatClient(a.base_url, a.model, api_key_env=a.api_key_env, provider=a.provider,
                           transport=transport, log_path=log, timeout=a.timeout, extra_body=extra)
    return c, rstats


def run(a) -> int:
    bench_dir = Path(a.bench_dir)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    params = {"max_tokens": a.max_tokens, "temperature": a.temperature, "thinking": a.thinking,
              "provider": a.provider, "base_url": a.base_url, "prompt_version": prompts.PROMPT_VERSION}
    client = rstats = None
    if a.model not in ("engine", "rule"):
        client, rstats = make_client(a)
    done = set()
    if out.exists() and a.resume:
        done = {json.loads(line)["id"] for line in out.open(encoding="utf-8") if line.strip()}
    n = 0
    with out.open("a" if a.resume else "w", encoding="utf-8") as f:
        for split, ex, gold in _bench_rows(bench_dir, a.splits, a.limit):
            if ex.id in done:
                continue
            rec = {"id": ex.id, "split": split, "model": a.model, "params": params,
                   "date": datetime.now(timezone.utc).isoformat()}
            t0 = time.perf_counter()
            if a.model == "engine":
                rec["text"] = ex.messages[-1]["content"]
            elif a.model == "rule":
                rec["text"] = rule_predict(gold) if gold.get("schema") == "diagnose" else ""
            else:
                schema = prompts.SCHEMAS[ex.meta.get("schema", ex.task)]
                try:
                    before = rstats["reasoning_chars"] if rstats else 0
                    r = client.chat(ex.messages[:-1], json_schema=schema, max_tokens=a.max_tokens,
                                    temperature=a.temperature)
                    rec.update(text=r["text"], usage=r.get("usage"), response_model=r.get("model"))
                    if rstats:
                        rec["reasoning_chars"] = rstats["reasoning_chars"] - before
                except Exception as e:  # noqa: BLE001  单条失败记下来，继续跑
                    rec.update(text="", error=f"{type(e).__name__}: {e}")
            rec["latency_s"] = round(time.perf_counter() - t0, 3)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="在物理 bench 上跑模型（手动启动）")
    ap.add_argument("--model", required=True, help="engine / rule / 接口上的模型 id")
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--bench-dir", default=str(OUT))
    ap.add_argument("--out", default=None, help="默认 results/bench/physics/preds/<模型>.jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--provider", default="mlx", choices=["mlx", "deepseek", "openai"])
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--thinking", default="off", choices=["off", "high"])
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=None, help="每个划分最多跑几条")
    ap.add_argument("--resume", action="store_true", help="跳过输出文件里已有的样本，接着跑")
    ap.add_argument("--log-calls", action="store_true", help="把每次调用（已去掉密钥）记到 <out>.calls.jsonl")
    a = ap.parse_args(argv)
    if a.out is None:
        tag = a.model.replace("/", "_") + ("-thinking" if a.thinking == "high" else "")
        a.out = str(Path(a.bench_dir) / "preds" / f"{tag}.jsonl")
    n = run(a)
    print(f"写了 {n} 条预测 → {a.out}")


if __name__ == "__main__":
    main()
