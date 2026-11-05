"""合并适配器 + bench 贪心批量生成（服务器）。

子命令：
  merge  把 PEFT 适配器合并进 bf16 底座，存成完整模型目录（给 vLLM 用）。在训练环境（venv-train）里跑。
         合并后核对：权重键集合与底座一致（底座里加载时本就忽略的 mtp.* 除外），处理器 / 模板文件齐全。
  eval   在 bench JSONL（同 §2.5 格式，最后一条助手消息 = 金标准）上贪心生成，写预测 JSONL。
         --backend vllm（服务环境 venv-serve，用合并后的目录）或 hf（训练环境，可直接 --adapter 不合并）。
         关闭思考：模板参数 enable_thinking=False，并断言 prompt 以空思考块结尾；每条回复检查有没有思考标记。
         --shots K --shots-from 训练池：3-shot 基线（示例只取自训练池、同 task、只用纯文字样本，按 id 确定性挑选）。
         --resume：跳过输出文件里已有的 id，逐块追加（断点续跑）。

预测 JSONL 每行：id、text（模型原始输出，bench 打分器 recognition_bench score 读这个键）、task、source、experiment_id、
split、bench_file、arm、backend、model、adapter、prediction（同 text，旧名）、gold、has_think、gold_is_json、json_ok、
exact、n_prompt_tokens、n_gen_tokens、finish_reason、latency_s（整块平均）、error。
识别 bench 的图必须是带编号框的版本：图片字节等于 meta.image_sha1（没画框的原图）就直接停。
旁边写 <输出>.summary.json：参数、bench 文件哈希、计数、JSON 合法率、思考标记率、耗时、版本。
评分不在这里做（bench 模块负责）。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

AUX_FILES = ("chat_template.jinja", "chat_template.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
             "merges.txt", "special_tokens_map.json", "added_tokens.json", "preprocessor_config.json",
             "video_preprocessor_config.json", "processor_config.json", "generation_config.json")
THINK_RE = re.compile(r"</?think>")
IGNORED_BASE_KEYS = re.compile(r"^mtp\.")  # 底座检查点里的多 token 预测层，transformers 加载时就忽略


# ---------------------------------------------------------------- merge

def _load_hf(model_dir, device):
    import torch
    from transformers import AutoModelForImageTextToText

    kw = dict(device_map={"": device})
    try:
        return AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, **kw)
    except TypeError:
        return AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch.bfloat16, **kw)


def merge(base: str, adapter: str, out: str, device: str = "cuda:0") -> dict:
    from peft import PeftModel
    from transformers import AutoProcessor

    outp = Path(out)
    if outp.exists() and any(outp.iterdir()):
        raise SystemExit(f"{out} 已存在且非空")
    model = _load_hf(base, device)
    model = PeftModel.from_pretrained(model, adapter)
    n_b = sum(int(p.abs().sum() > 0) for n, p in model.named_parameters() if "lora_B" in n)
    if n_b == 0:
        raise SystemExit("适配器的 lora_B 全为 0（没训练过？）")
    model = model.merge_and_unload()
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(base).save_pretrained(out)
    for name in AUX_FILES:
        src = Path(base) / name
        if src.is_file() and not (outp / name).exists():
            shutil.copy2(src, outp / name)
    base_shapes = {k: v for k, v in C.model_weight_shapes(base).items() if not IGNORED_BASE_KEYS.match(k)}
    out_shapes = C.model_weight_shapes(out)
    missing = sorted(set(base_shapes) - set(out_shapes))
    extra = sorted(set(out_shapes) - set(base_shapes))
    shape_diff = sorted(k for k in set(base_shapes) & set(out_shapes) if base_shapes[k] != out_shapes[k])
    rep = {"base": base, "adapter": adapter, "out": out, "n_lora_B_nonzero": n_b, "n_keys": len(out_shapes),
           "missing": missing[:20], "extra": extra[:20], "shape_diff": shape_diff[:20],
           "adapter_sha256": C.file_sha256(Path(adapter) / "adapter_model.safetensors"),
           "has_chat_template": (outp / "chat_template.jinja").is_file() or (outp / "chat_template.json").is_file()}
    (outp / "merge_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    if missing or extra or shape_diff:
        raise SystemExit(f"合并后的权重键和底座对不上：缺 {len(missing)}、多 {len(extra)}、形状不同 {len(shape_diff)}。"
                         f"见 {outp / 'merge_report.json'}")
    return rep


# ---------------------------------------------------------------- 提示构造

def pick_shots(rec: dict, pool: list[dict], k: int, seed: int) -> list[dict]:
    cands = [p for p in pool if p["task"] == rec["task"] and not C.image_refs(p["messages"]) and p["id"] != rec["id"]]
    cands.sort(key=lambda p: (C._h(f"shot:{seed}:{rec['id']}:{p['id']}"), p["id"]))
    return cands[:k]


def with_shots(prompt_msgs: list[dict], shots: list[dict]) -> list[dict]:
    """系统消息之后插入示例（各自的用户消息 + 金标准回复），再接 bench 自己的对话。"""
    sys_msgs = [m for m in prompt_msgs if m["role"] == "system"]
    rest = [m for m in prompt_msgs if m["role"] != "system"]
    demo = []
    for s in shots:
        demo.extend(m for m in s["messages"] if m["role"] != "system")
    return sys_msgs + demo + rest


def build_items(records, bench_file, codec, args, pool):
    items = []
    for rec in records:
        prompt_msgs, gold = C.split_prompt_target(rec)
        if args.shots:
            prompt_msgs = with_shots(prompt_msgs, pick_shots(rec, pool, args.shots, args.seed))
        text = C.render_prompt(codec, prompt_msgs)  # 断言关闭思考
        items.append({"rec": rec, "bench_file": bench_file, "gold": gold, "prompt": text,
                      "images": C.image_refs(prompt_msgs)})
    return items


def is_json(s: str) -> bool:
    try:
        json.loads(s.strip())
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def make_row(it, args, text, n_prompt, n_gen, finish, error=None, latency=None):
    rec, meta = it["rec"], C.meta_of(it["rec"])
    gold_json = is_json(it["gold"])
    pred = text or ""
    return {"id": rec["id"], "text": pred, "task": rec["task"], "source": rec.get("source"),
            "experiment_id": meta.get("experiment_id"), "split": meta.get("split"), "bench_file": it["bench_file"],
            "arm": args.arm, "backend": args.backend, "model": args.model, "adapter": args.adapter,
            "shots": args.shots, "prediction": pred, "gold": it["gold"], "has_think": bool(THINK_RE.search(pred)),
            "gold_is_json": gold_json, "json_ok": is_json(pred) if gold_json else None,
            "exact": pred.strip() == it["gold"].strip(), "n_prompt_tokens": n_prompt, "n_gen_tokens": n_gen,
            "finish_reason": finish, "latency_s": latency, "error": error}


# ---------------------------------------------------------------- 后端

def run_vllm(items, args, write):
    from vllm import LLM, SamplingParams

    max_img = max((len(it["images"]) for it in items), default=0)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, seed=0,
              gpu_memory_utilization=args.gpu_mem_util, limit_mm_per_prompt={"image": max(1, max_img)})
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, skip_special_tokens=True)

    def to_input(it):
        if not it["images"]:
            return {"prompt": it["prompt"]}
        imgs = [C.load_pil(C.resolve_image(r, args.data_root)) for r in it["images"]]
        return {"prompt": it["prompt"], "multi_modal_data": {"image": imgs}}

    for i in range(0, len(items), args.chunk):
        chunk = items[i:i + args.chunk]
        t0 = time.time()
        try:
            outs = llm.generate([to_input(it) for it in chunk], sp)
            pairs = list(zip(chunk, outs))
        except Exception:  # noqa: BLE001 — 整块失败就逐条重试，记下单条错误
            pairs = []
            for it in chunk:
                try:
                    pairs.append((it, llm.generate([to_input(it)], sp)[0]))
                except Exception as e:  # noqa: BLE001
                    write(make_row(it, args, "", None, 0, "error", repr(e)[:500]))
        lat = (time.time() - t0) / max(len(chunk), 1)
        for it, o in pairs:
            c = o.outputs[0]
            write(make_row(it, args, c.text, len(o.prompt_token_ids or []), len(c.token_ids), c.finish_reason,
                           latency=lat))


def run_hf(items, args, write):
    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    tok.padding_side = "left"  # 批量生成必须左填充
    model = _load_hf(args.model, "cuda:0")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    pad = tok.pad_token_id if tok.pad_token_id is not None else next(iter(eos))
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prompt"]))
    for s in range(0, len(order), args.batch_size):
        batch = [items[i] for i in order[s:s + args.batch_size]]
        imgs = [C.load_pil(C.resolve_image(r, args.data_root)) for it in batch for r in it["images"]]
        t0 = time.time()
        try:
            enc = processor(text=[it["prompt"] for it in batch], images=imgs or None, padding=True,
                            padding_side="left", return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None,
                                     top_p=None, top_k=None, pad_token_id=pad)
        except Exception as e:  # noqa: BLE001
            for it in batch:
                write(make_row(it, args, "", None, 0, "error", repr(e)[:500]))
            continue
        n_in = enc["input_ids"].shape[1]
        n_prompt = enc["attention_mask"].sum(1).tolist()
        lat = (time.time() - t0) / len(batch)
        for j, it in enumerate(batch):
            new = gen[j, n_in:].tolist()
            stop = next((k for k, t in enumerate(new) if t in eos), None)
            body = new if stop is None else new[:stop]
            text = tok.decode(body, skip_special_tokens=True)
            write(make_row(it, args, text, int(n_prompt[j]), len(body) + (stop is not None),
                           "stop" if stop is not None else "length", latency=lat))


# ---------------------------------------------------------------- eval 主流程

def evaluate(args) -> dict:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    if not C.template_has_thinking_switch(processor):
        raise SystemExit("模板里没有 enable_thinking 开关，停止")
    codec = C.make_hf_codec(processor, enable_thinking=False)
    pool = C.load_records(args.shots_from) if args.shots else []
    if args.shots and not pool:
        raise SystemExit("--shots 需要 --shots-from 训练池文件")
    items = []
    for bf in args.bench:
        recs = C.load_records([bf])
        if args.limit:
            recs = recs[: args.limit]
        unmarked = C.unmarked_image_hits(recs, args.data_root)
        if unmarked:
            raise SystemExit(f"{bf}：{len(unmarked)} 条的图片是没画编号框的原图（字节等于 meta.image_sha1），"
                             f"例 {unmarked[:3]}。打包时 bench 图片取错了目录：把 --data-root 指向 bench 自己的目录"
                             f"（里面的 images/ 是带框的），或重新打包。")
        items.extend(build_items(recs, Path(bf).name, codec, args, pool))
    out = Path(args.out)
    done = set()
    if out.exists() and args.resume:
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    elif out.exists():
        raise SystemExit(f"{out} 已存在；续跑加 --resume，重跑请换文件名")
    todo = [it for it in items if it["rec"]["id"] not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    counts = {"n": 0, "think": 0, "json_ok": 0, "gold_json": 0, "error": 0, "length": 0}
    with open(out, "a", encoding="utf-8") as f:
        def write(row):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            counts["n"] += 1
            counts["think"] += int(row["has_think"])
            counts["error"] += int(row["error"] is not None)
            counts["length"] += int(row["finish_reason"] == "length")
            if row["gold_is_json"]:
                counts["gold_json"] += 1
                counts["json_ok"] += int(bool(row["json_ok"]))

        print(f"共 {len(items)} 条，已完成 {len(done)}，本次 {len(todo)}（{args.backend}）", flush=True)
        (run_vllm if args.backend == "vllm" else run_hf)(todo, args, write)
    summ = {"args": vars(args), "bench_sha256": {bf: C.file_sha256(bf) for bf in args.bench},
            "n_items": len(items), "n_written_this_run": counts["n"], "elapsed_min": (time.time() - t0) / 60,
            "think_rate": counts["think"] / max(counts["n"], 1),
            "json_ok_rate": counts["json_ok"] / max(counts["gold_json"], 1),
            "error_rate": counts["error"] / max(counts["n"], 1),
            "length_cut_rate": counts["length"] / max(counts["n"], 1), "versions": _versions()}
    Path(str(out) + ".summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k not in ("args", "versions")}, ensure_ascii=False), flush=True)
    if counts["think"]:
        print(f"警告：{counts['think']} 条回复带思考标记", file=sys.stderr)
    return summ


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for dist in ("torch", "transformers", "peft", "vllm"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = None
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="合并适配器 / bench 贪心生成")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--base", required=True)
    m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--device", default="cuda:0")
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True, help="底座或合并后的模型目录")
    e.add_argument("--adapter", default=None, help="只用于 --backend hf：不合并直接挂适配器")
    e.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    e.add_argument("--bench", nargs="+", required=True)
    e.add_argument("--data-root", default=None, help="bench 图片根目录（默认第一个 bench 文件所在目录）")
    e.add_argument("--out", required=True)
    e.add_argument("--arm", required=True, help="对比组名，如 S1 / shuffled / base0 / base3shot")
    e.add_argument("--shots", type=int, default=0)
    e.add_argument("--shots-from", nargs="*", default=[])
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--max-new-tokens", type=int, default=1536,
                   help="诊断 JSON 约 700 字 ≈ 400–600 token；留足余量，截断的回复 JSON 不合法")
    e.add_argument("--max-model-len", type=int, default=16384,
                   help="vLLM 上下文长度：诊断提示约 2,000–2,500 token，3-shot 再加 3 个示例约 1 万")
    e.add_argument("--gpu-mem-util", type=float, default=0.85)
    e.add_argument("--batch-size", type=int, default=16, help="hf 后端的批大小")
    e.add_argument("--chunk", type=int, default=512, help="vllm 每次提交的条数（也是断点粒度）")
    e.add_argument("--limit", type=int, default=0, help="每个 bench 文件只取前 N 条（冒烟用）")
    e.add_argument("--resume", action="store_true")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "merge":
        print(json.dumps(merge(a.base, a.adapter, a.out, a.device), ensure_ascii=False, indent=1))
        return 0
    if a.adapter and a.backend == "vllm":
        raise SystemExit("vllm 后端请先 merge，再用 --model 指向合并后的目录")
    a.data_root = a.data_root or str(Path(a.bench[0]).resolve().parent)
    evaluate(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
