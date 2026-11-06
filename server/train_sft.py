"""LoRA SFT（RTX 5090）：Qwen3.5-4B 的语言部分，引擎标签监督。

- LoRA r=16、α=32、dropout 0.05，只挂在语言模型的线性层上。默认 all-linear = 全注意力层 q/k/v/o + MLP gate/up/down
  + 线性注意力（Gated DeltaNet）投影 in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj：Qwen3.5 大部分层是线性注意力，
  只挂 q/k/v/o 的话这些层只有 MLP 被微调。开训前在 Mac 上用 convert_adapter.py check-names 核对 MLX 同名；
  不通过就 S1 和对照都用 --targets attn-mlp（只有注意力 + MLP）。视觉塔、词嵌入、lm_head 全部冻结。
- bf16、梯度检查点（use_reentrant=False）、只对最后一条助手回复算损失（见 common.py 文件头）。
- 损失按 token 加权：一个优化步里所有微批次的目标 token 求和后再除以总数（所以微批次怎么切不影响目标函数）。
- 只算目标位置的 logits（logits_to_keep）：词表 248k，4096 长度整段 logits 要 4 GB，前向 + 反向翻倍。
- 默认 max_len 4096、微批次 1 × 累积 16：诊断样本约 2,200–3,000 token；超长样本不截断而是丢弃，
  正式训练时只要有一条超长（答案会被截掉）就拒绝开训（--allow-truncation 才放行），
  dry-run 打印各 task 的 prompt / 答案 / 总长分位数（p50/p90/p99/max）和答案会被截掉的条数。
- 显存估算（4B、bf16、LoRA、梯度检查点、max_len 4096、微批次 1）：权重约 9–10 GB（含视觉塔）；LoRA + AdamW 状态 < 1 GB；
  检查点激活约 层数 × L × hidden × 2 字节 ≈ 0.7 GB，加重算的一层约 1–2 GB；只算答案位置的 logits
  （约 1k × 词表 248k，bf16 + fp32 交叉熵副本 + 梯度）约 2–3 GB；合计约 15–20 GB，32 GB 卡余量充足。
  没有 fla 快速路径时 torch 版 gated-delta 更费显存，以吞吐探针日志里的 gpu_mem_gb 为准。
- 数据顺序只由 (seed, epoch, id) 决定；--shuffle-labels 的打乱标签对照看到完全相同的批次。
- 学习率调度由"计划总步数"决定；对照用 --match-steps-from <S1 输出目录> 读回 S1 的计划总步数和最终步数，
  两边学习率轨迹逐步相同。--max-minutes 只是安全闸。
- 训练前必做泄漏检查（--leak-check bench 文件），撞到就停。
- --dry-run：只加载处理器做分词，报告 token 统计、丢弃数、泄漏、打乱统计和 LoRA 模块清单，不碰 GPU。

用法见 server/README.md。
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

# DataLoader 用 fork 起工作进程，父进程已经用过 tokenizers 的线程池：关掉它的并行，免得子进程死锁
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from convert_adapter import ATTN_MLP, LANG_PREFIXES, expected_lora_modules, hf_module_to_mlx  # noqa: E402

LINEAR_ATTN = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
EVAL_MAX_NEW_TOKENS = 1536  # 与 merge_and_eval.py 的默认值一致：答案比它长，评测时也会被截断
# 对照必须和 S1 一致的配置项
MATCH_KEYS = ("model", "seed", "lr", "batch_size", "grad_accum", "epochs", "max_len", "lora_r", "lora_alpha",
              "lora_dropout", "targets", "warmup_ratio", "min_lr_ratio", "weight_decay", "max_grad_norm", "tasks")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Qwen3.5 LoRA SFT（引擎标签）")
    ap.add_argument("--model", required=True, help="底座目录（ModelScope / HF 下载的 Qwen3.5-4B）")
    ap.add_argument("--train", nargs="+", required=True, help="训练 JSONL（ARCHITECTURE §2.5 格式）")
    ap.add_argument("--val", nargs="*", default=[], help="验证 JSONL，只记录验证损失，不用来挑检查点")
    ap.add_argument("--data-root", default=None, help="数据包根目录（图片路径相对它）；默认取第一个训练文件所在目录")
    ap.add_argument("--output", required=True)
    ap.add_argument("--tasks", default="", help="只用这些 task（逗号分隔），默认全部")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条（按确定性顺序），调试用")
    # 训练超参
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="计划总步数上限（0 = 由 epochs 决定）")
    ap.add_argument("--max-minutes", type=float, default=120.0, help="时间闸（分钟），到点保存并停止")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1, help="微批次大小（等效批 = batch-size × grad-accum = 16）")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096, help="超过的样本直接丢弃（不截断）")
    ap.add_argument("--allow-truncation", action="store_true",
                    help="允许有超长样本（它们会被丢弃）；默认只要有一条超长就拒绝开训")
    ap.add_argument("--seed", type=int, default=1234)
    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--targets", default="all-linear",
                    help="LoRA 模块名后缀，逗号分隔；'all-linear'（默认）= 注意力 + MLP + 线性注意力投影；"
                         "'attn-mlp' = 只有注意力 q/k/v/o + MLP（Mac 上 check-names 不通过时用）")
    # 对照与匹配
    ap.add_argument("--shuffle-labels", action="store_true", help="打乱标签对照：每个诊断样本独立置换原因标签")
    ap.add_argument("--shuffle-tasks", default="diagnose")
    ap.add_argument("--match-steps-from", default=None, help="S1 输出目录：读回计划总步数和最终步数，并核对配置")
    ap.add_argument("--allow-mismatch", action="store_true")
    # 泄漏
    ap.add_argument("--leak-check", nargs="*", default=None, help="bench JSONL / id / 哈希清单")
    ap.add_argument("--bench-root", default=None, help="bench 图片根目录（默认同 --data-root）")
    ap.add_argument("--no-leak-check", action="store_true", help="跳过泄漏检查（只许冒烟用）")
    # 保存与日志
    ap.add_argument("--save-every", type=int, default=200, help="每多少步存一次检查点（断点续跑用）")
    ap.add_argument("--save-every-minutes", type=float, default=30.0)
    ap.add_argument("--keep-checkpoints", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=0, help="每多少步算一次验证损失（0 = 只在最后算）")
    ap.add_argument("--val-limit", type=int, default=200)
    ap.add_argument("--resume", default=None, help="从检查点目录续跑")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--attn", default="sdpa", help="attn_implementation")
    ap.add_argument("--backend", choices=("hf", "unsloth"), default="hf")
    ap.add_argument("--dry-run", action="store_true", help="只分词并报告，不加载权重")
    ap.add_argument("--run-name", default=None)
    return ap.parse_args(argv)


# ---------------------------------------------------------------- 工具

class JsonlLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, event: str, **kw):
        row = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kw}
        self.f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self.f.flush()
        short = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in kw.items()
                 if not isinstance(v, (dict, list))}
        print(f"[{event}] " + json.dumps(short, ensure_ascii=False, default=str), flush=True)


def resolve_targets(spec: str) -> tuple[str, ...]:
    if spec == "all-linear":
        return ATTN_MLP + LINEAR_ATTN
    if spec == "attn-mlp":
        return ATTN_MLP
    return tuple(s.strip() for s in spec.split(",") if s.strip())


def ids_digest(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def versions() -> dict:
    """按发行包查版本，不导入（导入 unsloth 会给 transformers 打补丁）。"""
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": sys.version.split()[0]}
    for dist in ("torch", "transformers", "peft", "trl", "accelerate", "safetensors", "unsloth",
                 "flash-linear-attention", "causal-conv1d", "vllm"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = None
    return out


def fast_path_selftest(which: str, device: str = "cuda") -> dict:
    """在 GPU 上真跑一遍线性注意力快速路径的核（前向 + 反向），和 transformers 自带的 torch 实现比对。
    导入成功不等于能用：预编译的 causal-conv1d 轮子可能没有 sm_120 内核，fla 的 Triton 核也可能在
    消费级 Blackwell 上超共享内存。which = "fla" | "causal_conv1d"。ok=None 表示没装。
    每个包单独开进程调用（CUDA 非法访存之类的错误会弄坏整个上下文）。"""
    import torch
    import torch.nn.functional as F

    dev, bf = device, torch.bfloat16
    gen = torch.Generator().manual_seed(0)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen)

    def rel(a, b):
        a, b = a.detach().float(), b.detach().float()
        return float((a - b).norm() / b.norm().clamp_min(1e-6))

    try:
        if which == "fla":
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            except ImportError as e:
                return {"ok": None, "error": f"没装：{e!r}"[:300]}
            from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

            # Qwen3.5-4B 线性注意力：16 个 k 头扩成 32 个 v 头，头维 128；T 不是 64 的倍数，测补齐
            B, T, H, D = 1, 200, 4, 128
            q, k, v = rnd(B, T, H, D), rnd(B, T, H, D), rnd(B, T, H, D)
            g = -F.softplus(rnd(B, T, H)) * 0.5
            beta = torch.sigmoid(rnd(B, T, H))
            xs = [t.to(dev, bf).requires_grad_() for t in (q, k, v)]
            o, _ = chunk_gated_delta_rule(*xs, g=g.to(dev), beta=beta.to(dev, bf), initial_state=None,
                                          output_final_state=False, use_qk_l2norm_in_kernel=True)
            o.float().pow(2).mean().backward()
            ref, _ = torch_chunk_gated_delta_rule(q.to(dev), k.to(dev), v.to(dev), g=g.to(dev), beta=beta.to(dev),
                                                  use_qk_l2norm_in_kernel=True)
            err = rel(o, ref)
            grads_ok = all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in xs)
            res = {"rel_err": err, "grads_finite": grads_ok}
            try:  # 有 fla 时 transformers 还会用它的融合门控 RMSNorm
                from fla.modules import FusedRMSNormGated

                norm = FusedRMSNormGated(D, eps=1e-6, activation="silu").to(dev)
                x, z = rnd(64, D).to(dev, bf).requires_grad_(), rnd(64, D).to(dev, bf).requires_grad_()
                y = norm(x, z)
                y.float().sum().backward()
                xf = x.detach().float()
                ref_n = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * F.silu(z.detach().float())
                res["norm_rel_err"] = rel(y, ref_n)
            except ImportError:
                res["norm_rel_err"] = None
            res["ok"] = bool(math.isfinite(err) and err < 0.1 and grads_ok
                             and (res["norm_rel_err"] is None or res["norm_rel_err"] < 0.1))
            return res
        if which == "causal_conv1d":
            try:
                from causal_conv1d import causal_conv1d_fn
            except ImportError as e:
                return {"ok": None, "error": f"没装：{e!r}"[:300]}
            Bc, Dc, Tc, K = 2, 256, 77, 4
            x = rnd(Bc, Dc, Tc).to(dev, bf).requires_grad_()
            w = rnd(Dc, K).to(dev, bf)
            y = causal_conv1d_fn(x=x, weight=w, bias=None, activation="silu")
            y.float().sum().backward()
            ref = F.silu(F.conv1d(x.detach().float(), w.float().unsqueeze(1), padding=K - 1, groups=Dc)[..., :Tc])
            err = rel(y, ref)
            ok = math.isfinite(err) and err < 0.05 and x.grad is not None and bool(torch.isfinite(x.grad).all())
            return {"ok": bool(ok), "rel_err": err}
        raise ValueError(which)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)[:500]}


def fast_path_info() -> dict:
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m

        return {"is_fast_path_available": bool(getattr(m, "is_fast_path_available", False)),
                "causal_conv1d": getattr(m, "causal_conv1d_fn", None) is not None,
                "fla_gated_delta_rule": getattr(m, "chunk_gated_delta_rule", None) is not None}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


# ---------------------------------------------------------------- 数据

def prepare_records(args, log):
    recs = C.load_records(args.train)
    tasks = [t for t in args.tasks.split(",") if t]
    if tasks:
        recs = [r for r in recs if r["task"] in tasks]
    if args.limit:
        order = C.deterministic_order([r["id"] for r in recs], args.seed)
        recs = [recs[i] for i in order[: args.limit]]
    report = {"summary": C.dataset_summary(recs), "train_sha256": {p: C.file_sha256(p) for p in args.train}}
    if args.leak_check:
        bench = C.load_bench_index(args.leak_check, args.bench_root or args.data_root)
        leak = C.leakage_report(recs, bench, args.data_root)
        report["leakage"] = {k: v for k, v in leak.items() if k not in ("id_hits", "hash_hits")}
        report["leakage"].update(id_hits=leak["id_hits"][:20], hash_hits=leak["hash_hits"][:20],
                                 n_id_hits=len(leak["id_hits"]), n_hash_hits=len(leak["hash_hits"]),
                                 files={p: C.file_sha256(p) for p in args.leak_check})
        log("leak_check", **report["leakage"])
        if not leak["clean"]:
            raise SystemExit(f"泄漏检查不通过：id 撞 {len(leak['id_hits'])} 条，内容撞 {len(leak['hash_hits'])} 条。停止。")
    elif not args.no_leak_check:
        raise SystemExit("没有给 --leak-check（bench 文件）。正式训练必须做泄漏检查；只有冒烟可以用 --no-leak-check。")
    else:
        report["leakage"] = None
    originals = {r["id"]: r for r in recs}
    if args.shuffle_labels:
        recs, st = C.apply_label_shuffle(recs, args.seed, [t for t in args.shuffle_tasks.split(",") if t])
        report["label_shuffle"] = st
        log("label_shuffle", **st)
        if st["n_target_task"] == 0 or st["frac_changed"] < 0.5:
            raise SystemExit(f"打乱标签只改动了 {st['frac_changed']:.0%} 的诊断样本，对照无效。检查 meta.causes。")
    return recs, originals, report


def encode_all(recs, originals, codec, args, log):
    """全部分词一遍：丢弃超长样本（不截断）。打乱标签时按原始目标的长度决定去留，保证和 S1 留下同一批样本。"""
    items, dropped, lens, plens, tlens = [], [], [], [], []
    by_task: dict[str, dict[str, list]] = {}  # 全部样本（含超长）的长度，dry-run 看分位数用
    n_render_mismatch = 0
    for i, rec in enumerate(recs):
        enc = C.encode_example(rec, codec, args.data_root)
        n_decide = enc["n_prompt"] + enc["n_target"]
        orig = originals[rec["id"]]
        if orig is not rec:
            _, ot = C.split_prompt_target(orig)
            n_decide = enc["n_prompt"] + len(codec.tokenize(ot + codec.end_marker, [])["input_ids"])
        if i < 20 and not C.check_full_render(rec, codec):
            n_render_mismatch += 1
        bt = by_task.setdefault(rec["task"], {"total": [], "prompt": [], "target": [], "over": [], "prompt_over": []})
        bt["total"].append(n_decide)
        bt["prompt"].append(enc["n_prompt"])
        bt["target"].append(n_decide - enc["n_prompt"])
        if n_decide > args.max_len:
            bt["over"].append(rec["id"])
            if enc["n_prompt"] >= args.max_len:  # 连 prompt 都放不下：答案一个 token 都留不住
                bt["prompt_over"].append(rec["id"])
            dropped.append(rec["id"])
            continue
        n = len(enc["input_ids"])
        lens.append(n)
        plens.append(enc["n_prompt"])
        tlens.append(enc["n_target"])
        light = None
        if enc["n_images"] == 0:
            light = {k: enc[k] for k in ("input_ids", "labels", "seq")}
            light["mm"] = {}
        items.append({"rec": rec, "enc": light})
    stats = {"max_len": args.max_len, "n_kept": len(items), "n_dropped_too_long": len(dropped),
             "dropped_ids": dropped[:50],
             "tokens_total": C.length_stats(lens), "tokens_prompt": C.length_stats(plens),
             "tokens_target": C.length_stats(tlens),
             "by_task": {k: {"n": len(v["total"]), "n_over_max_len": len(v["over"]),
                             "n_prompt_over_max_len": len(v["prompt_over"]),
                             "total": C.length_stats(v["total"]), "prompt": C.length_stats(v["prompt"]),
                             "target": C.length_stats(v["target"])} for k, v in sorted(by_task.items())},
             "full_render_mismatch_first20": n_render_mismatch}
    if n_render_mismatch:
        log("warning", msg=f"前 20 条里有 {n_render_mismatch} 条：完整对话渲染 ≠ prompt + 目标 + 结束符。检查模板。")
    return items, stats


def length_table(tok_stats, eval_max_new_tokens: int = EVAL_MAX_NEW_TOKENS) -> str:
    """dry-run 打印用：每个 task 的 prompt / 答案 / 总长 token 分位数（p50/p90/p99/max，按全部样本统计，含超长的），
    以及答案会被截掉的条数（prompt + 答案 > max_len：训练时整条丢弃，等于这条答案被截光）。"""
    q = (lambda d: "/".join(str(d.get(k, "-")) for k in ("p50", "p90", "p99", "max")))
    rows = [f"max_len={tok_stats['max_len']}  保留 {tok_stats['n_kept']}  "
            f"答案会被截掉（整条丢弃）{tok_stats['n_dropped_too_long']}",
            # 中文占两格：表头手工对齐数据列（task 10、n 6、答案被截 10、prompt超长 12）
            "task           n  答案被截  prompt超长  prompt p50/p90/p99/max      答案 p50/p90/p99/max      "
            "总长 p50/p90/p99/max"]
    long_targets = []
    for task, s in tok_stats["by_task"].items():
        rows.append(f"{task:<10}{s['n']:>6}{s['n_over_max_len']:>10}{s.get('n_prompt_over_max_len', 0):>12}  "
                    f"{q(s['prompt']):<28}{q(s['target']):<26}{q(s['total'])}")
        if s["target"].get("max", 0) > eval_max_new_tokens:
            long_targets.append(f"{task}（最长 {s['target']['max']}）")
    if long_targets:
        rows.append(f"警告：这些 task 的答案超过评测的 --max-new-tokens {eval_max_new_tokens}，评测时会被截断、"
                    f"JSON 不合法：{'、'.join(long_targets)}。评测时加大 --max-new-tokens。")
    return "\n".join(rows)


def truncation_gate(tok_stats, allow_truncation: bool) -> tuple[bool, str]:
    """返回 (是否拒绝开训, 说明)。有任何一条答案会被截掉（prompt + 答案 > max_len）就拒绝，
    除非 --allow-truncation（这些样本会被整条丢弃；S1 和对照要用同一设置）。"""
    n = tok_stats["n_dropped_too_long"]
    if n == 0:
        return False, f"全部 {tok_stats['n_kept']} 条都在 max_len={tok_stats['max_len']} 以内，可以开训。"
    per = {t: s["n_over_max_len"] for t, s in tok_stats["by_task"].items() if s["n_over_max_len"]}
    msg = (f"{n} 条样本的答案会被截掉（prompt + 答案 > max_len={tok_stats['max_len']}；按 task：{per}；"
           f"例 {tok_stats['dropped_ids'][:5]}）。")
    if allow_truncation:
        return False, msg + "已加 --allow-truncation：这些样本整条丢弃后开训（对照必须同样设置）。"
    return True, msg + "正式训练会拒绝开训：加大 --max-len（如 6144），或确认后加 --allow-truncation（S1 和对照一致）。"


def build_plan(items, args, start_step=0):
    ids = [it["rec"]["id"] for it in items]
    plan = C.epoch_plan(ids, args.seed, args.epochs)
    micro = [plan[i:i + args.batch_size] for i in range(0, len(plan), args.batch_size)]
    steps_full = math.ceil(len(micro) / args.grad_accum)
    planned = min(steps_full, args.max_steps) if args.max_steps else steps_full
    stop_at = planned
    if args.match_steps_from:
        s = json.loads((Path(args.match_steps_from) / "summary.json").read_text(encoding="utf-8"))
        planned, stop_at = int(s["planned_total_steps"]), int(s["final_step"])
        if planned > steps_full:
            raise SystemExit(f"S1 的计划步数 {planned} 超过本次数据能提供的 {steps_full} 步")
    return micro, planned, stop_at


def check_match(args, cfg, kept_digest, log):
    if not args.match_steps_from:
        return None
    s = json.loads((Path(args.match_steps_from) / "summary.json").read_text(encoding="utf-8"))
    diffs = {k: (s["config"].get(k), cfg.get(k)) for k in MATCH_KEYS if s["config"].get(k) != cfg.get(k)}
    if s.get("kept_ids_sha256") != kept_digest:
        diffs["kept_ids_sha256"] = (s.get("kept_ids_sha256"), kept_digest)
    log("match_check", source=args.match_steps_from, diffs=diffs)
    if diffs and not args.allow_mismatch:
        raise SystemExit(f"和 S1 的配置 / 样本不一致：{diffs}")
    return diffs


def load_codec(model_path: str):
    """处理器 → (codec, pad_id)。断言模板能关闭思考、分词器认识回合结束符。"""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    if not C.template_has_thinking_switch(processor):
        raise SystemExit("模板里没有 enable_thinking 开关：无法保证关闭思考，停止。")
    codec = C.make_hf_codec(processor, enable_thinking=False)
    tok = getattr(processor, "tokenizer", processor)
    if tok.convert_tokens_to_ids(codec.end_marker) in (None, tok.unk_token_id):
        raise SystemExit(f"分词器里没有回合结束符 {codec.end_marker}")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids(codec.end_marker)
    return codec, pad_id


# ---------------------------------------------------------------- 模型

def load_model(args, targets, log, need_vision: bool = True):
    import torch

    if args.backend == "unsloth":
        from unsloth import FastVisionModel  # 实验性：装了才用

        model, _ = FastVisionModel.from_pretrained(args.model, load_in_4bit=False, dtype=torch.bfloat16,
                                                   use_gradient_checkpointing="unsloth")
        model = FastVisionModel.get_peft_model(
            model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True,
            finetune_mlp_modules=True, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", random_state=args.seed, target_modules=list(targets))
        return model
    from peft import LoraConfig, get_peft_model

    model = C.load_base_model(args.model, attn=args.attn, need_vision=need_vision)
    model.config.use_cache = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    names = lora_module_names(model, targets)
    if not names:
        raise SystemExit(f"在语言模型里找不到后缀为 {targets} 的线性层")
    for n in names:
        hf_module_to_mlx(n)  # 映射不了就现在报错，别等训完
    torch.manual_seed(args.seed)
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                     target_modules=names, bias="none")
    return get_peft_model(model, cfg)


def is_language_linear(name: str) -> bool:
    if any(x in name for x in ("visual", "vision", "lm_head", "embed", "merger")):
        return False
    return name.startswith(LANG_PREFIXES)


def lora_module_names(model, targets) -> list[str]:
    import torch.nn as nn

    return [n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and is_language_linear(n) and n.rsplit(".", 1)[-1] in targets]


def audit_trainable(model) -> dict:
    """可训练参数只能是 LoRA；视觉塔、嵌入、lm_head 一个都不能可训练。"""
    n_train, n_all, bad, lora_mods = 0, 0, [], set()
    for n, p in model.named_parameters():
        n_all += p.numel()
        if p.requires_grad:
            n_train += p.numel()
            if "lora_" not in n:
                bad.append(n)
            else:
                lora_mods.add(n.split(".lora_")[0])
    vision_trainable = [n for n, p in model.named_parameters() if p.requires_grad and ("visual" in n or "vision" in n)]
    if n_train == 0 or bad or vision_trainable:
        raise SystemExit(f"可训练参数异常：总数 {n_train}，非 LoRA {bad[:5]}，视觉塔 {vision_trainable[:5]}")
    mods = sorted(m.removeprefix("base_model.model.") for m in lora_mods)
    by_suffix = {}
    for m in mods:
        s = m.rsplit(".", 1)[-1]
        by_suffix[s] = by_suffix.get(s, 0) + 1
    return {"trainable": n_train, "total": n_all, "frac": n_train / n_all, "n_lora_modules": len(mods),
            "by_suffix": by_suffix, "modules": mods}


def lora_b_norms(model) -> dict:
    tot, n_zero, n = 0.0, 0, 0
    for name, p in model.named_parameters():
        if "lora_B" in name:
            v = float(p.detach().float().norm())
            tot += v * v
            n += 1
            n_zero += int(v == 0.0)
    return {"lora_B_norm": math.sqrt(tot), "n_lora_B": n, "n_lora_B_zero": n_zero}


def collate(batch, pad_id):
    import torch

    L = max(len(x["input_ids"]) for x in batch)
    B = len(batch)
    out = {"input_ids": torch.full((B, L), pad_id, dtype=torch.long),
           "labels": torch.full((B, L), C.IGNORE_INDEX, dtype=torch.long),
           "attention_mask": torch.zeros((B, L), dtype=torch.long)}
    # 取所有样本的并集：某条缺的键补 0（mm_token_type_ids 的 0 = 文字，纯文字样本本来就全是 0）
    seq_keys = sorted({k for x in batch for k in x["seq"] if k != "attention_mask"})
    for k in seq_keys:
        out[k] = torch.zeros((B, L), dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        out["input_ids"][i, :n] = torch.tensor(x["input_ids"])
        out["labels"][i, :n] = torch.tensor(x["labels"])
        out["attention_mask"][i, :n] = 1
        for k in seq_keys:
            if k in x["seq"]:
                out[k][i, :n] = torch.tensor(x["seq"][k])
    mm_keys = sorted({k for x in batch for k in x["mm"]})
    for k in mm_keys:
        out[k] = torch.cat([x["mm"][k] for x in batch if k in x["mm"]], dim=0)
    return out


def token_ce_sum(logits, labels):
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:].to(logits.device)
    mask = shift_labels != C.IGNORE_INDEX
    return F.cross_entropy(shift_logits[mask].float(), shift_labels[mask], reduction="sum")


def logits_keep(labels) -> int:
    """只需要从"第一个目标 token 的前一位"到序列末尾的 logits（右填充，目标在每行末尾）。"""
    valid = labels != C.IGNORE_INDEX
    L = labels.shape[1]
    if not bool(valid.any()):
        return 1
    first = int(valid.float().argmax(dim=1)[valid.any(dim=1)].min())
    return max(1, min(L, L - first + 1))


def forward_loss(model, batch, device):
    labels = batch.pop("labels")
    keep = logits_keep(labels)
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    out = model(**batch, use_cache=False, logits_to_keep=keep)
    logits = out.logits
    if logits.shape[1] != keep:  # 模型没认 logits_to_keep 时退回整段
        keep = logits.shape[1]
    return token_ce_sum(logits, labels[:, labels.shape[1] - keep:])


class SFTData:
    def __init__(self, items, codec, root):
        self.items, self.codec, self.root = items, codec, root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        if it["enc"] is not None:
            return it["enc"]
        enc = C.encode_example(it["rec"], self.codec, self.root)
        return {k: enc[k] for k in ("input_ids", "labels", "seq", "mm")}


def lr_lambda_factory(planned, warmup_ratio, min_ratio):
    warm = max(1, math.ceil(warmup_ratio * planned)) if warmup_ratio > 0 else 0

    def f(step):
        if warm and step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, planned - warm))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    return f


def evaluate_loss(model, data, idx, pad_id, bs, device):
    import torch

    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = collate([data[j] for j in idx[i:i + bs]], pad_id)
            ntok = int((b["labels"][:, 1:] != C.IGNORE_INDEX).sum())
            tot += float(forward_loss(model, b, device))
            n += ntok
    model.train()
    return tot / max(n, 1)


def save_checkpoint(model, opt, sched, step, elapsed, ckpt_dir: Path, keep: int):
    import torch

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step, "elapsed": elapsed,
                "cuda_rng": torch.cuda.get_rng_state_all(), "cpu_rng": torch.get_rng_state()},
               ckpt_dir / "trainer_state.pt")
    (ckpt_dir / "state.json").write_text(json.dumps({"step": step, "elapsed_s": elapsed}), encoding="utf-8")
    ckpts = sorted(ckpt_dir.parent.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for old in ckpts[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)


def export_bf16(src: Path, dst: Path):
    """下载用的小份：适配器权重转 bf16（体积减半）。"""
    import torch
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    t = load_file(str(src / "adapter_model.safetensors"))
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in t.items()}, str(dst / "adapter_model.safetensors"))
    shutil.copy2(src / "adapter_config.json", dst / "adapter_config.json")


# ---------------------------------------------------------------- 主流程

def main(argv=None):
    args = parse_args(argv)
    args.data_root = args.data_root or str(Path(args.train[0]).resolve().parent)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log = JsonlLog(out / ("dryrun_log.jsonl" if args.dry_run else "train_log.jsonl"))
    targets = resolve_targets(args.targets)
    cfg = {k: getattr(args, k) for k in ("model", "seed", "lr", "batch_size", "grad_accum", "epochs", "max_len",
                                         "lora_r", "lora_alpha", "lora_dropout", "warmup_ratio", "min_lr_ratio",
                                         "weight_decay", "max_grad_norm", "tasks", "max_steps", "max_minutes",
                                         "shuffle_labels", "shuffle_tasks", "backend", "train", "val", "data_root",
                                         "allow_truncation")}
    cfg["model"] = os.path.basename(os.path.normpath(args.model))
    cfg["targets"] = list(targets)
    log("start", config=cfg, run_name=args.run_name, versions=versions(), fast_path=fast_path_info())

    codec, pad_id = load_codec(args.model)

    recs, originals, data_report = prepare_records(args, log)
    t0 = time.time()
    items, tok_stats = encode_all(recs, originals, codec, args, log)
    kept_digest = ids_digest(it["rec"]["id"] for it in items)
    micro, planned, stop_at = build_plan(items, args)
    log("data", encode_seconds=round(time.time() - t0, 1), kept_ids_sha256=kept_digest, planned_total_steps=planned,
        stop_at_step=stop_at, n_micro_batches=len(micro), **{k: v for k, v in tok_stats.items() if k != "dropped_ids"})
    match_diffs = check_match(args, cfg, kept_digest, log)
    print(length_table(tok_stats), flush=True)
    refuse, gate_msg = truncation_gate(tok_stats, args.allow_truncation)
    log("truncation_gate", refuse=refuse, msg=gate_msg)

    if args.dry_run:
        hf_keys = list(C.model_weight_shapes(args.model))
        mods = expected_lora_modules(hf_keys, targets)
        lora = [{"hf": m, "mlx": hf_module_to_mlx(m)} for m in mods]
        (out / "lora_targets.json").write_text(json.dumps(lora, ensure_ascii=False, indent=1), encoding="utf-8")
        rep = {"config": cfg, "data": data_report, "tokens": tok_stats, "planned_total_steps": planned,
               "stop_at_step": stop_at, "kept_ids_sha256": kept_digest, "n_lora_modules": len(lora),
               "lora_by_suffix": {s: sum(m["hf"].endswith("." + s) for m in lora) for s in targets},
               "match_diffs": match_diffs, "would_refuse_training": refuse, "truncation_gate": gate_msg}
        (out / "dryrun.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        log("dry_run_done", n_kept=tok_stats["n_kept"], dropped=tok_stats["n_dropped_too_long"],
            planned_total_steps=planned, n_lora_modules=len(lora))
        if not lora:
            print(f"警告：底座权重里找不到后缀为 {targets} 的语言模型线性层（--model 指错了？）", flush=True)
        print(("不能开训：" if refuse else "dry-run 结论：") + gate_msg, flush=True)
        return 0

    if refuse:
        raise SystemExit("拒绝开训：" + gate_msg)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("没有 GPU")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    need_vision = any(C.image_refs(it["rec"]["messages"]) for it in items)
    model = load_model(args, targets, log, need_vision=need_vision)
    audit = audit_trainable(model)
    (out / "lora_modules.json").write_text(json.dumps(
        [{"hf": m, "mlx": hf_module_to_mlx(m)} for m in audit["modules"]], ensure_ascii=False, indent=1),
        encoding="utf-8")
    log("model", gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
        **{k: v for k, v in audit.items() if k != "modules"})
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(planned, args.warmup_ratio, args.min_lr_ratio))

    start_step, elapsed_before = 0, 0.0
    if args.resume:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        rd = Path(args.resume)
        set_peft_model_state_dict(model, load_file(str(rd / "adapter_model.safetensors")))
        st = torch.load(rd / "trainer_state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        start_step, elapsed_before = int(st["step"]), float(st["elapsed"])
        torch.set_rng_state(st["cpu_rng"])
        torch.cuda.set_rng_state_all(st["cuda_rng"])
        log("resume", step=start_step, elapsed_s=elapsed_before)

    data = SFTData(items, codec, args.data_root)
    # 显式用 fork：处理器闭包不能 pickle（Python 3.14 起 Linux 默认不再是 fork）
    loader = torch.utils.data.DataLoader(
        data, batch_sampler=micro[start_step * args.grad_accum:], num_workers=args.num_workers,
        collate_fn=functools.partial(collate, pad_id=pad_id), pin_memory=True,
        multiprocessing_context="fork" if args.num_workers > 0 else None)
    val_items, val_idx = [], []
    if args.val:
        vrecs = C.load_records(args.val)
        vrecs = [vrecs[i] for i in C.deterministic_order([r["id"] for r in vrecs], args.seed)[: args.val_limit]]
        val_items, _ = encode_all(vrecs, {r["id"]: r for r in vrecs}, codec, args, log)
        val_idx = list(range(len(val_items)))
    vdata = SFTData(val_items, codec, args.data_root)

    t_start = time.time()
    last_save = time.time()
    it = iter(loader)
    step = start_step
    stopped_by = "steps"
    win_loss, win_tok, win_all_tok, win_t = 0.0, 0, 0, time.time()
    seen_tokens = 0
    try:
        while step < stop_at:
            window = []
            for _ in range(args.grad_accum):
                try:
                    window.append(next(it))
                except StopIteration:
                    break
            if not window:
                break
            n_tok = sum(int((b["labels"][:, 1:] != C.IGNORE_INDEX).sum()) for b in window)
            for b in window:
                n_all = int(b["attention_mask"].sum())
                loss_sum = forward_loss(model, b, device)
                (loss_sum / max(n_tok, 1)).backward()
                win_loss += float(loss_sum.detach())
                win_all_tok += n_all
            win_tok += n_tok
            seen_tokens += sum(int(b["input_ids"].numel()) for b in window)
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            elapsed = elapsed_before + (time.time() - t_start)
            if not math.isfinite(gnorm) or not math.isfinite(win_loss):
                raise SystemExit(f"第 {step} 步出现非有限的损失或梯度")
            if step % args.log_every == 0 or step == stop_at:
                dt = time.time() - win_t
                log("step", step=step, of=stop_at, loss=win_loss / max(win_tok, 1), lr=sched.get_last_lr()[0],
                    grad_norm=gnorm, tokens_per_s=win_all_tok / max(dt, 1e-6), target_tokens=win_tok,
                    elapsed_min=elapsed / 60, gpu_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
                win_loss, win_tok, win_all_tok, win_t = 0.0, 0, 0, time.time()
            if args.val_every and val_idx and step % args.val_every == 0:
                log("val", step=step, val_loss=evaluate_loss(model, vdata, val_idx, pad_id, args.batch_size, device))
            time_up = elapsed > args.max_minutes * 60
            if step % args.save_every == 0 or (time.time() - last_save) > args.save_every_minutes * 60:
                save_checkpoint(model, opt, sched, step, elapsed, out / f"checkpoint-{step}", args.keep_checkpoints)
                last_save = time.time()
                log("checkpoint", step=step)
            if time_up and step < stop_at:
                stopped_by = "time"
                break
    except KeyboardInterrupt:
        stopped_by = "interrupted"
    elapsed = elapsed_before + (time.time() - t_start)

    final = out / "final"
    model.save_pretrained(str(final))
    export_bf16(final, out / "final_bf16")
    bn = lora_b_norms(model)
    if bn["lora_B_norm"] == 0.0:
        raise SystemExit("训练后 LoRA B 全为 0：梯度没有传到适配器（检查梯度检查点 / enable_input_require_grads）")
    val_loss = evaluate_loss(model, vdata, val_idx, pad_id, args.batch_size, device) if val_idx else None
    summary = {"run_name": args.run_name, "config": cfg, "planned_total_steps": planned, "final_step": step,
               "stop_at_step": stop_at, "stopped_by": stopped_by, "elapsed_min": elapsed / 60,
               "seen_tokens": seen_tokens, "kept_ids_sha256": kept_digest, "data": data_report,
               "tokens": tok_stats, "val_loss": val_loss, "lora": {k: v for k, v in audit.items() if k != "modules"},
               **bn, "match_steps_from": args.match_steps_from, "match_diffs": match_diffs,
               "versions": versions(), "fast_path": fast_path_info(),
               "gpu": torch.cuda.get_device_name(0), "gpu_mem_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
               "final_adapter": str(final), "final_adapter_sha256": C.file_sha256(final / "adapter_model.safetensors")}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log("done", final_step=step, stopped_by=stopped_by, elapsed_min=elapsed / 60, val_loss=val_loss, **bn)
    return 0 if stopped_by != "interrupted" else 130


if __name__ == "__main__":
    raise SystemExit(main())
