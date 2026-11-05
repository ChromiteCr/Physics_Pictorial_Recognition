"""PEFT LoRA 适配器 → mlx-vlm 适配器（在 Mac 上运行；也能在服务器上跑）。

数学（两边必须逐元素一致）：
  PEFT：  y = x Wᵀ + s · x Aᵀ Bᵀ，     A: (r, in)，B: (out, r)，s = α/r（rslora 时 s = α/√r）
  MLX：   y = x Wᵀ + s · (x @ a) @ b，  a = Aᵀ: (in, r)，b = Bᵀ: (r, out)

模块名：HF 的 model.language_model.layers.N.… → mlx-vlm 的 language_model.model.layers.N.…
（前缀可用 --mlx-prefix 改；给了 --mlx-model 时从 MLX 权重里自动识别）。

两种输出格式（mlx-vlm 不同版本读法不同，先用 `inspect` 子命令看本机装的 mlx_vlm 源码再选）：
  --format lora    （默认）键 <模块>.lora_a / .lora_b，adapter_config.json 写 rank、alpha、scale=α/r，
                   并附 mlx-lm 风格的 lora_parameters 块。
  --format legacy  键 <模块>.A / .B；adapter_config.json 只写 {"rank", "alpha", "dropout"}，
                   其中 alpha 写的是缩放系数 s（老版 mlx-vlm 的 LoRaLayer 直接乘 alpha，不除以 r）。
真正的检验是 Mac 上的 K0（同一输入，合并前后 / 服务器与 Mac 的 logits 一致性）。

任何映射不了的 PEFT 键都直接报错，绝不悄悄丢弃。

用法：
  python server/convert_adapter.py convert --peft runs/s1/final --out adapters/s1_mlx [--mlx-model <MLX 4B 目录>]
  python server/convert_adapter.py check-names --hf <HF 4B 目录或 index.json> --mlx-model <MLX 4B 目录>
  python server/convert_adapter.py inspect          # 看本机 mlx_vlm 的 LoRA 读法
  python server/convert_adapter.py selftest         # 随机小张量往返
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import model_weight_shapes  # noqa: E402

DEFAULT_MLX_PREFIX = "language_model.model."
LANG_PREFIXES = ("model.language_model.layers.", "model.layers.", "language_model.model.layers.")
ATTN_MLP = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# PEFT 键：<前缀><模块>.lora_A[.<适配器名>].weight
_PEFT_KEY = re.compile(r"^(?P<mod>.+)\.lora_(?P<ab>[AB])(?:\.(?P<name>[^.]+))?\.weight$")


# ---------------------------------------------------------------- 名称映射

def strip_peft_prefix(key: str) -> str:
    return key[len("base_model.model."):] if key.startswith("base_model.model.") else key


def hf_module_to_mlx(hf_module: str, mlx_prefix: str = DEFAULT_MLX_PREFIX) -> str:
    """HF（transformers qwen3_5）模块名 → mlx-vlm 模块名。只允许语言模型里的层。"""
    m = strip_peft_prefix(hf_module)
    for pre in ("model.language_model.", "language_model.model.", "model."):
        if m.startswith(pre + "layers."):
            return mlx_prefix + m[len(pre):]
    if m.startswith(("model.visual.", "visual.", "vision_tower.")):
        raise ValueError(f"视觉塔不应有 LoRA：{hf_module}")
    if "lm_head" in m or "embed_tokens" in m:
        raise ValueError(f"词嵌入 / lm_head 应冻结，不应有 LoRA：{hf_module}")
    raise ValueError(f"映射不了的模块名：{hf_module}")


def parse_peft_key(key: str) -> tuple[str, str]:
    """→ (HF 模块名, "A"|"B")。DoRA、嵌入 LoRA 等不支持的键直接报错。"""
    if "lora_magnitude_vector" in key or "lora_embedding" in key:
        raise ValueError(f"不支持的适配器键（DoRA / 嵌入 LoRA）：{key}")
    m = _PEFT_KEY.match(key)
    if not m:
        raise ValueError(f"不认识的适配器键：{key}")
    return strip_peft_prefix(m.group("mod")), m.group("ab")


def detect_mlx_prefix(mlx_keys: Sequence[str]) -> str:
    for k in mlx_keys:
        i = k.find("layers.0.")
        if i >= 0 and "vision" not in k[:i] and "visual" not in k[:i]:
            return k[:i]
    raise ValueError("MLX 权重里找不到 layers.0.，无法识别前缀")


# ---------------------------------------------------------------- 读写张量

def load_tensors(path: str | Path) -> dict[str, np.ndarray]:
    """读 safetensors（优先 torch，能读 bf16）或 .npz，统一转成 float32 numpy。"""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            return {k: z[k].astype(np.float32) for k in z.files}
    try:
        import torch
        from safetensors.torch import load_file

        return {k: v.to(torch.float32).numpy() for k, v in load_file(str(path)).items()}
    except ImportError:
        from safetensors.numpy import load_file  # 读不了 bf16

        return {k: v.astype(np.float32) for k, v in load_file(str(path)).items()}


def save_tensors(path: str | Path, tensors: dict[str, np.ndarray], dtype: str = "float32",
                 metadata: dict | None = None) -> Path:
    """写 adapters.safetensors；torch / safetensors 都没有时退回 .npz（mlx 也能读，但需改名）。"""
    path = Path(path)
    meta = {k: str(v) for k, v in (metadata or {}).items()}
    try:
        import torch
        from safetensors.torch import save_file

        td = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
        save_file({k: torch.from_numpy(np.ascontiguousarray(v)).to(td).contiguous() for k, v in tensors.items()},
                  str(path), metadata=meta)
        return path
    except ImportError:
        pass
    try:
        from safetensors.numpy import save_file

        if dtype == "bfloat16":
            raise ValueError("没有 torch 时不能写 bfloat16，改用 --dtype float16 或 float32")
        save_file({k: np.ascontiguousarray(v.astype(dtype)) for k, v in tensors.items()}, str(path), metadata=meta)
        return path
    except ImportError:
        alt = path.with_suffix(".npz")
        np.savez(alt, **{k: v.astype(np.float32 if dtype == "bfloat16" else dtype) for k, v in tensors.items()})
        print(f"警告：没有 safetensors，已写成 {alt}（mlx-vlm 需要 adapters.safetensors，到 Mac 上再转一次）",
              file=sys.stderr)
        return alt


# ---------------------------------------------------------------- 转换

def peft_scale(cfg: dict) -> float:
    r, alpha = cfg["r"], cfg.get("lora_alpha", cfg["r"])
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError("适配器用了 rank_pattern / alpha_pattern（每层不同的 r 或 α），转换器不支持")
    if cfg.get("use_dora"):
        raise ValueError("适配器是 DoRA，转换器不支持")
    return alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r


def convert_state(peft_tensors: dict[str, np.ndarray], mlx_prefix: str = DEFAULT_MLX_PREFIX,
                  fmt: str = "lora") -> tuple[dict[str, np.ndarray], list[dict]]:
    """PEFT 张量 → MLX 张量。返回 (张量, 每个模块的映射记录)。"""
    pairs: dict[str, dict[str, np.ndarray]] = {}
    for key, val in peft_tensors.items():
        mod, ab = parse_peft_key(key)
        if ab in pairs.setdefault(mod, {}):
            raise ValueError(f"模块 {mod} 的 lora_{ab} 出现两次（多个适配器名？）")
        pairs[mod][ab] = val
    names = ("lora_a", "lora_b") if fmt == "lora" else ("A", "B")
    out, rows = {}, []
    for mod in sorted(pairs):
        p = pairs[mod]
        if set(p) != {"A", "B"}:
            raise ValueError(f"模块 {mod} 缺少 lora_A 或 lora_B")
        A, B = p["A"], p["B"]
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
            raise ValueError(f"模块 {mod} 形状不对：A{A.shape} B{B.shape}")
        mlx_mod = hf_module_to_mlx(mod, mlx_prefix)
        out[f"{mlx_mod}.{names[0]}"] = np.ascontiguousarray(A.T)  # (in, r)
        out[f"{mlx_mod}.{names[1]}"] = np.ascontiguousarray(B.T)  # (r, out)
        rows.append({"hf": mod, "mlx": mlx_mod, "in": int(A.shape[1]), "out": int(B.shape[0]),
                     "r": int(A.shape[0])})
    return out, rows


def check_against_mlx(rows: Sequence[dict], mlx_shapes: dict, group_size: int | None = None) -> list[str]:
    """每个映射后的模块必须在 MLX 权重里有 <模块>.weight，且输出 / 输入维度对得上（兼容量化权重）。"""
    probs = []
    for r in rows:
        w = r["mlx"] + ".weight"
        if w not in mlx_shapes:
            probs.append(f"MLX 里没有 {w}")
            continue
        shp = mlx_shapes[w]
        if shp is None:
            continue
        if shp[0] != r["out"]:
            probs.append(f"{w} 输出维度 {shp[0]} ≠ LoRA 的 {r['out']}")
        sc = mlx_shapes.get(r["mlx"] + ".scales")
        if sc is not None and group_size:
            n_in = sc[1] * group_size
        elif sc is None:
            n_in = shp[1]
        else:
            n_in = None
        if n_in is not None and n_in != r["in"]:
            probs.append(f"{w} 输入维度 {n_in} ≠ LoRA 的 {r['in']}")
    return probs


def mlx_group_size(mlx_dir: Path) -> int | None:
    cfg = mlx_dir / "config.json"
    if not cfg.is_file():
        return None
    c = json.loads(cfg.read_text(encoding="utf-8"))
    q = c.get("quantization") or c.get("quantization_config") or {}
    return q.get("group_size")


def convert(peft_dir: str | Path, out_dir: str | Path, fmt: str = "lora", mlx_model: str | Path | None = None,
            mlx_prefix: str | None = None, dtype: str = "float32") -> dict:
    peft_dir, out_dir = Path(peft_dir), Path(out_dir)
    cfg = json.loads((peft_dir / "adapter_config.json").read_text(encoding="utf-8"))
    wfile = next((peft_dir / n for n in ("adapter_model.safetensors", "adapter_model.npz")
                  if (peft_dir / n).is_file()), None)
    if wfile is None:
        raise FileNotFoundError(f"{peft_dir} 里没有 adapter_model.safetensors")
    scale = peft_scale(cfg)
    mlx_shapes = None
    if mlx_model is not None:
        mlx_shapes = model_weight_shapes(mlx_model)
        mlx_prefix = mlx_prefix or detect_mlx_prefix(list(mlx_shapes))
    mlx_prefix = mlx_prefix or DEFAULT_MLX_PREFIX
    tensors, rows = convert_state(load_tensors(wfile), mlx_prefix, fmt)
    problems = []
    if mlx_shapes is not None:
        problems = check_against_mlx(rows, mlx_shapes, mlx_group_size(Path(mlx_model)))
        if problems:
            raise ValueError("和 MLX 底座对不上：\n" + "\n".join(problems[:40]))
    out_dir.mkdir(parents=True, exist_ok=True)
    wpath = save_tensors(out_dir / "adapters.safetensors", tensors, dtype,
                         {"format": fmt, "scale": scale, "source": "peft"})
    r = int(cfg["r"])
    suffixes = sorted({row["mlx"].rsplit(".", 1)[-1] for row in rows})
    # mlx-lm 风格的 keys 是相对每层的路径（如 self_attn.q_proj）；两种写法都放，哪种匹配方式都不会漏
    rel = sorted({re.sub(r"^.*?layers\.\d+\.", "", row["mlx"]) for row in rows})
    layers = sorted({int(m.group(1)) for row in rows if (m := re.search(r"layers\.(\d+)\.", row["mlx"]))})
    if fmt == "lora":
        acfg = {"format": "mlx-lora", "rank": r, "alpha": cfg.get("lora_alpha"), "scale": scale, "dropout": 0.0,
                "fine_tune_type": "lora", "num_layers": len(layers),
                "lora_parameters": {"rank": r, "scale": scale, "dropout": 0.0, "keys": rel + suffixes},
                "target_modules": [row["mlx"] for row in rows],
                "base_model_name_or_path": cfg.get("base_model_name_or_path"),
                "peft_lora_dropout": cfg.get("lora_dropout")}
    else:
        acfg = {"rank": r, "alpha": scale, "dropout": 0.0}
    (out_dir / "adapter_config.json").write_text(json.dumps(acfg, ensure_ascii=False, indent=1), encoding="utf-8")
    report = {"peft_dir": str(peft_dir), "weights": str(wpath), "format": fmt, "scale": scale, "rank": r,
              "mlx_prefix": mlx_prefix, "n_modules": len(rows), "suffixes": suffixes, "layers": layers,
              "checked_against_mlx": mlx_shapes is not None, "modules": rows}
    (out_dir / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    return report


# ---------------------------------------------------------------- 名称核对（训练前做）

def expected_lora_modules(hf_keys: Sequence[str], suffixes: Sequence[str]) -> list[str]:
    """HF 底座里会被 LoRA 的模块（语言模型里、名字以 suffixes 结尾的线性层）。"""
    mods = set()
    for k in hf_keys:
        if not k.endswith(".weight"):
            continue
        mod = k[: -len(".weight")]
        if mod.rsplit(".", 1)[-1] not in suffixes:
            continue
        if "visual" in mod or "vision" in mod or "lm_head" in mod or "embed" in mod:
            continue
        # 只要语言模型的层（检查点里的 mtp.* 多 token 预测层加载时会被忽略，不挂 LoRA）
        if mod.startswith(LANG_PREFIXES):
            mods.add(mod)
    return sorted(mods)


def check_names(hf_model: str | Path, mlx_model: str | Path, suffixes: Sequence[str] = ATTN_MLP) -> dict:
    hf_shapes = model_weight_shapes(hf_model)
    mlx_shapes = model_weight_shapes(mlx_model)
    prefix = detect_mlx_prefix(list(mlx_shapes))
    mods = expected_lora_modules(list(hf_shapes), suffixes)
    rows, missing = [], []
    for mod in mods:
        mlx_mod = hf_module_to_mlx(mod, prefix)
        shp = hf_shapes.get(mod + ".weight")
        rows.append({"hf": mod, "mlx": mlx_mod, "out": shp[0] if shp else None, "in": shp[1] if shp else None})
        if mlx_mod + ".weight" not in mlx_shapes:
            missing.append(mlx_mod)
    probs = check_against_mlx([r for r in rows if r["out"] is not None], mlx_shapes, mlx_group_size(Path(mlx_model)))
    by_suffix = {}
    for r in rows:
        s = r["hf"].rsplit(".", 1)[-1]
        by_suffix[s] = by_suffix.get(s, 0) + 1
    return {"mlx_prefix": prefix, "n_modules": len(rows), "by_suffix": by_suffix, "missing": missing,
            "problems": probs, "ok": not missing and not probs}


# ---------------------------------------------------------------- 本机 mlx_vlm 读法探测

def inspect_mlx_vlm() -> dict:
    """不导入 mlx_vlm，只找到它的源码目录，搜 LoRA 层和适配器加载的写法，给出 --format 建议。"""
    import importlib.util

    spec = importlib.util.find_spec("mlx_vlm")
    if spec is None or not spec.submodule_search_locations:
        return {"found": False, "advice": "本机没装 mlx_vlm；装好后再运行 inspect"}
    root = Path(list(spec.submodule_search_locations)[0])
    hits = []
    pat = re.compile(r"self\.(lora_a|lora_b|A|B|scale|alpha)\s*=|adapter_config|adapters\.safetensors|"
                     r"def (apply_lora_layers|get_peft_model|load_adapter)|class \w*Lo[Rr][Aa]\w*|"
                     r"linear_to_lora_layers|lora_parameters|\bkeys\b|num_layers")
    for f in sorted(root.rglob("*.py")):
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(lines, 1):
            if pat.search(line):
                hits.append(f"{f.relative_to(root)}:{i}: {line.strip()[:120]}")
    text = "\n".join(hits)
    has_lora_a = "self.lora_a" in text
    has_A = re.search(r"self\.A\s*=", text) is not None
    advice = ("lora" if has_lora_a and not has_A else "legacy" if has_A and not has_lora_a else "不确定：两种写法都有/都没有，读下面的源码行再定")
    return {"found": True, "root": str(root), "advice_format": advice, "hits": hits[:80]}


# ---------------------------------------------------------------- 自检（与测试同一套数学）

def peft_forward(x, W, A, B, scale):
    return x @ W.T + scale * (x @ A.T) @ B.T


def mlx_forward(x, W, lora_a, lora_b, scale):
    return x @ W.T + scale * (x @ lora_a) @ lora_b


def selftest(seed: int = 0) -> float:
    import tempfile

    rng = np.random.default_rng(seed)
    r, d_in, d_out, alpha = 4, 12, 10, 8
    W = rng.normal(size=(d_out, d_in)).astype(np.float32)
    A = rng.normal(size=(r, d_in)).astype(np.float32)
    B = rng.normal(size=(d_out, r)).astype(np.float32)
    x = rng.normal(size=(3, d_in)).astype(np.float32)
    mod = "model.language_model.layers.0.self_attn.q_proj"
    with tempfile.TemporaryDirectory() as td:
        pd = Path(td) / "peft"
        pd.mkdir()
        save_tensors(pd / "adapter_model.safetensors",
                     {f"base_model.model.{mod}.lora_A.weight": A, f"base_model.model.{mod}.lora_B.weight": B})
        (pd / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha}), encoding="utf-8")
        rep = convert(pd, Path(td) / "mlx")
        t = load_tensors(Path(td) / "mlx" / "adapters.safetensors")
        mlx_mod = DEFAULT_MLX_PREFIX + "layers.0.self_attn.q_proj"
        y0 = peft_forward(x, W, A, B, alpha / r)
        y1 = mlx_forward(x, W, t[mlx_mod + ".lora_a"], t[mlx_mod + ".lora_b"], rep["scale"])
    return float(np.max(np.abs(y0 - y1)))


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="PEFT LoRA → mlx-vlm 适配器")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--peft", required=True, help="PEFT 适配器目录（adapter_config.json + adapter_model.safetensors）")
    c.add_argument("--out", required=True)
    c.add_argument("--format", choices=("lora", "legacy"), default="lora")
    c.add_argument("--mlx-model", default=None, help="MLX 底座目录：核对模块名和形状，并自动识别前缀")
    c.add_argument("--mlx-prefix", default=None)
    c.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    n = sub.add_parser("check-names", help="训练前：HF 底座里要 LoRA 的模块在 MLX 底座里是否都有")
    n.add_argument("--hf", required=True, help="HF 底座目录、index.json 或 .safetensors")
    n.add_argument("--mlx-model", required=True)
    n.add_argument("--targets", default=",".join(ATTN_MLP), help="逗号分隔的模块名后缀")
    sub.add_parser("inspect", help="看本机 mlx_vlm 的 LoRA 读法")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        rep = convert(a.peft, a.out, a.format, a.mlx_model, a.mlx_prefix, a.dtype)
        rep.pop("modules")
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    elif a.cmd == "check-names":
        rep = check_names(a.hf, a.mlx_model, [s for s in a.targets.split(",") if s])
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 0 if rep["ok"] else 1
    elif a.cmd == "inspect":
        print(json.dumps(inspect_mlx_vlm(), ensure_ascii=False, indent=1))
    else:
        err = selftest()
        print(f"往返最大误差 {err:.2e}")
        return 0 if err < 1e-4 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
