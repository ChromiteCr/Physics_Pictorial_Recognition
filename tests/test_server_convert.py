"""PEFT → mlx-vlm 适配器转换：随机小张量往返、名称映射、出错路径、和 MLX 底座的核对。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import convert_adapter as ca  # noqa: E402

LAYER_MODS = ["self_attn.q_proj", "self_attn.k_proj", "mlp.down_proj"]


def _fake_peft(tmp_path, r=4, alpha=8, n_layers=2, d=12, seed=0, extra=None, cfg_extra=None):
    """造一个小 PEFT 适配器，返回 (目录, {HF 模块: (A, B)})。"""
    rng = np.random.default_rng(seed)
    tensors, mats = {}, {}
    for i in range(n_layers):
        for j, m in enumerate(LAYER_MODS):
            d_out = d + 2 * j  # 不同模块输出维度不同，便于发现转置错误
            mod = f"model.language_model.layers.{i}.{m}"
            A = rng.normal(size=(r, d)).astype(np.float32)
            B = rng.normal(size=(d_out, r)).astype(np.float32)
            tensors[f"base_model.model.{mod}.lora_A.weight"] = A
            tensors[f"base_model.model.{mod}.lora_B.weight"] = B
            mats[mod] = (A, B)
    tensors.update(extra or {})
    pd = tmp_path / "peft"
    pd.mkdir()
    ca.save_tensors(pd / "adapter_model.safetensors", tensors)
    cfg = {"r": r, "lora_alpha": alpha, "lora_dropout": 0.05, "base_model_name_or_path": "Qwen/Qwen3.5-4B"}
    cfg.update(cfg_extra or {})
    (pd / "adapter_config.json").write_text(json.dumps(cfg))
    return pd, mats


def test_round_trip_numeric(tmp_path):
    r, alpha = 4, 8
    pd, mats = _fake_peft(tmp_path, r, alpha)
    rep = ca.convert(pd, tmp_path / "mlx")
    assert rep["scale"] == pytest.approx(alpha / r)
    t = ca.load_tensors(tmp_path / "mlx" / "adapters.safetensors")
    rng = np.random.default_rng(1)
    for mod, (A, B) in mats.items():
        W = rng.normal(size=(B.shape[0], A.shape[1])).astype(np.float32)
        x = rng.normal(size=(5, A.shape[1])).astype(np.float32)
        mlx_mod = ca.hf_module_to_mlx(mod)
        a, b = t[mlx_mod + ".lora_a"], t[mlx_mod + ".lora_b"]
        assert a.shape == (A.shape[1], r) and b.shape == (r, B.shape[0])
        y_peft = x @ W.T + (alpha / r) * x @ A.T @ B.T
        y_mlx = x @ W.T + rep["scale"] * (x @ a) @ b
        np.testing.assert_allclose(y_mlx, y_peft, rtol=1e-5, atol=1e-5)
    cfg = json.loads((tmp_path / "mlx" / "adapter_config.json").read_text())
    assert cfg["rank"] == r and cfg["scale"] == pytest.approx(2.0)
    assert cfg["lora_parameters"]["keys"] == ["down_proj", "k_proj", "q_proj"]


def test_selftest_cli_math():
    assert ca.selftest() < 1e-4


def test_legacy_format_scale_goes_into_alpha(tmp_path):
    pd, mats = _fake_peft(tmp_path, r=16, alpha=32)
    ca.convert(pd, tmp_path / "mlx", fmt="legacy")
    cfg = json.loads((tmp_path / "mlx" / "adapter_config.json").read_text())
    assert cfg == {"rank": 16, "alpha": pytest.approx(2.0), "dropout": 0.0}  # 老版直接乘 alpha
    t = ca.load_tensors(tmp_path / "mlx" / "adapters.safetensors")
    mod = "model.language_model.layers.1.mlp.down_proj"
    np.testing.assert_allclose(t[ca.hf_module_to_mlx(mod) + ".A"], mats[mod][0].T)
    np.testing.assert_allclose(t[ca.hf_module_to_mlx(mod) + ".B"], mats[mod][1].T)


def test_rslora_scale():
    assert ca.peft_scale({"r": 16, "lora_alpha": 32, "use_rslora": True}) == pytest.approx(8.0)
    with pytest.raises(ValueError):
        ca.peft_scale({"r": 16, "lora_alpha": 32, "rank_pattern": {"q_proj": 8}})
    with pytest.raises(ValueError):
        ca.peft_scale({"r": 16, "lora_alpha": 32, "use_dora": True})


def test_name_map():
    f = ca.hf_module_to_mlx
    assert f("base_model.model.model.language_model.layers.3.self_attn.q_proj") == \
        "language_model.model.layers.3.self_attn.q_proj"
    assert f("model.language_model.layers.0.linear_attn.in_proj_qkv") == \
        "language_model.model.layers.0.linear_attn.in_proj_qkv"
    assert f("model.layers.7.mlp.gate_proj") == "language_model.model.layers.7.mlp.gate_proj"  # 纯文本 CausalLM
    assert f("model.language_model.layers.1.mlp.up_proj", "model.") == "model.layers.1.mlp.up_proj"
    for bad in ("model.visual.blocks.0.attn.qkv", "lm_head", "model.language_model.embed_tokens"):
        with pytest.raises(ValueError):
            f(bad)


def test_parse_peft_key_variants():
    assert ca.parse_peft_key("base_model.model.model.language_model.layers.0.mlp.up_proj.lora_B.weight") == \
        ("model.language_model.layers.0.mlp.up_proj", "B")
    assert ca.parse_peft_key("base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight") == \
        ("model.language_model.layers.0.mlp.up_proj", "A")
    for bad in ("base_model.model.x.lora_magnitude_vector.weight", "base_model.model.x.bias"):
        with pytest.raises(ValueError):
            ca.parse_peft_key(bad)


def test_unmappable_key_fails_loudly(tmp_path):
    extra = {"base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight": np.zeros((4, 12), np.float32),
             "base_model.model.model.visual.blocks.0.attn.qkv.lora_B.weight": np.zeros((36, 4), np.float32)}
    pd, _ = _fake_peft(tmp_path, extra=extra)
    with pytest.raises(ValueError, match="视觉塔"):
        ca.convert(pd, tmp_path / "mlx")


def test_missing_half_fails(tmp_path):
    extra = {"base_model.model.model.language_model.layers.5.mlp.up_proj.lora_A.weight": np.zeros((4, 12), np.float32)}
    pd, _ = _fake_peft(tmp_path, extra=extra)
    with pytest.raises(ValueError, match="缺少"):
        ca.convert(pd, tmp_path / "mlx")


def _fake_mlx_model(tmp_path, mods_shapes, quant=False):
    """造一个只有权重头的 MLX 底座目录（值无所谓）。"""
    md = tmp_path / "mlx_base"
    md.mkdir()
    t = {}
    for mod, (d_out, d_in) in mods_shapes.items():
        if quant:  # 4 bit、group 4：weight (out, in/8)，scales (out, in/4)
            t[mod + ".weight"] = np.zeros((d_out, d_in // 8), np.float32)
            t[mod + ".scales"] = np.zeros((d_out, d_in // 4), np.float32)
        else:
            t[mod + ".weight"] = np.zeros((d_out, d_in), np.float32)
    t["language_model.model.embed_tokens.weight"] = np.zeros((3, 4), np.float32)
    ca.save_tensors(md / "model.safetensors", t)
    if quant:
        (md / "config.json").write_text(json.dumps({"quantization": {"group_size": 4, "bits": 4}}))
    return md


@pytest.mark.parametrize("quant", [False, True])
def test_check_against_mlx_model(tmp_path, quant):
    pd, mats = _fake_peft(tmp_path, d=16)
    shapes = {ca.hf_module_to_mlx(m): (B.shape[0], A.shape[1]) for m, (A, B) in mats.items()}
    md = _fake_mlx_model(tmp_path, shapes, quant)
    rep = ca.convert(pd, tmp_path / "out", mlx_model=md)
    assert rep["checked_against_mlx"] and rep["mlx_prefix"] == "language_model.model."
    # 少一个模块 → 报错
    md2 = tmp_path / "less"
    (tmp_path / "less").mkdir()
    shapes.pop(next(iter(shapes)))
    t = {k + ".weight": np.zeros(v, np.float32) for k, v in shapes.items()}
    ca.save_tensors(md2 / "model.safetensors", t)
    with pytest.raises(ValueError, match="MLX 里没有"):
        ca.convert(pd, tmp_path / "out2", mlx_model=md2)


def test_check_names(tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    keys = {}
    for i in range(2):
        for m in ("self_attn.q_proj", "mlp.gate_proj", "linear_attn.in_proj_qkv"):
            keys[f"model.language_model.layers.{i}.{m}.weight"] = np.zeros((8, 6), np.float32)
    keys["model.visual.blocks.0.attn.qkv.weight"] = np.zeros((9, 3), np.float32)
    keys["lm_head.weight"] = np.zeros((5, 6), np.float32)
    ca.save_tensors(hf / "model.safetensors", keys)
    mlx_ok = _fake_mlx_model(tmp_path, {f"language_model.model.layers.{i}.{m}": (8, 6)
                                        for i in range(2) for m in ("self_attn.q_proj", "mlp.gate_proj")})
    rep = ca.check_names(hf, mlx_ok)
    assert rep["ok"] and rep["n_modules"] == 4 and rep["by_suffix"] == {"q_proj": 2, "gate_proj": 2}
    # 想加 linear_attn 的投影，但 MLX 底座里没有同名层（例如被融合）→ 不通过
    rep2 = ca.check_names(hf, mlx_ok, ["q_proj", "gate_proj", "in_proj_qkv"])
    assert not rep2["ok"] and len(rep2["missing"]) == 2


def test_safetensors_bf16_read(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    x = torch.randn(3, 4)
    save_file({"a": x.to(torch.bfloat16)}, str(tmp_path / "w.safetensors"))
    t = ca.load_tensors(tmp_path / "w.safetensors")
    np.testing.assert_allclose(t["a"], x.to(torch.bfloat16).float().numpy())
