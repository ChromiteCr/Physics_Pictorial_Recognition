#!/usr/bin/env bash
# RTX 5090 服务器一次性引导：探网 → 两个 venv（训练 / 服务）→ 下载权重 → GPU 检查 → 0.8B 冒烟（输出丢弃）。
# 用法（在 tmux 里）：  bash ~/ld/server/bootstrap.sh
# 只跑部分步骤：        STEPS="download check" bash ~/ld/server/bootstrap.sh
# 可调环境变量：
#   WORK=~/ld                               工作目录（环境、模型、日志都在这里）
#   PYTHON=python3                          建 venv 用的解释器（≥ 3.10）
#   PIP_MIRROR=auto|tuna|aliyun|ustc|pypi   PyPI 源（auto = 依次试 tuna、aliyun、ustc、pypi）
#   TORCH_CUDA=cu128                        torch 轮子的 CUDA 版本（sm_120 需要 ≥ cu128）
#   TORCH_SPEC="torch torchvision"          例如 "torch==2.9.1 torchvision==0.24.1"（固定版本）
#   TRANSFORMERS_SPEC="transformers>=5.2"   qwen3_5 架构需要 ≥ 5.2
#   VLLM_SPEC=vllm                          例如 "vllm==0.x.y"
#   MODELS="Qwen/Qwen3.5-4B Qwen/Qwen3.5-0.8B"
#   HF_MIRROR=https://hf-mirror.com         ModelScope 失败时 HF 兜底用的端点（空 = 官方）
#   SKIP_FASTPATH=1                         不装 flash-linear-attention / causal-conv1d
#   SKIP_VLLM=1                             不建服务环境（评测改用 HF 后端）
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HOME/ld}
PY=${PYTHON:-python3}
mkdir -p "$WORK"/logs "$WORK"/models "$WORK"/envs
TS=$(date +%Y%m%d_%H%M%S)
LOG="$WORK/logs/bootstrap_$TS.log"
exec > >(tee -a "$LOG") 2>&1
STEPS=${STEPS:-"preflight net venv-train venv-serve download check smoke"}
TORCH_CUDA=${TORCH_CUDA:-cu128}
TORCH_SPEC=${TORCH_SPEC:-"torch torchvision"}
TRANSFORMERS_SPEC=${TRANSFORMERS_SPEC:-"transformers>=5.2"}
VLLM_SPEC=${VLLM_SPEC:-vllm}
MODELS=${MODELS:-"Qwen/Qwen3.5-4B Qwen/Qwen3.5-0.8B"}
HF_MIRROR=${HF_MIRROR:-https://hf-mirror.com}
T="$WORK/envs/train"
S="$WORK/envs/serve"
NET_ENV="$WORK/logs/net.env"

step() { echo; echo "======== [$(date +%H:%M:%S)] $* ========"; }
die() { echo "错误：$*"; echo "日志：$LOG"; exit 1; }
warn() { echo "警告：$*"; }
has_step() { [[ " $STEPS " == *" $1 "* ]]; }
probe() {  # 名称 URL → 打印 HTTP 码和耗时，返回 0 表示可达
  local r
  r=$(curl -s -o /dev/null -w "%{http_code} %{time_total}s" --max-time 10 "$2" 2>/dev/null) || r="fail"
  echo "  $1  $2  →  $r"
  [[ "$r" == 2* || "$r" == 3* ]]
}
pipi() {  # venv 里 pip install（带源和 torch 约束）
  local venv=$1; shift
  "$venv/bin/python" -m pip install --progress-bar off -i "$PIP_INDEX" "$@"
}

echo "bootstrap 开始：$(date)  WORK=$WORK  STEPS=$STEPS"

# ------------------------------------------------------------------ 1. 预检
if has_step preflight; then
  step "1/7 预检：GPU、驱动、磁盘、Python"
  command -v nvidia-smi >/dev/null || die "没有 nvidia-smi"
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader || die "nvidia-smi 失败"
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  DRV_CUDA=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | awk '{print $3}')
  echo "  计算能力 $CAP，驱动支持的 CUDA $DRV_CUDA"
  [[ "$CAP" == "12.0" ]] || warn "不是 sm_120（RTX 5090 应为 12.0），下面的版本选择可能不合适"
  if [[ -n "$DRV_CUDA" ]] && awk "BEGIN{exit !($DRV_CUDA < 12.8)}"; then
    die "驱动只支持 CUDA $DRV_CUDA，5090 需要 ≥ 12.8：换镜像或换机器"
  fi
  df -h "$WORK" | tail -1
  FREE_G=$(df -BG "$WORK" | tail -1 | awk '{print $4}' | tr -d G)
  [[ "${FREE_G:-0}" -ge 60 ]] || warn "可用磁盘 ${FREE_G}G < 60G（两个底座 + 两个合并模型 + 环境约需 50–60G）"
  free -g | head -2; echo "  CPU 核数 $(nproc)"
  $PY -c 'import sys; assert sys.version_info >= (3, 10), sys.version; print("  Python", sys.version.split()[0])' \
    || die "需要 Python ≥ 3.10（设 PYTHON=...）"
  command -v nvcc >/dev/null && nvcc --version | tail -1 || echo "  没有 nvcc（causal-conv1d 只能用预编译轮子）"
  command -v tmux >/dev/null || warn "没有 tmux：正式训练务必在 tmux / screen 里跑"
fi

# ------------------------------------------------------------------ 2. 探网、选源
if has_step net; then
  step "2/7 探网：PyPI 镜像、PyTorch 源、ModelScope、Hugging Face"
  declare -A IDX=([tuna]=https://pypi.tuna.tsinghua.edu.cn/simple [aliyun]=https://mirrors.aliyun.com/pypi/simple
                  [ustc]=https://mirrors.ustc.edu.cn/pypi/simple [pypi]=https://pypi.org/simple)
  MIRROR=${PIP_MIRROR:-auto}
  if [[ "$MIRROR" == auto ]]; then
    for m in tuna aliyun ustc pypi; do
      if probe "$m" "${IDX[$m]}/pip/"; then MIRROR=$m; break; fi
    done
  fi
  [[ "$MIRROR" != auto ]] || die "所有 PyPI 源都连不上"
  PIP_INDEX=${IDX[$MIRROR]}
  TORCH_MODE=index; TORCH_URL=https://download.pytorch.org/whl/$TORCH_CUDA
  if ! probe torch-official "$TORCH_URL/torch/"; then
    TORCH_MODE=findlinks; TORCH_URL=https://mirrors.aliyun.com/pytorch-wheels/$TORCH_CUDA/
    probe torch-aliyun "$TORCH_URL" || warn "两个 PyTorch 源都不通，torch 只能从 PyPI 镜像装（再核对 CUDA 版本）"
  fi
  probe modelscope https://www.modelscope.cn/api/v1/models/Qwen/Qwen3.5-4B && MS_OK=1 || MS_OK=0
  probe huggingface https://huggingface.co/api/models/Qwen/Qwen3.5-4B && HF_OK=1 || HF_OK=0
  probe hf-mirror "$HF_MIRROR/api/models/Qwen/Qwen3.5-4B" && HFM_OK=1 || HFM_OK=0
  probe github https://github.com || true
  cat > "$NET_ENV" <<EOF
PIP_INDEX=$PIP_INDEX
TORCH_MODE=$TORCH_MODE
TORCH_URL=$TORCH_URL
MS_OK=$MS_OK
HF_OK=$HF_OK
HFM_OK=$HFM_OK
EOF
  echo "  选用：PyPI=$PIP_INDEX  torch=$TORCH_MODE $TORCH_URL  ModelScope=$MS_OK HF=$HF_OK HF镜像=$HFM_OK"
  [[ "$MS_OK$HF_OK$HFM_OK" != 000 ]] || warn "ModelScope 和 HF 都连不上：权重拉不到时按 README 改走 S1-mac"
fi
[[ -f "$NET_ENV" ]] && source "$NET_ENV"
PIP_INDEX=${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}

mkvenv() {
  local d=$1
  if [[ ! -x "$d/bin/python" ]]; then
    $PY -m venv "$d" 2>/dev/null || { $PY -m pip install -q --user virtualenv -i "$PIP_INDEX" && $PY -m virtualenv "$d"; } \
      || die "建 venv 失败：$d（Debian 系可 apt install python3-venv，或用 conda 的 python）"
  fi
  "$d/bin/python" -m pip install -q --progress-bar off -U pip wheel setuptools -i "$PIP_INDEX" || die "升级 pip 失败"
}

# ------------------------------------------------------------------ 3. 训练环境
if has_step venv-train; then
  step "3/7 训练环境 $T：torch($TORCH_CUDA) + transformers + peft + trl + accelerate + datasets"
  mkvenv "$T"
  if [[ "${TORCH_MODE:-index}" == index ]]; then
    "$T/bin/python" -m pip install --progress-bar off $TORCH_SPEC --index-url "$TORCH_URL" \
      || pipi "$T" $TORCH_SPEC -f "https://mirrors.aliyun.com/pytorch-wheels/$TORCH_CUDA/" || die "torch 安装失败"
  else
    pipi "$T" $TORCH_SPEC -f "$TORCH_URL" || die "torch 安装失败"
  fi
  "$T/bin/python" -m pip freeze | grep -E '^(torch|torchvision|triton)==' > "$WORK/envs/train_torch_constraints.txt"
  echo "  torch 约束：$(tr '\n' ' ' < "$WORK/envs/train_torch_constraints.txt")"
  "$T/bin/python" - <<'EOF' || die "torch 不支持 sm_120 或 CUDA < 12.8：换 TORCH_CUDA / TORCH_SPEC"
import torch
print("  torch", torch.__version__, "CUDA", torch.version.cuda, "arch", torch.cuda.get_arch_list())
major, minor = map(int, (torch.version.cuda or "0.0").split(".")[:2])
assert (major, minor) >= (12, 8), torch.version.cuda
assert any(a.endswith("_120") for a in torch.cuda.get_arch_list()), "没有 sm_120 内核"
EOF
  pipi "$T" -c "$WORK/envs/train_torch_constraints.txt" "$TRANSFORMERS_SPEC" peft trl accelerate datasets pillow \
    safetensors modelscope huggingface_hub numpy || die "训练依赖安装失败"
  if [[ "${SKIP_FASTPATH:-0}" != 1 ]]; then
    echo "  装线性注意力快速路径（失败不致命，只是慢）"
    pipi "$T" -c "$WORK/envs/train_torch_constraints.txt" flash-linear-attention || warn "flash-linear-attention 没装上"
    timeout 1800 "$T/bin/python" -m pip install --progress-bar off -i "$PIP_INDEX" \
      -c "$WORK/envs/train_torch_constraints.txt" --no-build-isolation causal-conv1d || warn "causal-conv1d 没装上"
    if ! "$T/bin/python" -c "from transformers.models.qwen3_5 import modeling_qwen3_5" 2>/dev/null; then
      warn "装了快速路径后 transformers 的 qwen3_5 导入失败，卸掉它们"
      "$T/bin/python" -m pip uninstall -y flash-linear-attention fla-core causal-conv1d
    fi
  fi
  "$T/bin/python" -c "from transformers.models.qwen3_5 import modeling_qwen3_5" || die "transformers 没有 qwen3_5（需要 ≥ 5.2）"
  "$T/bin/python" -c "import sys; sys.path.insert(0, '$HERE'); import train_sft, json; print('  快速路径', json.dumps(train_sft.fast_path_info()))"
  "$T/bin/python" -c "import sys; sys.path.insert(0, '$HERE'); import train_sft, json; print(json.dumps(train_sft.versions()))" \
    > "$WORK/logs/versions_train.json"
  "$T/bin/python" -m pip freeze > "$WORK/logs/pip_freeze_train.txt"
  echo "  版本：$(cat "$WORK/logs/versions_train.json")"
fi

# ------------------------------------------------------------------ 4. 服务环境（vLLM）
SERVE_BACKEND=hf
if has_step venv-serve && [[ "${SKIP_VLLM:-0}" != 1 ]]; then
  step "4/7 服务环境 $S：vLLM（装不上就用 HF 生成兜底）"
  mkvenv "$S"
  if pipi "$S" "$VLLM_SPEC" pillow modelscope && "$S/bin/python" - <<'EOF'
import torch, vllm, transformers
print("  vllm", vllm.__version__, "torch", torch.__version__, "CUDA", torch.version.cuda, "transformers", transformers.__version__)
assert any(a.endswith("_120") for a in torch.cuda.get_arch_list()), "vLLM 带的 torch 没有 sm_120"
EOF
  then
    SERVE_BACKEND=vllm
    "$S/bin/python" -m pip freeze > "$WORK/logs/pip_freeze_serve.txt"
    "$S/bin/python" -c "import json, vllm, torch, transformers; print(json.dumps({'vllm': vllm.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__}))" \
      > "$WORK/logs/versions_serve.json"
  else
    warn "vLLM 不可用：评测用训练环境 + --backend hf（慢一些，结果同为贪心）"
  fi
fi
if has_step venv-serve; then
  echo "SERVE_BACKEND=$SERVE_BACKEND" > "$WORK/logs/serve.env"
elif [[ -f "$WORK/logs/serve.env" ]]; then
  source "$WORK/logs/serve.env"
fi

# ------------------------------------------------------------------ 5. 下载权重
if has_step download; then
  step "5/7 下载权重：$MODELS（ModelScope 优先，HF 兜底）"
  for MID in $MODELS; do
    DST="$WORK/models/${MID##*/}"
    if [[ -f "$DST/DOWNLOAD.json" ]]; then echo "  已有 $DST"; continue; fi
    "$T/bin/python" - "$MID" "$DST" <<'EOF' && OK=1 || OK=0
import sys
from modelscope import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
EOF
    SRC=modelscope
    if [[ $OK != 1 ]]; then
      warn "ModelScope 下载 $MID 失败，改走 HF（$HF_MIRROR）"
      HF_ENDPOINT=$HF_MIRROR "$T/bin/python" - "$MID" "$DST" <<'EOF' && OK=1 || OK=0
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
EOF
      SRC="hf:$HF_MIRROR"
    fi
    [[ $OK == 1 ]] || die "$MID 下载失败：按 README「权重拉不到」处理（S1-mac）"
    "$T/bin/python" - "$DST" "$MID" "$SRC" <<'EOF' || die "$MID 校验失败"
import hashlib, json, sys
from pathlib import Path
d, mid, src = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
cfg = json.loads((d / "config.json").read_text())
assert cfg.get("model_type") == "qwen3_5", f"model_type={cfg.get('model_type')}，不是 qwen3_5"
files = sorted(p for p in d.rglob("*") if p.is_file() and ".git" not in p.parts)
st = [p for p in files if p.suffix == ".safetensors"]
assert st, "没有 safetensors 权重"
assert (d / "chat_template.jinja").is_file() or "chat_template" in (d / "tokenizer_config.json").read_text(), "没有对话模板"
small = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.stat().st_size < 50_000_000}
info = {"model_id": mid, "source": src, "total_gb": sum(p.stat().st_size for p in files) / 2**30,
        "files": {str(p.relative_to(d)): p.stat().st_size for p in files}, "sha256_small_files": small}
(d / "DOWNLOAD.json").write_text(json.dumps(info, indent=1))
print(f"  {mid}: {info['total_gb']:.2f} GB，{len(st)} 个权重分片，来源 {src}")
EOF
  done
fi

# ------------------------------------------------------------------ 6. GPU 检查
if has_step check; then
  step "6/7 GPU 检查（训练环境；有 vLLM 时服务环境也查）"
  for V in "$T" "$S"; do
    [[ -x "$V/bin/python" ]] || continue
    "$V/bin/python" - <<'EOF' || die "GPU 检查失败（$V）"
import time, torch
assert torch.cuda.is_available(), "CUDA 不可用"
cap = torch.cuda.get_device_capability(0)
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(3): a @ b
torch.cuda.synchronize(); t = time.time()
for _ in range(20): a @ b
torch.cuda.synchronize(); dt = time.time() - t
free, total = torch.cuda.mem_get_info()
print(f"  {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}  bf16 matmul {20 * 2 * 8192**3 / dt / 1e12:.0f} TFLOPS  "
      f"显存 {free / 2**30:.1f}/{total / 2**30:.1f} GB 空闲")
EOF
  done
fi

# ------------------------------------------------------------------ 7. 0.8B 冒烟（输出丢弃）
if has_step smoke; then
  step "7/7 冒烟：0.8B 上 10 步 LoRA + 打乱标签对照 10 步 + 评测 + 合并 + 转 MLX（输出全部丢弃，只留日志）"
  M08="$WORK/models/Qwen3.5-0.8B"
  [[ -d "$M08" ]] || die "没有 $M08"
  SM="$WORK/smoke"; rm -rf "$SM"; mkdir -p "$SM"
  KEEP="$WORK/logs/smoke_$TS"; mkdir -p "$KEEP"
  "$T/bin/python" "$HERE/common.py" synth --out "$SM/data" --n 24 || die "合成数据失败"
  COMMON_ARGS=(--model "$M08" --train "$SM/data/train.jsonl" --leak-check "$SM/data/bench.jsonl"
               --batch-size 2 --grad-accum 2 --max-steps 10 --save-every 5 --log-every 1 --max-minutes 15 --num-workers 0)
  "$T/bin/python" "$HERE/train_sft.py" "${COMMON_ARGS[@]}" --output "$SM/dry" --dry-run || die "dry-run 失败"
  "$T/bin/python" "$HERE/train_sft.py" "${COMMON_ARGS[@]}" --output "$SM/s1" --run-name smoke-s1 || die "冒烟训练失败"
  "$T/bin/python" "$HERE/train_sft.py" "${COMMON_ARGS[@]}" --output "$SM/shuf" --run-name smoke-shuffled \
    --shuffle-labels --match-steps-from "$SM/s1" || die "打乱标签对照冒烟失败"
  "$T/bin/python" "$HERE/merge_and_eval.py" eval --model "$M08" --adapter "$SM/s1/final" --backend hf \
    --bench "$SM/data/bench.jsonl" --out "$SM/pred_hf.jsonl" --arm smoke --max-new-tokens 64 || die "HF 评测失败"
  "$T/bin/python" "$HERE/merge_and_eval.py" merge --base "$M08" --adapter "$SM/s1/final" --out "$SM/merged" \
    || die "合并失败"
  if [[ "$SERVE_BACKEND" == vllm ]]; then
    "$S/bin/python" "$HERE/merge_and_eval.py" eval --model "$SM/merged" --backend vllm --bench "$SM/data/bench.jsonl" \
      --out "$SM/pred_vllm.jsonl" --arm smoke --max-new-tokens 64 --max-model-len 4096 --gpu-mem-util 0.5 \
      || { warn "vLLM 评测失败：正式评测改用 --backend hf"; echo "SERVE_BACKEND=hf" > "$WORK/logs/serve.env"; }
  fi
  "$T/bin/python" "$HERE/convert_adapter.py" convert --peft "$SM/s1/final_bf16" --out "$SM/mlx" || die "转 MLX 失败"
  cp "$SM/dry/dryrun.json" "$SM/dry/lora_targets.json" "$KEEP/" 2>/dev/null
  for R in s1 shuf; do
    cp "$SM/$R/summary.json" "$KEEP/${R}_summary.json"; cp "$SM/$R/train_log.jsonl" "$KEEP/${R}_train_log.jsonl"
  done
  cp "$SM"/pred_*.jsonl.summary.json "$SM/merged/merge_report.json" "$SM/mlx/conversion_report.json" "$KEEP/" 2>/dev/null
  "$T/bin/python" - "$KEEP" <<'EOF'
import json, sys
from pathlib import Path
k = Path(sys.argv[1])
s1, sh = (json.loads((k / f"{r}_summary.json").read_text()) for r in ("s1", "shuf"))
steps = [json.loads(l) for l in (k / "s1_train_log.jsonl").read_text().splitlines() if '"step"' in l]
tps = [r["tokens_per_s"] for r in steps if r.get("event") == "step"]
print(f"  S1 冒烟：{s1['final_step']} 步，B 范数 {s1['lora_B_norm']:.3g}，LoRA 模块 {s1['lora']['n_lora_modules']}，"
      f"峰值显存 {s1['gpu_mem_peak_gb']:.1f} GB，吞吐中位 {sorted(tps)[len(tps) // 2] if tps else 0:.0f} tok/s")
print(f"  快速路径：{s1['fast_path']}")
print(f"  对照：{sh['final_step']} 步（S1 {s1['final_step']}），打乱改动 {sh['data']['label_shuffle']['frac_changed']:.0%}")
assert sh["final_step"] == s1["final_step"] and sh["kept_ids_sha256"] == s1["kept_ids_sha256"]
EOF
  rm -rf "$SM"
  echo "  冒烟输出已丢弃；摘要留在 $KEEP"
fi

cat > "$WORK/env.sh" <<EOF
# source ~/ld/env.sh
export WORK=$WORK
export T=$T S=$S
export M4=$WORK/models/Qwen3.5-4B M08=$WORK/models/Qwen3.5-0.8B
export SRV=$HERE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
$(cat "$WORK/logs/serve.env" 2>/dev/null)
EOF
step "完成：$(date)。环境变量见 $WORK/env.sh，日志 $LOG"
