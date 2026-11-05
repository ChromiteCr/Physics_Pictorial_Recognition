# 服务器运行手册（RTX 5090）

这份手册是明天在租来的 5090 上要做的全部步骤，按顺序做。命令可以直接复制。
标了〔你亲自启动〕的是正式运行，必须由你本人在终端里按回车；其余步骤（环境、冒烟、dry-run）可以让助手代劳。

## 0. 一览

| 步骤 | 在哪 | 预计用时 | 谁启动 |
|---|---|---|---|
| 1 租机、登录、开 tmux | 浏览器 + Mac 终端 | 10–15 分钟 | 你 |
| 2 上传脚本和数据包 | Mac | 脚本几秒；数据包每 MB 约 10 秒（100 KB/s） | 你或助手 |
| 3 bootstrap（两个环境 + 下载权重 + 0.8B 冒烟） | 服务器 | 30–60 分钟（编译 causal-conv1d 时再加 ≤ 30 分钟） | 你或助手 |
| 4 名称核对（训练前必做） | Mac | 5 分钟 | 你或助手 |
| 5 dry-run + 4B 吞吐探针 | 服务器 | 10–15 分钟 | 你或助手 |
| 6 训练 S1 | 服务器 | ≤ 2 小时 | 〔你亲自启动〕 |
| 7 训练打乱标签对照 | 服务器 | 与 S1 相同步数，约 ≤ 2 小时 | 〔你亲自启动〕 |
| 8 合并 + 评测（未微调 0-shot / 3-shot、S1、对照） | 服务器 | vLLM 每组 10–40 分钟；HF 兜底每组 1–2 小时 | 〔你亲自启动〕 |
| 9 下载适配器和预测 | Mac | 每个适配器约 40 MB（bf16），预测几 MB | 你或助手 |
| 10 在 Mac 上转成 mlx-vlm 格式 | Mac | 几分钟 | 你或助手 |

S1 和对照可以接着跑（同一张卡串行）。整条链大约 6–8 个 GPU 小时。

## 1. 租机、登录

要求：RTX 5090 32 GB；驱动支持 CUDA ≥ 12.8（`nvidia-smi` 右上角）；系统盘或数据盘可用空间 ≥ 80 GB；
Ubuntu 22.04/24.04，自带 Python ≥ 3.10（平台的"PyTorch / CUDA 12.8"基础镜像都可以）。地区选国内，ModelScope 快。

```bash
# Mac 上（端口、用户、主机按平台给的填）
ssh -p <端口> <用户>@<主机>
# 服务器上：所有长任务都放进 tmux，断线不会中断
tmux new -s ld            # 断开：Ctrl-b 再按 d；重连：tmux attach -t ld
mkdir -p ~/ld/pkg
```

## 2. 上传（上行只有约 100 KB/s，只传代码和数据包）

```bash
# Mac 上，在仓库根目录
tar czf /tmp/server_scripts.tar.gz server
scp -P <端口> /tmp/server_scripts.tar.gz results/package_<标签>.tar.gz <用户>@<主机>:~/ld/
shasum -a 256 /tmp/server_scripts.tar.gz results/package_<标签>.tar.gz   # 记下哈希

# 服务器上
cd ~/ld && sha256sum server_scripts.tar.gz package_<标签>.tar.gz          # 和 Mac 上的对一遍
tar xzf server_scripts.tar.gz && tar xzf package_<标签>.tar.gz -C pkg
ls pkg/                       # 确认数据包根目录，下面记作 $PKG
```

数据包约定（由 `lab_detective/training` 打包，格式见 `docs/ARCHITECTURE.md` §2.5）：根目录下有训练 JSONL、验证 JSONL、
冻结的 bench JSONL 和 `images/`。图片路径相对数据包根目录。下面假设：

```bash
export PKG=~/ld/pkg/package_<标签>       # 数据包根目录
export TRAIN=$PKG/train.jsonl VAL=$PKG/val.jsonl
export BENCH="$PKG/bench/bench_*.jsonl"   # 冻结的 bench（11/8 冻结后单独上传时改成那个路径）
```

文件名不一样就照实际改这几个变量，后面的命令不用动。

## 3. bootstrap

```bash
cd ~/ld && bash server/bootstrap.sh
```

脚本会依次做这些事，全部写进 `~/ld/logs/bootstrap_<时间>.log`：

1. 预检：GPU 是 sm_120、驱动 CUDA ≥ 12.8、磁盘 ≥ 60 GB、Python ≥ 3.10。
2. 探网：PyPI 镜像（清华 → 阿里 → 中科大 → 官方，选第一个通的）、PyTorch 源（官方 cu128 → 阿里镜像）、ModelScope、HF、hf-mirror。
3. 训练环境 `~/ld/envs/train`：torch（cu128，检查有 sm_120 内核）、transformers ≥ 5.2、peft、trl、accelerate、datasets，
   以及线性注意力快速路径 flash-linear-attention、causal-conv1d（装不上不致命，只是慢）。
4. 服务环境 `~/ld/envs/serve`：vLLM（装不上就记下 `SERVE_BACKEND=hf`，评测改用 HF 生成）。
5. 下载 Qwen/Qwen3.5-4B 和 Qwen/Qwen3.5-0.8B（ModelScope 优先，失败走 hf-mirror），校验 `model_type=qwen3_5`，写 `DOWNLOAD.json`。
6. GPU 检查：bf16 矩阵乘吞吐、显存。
7. 冒烟（0.8B，输出全部丢弃，只留摘要在 `~/ld/logs/smoke_<时间>/`）：dry-run → 10 步 LoRA → 打乱标签对照 10 步
   （核对步数和样本完全一致）→ HF 评测 → 合并 → vLLM 评测 → 转 MLX。

结束时打印一行"S1 冒烟：… 吞吐中位 … tok/s"，并写 `~/ld/env.sh`。之后每开一个新终端先：

```bash
source ~/ld/env.sh     # 设 $T（训练环境）、$S（服务环境）、$M4（4B 底座）、$SRV（脚本目录）、SERVE_BACKEND，并开离线模式
```

要看的东西：日志里没有"错误"；"快速路径"一行里 `fla_gated_delta_rule` 为 true（为 false 训练会慢几倍，
可以 `STEPS="venv-train" bash server/bootstrap.sh` 重装）；vLLM 冒烟通过。只重跑某几步：`STEPS="download check" bash server/bootstrap.sh`。

常见问题：
- 某个源慢或不通：`PIP_MIRROR=aliyun bash server/bootstrap.sh`；torch 版本要固定：`TORCH_SPEC="torch==<版本> torchvision==<版本>"`。
- vLLM 和 Qwen3.5 不兼容：先 `VLLM_SPEC="vllm==<更新的版本>" STEPS="venv-serve smoke" bash server/bootstrap.sh`；不行就全程 `--backend hf`。
- **权重拉不到**（ModelScope 和 HF 都失败）：按计划改走 Mac 兜底训练，单独登记为 "S1-mac"，不冒充 S1。退租。

## 4. 名称核对（Mac，训练 S1 之前必做）

确认要挂 LoRA 的每一层在 mlx-vlm 的 4B 底座里都有同名、同形状的层，否则适配器在 Mac 上加载不上。

```bash
# 服务器上的 4B 权重索引只有几十 KB，拿回 Mac
scp -P <端口> <用户>@<主机>:~/ld/models/Qwen3.5-4B/model.safetensors.index.json /tmp/qwen35_4b_index.json
# Mac 上，仓库根目录；<MLX4B> = LM Studio 里 Qwen3.5-4B MLX 版的文件夹
.venv/bin/python server/convert_adapter.py check-names --hf /tmp/qwen35_4b_index.json --mlx-model <MLX4B>
.venv/bin/python server/convert_adapter.py check-names --hf /tmp/qwen35_4b_index.json --mlx-model <MLX4B> \
    --targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
```

- 第一条（默认：注意力 q/k/v/o + MLP）必须 `"ok": true`，否则不要开训，先把输出发给助手。
- 第二条（再加线性注意力投影）也 `"ok": true` 的话，可以在训练时用 `--targets all-linear` 覆盖语言部分全部线性层；
  不通过（例如 MLX 里这些投影被融合了）就保持默认。**选哪个要在 S1 之前定死，对照用同一个。**
- 索引里只有名字没有形状时只核对名字；有分片文件时也核对形状。

## 5. dry-run 和吞吐探针（服务器）

```bash
source ~/ld/env.sh && cd ~/ld
# 只分词：token 统计、超长丢弃数、泄漏检查、LoRA 模块清单，不碰 GPU（2–5 分钟）
$T/bin/python $SRV/train_sft.py --model $M4 --train $TRAIN --data-root $PKG --leak-check $BENCH \
    --output runs/dry_s1 --dry-run
# 对照的打乱统计
$T/bin/python $SRV/train_sft.py --model $M4 --train $TRAIN --data-root $PKG --leak-check $BENCH \
    --output runs/dry_shuf --dry-run --shuffle-labels
```

检查 `runs/dry_s1/dryrun.json`：
- `data.leakage.clean` 为 true（不干净脚本会直接停）。
- `tokens.n_dropped_too_long` 接近 0（超过 1% 就考虑 `--max-len 3072`，S1 和对照用同一个值）。
- `tokens.full_render_mismatch_first20` 为 0。
- `n_lora_modules` 和第 4 步核对的数量一致。
- `runs/dry_shuf/dryrun.json` 里 `data.label_shuffle.frac_changed` 接近 1、`frac_top_changed` 约等于 1 − 1/候选数；
  `by_how.meta` 应该是全部诊断样本（`experiment` 多说明数据缺 `meta.causes`，对照会偏弱，先告诉助手）。

吞吐探针（可选，约 5 分钟，输出丢弃），用来定 S1 的步数，让学习率调度在 2 小时内走完：

```bash
$T/bin/python $SRV/train_sft.py --model $M4 --train $TRAIN --data-root $PKG --leak-check $BENCH \
    --output runs/probe --max-steps 20 --log-every 5
grep '"step"' runs/probe/train_log.jsonl | tail -2      # 看 tokens_per_s 和每 5 步耗时
rm -rf runs/probe
```

每步秒数 × `planned_total_steps`（dry-run 里有）超过 110 分钟时，给 S1 加 `--max-steps <110 分钟能跑的步数>`。
这一步定下的数字写进研究计划。

## 6. 训练 S1〔你亲自启动〕

前提：研究计划已写完、表格签好、物理卡审核完、bench 已冻结并核对哈希。

```bash
source ~/ld/env.sh && cd ~/ld
$T/bin/python $SRV/train_sft.py --model $M4 --train $TRAIN --val $VAL --data-root $PKG --leak-check $BENCH \
    --output runs/s1 --run-name S1 --epochs 2 --max-minutes 120 --seed 1234
# 第 5 步定了步数上限就在末尾加 --max-steps <N>；第 4 步决定覆盖全部线性层就加 --targets all-linear
```

固定超参（默认值，写进计划）：LoRA r=16、α=32、dropout 0.05；lr 2e-4，3% 预热后余弦降到 10%；微批次 2 × 累积 8；
max_len 2048（超长丢弃）；bf16；梯度检查点；只对最后一条助手回复算损失；种子 1234。只用最终检查点 `runs/s1/final`。

- 进度：`tail -f runs/s1/train_log.jsonl`（每 10 步一行：loss、lr、grad_norm、tokens_per_s、显存）。
- 每 200 步或 30 分钟存一次断点（`runs/s1/checkpoint-*`，只留最近 2 个）。断线、崩溃后续跑：
  加 `--resume runs/s1/checkpoint-<步数>`，其余参数原样。
- 爆显存：改 `--batch-size 1 --grad-accum 16`（等效批大小不变），对照也要同样改。
- 结束后看 `runs/s1/summary.json`：`stopped_by`（steps 最好；time 说明到了时间闸）、`final_step`、`planned_total_steps`、
  `lora_B_norm` > 0、`val_loss`、`final_adapter_sha256`。

## 7. 打乱标签对照〔你亲自启动〕

同样的提示、同样的样本和顺序、同样的种子和步数；每条诊断样本的原因标签（含"只有随机误差"）独立随机置换。

```bash
$T/bin/python $SRV/train_sft.py --model $M4 --train $TRAIN --val $VAL --data-root $PKG --leak-check $BENCH \
    --output runs/shuffled --run-name shuffled --epochs 2 --max-minutes 180 --seed 1234 \
    --shuffle-labels --match-steps-from runs/s1
# S1 加过的 --max-steps / --targets / --batch-size / --grad-accum / --max-len 这里原样加上（配置不一致会直接停）
```

`--match-steps-from` 读回 S1 的计划总步数和最终步数（学习率轨迹逐步相同），并核对配置和样本 id 哈希；
有任何不一致直接停。`--max-minutes 180` 只是安全闸，不应触发。

## 8. 合并 + 评测〔你亲自启动〕

```bash
source ~/ld/env.sh && cd ~/ld
# 合并（训练环境，各约 3 分钟；合并后核对权重键和底座一致）
$T/bin/python $SRV/merge_and_eval.py merge --base $M4 --adapter runs/s1/final --out merged/s1
$T/bin/python $SRV/merge_and_eval.py merge --base $M4 --adapter runs/shuffled/final --out merged/shuffled

# 评测（SERVE_BACKEND=vllm 时用服务环境）；贪心、关闭思考、每条检查思考标记
EV="$S/bin/python $SRV/merge_and_eval.py eval --backend vllm --bench $BENCH --data-root $PKG"
$EV --model $M4           --arm base0     --out preds/base0.jsonl
$EV --model $M4           --arm base3shot --out preds/base3shot.jsonl --shots 3 --shots-from $TRAIN
$EV --model merged/s1       --arm S1        --out preds/s1.jsonl
$EV --model merged/shuffled --arm shuffled  --out preds/shuffled.jsonl
```

- `SERVE_BACKEND=hf` 时把 `EV` 换成：`$T/bin/python $SRV/merge_and_eval.py eval --backend hf --bench $BENCH --data-root $PKG`，
  S1 和对照可以不合并：`--model $M4 --adapter runs/s1/final`。
- 中断后加 `--resume` 原样重跑，跳过已完成的 id。
- 每个输出旁边有 `<文件>.summary.json`：`think_rate` 必须是 0，`json_ok_rate`、`length_cut_rate`、`error_rate` 记进实验记录。
- 3-shot 的示例只取自训练池（同 task、纯文字、按 id 确定性挑选）。打分由 bench 模块在 Mac 上做，这里只生成。
- bench 图片根目录和数据包不同时改 `--data-root`。

## 9. 下载

```bash
# 服务器上：打成一个包（适配器用 bf16 版，每个约 40 MB）
cd ~/ld && tar czf results_$(date +%m%d).tar.gz runs/s1/final_bf16 runs/shuffled/final_bf16 \
    runs/s1/summary.json runs/s1/train_log.jsonl runs/s1/lora_modules.json \
    runs/shuffled/summary.json runs/shuffled/train_log.jsonl merged/*/merge_report.json preds logs
sha256sum results_*.tar.gz
# Mac 上，仓库根目录
mkdir -p results/server && scp -P <端口> <用户>@<主机>:~/ld/results_<日期>.tar.gz results/server/
cd results/server && shasum -a 256 results_<日期>.tar.gz && tar xzf results_<日期>.tar.gz
```

## 10. 在 Mac 上转成 mlx-vlm 适配器

```bash
# 回到仓库根目录再做
# 1) 看本机装的 mlx_vlm 怎么读适配器，决定 --format（lora = lora_a/lora_b + scale；legacy = A/B + alpha 当缩放）
.venv/bin/python server/convert_adapter.py inspect
# 2) 转换，同时和 MLX 4B 底座核对模块名、形状
.venv/bin/python server/convert_adapter.py convert --peft results/server/runs/s1/final_bf16 \
    --out results/server/s1_mlx --mlx-model <MLX4B> --format <inspect 建议的格式>
.venv/bin/python server/convert_adapter.py convert --peft results/server/runs/shuffled/final_bf16 \
    --out results/server/shuffled_mlx --mlx-model <MLX4B> --format <同上>
```

转换只做两件事：改名（`model.language_model.layers.N…` → `language_model.model.layers.N…`）和转置
（lora_A (r, in) → lora_a (in, r)，lora_B (out, r) → lora_b (r, out)），缩放 = α/r = 2。映射不了的键直接报错。
真正的验收是 K0：同一输入下 Mac（mlx-vlm + 适配器）和服务器（合并模型）的 token 一致，以及"适配器效应检查"
（挂和不挂适配器输出不同）。K0 不过时先换 `--format` 重转一次。

## 11. 退租前清单

- [ ] 第 9 步的文件都下载了，`summary.json` 里的 `final_adapter_sha256` 和本地文件对得上。
- [ ] `preds/*.jsonl` 条数等于 bench 条数，`*.summary.json` 也下载了。
- [ ] `~/ld/logs/` 整个下载（bootstrap、版本、pip freeze、冒烟摘要）。
- [ ] 后面还要跑（消融、思考参考组、PhyX）就先别退租。
