# 开发进度与续做说明（2026-11-07）

分支 `p2-lab-detective`。第二轮开发（界面、服务器脚本、引导/识别/实验卡样本与蒸馏任务、物理 bench、文档重写）
和集成都已完成并提交；全量训练数据、两个 bench 和 5090 数据包已生成并核对通过（都在 gitignore 的 `results/` 下）。

## 已完成（都已提交，测试通过）
- 仓库重构：YOLO 工具在 `tools/yolo/`，权重在 `assets/weights/`（不进库），完整数据集在 `datasets/`（gitignore）。
- 物理核心：`labsim/` 精选实验孪生（训练 + 留出两组，各经独立物理审查）、贝叶斯引擎、特征面板、程序化任务族；
  高保真错配世界 `labsim/hifi.py`（bench 用）。
- 物理组件层 `lab_detective/physics/`：组件、仪器、规律模板、组合器；镜像实验与精选版逐项一致，另有留出的新组合。
- 信息增益顾问与可辨识性图谱、不确定度预算、孪生与幽灵曲线、事实单；知识点图谱与实验卡；识别流水线与本人照片数据集。
- 探究式导师（状态机、按阶段组装上下文、模板讲解、声明校验器、脚本导师）与模型客户端。
- 训练样本生成器：诊断、设计、讲解、知识、引导、实验卡、识别七类；蒸馏任务（DeepSeek / Hy4，封顶 5,248）；打包脚本。
- 评测：物理 bench（10 个划分 + 真值旁注 + 事实单）、打分 `bench/score.py`、对照模型运行器 `bench/run_models.py`；
  识别 bench（按拍摄时段切分）；统计库与冻结工具。
- 服务器脚本（`server/`）：max_len 4096、dry-run 按 task 报 token 分位数并拒绝截断、LoRA 默认全部线性层、
  打乱标签对照、开训前数据包核对 `server/check_package.py`、手册开头的"上机后前 30 分钟"清单。
- 新界面（`app.py`、`ui/`）：五个标签页，离线，默认脚本模式，探测不到本机模型服务时自动退回。
- 真实故障注入工具包与学生材料（`lab_detective/fieldkit/`、`docs/student/`）。

## 集成时修的两处（11/6）
- `training/package.py`：bench 引用的图片先取 bench 文件旁边的 `images/`。以前识别 bench 的 `images/<id>.jpg`
  会按 `results/recognition` 解析，打进包的是没画编号框的原图。同名不同内容时报错。
- `training/generate.py` 新增 `--exclude <bench 文件…>`：输入与任一 bench 题相同的训练/验证样本删掉并计数。
  第一次全量生成时，设计题撞了 1 道 bench 题（精选实验的设计提示组合有限），换种子只会换一道撞，所以改为按 bench 排除。

## 全量数据（2026-11-06 生成）

生成命令（6 进程，墙钟 147 s，主进程峰值内存约 0.76 GB）：

```bash
.venv/bin/python -m lab_detective.training.generate --n-diagnose 9000 --n-design 1500 --n-explain 2000 \
    --n-knowledge 1000 --n-guide 3000 --n-card 150 --workers 6 --out results \
    --exclude results/bench/physics/bench_*.jsonl results/bench/recognition/test.jsonl
.venv/bin/python -m lab_detective.training.generate --n-recognize 9999 --out results   # 识别样本，单独跑
```

| task | train | val | 来源 | 吞吐（条/秒，6 进程 / 单进程） | 备注 |
|---|---:|---:|---|---|---|
| diagnose | 8,549 | 450 | engine | 188 / 32 | 程序化 4,262、精选/组合 4,287；变体 list 6,355 / propose 1,702 / omit_true 492 |
| design | 1,424 | 75 | engine | 20 / 3.6 | 程序化 1,002；撞 bench 删 1 |
| explain | 1,895 | 100 | template | 184 / 32 | 讲解过不了校验器丢 5 |
| knowledge | 492 | 30 | template | — | 题库用完（请求 1,000） |
| guide | 2,367 | 132 | template | 284 / 49 | 过滤丢 306，与验证重叠删 195 |
| card | 112 | 8 | engine | — | 不同提示只有约 50 种，所以只要 150（请求 800 会大量重复） |
| recognize | 51 | 3 | photo | — | **没打进包**：18 张合格照片全是 E，会教模型永远答 E |

训练合计（不含识别）14,839 条，验证 795 条。报告在 `results/generate_report.json`。

**bench**（都在 `results/bench/`，未冻结）：
- 物理 bench `physics/`：3,900 题（in_distribution 600、held_out 1,200、noise_shift 300、withheld_cause 300、hifi 300、
  clean 150、unmodelled 150、swapped_table 600、oracle_in_prompt 100、design 200），5 进程约 64 s；重建两次文件逐字节相同。
- 识别 bench `recognition/test.jsonl`：54 张，金标签全是暂定的（C 12、Z 42），学生审核前只能当草稿。

**泄漏**：`format.leakage_report` 对 10 个物理 bench 提示文件和识别 test 全部 clean（id 0、内容 0）；
物理 bench 的 manifest 里自带的检查也是 clean；`check_package.py` 的照片级检查 0。

## 5090 数据包

| 项 | 值 |
|---|---|
| 压缩包 | `results/package_20261106.tar.gz` |
| 大小 | 17,516,890 字节（16.7 MB；解包后 129 MB） |
| 上传时间 | 约 2.9 分钟（100 KB/s） |
| sha256 | `6543dbef9b8c14149967dfc722bdec06b70f0caf849f0a22d06f5b58c40b6cf6` |
| 内容 | train 14,839、val 795、bench 11 个文件 3,954 题、图片 54 张（识别 bench 带框图）、server/ 脚本、MANIFEST.json |
| 核对 | `python3 server/check_package.py results/package_20261106` → 结论：通过（0 错误 0 警告） |

打包命令：

```bash
.venv/bin/python -m lab_detective.training.package \
    --train $(ls results/train/*.jsonl | grep -v recognize) --val $(ls results/val/*.jsonl | grep -v recognize) \
    --bench $(ls results/bench/physics/bench_*.jsonl | grep -v -e '\.gold\.' -e '\.facts\.') results/bench/recognition/test.jsonl \
    --images-root results/recognition --out results --tag 20261106
```

注意：bench 的 `.gold.jsonl` / `.facts.jsonl` 旁注不能传给 `--bench`（不是样本格式）；它们留在 Mac 上，
服务器评测输出拿回来后在本机用 `lab_detective.bench.score` 打分。

用测试里的假分词器（按字符计，偏保守）在这个包上跑过 `train_sft.py --dry-run --shuffle-labels`：14,839 条全部在 4096 以内
（最长 3,980），答案被截 0 条，打乱标签对照改动 98.9% 的诊断样本。真实 token 长度以服务器上的 dry-run 为准。
按 2 个 epoch 计划 1,855 步；总量约 8 千万字符级 token，默认 120 分钟时间闸可能先到（到点会保存检查点）。

## 最新数据包（2026-11-07，含蒸馏样本）——上 5090 用这个

| 项 | 值 |
|---|---|
| 压缩包 | `results/package_20261107.tar.gz` |
| 大小 | 18,118,852 字节（17.3 MB），上传约 2.9 分钟（100 KB/s） |
| sha256 | `5bd8cce7bc23e38d8ea48f5631f839da24452a436f03ec5e1b309d555ca444c1` |
| 训练 | 16,010 条 = 引擎/模板 14,839 + DeepSeek 蒸馏 1,171（引导 937、诊断 234）；验证 795；bench 11 个文件 |
| 核对 | `python3 server/check_package.py results/package_20261107` → 通过（0 警告）；诊断样本 8,783/8,783 带 meta.causes；泄漏 0 |

旧包 `package_20261106` 不含蒸馏样本，作废。

## 蒸馏（2026-11-06 夜，因 DeepSeek 额度用完停止）

- 教师 `deepseek-flash`（开思考）。调用：诊断 977 次保留 234（约 24%），引导 2,000 次保留 1,322，
  转训练时再剔除内部术语/"照抄"措辞 262 条、与训练或验证提示重复 151 条，最终 1,171 条（上限 5,248）。
- 诊断题被拒的主要原因是教师判断的最可能原因与物理引擎不一致——这本身说明强模型在这个任务上不是贝叶斯最优，
  可以作为 bench 对比的一个观察。
- **费用为什么快**：（1）思考 token：诊断每条约 5,000 个思考 token（诊断输出 507 万 token 中 479 万是思考），
  输出单价是输入的 4 倍，这是主因；（2）缓存命中率只有 13–14%：提示里长而固定的部分（特征词表、输出要求）放在每题数据之后，
  能共享的前缀只有约 250 字的系统提示。
- **以后再蒸馏前先改**：把固定说明移到提示最前面（训练和推理用同一份 `training/prompts.py`，改完要重新生成数据），
  预计命中率 > 60%；引导题用 `--no-think`；诊断题限制思考长度或只蒸馏讲解字段。现在不再调用任何 API。
- Hy4 题包已按新规则导出：`results/distill/hy4_pack/diagnose/`（400 题）、`results/distill/hy4_pack/guide/`（300 题），
  每个文件夹有 README；跑完把 `responses.jsonl` 放回对应文件夹，用 `python -m lab_detective.training.distill_jobs hy4-import`
  导入，再 `to-train --exclude …` 并重新打包。

## 还没做 / 只有学生能做
1. **蒸馏**：DeepSeek 已停（额度用完），1,171 条已打进 `package_20261107`；Hy4 可选，见上一节。
2. **照片标签审核**：在 `results/recognition/review.html` 审核并导出 `review.json`，然后重新生成识别 bench 和识别训练样本
   （`--n-recognize 9999`），确认不止一个类别后再重新打包。另有 35 张只有 HEIC 的照片可以用 `sips` 转 JPEG 后加入。
   识别切分现在还是草稿（`split.json status=draft`）。
3. **冻结 bench**：由学生本人做（`python -m lab_detective.bench.freeze freeze <目录> --who <姓名>`）。
   冻结后如果 bench 重建过，训练数据要用 `--exclude` 重新生成并重新打包。
4. **正式运行**：今天 5090 上的训练属于开发运行；正式的 S1 在研究计划与表格签字、bench 冻结之后由学生本人启动。
5. 已知限制：引导样本 predict / measure 阶段的上下文里没有要问的那一行设定值、reflect 阶段没有实测数据（这两阶段权重已压低）；
   引导样本没用组合实验；模板讲解 225–320 字，长于系统提示的 120 字。

## 续做方法
- 打包前先跑全部测试：`.venv/bin/python -m pytest -q -p no:cacheprovider`。
- 服务器上的步骤看 `server/README.md` 开头的"上机后前 30 分钟"清单。
