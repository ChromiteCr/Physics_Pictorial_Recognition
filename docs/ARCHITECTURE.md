# P2 实验侦探：架构与模块接口（开发者必读）

> 这份文件是并行开发的唯一接口约定。改接口先改这里。所有文字面向中学生的部分用中文。
> 状态以 `docs/PROGRESS.md` 为准；本文描述接口和已有模块的职责，"已定稿"表示接口不再改动。

## 0. 产品与创新点（物理为主）

**产品**：拍一张力学实验装置的照片 → 认出是什么实验（开放集，力学优先）→ 把这台装置变成一个**参数化物理孪生**
→ 实验前给出预测和不确定度预算 → 带学生一步步测量，每填一行数据就和孪生对照 → 数据偏离时用贝叶斯推断找出系统误差来源，
推荐最能区分各原因的下一次测量 → 依据物理事实单讲解，并关联知识点。完全离线，本地小模型。

**自己的创新点**（与格物论文无任何相似；格物是文本题库上的解题辅导 + 代数沙箱 + 投票）：
1. **误差物理孪生**：中学力学实验的可执行前向模型，带系统误差物理（摩擦、车轮/滑轮转动惯量、绳子质量、遮光片宽度、弹性限度、调零方向……）和仪器噪声模型。已精修的课内与留出实验只是起点，其他装置由物理组件层（`physics/`：物体、连接件、表面、仪器 + 规律模板，见 §2.8）组合出来；组件也拼不出的装置再退到通用关系族 + 通用故障算子（`labsim/generic.py`）。
2. **可辨识性图谱**（测量物理）：课本标准方案能/不能暴露哪些误差，哪一次追加测量能把它们分开（贝叶斯实验设计）。
3. **看图建模**：照片 → 器材清单与构型 → 匹配精选孪生，或用物理组件组合出这台装置的孪生（`physics/from_photo.py`），并列出还要学生补的已知量（质量、倾角、遮光片宽度……）。
4. **物理引擎当老师**：模型的主要训练标签来自对含噪数据做贝叶斯逆推断的物理引擎，不是教师模型的解题轨迹。
5. **真实器材闭环**：故障注入实验检验孪生的预测（sim-to-real）。
6. **探究式教学**：实验前预测、实验中实时一致性检查、按信息增益选"下一个问题/下一次测量"、误差与知识点关联。

**模型从不自己算物理**：所有数值来自物理引擎（确定性代码），模型负责看图、规划、解释和对话。
没有代码沙箱、没有工具调用循环、没有投票。蒸馏只借鉴格物的一点：教师盲做，只保留和**物理引擎判定一致**的输出；总条数 ≤ 5,248。

## 1. 目录与模块归属

```
lab_detective/
  paths.py                  仓库、数据集、结果目录的统一路径
  labsim/core.py            公共接口（已定稿，见文件头建模约定）
  labsim/registry.py        自动发现 experiments/*.py 里的 Experiment 子类（新增实验只要新建文件）
  labsim/experiments/*.py   精选实验的孪生，每个实验一个文件（hooke.py 是样板）；类属性 held_out 区分训练 / 留出
  labsim/generic.py         通用关系族 + 通用故障算子（"其他实验"和程序化任务族）
  physics/                  物理组件层：把精选实验拆成可复用组件，再组合出新装置的孪生（见 §2.8）
    base.py                 Effect（误差机制）、Ctx、Component、AuxMeasurement、Variant、Control、Setup、Affine
    components.py           物体（小车、木块、小球、摆球、挂钩）、连接件（弹簧、细线、滑轮、手拉、释放）、表面（导轨、斜面、木板、杠杆）
    instruments.py          刻度尺、弹簧测力计、秒表、手机计时、光电门、打点计时器、天平、量角器、游标卡尺
    templates.py            规律模板：静态弹簧、拉动摩擦、匀加速系统、斜面/系统机械能、单摆、弹簧振子、光电门自由落体、
                            由静止下滑、斜槽平抛、杠杆平衡、一维碰撞
    compose.py              Setup → ComposedExperiment（遵守 labsim/core.py 接口，引擎等模块不用改）
    catalog.py              镜像精选实验的装置 + 训练集里没有的新组合（默认留出）
    from_photo.py           识别结果（器材 + 线索）→ 候选装置 + 未决问题
  engine.py                 贝叶斯后验：非线性参数放网格、线性参数解析积分，并给"不在清单里"用的失配统计量（已定稿）
  panel.py                  特征面板：只用可观测量（截距、斜率比、曲率、散布偏大/偏小、漂移）（已定稿）
  eig.py                    期望信息增益、模拟闭环（基础数据流与规划流分开，策略配对比较）
  atlas.py                  可辨识性图谱：某方案下哪些原因分得开，哪次追加测量能分开分不开的一对
  uncertainty.py            不确定度预算（实验前）与传播
  twin.py                   孪生实例化：实验 id + 参数 → 预测曲线、预期读数、幽灵曲线
  facts.py                  事实单 FactSheet（引擎输出的唯一出口，也是讲解校验的依据）
  knowledge/                实验卡 + 知识点图谱（JSON 数据 + 加载器；数据随包安装）
  vision/                   照片 → 构型 → 实验：cues（YOLO 线索与规则）、som（编号框）、letters（字母表）、
                            prompt/schema/pipeline（模型兜底 + 澄清问题 + 手动选择）、build_dataset（识别样本）
  tutor/                    探究式引导：session（会话状态机）、context（按阶段组装上下文）、policy（脚本导师）、
                            templates（模板讲解）、verifier（声明校验器）
  llm/                      模型客户端 client.py：OpenAI 兼容接口（mlx_vlm.server / DeepSeek / OpenAI），
                            一律关闭思考，所有请求与回复写 JSONL 日志；产品里只连本机服务
  training/                 训练样本生成、蒸馏流水线、打包
    format.py               样本格式（§2.5）、读写、泄漏检查
    prompts.py              提示词的唯一来源（训练、蒸馏、bench、推理共用）+ 输出 JSON schema（见 §2.7）
    splits.py               训练/留出实验、每个训练实验扣下的原因、种子命名空间、训练/验证划分
    tasks_diagnose.py       诊断样本（含"自己提出原因"变体）；tasks_design.py 实验前规划；
    tasks_explain.py        事实单讲解（3 种措辞/槽位，必须过校验器）；tasks_knowledge.py 知识点问答
    generate.py             命令行：多进程生成 results/{train,val}/<任务>.jsonl
    distill.py              离线蒸馏：教师盲做，只保留与物理引擎判定一致的输出，总数 ≤ 5,248（见 §2.9）
    package.py              打包给 5090：样本、被引用的图片（长边 ≤ 768）、冻结的 bench、服务器脚本、清单与哈希
  bench/                    评测基准
    freeze.py               冻结：文件逐个算 SHA-256 写清单（只能由学生本人运行）
    stats.py                精确 McNemar、Wilson 区间、精确符号检验、配对 bootstrap、TOST、Holm、Brier、ECE
    recognition_bench.py    识别 bench：按拍摄时段切分、生成 test.jsonl、打分、规则基线（见 §2.6）
    review_gallery.py       识别照片审核图册 review.html（学生逐张确认标签，导出 review.json）
  fieldkit/                 真实故障注入实验工具包（见 §2.10）
server/                     5090 上的脚本（见 §2.11）：bootstrap.sh、common.py、train_sft.py、merge_and_eval.py、convert_adapter.py
app.py + ui/                Streamlit 界面（见 §2.12）
tools/yolo/                 YOLO 数据与训练工具（P1 起）；detect.py 为 YOLO 器材识别入口
tests/                      pytest，每个模块一个文件
```

物理 bench（诊断、下一次测量、迁移到留出实验的冻结测试集及其打分）与高保真错配模拟器、对照模型运行器按同一套约定放在 `bench/` 下，
落地后在 §2.6 旁边补接口；在此之前它们的目标与规格见 `docs/isef/2026-10-31-isef-upgrade-design.md` §16。

## 2. 关键数据结构（JSON 可序列化）

### 2.1 FactSheet（facts.py）
```json
{"experiment_id": "hooke", "experiment_name": "...", "protocol": "standard",
 "data": {"n_points": 5, "table": [{"setting": 50, "values": [1.2]}]},
 "panel": {...Panel.to_dict()...},
 "agreement": {"quantity": "k", "value": 24.1, "uncertainty": 0.6, "unit": "N/m", "z": 2.4, "band": "consistent|tension|inconsistent"},
 "diagnosis": {"ranked": [{"cause_id": "...", "name": "...", "prob": 0.71, "evidence": "截距 +1.5 cm（z=6.2）"}],
               "not_in_list": false, "misfit_p": 0.4},
 "next_measurement": {"id": "weigh_hanger", "label": "...", "eig_bits": 0.8,
                      "predicted": [{"cause_id": "...", "value": 23.0, "unit": "g"}]},
 "fix": {"cause_id": "...", "text": "...", "predicted_rerun": "..."},
 "knowledge_points": ["kp.hooke_law", "kp.systematic_error"],
 "numbers": [{"name": "intercept", "value": 1.52, "unit": "cm"}]}
```
`numbers` 列出事实单里允许出现在讲解文字中的所有数字（校验器用）。

### 2.2 实验卡与知识点（knowledge/）
- `knowledge/data/cards/<experiment_id>.json`：
  `id, name_zh, name_en, grade("初中"|"高中"), goal, principle(文字), formula(纯文本), variables[{symbol,name,unit,role:"control|measured|derived"}],
   apparatus[], steps[], data_table{columns[]}, safety[], common_causes[cause_id], knowledge_points[kp_id], relation_family, labsim_id, yolo_cues[]`
- `knowledge/data/knowledge_points.json`：`[{id, name_zh, statement, formula, prerequisites[kp_id], misconceptions[{id, text, symptom}], experiments[id]}]`
- 加载器：`knowledge.load_card(id)`, `knowledge.all_cards()`, `knowledge.kp(id)`, `knowledge.graph()`。

### 2.3 识别结果（vision/schema.py）
```json
{"letter": "A"...,"experiment_id": "hooke"|null, "open_name": "单摆测重力加速度", "cues": ["弹簧竖直悬挂", "下端挂钩码"],
 "apparatus": ["spring", "slotted_mass", "ruler"], "confidence": 0.8, "needs_question": false}
```
字母表：课内与留出实验各一个大写字母，`Z` = 其他。所有模型（本地、API）共用同一份字母表和 JSON schema。

### 2.4 会话状态（tutor/session.py）
阶段：`identify → plan → predict → measure → check → diagnose → explain → reflect`。
`LabSession`：experiment_id、card、protocol、observations、student_predictions、fact_sheet、phase、hints_used、kp_touched、turns（只存精简摘要）。
上下文组装 `build_context(session, budget_tokens)`：按阶段只放必要信息（实验卡要点 + 最新事实单 + 最近 2 轮），**不放完整对话历史**。

### 2.5 训练样本（training/format.py）——服务器端训练脚本只认这个格式
JSONL，每行：
```json
{"id": "diag-hooke-000123", "task": "diagnose|design|recognize|guide|explain|card|knowledge",
 "source": "engine|template|distill-deepseek|distill-hy4|photo",
 "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": [{"type": "image", "image": "images/xxx.jpg"}, {"type": "text", "text": "..."}]},
              {"role": "assistant", "content": "{...JSON 或中文...}"}],
 "meta": {"experiment_id": "hooke", "split": "train|val", "seed": 123}}
```
- 图片路径相对数据包根目录；图片统一缩放到长边 ≤ 768 px、JPEG 质量 85。
- `task` 和 `source` 必填；测试集样本绝不进训练包（泄漏检查器按 id 和内容哈希双重检查）。
- 系统提示：训练时看到的指令必须和推理时一致（蒸馏样本里教师看到的指令也照样放进训练提示）。
- 引擎标注的四类文本样本（diagnose / design / explain / knowledge）怎么生成、提示里放什么、目标 JSON 长什么样，见 §2.7。

### 2.6 识别 bench 与审核图册（bench/recognition_bench.py、bench/review_gallery.py）
bench 由学生冻结，开发侧只生成、打分，**从不调用 `bench.freeze`**。切分间隔、"未知/Z 怎么计分"、主指标都由学生在冻结前定死。
生成物都在 `results/bench/recognition/`（`results/` 不进版本库）：`split.json`、`test.jsonl`、`images/`（带编号框的测试图）、`detections.jsonl`（YOLO 检测缓存）、`meta.json`；各 arm 的预测和报告放 `results/bench/recognition_runs/<arm>/`。

- **标签由学生本人核实**：`python -m lab_detective.bench.review_gallery` 生成 `results/recognition/review.html`（离线、缩略图内嵌）。
  页面**没有任何预选**；规则的暂定字母只是提示（规则没有候选时不给字母，暂定 Z 不等于"其他实验"），可开盲审隐藏提示。
  学生点字母（或键盘 A–L / Z）才算确认；导出的 `review.json` 恰好是 `{"<照片id>": "<字母>"}`，只含确认过的照片。
  生成 bench 时用 `recognition_bench build --review review.json`；没核实的照片保留暂定标签，并在 `meta.gold_status` 标 `provisional`。
- **按拍摄时段切分**：拍摄时间依次取 manifest 的 `taken_at`、原图 EXIF（`datasets/` 下的原图；`results/recognition/images/` 里的图不带 EXIF）、文件名时间戳、文件编号相邻。
  相邻照片间隔 ≤ `gap_s`（默认 1800 s）算同一时段，约 25% 的时段整段进测试集（按首张照片 id 的哈希排序取前 k 个，新增照片不会打乱已有时段）。
  现有 200 张照片全拍于 29 分钟内，默认间隔只得到 1 个时段，`split` 会拒绝并列出各间隔下的时段数，由学生用 `--gap-s N|auto` 定下来（写进 `split.json.params`）。
  时段太少时测试集与训练集的相邻照片仍可能是同一场景的连拍，`split.json.nearest_train_gap_s` 记录最近间隔。
  `split.json.status` 初始为 `draft`；学生定好间隔和种子后运行 `recognition_bench --out <bench 目录> confirm --who <姓名>` 改成 `confirmed`，在此之前 `meta.json` 带"切分还是草稿"的警告。命令行里 `--manifest`、`--out` 是顶层选项，写在子命令前面。
  开发识别模块时已经看过或跑过的照片（`recognition_bench.DEV_SEEN_STEMS`）所在的时段固定划进训练侧，不进测试集，记在 `split.json.dev_seen_forced_train`；可选时段不够 k 个时测试集相应缩小并警告，一个都不剩时拒绝切分。
  **训练样本生成必须排除 `split_ids(load_split(), "test")`**，并用 `check_no_overlap` 自查。
- **`test.jsonl`**：§2.5 的格式，`task=recognize`、`source=photo`、`id=recbench-<照片id>`。messages 由 `vision/prompt.py` 的 `build_messages` 拼，
  图是带编号框的 `images/<id>.jpg`（编号框来自 YOLO **检测**结果，与推理一致，不是真值框）。assistant 里只有 `letter` 是 gold，`open_name/cues/apparatus` 只是占位。
  `meta` 含：`photo_id, gold_letter, gold_status(reviewed|provisional), provisional_letter, yolo_split, yolo_seen, session, taken_at, gt_boxes, detections, legend, prompt_hash`。
- **预测文件**（每个 arm 一份 JSONL）：`{"id": "recbench-…", "text": "<模型原始输出>", "letter_logprobs"?: {"A": -0.1, …}, "letter_probs"?: {…}, "latency_s"?: 1.2}`。
  缺失的 id、解析不出字母的输出一律算错（所有 arm 都不用约束解码，非法输出照算错）。
  开放集 AUROC 的正类是 gold 为 Z 的照片，分数是 P(Z)：有 logprobs/概率向量就用它，否则用 `confidence`（预测 Z 取 conf，否则取 1 − conf）。
- **报告**（`recognition_bench score`）：top-1 + Wilson 区间、每类召回与混淆矩阵、按 `source` 和 `yolo_split` 分层、JSON 合法率、器材 F1（只含 YOLO 词表，仅作参考）、ECE、AUROC（只有一个类时为 null）。
  **只有 `reviewed` 一栏是证据**；`provisional` 一栏只用来检查流程——暂定标签就是规则对真值框的输出，规则基线在这一栏按构造满分。
  YOLO 训练时见过这批照片（`yolo_split` 为 train/val），"规则 + 检测框"要按 `yolo_split` 分层看，不和 VLM arm 混成一个数。
- **规则基线**：`recognition_bench rules` 分别在真值框和检测框上跑 YOLO + 规则，输出完整的概率向量（Z 用虚拟分数），并报告规则"有把握"的覆盖率和准确率。

### 2.7 引擎标注的训练样本（training/prompts.py、splits.py、tasks_*.py、generate.py）
所有标签都由确定性代码给出（贝叶斯引擎、EIG、孪生、不确定度预算、可辨识性图谱、知识点图谱），没有人工标签、API 输出或学生数据。

- **提示的唯一来源**：`training/prompts.py`（`PROMPT_VERSION = "lab-v1"`，改提示必须升版本）。训练、蒸馏、bench、推理都调
  `build_diagnose_messages / build_design_messages / build_explain_messages / build_knowledge_messages`，返回 `[system, user]`；
  system 一律是 `tutor/context.py` 的 `SYSTEM_PROMPT_ZH`。讲解提示就是 `build_context(..., phase="explain")`，与 `tutor/policy.guarded_explanation` 推理时完全相同。
- **诊断 / 设计提示只放观测量**：实验卡要点（名称、目的、公式、变量；程序化实例按关系族现场生成）、数据表（照学生记录的样子）、
  面板数字（与事实单同一套取整，讲解引用的数字都能在提示里找到）、一致性的数值和 z、候选原因的名称 + 物理描述（编号 `C1…` 每条样本随机打乱）、
  追加测量菜单（编号 `M1…` 随机打乱，附用时和会得到的读数）。**不放**：原因的预期特征、面板判出的特征、后验概率、EIG、一致性结论。
  特征词表、机制词表和输出格式是所有样本共用的固定文字（`prompts.fixed_blocks()`），不带样本信息。
- **输出 schema**（`prompts.SCHEMAS`；`prompts.validate_output(variant, obj)` 检查 schema 和 explanation ≤ 80 字）：
  - `diagnose`：`{"signatures": [特征词], "evidence": "证据句", "ranked": [{"cause_id": "C3", "prob": 0.7, "expect": [特征词]}, …], "not_in_list": false,
    "next_measurement": {"id": "M2", "predicted": [{"cause_id", "value", "unit"}（前两名原因）]} | null, "agreement": "consistent|tension|inconsistent" | null, "explanation": "≤ 80 字"}`。
    ranked 列出全部候选，按未取整的后验从大到小，prob 取 0.05 的倍数；expect = 原因在标准方案下的预期特征；第一个 `cause_id` 就是最可能的原因（打乱标签对照据此统计）。
  - `propose`（不给候选清单）：ranked 换成 `[{"name", "mechanism", "expect", "prob"}]`（引擎排名前 6），predicted 用 `name`，没有 `not_in_list`。
  - `design`：`{"expected": [{"setting", "value", "sd", "unit"}], "dominant_source": {"reading", "type": "random|resolution"},
    "recommend": {"n_points", "range": [最小, 最大], "unit", "repeat_each"}, "unresolved": [{"pair": ["C1", "C4"], "status": "confused|partial", "measurement": "M2" | null}], "explanation"}`。
    提示里给孪生预期读数和标准方案下斜率的相对不确定度（引擎算好），不给不确定度份额和图谱结果。
  - `knowledge`：`{"knowledge_points": [{"id", "name"}], "prerequisites": [{"id", "name"}], "answer": "≤ 80 字"}`。
  - `explain`：纯中文 ≤ 120 字（原因、数值证据、下一次测量及前两名原因的预测值、修正），每个槽位 3 种措辞，**必须通过 `tutor/verifier.verify`**，过不了就丢弃。
- **划分（`splits.py`）**：训练 / 留出实验完全按注册表的 `held_out`；留出实验的任何内容都不进训练数据。每个训练实验扣下一个原因
  （`WITHHELD_CAUSES`：`average_speed.cart_length`、`energy.wheel_inertia`、`friction.zero_vertical`、`hooke.total_length`、`momentum.friction`、`newton2.heavy_hanging`，
  理由见 `WITHHELD_REASONS_ZH`），它不用来模拟、不进候选清单和引擎原因集、不进知识问答，留给"已知实验里的新原因"测试。
  每条样本的种子 = `SeedSequence(基础种子, 任务, 划分, 序号)`；验证集 = 独立命名空间、总数的 5%；`SPLIT_CODES["bench"]` 留给冻结测试集，生成器从不使用。
- **诊断配比**：程序化任务族 50%（`labsim/generic.py`）+ 训练实验 50%；方案 单次 15% / 标准 60% / 标准 + 1 项追加测量 25%；真实原因约 25% 是"只有随机误差"；
  20% 是 propose 变体；8% 是"真实原因不在清单里"（目标照样来自引擎）。EIG 用 `n_samples = 64`。设计任务里程序化实例占 70%。
- **meta**：`experiment_id`（训练实验 id 或 `proc_<定律>`）、`split`、`seed`、`variant`、`schema`、`protocol`、`truth_cause`、`top_cause`、
  `causes`（`[{"id": 显示编号, "name": 中文名}]`，打乱标签对照按它置换）、`id_map`（显示编号 → 真实原因 id）、`menu_map`、`explanation_verified`、`prompt_version`。
- **生成**：`python -m lab_detective.training.generate --n-diagnose N --n-design N --n-explain N --n-knowledge N --workers 6`
  → `results/train/<任务>.jsonl`、`results/val/<任务>.jsonl`、`results/generate_report.json`（条数、丢弃数、吞吐）。每个 `--n` 是训练 + 验证的总数；
  输出与进程数无关，逐字节可复现。Mac 上 6 进程实测：诊断约 114 条/秒（单进程约 26）、设计约 14 条/秒（单进程约 3.7，图谱是大头）、
  讲解约 136 条/秒、知识问答题库约 500 题（用完即止）。

### 2.8 物理组件层（physics/）
精选实验（`labsim/experiments/`）是起点。照片里是别的器材组合时，从组件库取出物体、连接件、表面和仪器，配一条规律模板，
组合成 `ComposedExperiment`——它就是一个普通的 `labsim.core.Experiment`，引擎、面板、EIG、可辨识性图谱、孪生、不确定度预算和事实单都不用改。

- **四层**：组件（`components.py`，带物理参数和自己的误差机制 `Effect`）→ 仪器（`instruments.py`，真实量 → 读数，尽量仿射）
  → 规律模板（`templates.py`，算真实物理量并给出线性化）→ 装置 `Setup`（模板 + 组件和仪器 + 控制量 + 方案），`compose.py` 把它变成实验。
- **单故障**：一个原因 = 一个 `Effect`，"只有随机误差"时全部关闭。`Setup.exact_in_faults` 里的机制表示"课本近似不成立"，
  在任何故障原因下都打开（镜像牛顿第二定律：随机误差 = 课本模型 a = F/M，故障 = 精确连接体动力学 + 一个效应）。
- **线性参数**：组件和模板用 `Affine` 表示 c + U·θ；两个含 θ 的量不能相乘、不能开方，这时其中一个量只能放网格
  （例：系统机械能里的阻力；光电门测自由落体时宽度误差按比例放网格，1/√g 才能保持线性）。
- **命名**：参数 `<组件>.<量>`；机制 key `<组件>_<机制>`；原因 id `<装置 id>.<key 或 cause_alias>`。装置 id 不含点号、不与精选实验重名。
  镜像精选实验时用 `cause_alias` / `kind_alias` 沿用原来的原因后缀和观测种类名。
- **控制量**：`Control.quantity`（mass / force / height / length / distance）；组件的 `control()` 把记录值改成真实值
  （漏算挂钩质量、摆长量到球顶、斜面距离当高度、长度量错参考点）。
- **变体与附加测量**：`Variant`（flags：fixed、reverse、detached、loaded、no_push、careful；`n_count` 改计数次数）和
  `AuxMeasurement` 由组件提供，菜单自动生成。
- **网格**：`relevant_nuisance` 只放当前原因 `Effect.uses` 和已观测附加测量的 `uses`（网格是笛卡儿积）；
  同一物理量既线性又出现在分母里时用 `Setup.sync`（摩擦力斜拉）。
- **注册**：组合出来的实验**不进** `labsim` 注册表（训练/留出清单不变）；新组合默认 `held_out=True`，可作迁移测试。
- **验证**（`tests/test_physics_*.py`）：镜像（胡克、摩擦、牛二、机械能、单摆，含计时器和已知倾角等选项）对精选实验的每个原因、
  每种观测，读数均值和总标准差逐项一致（容差 1e-6，实测 ~1e-15），方案、菜单、线性化也一致；新组合零故障时复原物理常数，
  标准方案下可见的大故障 20 次模拟里引擎至少 14 次排第一，后验 < 0.5 s。

| 新组合 id | 装置 | 标准方案下能看出来的大故障 |
|---|---|---|
| `incline_newton2` | 斜面上用悬挂重物拉小车（槽码从车上取下挂到小盘，总质量不变），两光电门测 a | 遮光片宽度用错（斜率偏大） |
| `system_energy` | 悬挂重物通过滑轮带动水平导轨上的小车，验证系统机械能守恒 | 释放时推了一下（截距为正） |
| `spring_oscillator` | 弹簧振子周期与振子质量的关系（手机计时） | 漏算挂钩质量（T²–m 截距为正） |
| `photogate_free_fall` | 小球穿过光电门，v = d/Δt，验证 v² = 2gh 并测 g | 下落高度量错参考点（截距） |
| `incline_slide` | 木块沿斜面由静止下滑，释放即计时、光电门停表 | 起点标记量错（t–√(2s) 弯曲） |

**看图建模**（`from_photo.candidates_from_photo(rec)`）：输入 `RecognitionResult` 或同字段 dict，器材名先按 §2.3 词表归一化；
按必需器材组、可选器材、线索词、常见程度和识别出的 experiment_id 打分，由器材选计时器（光电门 / 打点计时器 / 秒表），
返回 `PhotoCandidate`（entry_id、score、options、reasons、open_questions、setup）。一个都匹配不上时返回空列表，交给"其他实验"通道。

### 2.9 蒸馏流水线（training/distill.py）
只在构建训练数据时运行；产品里从不调用任何外部模型。唯一借鉴格物论文的做法：**教师盲做**——教师看到的提示和学生训练时的提示完全相同
（同一个 `training/prompts.py`），看不到物理引擎的答案；**只保留与可核验标准一致的输出**——标准是物理引擎的判定（等价类内的最可能原因、
信息增益前两名的追加测量）以及声明校验器；总数不超过 `MAX_DISTILLED = 5248`。
- 教师来源：DeepSeek（API，key 放在 `~/.config/ppr/keys.env`，一行 `DEEPSEEK_API_KEY=...`，不进仓库、不打印）；
  没有 API 的教师（Hy4）走 `export_prompt_pack()` 导出提示词包，学生在别处跑完，`import_responses()` 用同一套过滤规则读回。
- 蒸馏样本 `source = "distill-deepseek" | "distill-hy4"`，训练提示里保留教师当时看到的指令；API 输出只能以这种方式（过滤后）进入训练，
  且在 bench 冻结之前定好；bench 样本、bench 变体不进蒸馏题目。

### 2.10 真实故障注入工具包（fieldkit/）
学生本人在真实器材上测量；成年人只做安全监督并按密封卡设置故障；记录表里没有任何关于人的信息。
流程：`schedule`（预先提交的随机表：公开表 + 密封表 + 密封卡 + 记录表模板，两个哈希写进研究计划；课次平衡、追加测量顺序拉丁方、图谱混淆检查）
→ `live`（每次基础方案做完，算信息增益推荐的追加测量，留回执与基础读数指纹）→ `ingest`（检查并哈希冻结记录表，转成引擎数据集）
→ `analysis`（学生本人启动预先规定的分析：拆封、核对、逐项追加后验、配对 Wilcoxon、Wilson 区间、图谱预测效度；只输出数字，解读由学生写）。
`cards.py` 生成可打印的离线密封卡，`sheet.py` 读写 CSV 记录表，`student_sheet.py` 生成手推清单，`catalog.py` 给出可注入的故障目录。
精选实验只是起点；单摆等走"其他实验"通道（通用关系族 + 通用故障算子），留出孪生做对照引擎。

### 2.11 服务器脚本（server/）
`server/` 自成一体（不导入 `lab_detective`），上传服务器时只需它和数据包；布局、命令、检查点全在 `server/README.md`。
- `bootstrap.sh`：预检（sm_120、CUDA ≥ 12.8、磁盘）、探网（PyPI 镜像 / ModelScope / HF）、训练环境与服务环境（vLLM 装不上则回退 HF 生成）、
  下载 Qwen3.5-4B 与 0.8B、0.8B 全链路冒烟（输出丢弃）。
- `common.py`：训练 JSONL 读取与校验、对话模板渲染和"只对最后一条助手回复算损失"的掩码（prompt 必须以空思考块结尾）、
  确定性打乱（只由 seed、epoch、id 决定）、泄漏检查、打乱标签对照。
- `train_sft.py`：LoRA r=16、α=32、dropout 0.05，挂在语言部分的线性层，视觉塔/词嵌入/lm_head 冻结；bf16 + 梯度检查点；
  默认 max_len 4096（诊断样本约 2,000–3,000 token），超长样本不截断——正式训练遇到超长即拒绝开训；
  `--max-minutes` 时间闸、断点续跑；`--shuffle-labels` 与 `--match-steps-from` 保证打乱标签对照与 S1 步数和批次一致；`--dry-run` 报 token 分位数。
- `merge_and_eval.py`：合并适配器；在 bench JSONL 上贪心批量生成（vLLM 或 HF），支持 3-shot（示例只取自训练池）、断点续跑，每条回复检查有无思考标记。
- `convert_adapter.py`：PEFT LoRA → mlx-vlm 适配器（转置 A/B、scale = α/r、改模块名前缀），自带名称核对子命令。

### 2.12 界面（app.py + ui/）
`app.py` 只负责页面骨架和侧栏；每一步一个文件：`step_identify`（拍照识别）、`step_design`（实验设计）、`step_measure`（测量）、
`step_diagnose`（诊断）、`step_explain`（讲解与知识点）。纯逻辑模块不 import streamlit，可单独测试：
`catalog`（实验目录：精选实验 + "其他实验"，实验键是字符串，同一份 spec 永远得到同一个孪生）、`lab`（数据表 ↔ 观测、逐行实时对照、
事实单与幽灵曲线封装、测量范围建议、修正后复测）、`llm`（探测本机模型服务，0.5 秒超时）、`charts`（Altair）。
`theme` 是黑板主题，只用系统字体，完全离线。
- **默认脚本模式**：模板讲解，不需要模型；侧栏打开"模型模式"才探测本机地址，探测不到就继续用脚本模式。
- **先预测再测量**：学生写下自己的预测（或明确跳过）之前，孪生的预测数字不显示。
- **手动选择永远可用**：流水线不覆盖学生的选择。
- `tests/test_app.py` 检查界面不 import `explain_api` / `zhipuai` / `intent_classifier`，页面文案里没有固定实验数。

## 3. 统一约定
- 纯 numpy/scipy，不引入新依赖（需要时在 pyproject 里集中加，由主会话处理）。
- 随机性只用 `np.random.Generator`，所有生成函数接受 `seed`。
- 每个模块配 `tests/test_<模块>.py`，单个测试 < 10 秒；整套测试 < 2 分钟。
- 本机内存上限：我们启动的所有进程合计 ≤ 24 GB；测试和生成任务控制在 < 2 GB。
- 产品和测试不调用任何外部 API；只有蒸馏脚本在用户提供 key 后手动运行，评测对照的 API 运行器同理且只由学生启动。
- 提交信息不写 AI 工具名；AI 写的代码记进 docs/AI_USAGE.md。
