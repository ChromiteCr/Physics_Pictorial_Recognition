# P2 实验侦探：架构与模块接口（开发者必读）

> 这份文件是并行开发的唯一接口约定。改接口先改这里。所有文字面向中学生的部分用中文。

## 0. 产品与创新点（物理为主）

**产品**：拍一张力学实验装置的照片 → 认出是什么实验（开放集，力学优先）→ 把这台装置变成一个**参数化物理孪生**
→ 实验前给出预测和不确定度预算 → 带学生一步步测量，每填一行数据就和孪生对照 → 数据偏离时用贝叶斯推断找出系统误差来源，
推荐最能区分各原因的下一次测量 → 依据物理事实单讲解，并关联知识点。完全离线，本地小模型。

**自己的创新点**（与格物论文无任何相似；格物是文本题库上的解题辅导 + 代数沙箱 + 投票）：
1. **误差物理孪生**：中学力学实验的可执行前向模型，带系统误差物理（摩擦、车轮/滑轮转动惯量、绳子质量、遮光片宽度、弹性限度、调零方向……）和仪器噪声模型。已精修的课内与留出实验只是起点，其他装置由物理组件（通用关系族 + 通用故障算子，见 `labsim/generic.py`）拼出来。
2. **可辨识性图谱**（测量物理）：课本标准方案能/不能暴露哪些误差，哪一次追加测量能把它们分开（贝叶斯实验设计）。
3. **看图建模**：照片 → 装置构型 → 选定实验孪生并实例化参数。
4. **物理引擎当老师**：模型的主要训练标签来自对含噪数据做贝叶斯逆推断的物理引擎，不是教师模型的解题轨迹。
5. **真实器材闭环**：故障注入实验检验孪生的预测（sim-to-real）。
6. **探究式教学**：实验前预测、实验中实时一致性检查、按信息增益选"下一个问题/下一次测量"、误差与知识点关联。

**模型从不自己算物理**：所有数值来自物理引擎（确定性代码），模型负责看图、规划、解释和对话。
没有代码沙箱、没有工具调用循环、没有投票。蒸馏只借鉴格物的一点：教师盲做，只保留和**物理引擎判定一致**的输出；总条数 ≤ 5,248。

## 1. 目录与模块归属

```
lab_detective/
  labsim/core.py            公共接口（已定稿，见文件头建模约定）
  labsim/registry.py        自动发现 experiments/*.py 里的 Experiment 子类
  labsim/experiments/*.py   每个实验一个文件（hooke.py 是样板）
  labsim/generic.py         通用关系族 + 通用故障算子（"其他实验"和程序化任务族）
  engine.py                 贝叶斯后验（已定稿）
  panel.py                  特征面板（已定稿）
  eig.py                    期望信息增益、模拟闭环
  atlas.py                  可辨识性图谱
  uncertainty.py            不确定度预算（实验前）与传播
  twin.py                   孪生实例化：实验 id + 参数 → 预测曲线、预期读数、幽灵曲线
  facts.py                  事实单 FactSheet（引擎输出的唯一出口）
  knowledge/                实验卡 + 知识点图谱（JSON 数据 + 加载器）
  vision/                   照片 → 构型 → 实验（YOLO 线索 + VLM 提示 + 解析 + 澄清问题）
  tutor/                    探究式引导：会话状态机、上下文组装、模板讲解、声明校验器
  llm/                      模型客户端（mlx_vlm.server / OpenAI 兼容 API / 脚本化兜底）
  training/                 训练样本生成、蒸馏流水线、打包
    prompts.py              提示词的唯一来源（训练、蒸馏、bench、推理共用）+ 输出 JSON schema（见 §2.7）
    splits.py               训练/留出实验、每个训练实验扣下的原因、种子命名空间、训练/验证划分
    tasks_diagnose.py       诊断样本（含"自己提出原因"变体）；tasks_design.py 实验前规划；
    tasks_explain.py        事实单讲解（3 种措辞/槽位，必须过校验器）；tasks_knowledge.py 知识点问答
    generate.py             命令行：多进程生成 results/{train,val}/<任务>.jsonl
  bench/                    物理 bench、识别 bench（生成、冻结、评分）
    recognition_bench.py    识别 bench：按拍摄时段切分、生成 test.jsonl、打分、规则基线（见 §2.6）
    review_gallery.py       识别照片审核图册 review.html（学生逐张确认标签，导出 review.json）
server/                     5090 上的脚本（引导、训练、合并、评测、转 MLX）
app.py + ui/                Streamlit 界面
tests/                      pytest，每个模块一个文件
```

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

## 3. 统一约定
- 纯 numpy/scipy，不引入新依赖（需要时在 pyproject 里集中加，由主会话处理）。
- 随机性只用 `np.random.Generator`，所有生成函数接受 `seed`。
  开发识别模块时已经看过或跑过的照片（`recognition_bench.DEV_SEEN_STEMS`）所在的时段固定划进训练侧，不进测试集，记在 `split.json.dev_seen_forced_train`；可选时段不够 k 个时测试集相应缩小并警告，一个都不剩时拒绝切分。
- 每个模块配 `tests/test_<模块>.py`，单个测试 < 10 秒；整套测试 < 2 分钟。
- 本机内存上限：我们启动的所有进程合计 ≤ 24 GB；测试和生成任务控制在 < 2 GB。
- 不调用任何外部 API（蒸馏脚本除外，且只由用户提供 key 后手动运行）。
- 提交信息不写 AI 工具名；AI 写的代码记进 docs/AI_USAGE.md。
