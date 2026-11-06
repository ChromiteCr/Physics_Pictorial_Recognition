# AI 使用台账（ISEF Form 2A）

> 记录本仓库中**由 AI 参与编写的代码**，用于 ISEF Form 2A 披露。
> 本文件由 AI 起草；Form 2A 的披露责任人是学生本人，**须由学生核对、修改并确认后才算数**。
> 凡无法从仓库或会话记录核实的信息，一律标「未核实」，不推测。

## 1. 使用政策

### 1.1 使用了哪些 AI 工具

| 工具 | 用途 | 范围限制 |
| ---- | ---- | -------- |
| 主编码助手（Anthropic Claude 系列模型，经 Claude Code 使用；具体型号以会话记录为准）及其在会话内派生的子代理 | 编写与修改工程代码、测试、工程文档（README、台账、设计文档的工程部分） | 不撰写研究计划、摘要、海报、论文、结论 |
| GPT 代理（另一台机器上运行；具体型号未核实） | **仅用于**代码审查，以及对关键模块做独立重新实现（用来交叉验证） | 不直接向本仓库写入未经审查的代码 |

### 1.2 与 ISEF 规则 8 的边界

- AI **不得**撰写学生的研究计划、摘要、海报、论文和结论，**不得**向学生提供引用文献或参考文献列表。上述内容全部由学生本人完成。
- 工程文档（README、本台账、代码注释、工程设计说明）允许由 AI 起草。
- 台账中标为「学生本人」的条目，AI 不代写、不代为实现。

### 1.3 披露方式

- 仓库所有者的规则：**提交信息里刻意不写任何 AI 工具名**。因此本台账承担披露责任：每次有 AI 参与的改动，都要在第 3 节「变更日志」里登记日期和范围。
- 提示词日志放在 `docs/isef/isef-upgrade-logs/`（规划阶段的对话、评审、调研原文，以及后续阶段的工作流脚本存档）；完整的会话记录保存在
  `~/.claude/projects/-Users-billgao-Documents-Coding-SCPKU-Proj/` 下的 jsonl 文件（两个工作流各自的 journal 在同目录 `subagents/workflows/` 下）。
- AI 编写的代码默认流程：AI 编写 → GPT 代理审查（另一台机器，材料由学生转交）→ 并入。「审查者」一列写"待审查"表示审查尚未发生或未核实。
- 产品不调用任何外部模型；训练样本的标签来自确定性代码。只有蒸馏脚本在学生提供 key 后手动运行时才会调用教师 API，其输出以过滤后的蒸馏样本形式进入训练（见 ARCHITECTURE §2.9），运行日志另存。

## 2. 模块归属表

说明：「作者」列只写事实：AI 实际改过的文件才写 AI；「审查者」一列未核实的写"待审查"。

| 模块/文件 | 作者（AI/学生） | 审查者 | 状态 |
| --------- | --------------- | ------ | ---- |
| `tools/yolo/*`、`detect.py` | 脚本本体为 P1 阶段原作（作者归属未核实）；2026-11-05 由 AI 移动文件、修改路径常量与默认权重路径 | 待审查 | 已完成 |
| `pyproject.toml`、`.gitignore` | AI | 待审查 | 已完成 |
| `README.md`、`docs/ARCHITECTURE.md`、`docs/AI_USAGE.md`、`docs/isef/*.md`（工程文档） | AI 起草（2026-11-05 起；2026-11-06 整体重写） | 学生核对（未核实是否已完成） | 已起草 |
| `lab_detective/labsim/`（`core.py`、`registry.py`、`generic.py`、`experiments/*`） | AI（物理推导与默认先验为 AI 起草的工程假设，未经学生审核） | 物理复核由 AI 子代理做过；GPT 与学生审核待做 | 已完成代码与测试；物理卡待学生审核 |
| `lab_detective/physics/*`（物理组件层） | AI | 待审查 | 已完成；新组合的装置细节待学生确认 |
| `engine.py`、`panel.py`、`eig.py`、`atlas.py`、`uncertainty.py`、`twin.py`、`facts.py` | AI | 待审查（GPT 差分测试待做） | 已完成 |
| `lab_detective/knowledge/*`（知识点图谱、实验卡） | AI 起草 | 待学生核对 | 已完成 |
| `lab_detective/vision/*`、`tutor/*`、`llm/*` | AI | 待审查 | 已完成 |
| `lab_detective/training/*`（`prompts.py`、`splits.py`、`tasks_*.py`、`generate.py`、`distill.py`、`format.py`、`package.py`） | AI | 待审查 | 已完成；扣下的原因与配比待学生确认 |
| `lab_detective/bench/*`（`freeze.py`、`stats.py`、`recognition_bench.py`、`review_gallery.py`） | AI | 待审查 | 已完成；冻结由学生本人运行，AI 从未运行冻结 |
| `lab_detective/fieldkit/*` | AI | 待审查 | 已完成；真实数据与分析由学生本人启动 |
| `server/*` | AI | 待审查 | 已完成脚本与单元测试；尚未在 5090 上由学生运行验证 |
| `app.py`、`ui/*` | AI | 待审查 | 已完成（含测试） |
| `tests/*` | AI | 待审查 | 已完成 |
| 物理 bench、高保真错配模拟器、对照模型运行器 | AI | 待审查 | 进行中 |
| `docs/student/*` | AI 起草的操作清单和决定清单（不含研究计划文字、不含文献） | 学生使用 | 已完成 |
| 研究计划、摘要、海报、论文、结论、未来工作、文献检索 | 学生本人 | — | 学生本人完成（进度未核实） |
| 手推原因特征 | 学生本人 | — | 学生本人完成（进度未核实） |
| 亲手实现的故障模型（胡克零点偏移） | 学生本人 | — | 学生本人完成（进度未核实） |
| 物理卡审核、识别照片标签审核、bench 冻结与哈希、全部阈值与登记项 | 学生本人 | — | 学生本人完成（进度未核实） |
| 真实故障注入实验的测量与分析 | 学生本人（成年人只做安全监督和按密封卡设置故障） | — | 未开始（未核实） |

## 3. 变更日志

按日期倒序，新条目加在最上面。每条写清：日期、改动范围、哪些由 AI 完成。

### 2026-11-06（五）：文档整体重写

- AI 重写 `README.md`（P2 概览在前、P1 历史在后；版本 P2a、P2b）、`docs/ARCHITECTURE.md`（目录补 `fieldkit/`、`bench/` 各模块、`training/distill.py`、`server/`、`ui/`，新增 §2.9–§2.12）、
  `docs/isef/2026-10-31-isef-upgrade-design.md`（按当前方向整体重写，文末附版本沿革）、`docs/isef/2026-11-05-implementation-plan.md`（已建成 / 进行中 / 待办 / 排期）、
  `docs/isef/innovation-and-boundaries.md`、`docs/isef/bench-comparators.md` 的措辞，以及本台账。
- 设计文档里所有阈值、目标和登记项保持为学生待定事项（§14），文中没有替学生填写任何答案；文献线索仍只是线索。

### 2026-11-06（五）：第二轮开发（界面、服务器脚本复核）

- AI 重写 `app.py` 并新建 `ui/` 五步界面与 `tests/test_app.py`（中断保存的提交里有未完成部分，以 `docs/PROGRESS.md` 和测试结果为准）；
  复核并修改 `server/*`（序列长度 4096、超长样本拒绝开训、dry-run 报告 token 分位数等）。
  蒸馏流水线、打包脚本、冻结与统计库属于更早的开发提交，细节以 git 历史为准。
- 蒸馏教师的 key 由学生提供，不进仓库。

### 2026-11-05（四）：真实故障注入工具包与学生材料

- AI 新建 `lab_detective/fieldkit/`（随机表、密封卡、记录表、实时推荐、冻结与预先规定的分析）、`docs/student/*`（手推清单由代码生成，只描述物理过程，不放预期特征和修正）。
- 待学生确认：随机表的种子与承诺方式、故障配方里的幅度选项（AI 按各原因的先验范围挑选）。

### 2026-11-05（五）：物理组件层

- AI 新建 `lab_detective/physics/`：`base.py`（误差机制 Effect、组件、附加测量、主测量变体、控制量、装置 Setup、仿射量 Affine）、
  `components.py`（小车、木块、小球、摆球、挂钩、弹簧、细线、滑轮、手拉、释放、导轨、斜面、木板、杠杆）、
  `instruments.py`（刻度尺、弹簧测力计、秒表、手机计时、光电门、打点计时器、天平、量角器、游标卡尺）、
  `templates.py`（11 条规律模板）、`compose.py`（Setup → 符合 labsim 接口的实验）、`catalog.py`（镜像 5 个精选实验 + 5 个新组合）、
  `from_photo.py`（识别结果 → 候选装置和未决问题），以及 `tests/test_physics_{core,mirror,compositions,from_photo}.py`。
- 组件的物理和默认先验是从已精修的精选实验里提取的（AI 提取，镜像测试证明与精选实验逐项一致）；新组合的物理推导写在代码注释里，
  其默认数值（小车 1 kg、倾角 5°、弹簧 20 g、钢球直径 2.00 cm、各误差的幅度范围等）是 AI 定的工程假设，**未经学生审核**。
- AI 修改 `docs/ARCHITECTURE.md`（创新点 1、3 的措辞，目录表补 `physics/`，新增 §2.8）、`docs/isef/innovation-and-boundaries.md`、
  `docs/isef/2026-11-05-implementation-plan.md`（labsim 一行不再写固定实验数，补组件层）和 `README.md` 版本记录。
- 待学生确认：5 个新组合的装置细节是否符合你们实验室的器材；它们默认是留出装置（迁移测试用），是否进训练由学生决定。

### 2026-11-05（四）：引擎标注的训练样本生成器

- AI 新建 `lab_detective/training/prompts.py`（提示词唯一来源和输出 schema）、`splits.py`（训练/留出划分、每个训练实验扣下一个原因、种子命名空间）、
  `tasks_diagnose.py`、`tasks_design.py`、`tasks_explain.py`、`tasks_knowledge.py`、`generate.py`（多进程命令行）和 `tests/test_training_tasks.py`。
- AI 修改 `docs/ARCHITECTURE.md`：目录表补这些模块，新增 §2.7（提示内容、输出 schema、划分、配比、生成命令和实测吞吐）。
- 样本标签全部来自确定性代码（引擎、EIG、孪生、不确定度预算、图谱、知识点图谱），生成过程不调用任何模型或 API。
- 待学生确认：6 个扣下的原因（`splits.WITHHELD_CAUSES`，AI 按"物理上值得考、能和随机误差分开、名称唯一"挑的）；诊断和设计的配比、
  设计任务里"相对不确定度超过 5% 时点数加倍"这条规划规则（AI 定的工程默认值，不是研究结论）。

### 2026-11-05（三）：识别 bench 与审核图册

- AI 新建 `lab_detective/bench/recognition_bench.py`（按拍摄时段切分、生成 test.jsonl、打分、规则基线）、`lab_detective/bench/review_gallery.py`（离线审核图册）、`tests/test_recognition_bench.py`。
- AI 修改 `docs/ARCHITECTURE.md`（新增 §2.6 识别 bench 接口；创新点 1 的措辞改成"起点而非固定集合"）和 `docs/isef/innovation-and-boundaries.md`（同一处措辞）。
- 照片标签**不由 AI 贴**：审核图册只给提示，没有预选，`review.json` 只含学生亲手确认的照片；bench 冻结由学生本人运行（AI 未运行 `freeze`）。
- 未经学生核实：切分间隔（现为 AI 为跑通流程生成的草稿，`split.json` 的 `status` 为 `draft`）、测试集里的 gold 标签（全部是规则暂定值，直到学生导出 `review.json`）。

### 2026-11-05（二）：README 与本台账

- AI 更新 `README.md`：新增 shields.io 徽章行、「P2：实验侦探」简介段、`uv` 安装说明、新的目录树、移动后的脚本/笔记/权重路径、版本记录新增 P2 行。
- AI 新建 `docs/AI_USAGE.md`（本文件）。

### 2026-11-05（一）：P2 仓库重构（提交 `9319109`）

- 提交说明：「P2 重构：YOLO 工具脚本移到 tools/yolo，旧笔记移到 docs/history，权重改放 assets/weights，新增 pyproject」。
- 由 AI 完成：
  - 8 个 YOLO 工具文件从仓库根目录移入 `tools/yolo/`，其中 `train.py`、`prepare_dataset.py`、`extract_video_frames.py`、`autolabel_glm.py`、`gen_label_gallery.py`、`pilot_yolo_probe.py` 的路径常量改为 `parents[2]`（`autolabel.py`、`class_descriptions.json` 仅移动）。
  - `2nd_step.md`、`3rd-training-output.md`、`tips.md` 移入 `docs/history/`。
  - `detect.py`：默认权重路径改为 `assets/weights/best.pt`，新增环境变量 `PPR_YOLO_WEIGHTS` 覆盖。
  - 新增 `pyproject.toml`（`uv` 环境；可选依赖组 `app`、`mac-llm`、`dev`）。
  - `.gitignore` 增加 `.venv/`、`assets/weights/`、`results/`、`*.egg-info/`、`.pytest_cache/`。
- 路径常量改动是否经过运行验证：未核实。
