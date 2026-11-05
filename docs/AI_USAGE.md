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
- 提示词日志放在 `docs/isef/isef-upgrade-logs/`（建设中），完整的会话记录另行留存（会话记录的保存位置未核实）。
- AI 编写的代码默认流程：AI 编写 → GPT 代理审查 → 并入。尚未落地的模块状态为「计划中」。

## 2. 模块归属表

说明：

- 「计划中」的模块文件**尚未存在**；其路径是规划位置，落地时按实际路径更正（未核实）。
- 「作者」列只写事实：AI 实际改过的文件才写 AI。

| 模块/文件 | 作者（AI/学生） | 审查者 | 状态 |
| --------- | --------------- | ------ | ---- |
| `tools/yolo/*`（`train.py`、`prepare_dataset.py`、`extract_video_frames.py`、`autolabel.py`、`autolabel_glm.py`、`gen_label_gallery.py`、`pilot_yolo_probe.py`、`class_descriptions.json`） | 脚本本体为 P1 阶段原作（作者归属未核实）；**2026-11-05 由 AI 移动文件并修改路径常量**（`PROJ` / `PROJ_DIR` / `DATA_DIR` 改为 `parents[2]`） | 未核实 | 已完成（2026-11-05 重构提交） |
| `detect.py` | P1 阶段原作（作者归属未核实）；**2026-11-05 AI 修改**：默认权重路径改为 `assets/weights/best.pt`，并加入环境变量 `PPR_YOLO_WEIGHTS` 覆盖 | 未核实 | 已完成（2026-11-05 重构提交） |
| `pyproject.toml` | AI（2026-11-05） | 未核实 | 已完成（2026-11-05 重构提交） |
| `.gitignore` | AI 追加 P2 相关忽略项（2026-11-05） | 未核实 | 已完成（2026-11-05 重构提交） |
| `README.md`、`docs/AI_USAGE.md` | AI 起草（2026-11-05） | 学生核对（未核实是否已完成） | 已起草 |
| `lab_detective/labsim/core.py` | AI | GPT 代理 | 计划中 / AI 编写，GPT 审查 |
| `engine.py` | AI | GPT 代理 | 计划中 / AI 编写，GPT 审查 |
| `panel.py` | AI | GPT 代理 | 计划中 / AI 编写，GPT 审查 |
| `eig.py` | AI | GPT 代理 | 计划中 / AI 编写，GPT 审查 |
| `experiments/*` | AI | GPT 代理 | 计划中 / AI 编写，GPT 审查 |
| `eval/*` | AI | GPT 代理 | 计划中 / AI 编写，GPT 审查 |
| `server/*` | AI | GPT 代理 | 计划中 / AI 编写，GPT 审查 |
| `convert_adapter.py` | AI | GPT 代理 | 计划中 / AI 编写，GPT 审查 |
| `local_llm.py` | AI | GPT 代理 | 计划中 / AI 编写，GPT 审查 |
| `watchdog.py` | AI | GPT 代理 | 计划中 / AI 编写，GPT 审查 |
| `app.py`（P2 新增 UI 部分） | AI | GPT 代理 | 计划中 / AI 编写，GPT 审查 |
| 研究计划 | 学生本人 | — | 学生本人完成（进度未核实） |
| 手推原因特征 | 学生本人 | — | 学生本人完成（进度未核实） |
| 亲手实现的故障模型（胡克零点偏移） | 学生本人 | — | 学生本人完成（进度未核实） |
| 物理卡审核 | 学生本人 | — | 学生本人完成（进度未核实） |

## 3. 变更日志

按日期倒序，新条目加在最上面。每条写清：日期、改动范围、哪些由 AI 完成。

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
