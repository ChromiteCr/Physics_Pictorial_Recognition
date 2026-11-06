# 实验侦探 Lab Detective（力学实验 AI 伙伴）

![version](https://img.shields.io/badge/version-P2b-blue)
![last commit](https://img.shields.io/github/last-commit/ChromiteCr/Physics_Pictorial_Recognition)
![commit activity](https://img.shields.io/github/commit-activity/m/ChromiteCr/Physics_Pictorial_Recognition)
![stars](https://img.shields.io/github/stars/ChromiteCr/Physics_Pictorial_Recognition)

一个**完全离线**的中学力学实验 AI 伙伴：拍一张实验装置的照片，它认出这是什么实验，把这台装置变成可计算的**物理孪生**，
带学生先预测、再测量、边测边对照；数据偏离理论时，用贝叶斯推断找出**系统误差的来源**，并告诉学生下一次最该测什么，
最后依据物理事实单讲解，并关联到知识点。仓库历史上的第一阶段（P1，YOLO 器材识别 + 规则 + 在线大模型讲解）见本文后半部分。

> 本项目是学生本人的 ISEF 延续项目（版本线 P2）。这里的文档是工程记录；研究计划、摘要、海报、论文和结论由学生本人亲笔撰写，
> AI 参与编写代码的披露见 [docs/AI_USAGE.md](docs/AI_USAGE.md)。

## P2：实验侦探

### 它是什么

| 步骤 | 做什么 | 代码在哪 |
| --- | --- | --- |
| ① 拍照识别 | YOLO 器材框 + 画布修正 + 规则 + 本地视觉语言模型 + 一个澄清问题；手动选择永远可用，不在表里的装置走"其他实验" | `lab_detective/vision/`、`ui/step_identify.py` |
| ② 实验设计 | 先写下自己的预测（再看孪生的预期）；不确定度预算；本方案分不开哪些原因；建议的测量范围 | `twin.py`、`uncertainty.py`、`atlas.py` |
| ③ 测量 | 多行数据表，填一行就和孪生实时对照（一致性 z 值与提示） | `ui/step_measure.py`、`ui/lab.py` |
| ④ 诊断 | 嫌疑人看板（概率 + 证据）、幽灵曲线、"下一次测量"卡（各原因的预测值）、修正后复测 | `engine.py`、`eig.py`、`facts.py` |
| ⑤ 讲解 | 依据事实单讲解，声明校验器逐个核对数字、单位、方向词；关联知识点与前置知识 | `tutor/`、`knowledge/` |

**模型从不自己算物理**：所有数字来自确定性的物理代码（孪生、贝叶斯引擎、EIG、不确定度预算），整理成事实单（FactSheet）；
语言模型只负责看图、规划、解释和对话，它写的每个数字都要过声明校验器，过不了就换成模板讲解。
产品默认是**脚本模式**（模板讲解，不需要任何模型）；侧栏打开"模型模式"时，只连用户填写的本机服务，从不连外部 API。

### 创新点

1. **误差物理孪生**：中学力学实验的可执行前向模型，带系统误差物理（摩擦、车轮/滑轮转动惯量、绳子质量、遮光片宽度、
   弹性限度、调零方向……）和仪器噪声模型。已精修的课内与留出实验（`labsim/experiments/`）只是起点。
2. **物理组件层**（`lab_detective/physics/`）：物体、连接件、表面、仪器和规律模板，可以组合出没精修过的装置的孪生；
   镜像精选实验时与精选版逐项一致，另有若干新组合默认留出做迁移测试。组件也拼不出的装置，退到通用关系族 + 通用故障算子。
3. **可辨识性图谱**：课本标准方案能/不能暴露哪些误差，哪一次追加测量能把它们分开（蒙特卡洛贝叶斯最优判别，带区间）。
4. **信息增益的下一次测量**：按期望信息增益（EIG）从追加测量菜单里选最能区分剩余嫌疑人的一项，并在学生动手前给出每个嫌疑人对应的预测读数。
5. **物理引擎当老师**：小模型（Qwen3.5-4B，LoRA）的主要训练标签来自对含噪、学生式数据做贝叶斯逆推断的物理引擎，
   不是教师模型的解题轨迹；另有不超过 5,248 条蒸馏样本（教师盲做，只保留和物理引擎判定一致的输出）。
6. **探究式教学**：实验前预测、实验中实时一致性检查、先让学生自己猜嫌疑人再揭示排名、误差与知识点关联、修正后复测验证。
7. **真实器材闭环**：故障注入工具包（预先提交的随机表、密封卡、记录表、预先规定的分析），检验孪生的预测（sim-to-real）。
   真实数据由学生本人采集和启动分析。

不宣称任何固定的"支持实验总数"：精选实验是起点，其余装置由组件层组合，识别不了的归入"其他实验"通道。
本项目与格物论文无相似之处：没有代码沙箱、没有工具调用循环、没有投票、不做文本题辅导。

### 评测

两个由学生冻结的 bench（开发侧只生成和打分，**从不调用冻结命令**）：
- **物理 bench**：诊断、下一次测量、迁移到留出实验；
- **识别 bench**：按拍摄时段切分的本人照片；标签由学生在审核图册里逐张确认。

对照：未微调的 4B、Qwen3.8-27B、DeepSeek V4.1 Flash 等（对照模型的调研见 [docs/isef/bench-comparators.md](docs/isef/bench-comparators.md)）。
目标由学生在冻结前亲笔预先登记，结果如实报告。API 模型只作评测对照和离线蒸馏教师，**不进入产品**。

### 现状（P2b）

已完成并有测试：物理孪生与贝叶斯引擎、特征面板、EIG 与可辨识性图谱、不确定度预算、事实单、物理组件层、知识点图谱与实验卡、
识别流水线、探究式导师与声明校验器、训练样本生成器（诊断/设计/讲解/知识四类）、蒸馏流水线、训练包打包、
5090 服务器脚本（引导、LoRA 训练、合并评测、适配器转 MLX）、识别 bench 与审核图册、真实故障注入工具包、新界面。
进行中：引导/识别训练样本与蒸馏任务、物理 bench 的生成与打分、高保真错配模拟器、对照模型运行器。
需要学生本人：研究计划与各项阈值、物理卡审核、识别照片标签审核、冻结 bench、正式运行与真实实验。进度细节见 [docs/PROGRESS.md](docs/PROGRESS.md)。

### 怎么跑

环境（Python >= 3.11，推荐 uv）：

```bash
uv venv && uv pip install -e '.[dev,app]'      # 物理核心 + 界面 + YOLO + 测试
uv pip install -e '.[mac-llm]'                 # 可选：Apple Silicon 上的本地视觉语言模型服务（mlx-vlm）
```

启动界面（完全离线，默认脚本模式）：

```bash
streamlit run app.py
```

YOLO 权重放在 `assets/weights/best.pt`（不进 git），或用环境变量 `PPR_YOLO_WEIGHTS` 指向。没有权重时识别步骤退回手动选择，其余步骤不受影响。
想用模型模式：在 Mac 上用 `mlx_vlm.server` 起本机服务，再在侧栏打开"模型模式"。

测试（单个测试 < 10 秒）：

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
```

生成训练数据（全部由确定性代码标注，逐字节可复现，与进程数无关）：

```bash
python -m lab_detective.training.generate --n-diagnose N --n-design N --n-explain N --n-knowledge N --workers 6
# 输出 results/train/<任务>.jsonl、results/val/<任务>.jsonl、results/generate_report.json
```

打包给 5090（只打包被引用的图片，报告大小和预计上传时间）：

```bash
python -m lab_detective.training.package --train results/train/*.jsonl --val results/val/*.jsonl \
    --bench <冻结后的 bench JSONL> --out results
```

服务器上的引导、LoRA 训练（S1 与打乱标签对照）、合并评测、适配器转 MLX，全部步骤见 [server/README.md](server/README.md)；
学生操作清单见 [docs/student/](docs/student/)。识别照片标签审核：`python -m lab_detective.bench.review_gallery`。
真实故障注入实验：`lab_detective/fieldkit/`（说明见 [docs/student/06_真实实验工具包.md](docs/student/06_真实实验工具包.md)）。

### 文档导航

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：模块布局与接口约定（开发者必读）
- [docs/isef/2026-10-31-isef-upgrade-design.md](docs/isef/2026-10-31-isef-upgrade-design.md)：当前设计（含 ISEF 合规）
- [docs/isef/2026-11-05-implementation-plan.md](docs/isef/2026-11-05-implementation-plan.md)：已建成 / 待办 / 排期
- [docs/isef/innovation-and-boundaries.md](docs/isef/innovation-and-boundaries.md)、[docs/isef/bench-comparators.md](docs/isef/bench-comparators.md)
- [docs/AI_USAGE.md](docs/AI_USAGE.md)：AI 使用台账（Form 2A）

### 项目结构

```text
Physics-Pictorial-Recognition/
├── app.py                    # Streamlit 入口（页面骨架，五步界面在 ui/）
├── ui/                       # 界面：每步一个文件 + 纯逻辑（catalog、lab、llm、charts）+ 黑板主题
├── lab_detective/            # P2 核心包
│   ├── labsim/               #   精选实验的物理孪生 + 通用关系族（接口见 labsim/core.py）
│   ├── physics/              #   物理组件层：组件、仪器、规律模板、组合器、看图建模
│   ├── engine.py panel.py eig.py atlas.py uncertainty.py twin.py facts.py   # 引擎、面板、信息增益、图谱、预算、孪生、事实单
│   ├── knowledge/            #   实验卡与知识点图谱
│   ├── vision/               #   照片 → 构型 → 实验
│   ├── tutor/                #   探究式导师、上下文组装、声明校验器
│   ├── llm/                  #   OpenAI 兼容模型客户端
│   ├── training/             #   训练样本、提示词唯一来源、蒸馏、打包
│   ├── bench/                #   冻结、统计、识别 bench、审核图册
│   └── fieldkit/             #   真实故障注入实验工具包
├── server/                   # 5090 脚本：bootstrap、train_sft、merge_and_eval、convert_adapter
├── tests/                    # pytest，每个模块一个文件
├── tools/yolo/               # YOLO 数据与训练工具（P1 起）
├── detect.py                 # YOLO 器材识别（默认权重 assets/weights/best.pt）
├── physics.py intent_classifier.py animate.py explain_api.py   # P1 遗留模块，新界面不再调用
├── docs/                     # ARCHITECTURE、AI_USAGE、PROGRESS、isef/、student/、history/
├── assets/weights/           # 本地权重（不进 git）
├── datasets/                 # 完整本地数据集（gitignore）
└── data/                     # P1 的 YOLO 数据集配置与数据
```

---

## P1 历史：器材识别 + 规则 + 在线讲解（已被 P2 取代）

P1 是一个 Streamlit 网页：**YOLOv8** 识别器材，人工核对/补标注，规则（拿不准时调 GLM 兜底）判断实验，手填实测数据算物理量，
再生成示意动画和 GLM 讲解。P2 的新界面不再调用 GLM，P1 的 `physics.py`、`intent_classifier.py`、`animate.py`、`explain_api.py`
作为遗留模块留在仓库根目录（新界面不 import 它们；`tests/test_app.py` 检查这一点）。YOLO 识别和训练工具仍在使用。

### YOLO 器材检测

训练集共 **8 类**编号（编号固定不挪用，即使某类不再用于推理也保留占位）：

| id | 类别 | 说明 |
| --- | --- | --- |
| 0 | `cart` | 小车 |
| 1 | `track` | 导轨/轨道（含底座支架类支撑结构） |
| 2 | `spring` | ~~弹簧~~（占位保留，标注/推理全程忽略） |
| 3 | `string` | 细线 |
| 4 | `ruler` | ~~标定尺~~（占位保留，标注/推理全程忽略，原因见 detect.py 注释） |
| 5 | `dynamometer` | 弹簧测力计整机 |
| 6 | `wooden_block` | 木块 |
| 7 | `iron_block` | 铁块/钢球 |

`spring`、`ruler` 两类在 `detect.py` 的 `EXCLUDED_FROM_DETECT` 里被过滤，推理结果不会输出，但编号不重新分配给别的类别。

**权重**：当前使用第三轮训练模型 `assets/weights/best.pt`（mAP@0.5 = 0.577），**不进 git**（`.gitignore` 排除 `*.pt`）；也可用 `PPR_YOLO_WEIGHTS` 指向别处。
第三轮训练（8 类 schema，809 图，V100 32GB）验证集 mAP@0.5 = **0.577**：

| 类别 | mAP50 | 备注 |
| --- | --- | --- |
| cart | 0.977 | 很好 |
| wooden_block | 0.800 | 好 |
| track | 0.814 | 好 |
| dynamometer | 0.493 | 一般，建议人工核对 |
| string | 0.260 | 差，建议人工核对 |
| iron_block | 0.121 | 差，建议人工核对 |

（数值与 [docs/history/3rd-training-output.md](docs/history/3rd-training-output.md) 训练日志一致；另 5 类（不含 iron_block）mAP@0.5 = 0.669）

### 训练一个新的 YOLO 模型

数据来源（标注格式统一为 YOLO `.txt`：`<class_id> <x_center> <y_center> <width> <height>`，坐标 0–1 归一化）：
`data/self_captured/`（自己拍摄）、`data/openclaw/`（网络补充）、`data/video_frames/`（视频抽帧，走
`tools/yolo/extract_video_frames.py` → `tools/yolo/autolabel_glm.py` 预标注 → `tools/yolo/gen_label_gallery.py` 网页核对 → 人工修正）。

```bash
python tools/yolo/prepare_dataset.py --val-ratio 0.15 --test-ratio 0.05      # 合并数据集
python tools/yolo/prepare_dataset.py --exclude openclaw                      # 不含网络图片
pip install -r requirements.txt && python tools/yolo/train.py                # 训练服务器用旧依赖清单
python detect.py 照片.jpg --conf 0.25                                         # 命令行单张测试
```

配置（模型大小、epoch、batch、设备）在 `tools/yolo/train.py` 顶部；输出权重在 `runs/detect/train/weights/best.pt`，
复制到 `assets/weights/best.pt` 才会被 `detect.py` 默认使用。5090（Blackwell，sm_120）需要 CUDA 12.8+ 的 PyTorch，见 `requirements.txt` 开头注释。

### 常见问题

**Q: 服务器上 `import cv2` 报 `libGL.so.1` 找不到？** 精简云镜像常缺这个库；`requirements.txt` 已用 `opencv-python-headless`。

**Q: 训练时出现 CUDA 错误？** 改 `tools/yolo/train.py` 里 `DEVICE = "cpu"`；GPU 架构和 CUDA 版本要求见脚本顶部注释。

**Q: 检测精度很低（尤其 string / iron_block）？** 这两类训练数据不够；界面的识别步骤里可以直接在图上拖框补标注，或往 `data/video_frames/` 补样本重训。

**Q: `tools/yolo/autolabel_glm.py` 报 429 限速？** 脚本内置指数退避重试并支持断点续标，直接重跑即可。这个脚本只用于 P1 阶段的离线标注，不属于产品。

## 📝 版本记录

| 版本 | 日期       | 变更内容                                                                                                   | 类型      |
| ---- | ---------- | ------------------------------------------------------------------------------------------------------------ | --------- |
| P2b  | 2026-11-06 | 训练数据与评测基准、5090 训练脚本、新界面：引擎标注的训练样本生成器与蒸馏流水线、打包脚本、识别 bench 与审核图册、冻结与统计库、服务器脚本（引导、LoRA 训练、合并评测、转 MLX）、真实故障注入工具包、五步新界面（离线、默认脚本模式）、文档按当前方向重写 | feat      |
| P2a  | 2026-11-06 | 物理孪生、贝叶斯引擎、信息增益与可辨识性图谱、组件组合层：精选实验的物理孪生与贝叶斯引擎、EIG 与图谱、不确定度预算、事实单、物理组件层（组合出新装置的孪生，镜像精选实验逐项一致） | feat      |
| P2   | 2026-11-05 | 进入 ISEF 升级阶段（实验侦探）：仓库重构（YOLO 工具移入 tools/yolo、旧笔记移入 docs/history、权重改放 assets/weights、新增 pyproject 与 uv 环境） | milestone |
| P1b3 | 2026-07-31 | 第二轮论文修改：§3.2 新增模型参数表、§4.2 结果改为表格、新增新类别退化分析（docx + main.tex 同步）                 | feat      |
| P1b2 | 2026-07-31 | 修正 README 第三轮 mAP 数值（0.743→0.577）与数据集规模（1296→809），与训练日志/论文对齐                    | fix       |
| P1b1 | 2026-07-14 | GLM 讲解的知识点清单改成"术语+简短展开"结构，不再是光秃秃的术语词条                                        | fix       |
| P1b  | 2026-07-14 | 所有耗时操作加黑板粉笔风格加载进度条；Part 2 计算结果改用指标卡片+徽章展示，不再用 st.json                     | feat      |
| P1a  | 2026-07-14 | 选项卡式导航替换侧边栏；黑板粉笔视觉主题；Part 1 支持拖框手动补标注（含 streamlit-drawable-canvas 兼容性修复） | feat      |
| P1   | 2026-07-14 | 打通 Part 1(8 类 YOLO 器材识别) + Part 2(意图识别/物理计算/动画/GLM讲解)全链路，第三轮训练 mAP@0.5=0.577      | milestone |

## 📄 许可证

MIT License
