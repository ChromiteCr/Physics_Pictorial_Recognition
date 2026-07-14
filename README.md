# 力学实验智能助教 Physics Pictorial Recognition

一个面向中学物理力学实验的智能识别与讲解系统：拍一张实验初状态照片，**YOLOv8** 自动识别器材，人工核对/画框补标注后，**规则+GLM** 判断这是哪种实验，手填实测数据算出物理量，再生成示意动画和 **GLM** 讲解。

## 📋 项目概述

系统分两部分，对应网页上的两个选项卡：

- **① 识别初状态**：上传照片 → YOLOv8 识别器材 → 网页上人工核对/修正（含拖框手动补标注）
- **② 实验分析**：确认后的器材 → 意图识别（规则表打分，拿不准时调 GLM 兜底，可随时手动改）→
  按实验类型动态生成输入表单，填实测数据 → 计算物理量 + 生成示意动画 + GLM 讲解/公式/知识点清单

物理量不再依赖 CV 在照片里量出来——小样本 YOLO 精度不够、标定链路也不稳，所以实测数据都是用户看着实物手动输入，YOLO 只负责"这张照片里有什么器材"这一步。

**网页交互细节**：

- 顶部是 `① 识别初状态` / `② 实验分析` 两个选项卡（不是侧边栏），黑板粉笔风格视觉主题
- Part 1 检测完之后，除了表格里改类别/删行，还能直接在图上拖框圈出模型漏检的器材
- YOLO 推理、GLM 调用、动画生成这些耗时操作，都有统一的黑板粉笔风格加载进度条（不是原生 spinner）
- Part 2 的计算结果用指标卡片 + 结论徽章 + 公式条展示，不是一坨 `st.json`，避免学生把结果误认成代码

## 🎯 支持的实验类型

| 实验 | 学段 | 必需器材 |
| --- | --- | --- |
| 探究摩擦力大小的影响因素 | 初中 | 测力计 + (小车 或 木块) |
| 测量平均速度 | 初中 | 小车 + 导轨 |
| 弹簧测力计原理 / 胡克定律初步 | 初中 | 测力计 |
| 验证牛顿第二定律 | 高中 | 小车 + 导轨 |
| 验证动量守恒定律 | 高中 | 2 辆小车 + 导轨 + 细线 |
| 验证机械能守恒定律 | 高中 | 小车 + 导轨 |

完整定义（输入字段、计算公式）见 [physics.py](physics.py) 里的 `EXPERIMENT_TYPES` 注册表。

## 🔍 目标检测（Part 1）

YOLOv8 训练集共 **8 类**编号（编号固定不挪用，即使某类不再用于推理也保留占位）：

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

第三轮训练（8 类，1296 图，V100 32GB）验证集 mAP@0.5 = **0.743**：

| 类别 | mAP50 | 备注 |
| --- | --- | --- |
| cart | 0.977 | 很好 |
| wooden_block | 0.800 | 好 |
| track | 0.814 | 好 |
| dynamometer | 0.793 | 好 |
| string | 0.560 | 好，由于内容过小，建议人工核对 |
| iron_block | 0.512 | 一般，建议人工核对 |

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.10
```

### 安装依赖

```bash
pip install -r requirements.txt
```

硬件相关的 torch 安装方式见 `requirements.txt` 开头注释（按租用 GPU 架构选 CUDA 版本）。

### 启动网页

```bash
streamlit run app.py
```

想用 GLM 相关功能（意图识别兜底、生成讲解）需要先设置环境变量 `GLM_API_KEY`（或 `ZHIPUAI_API_KEY`）；没配也能跑，只是这两处会显示警告，不影响器材识别/物理计算/动画。

### 项目结构

```text
Proj/
├── .streamlit/config.toml    # 网页主题配色 + 关闭热重载(避免 torch 内省报错)
├── app.py                    # Streamlit 网页入口（Part 1 + Part 2）
├── detect.py                 # Part 1：YOLO 器材识别
├── physics.py                # Part 2：EXPERIMENT_TYPES 注册表 + 物理量计算
├── intent_classifier.py      # Part 2：实验意图识别（规则 + GLM 兜底）
├── animate.py                # Part 2：示意动画生成
├── explain_api.py            # Part 2：GLM 讲解/公式/知识点生成
├── train.py                  # YOLOv8 训练脚本
├── prepare_dataset.py        # 合并各来源数据集，切分 train/val/test
├── extract_video_frames.py   # 从实验视频抽帧
├── autolabel_glm.py          # 用 GLM 多模态模型辅助标注抽出的帧
├── autolabel.py               # 用 Grounding DINO 零样本辅助标注
├── gen_label_gallery.py      # 生成标注结果的可视化网页，方便人工核对
├── class_descriptions.json   # GLM 自动标注用的类别描述（可编辑）
├── requirements.txt
└── data/
    ├── dataset.yaml               # YOLO 数据集配置（8 类）
    ├── self_captured/             # 自拍数据
    ├── openclaw/                  # 网络补充数据
    ├── video_frames/              # 视频抽帧 + GLM/人工标注
    │   ├── frame_hints.json       # 每张帧的"预期包含物体"提示（喂给 GLM）
    │   └── images/ labels/
    └── dataset/                   # 合并后的训练集（prepare_dataset.py 自动生成）
        ├── images/{train,val,test}/
        └── labels/{train,val,test}/
```

## 📚 训练一个新模型

### 1. 准备标注数据

三种数据来源，标注格式统一是 YOLO `.txt`（`<class_id> <x_center> <y_center> <width> <height>`，坐标 0-1 归一化）：

- `data/self_captured/` — 自己拍摄的照片，人工标注为主
- `data/openclaw/` — 网络补充图片
- `data/video_frames/` — 从实验视频抽帧，走 `extract_video_frames.py` → `autolabel_glm.py`
  自动预标注 → `gen_label_gallery.py` 生成网页核对 → 人工修正后放回对应 `labels/`

### 2. 合并数据集

```bash
python prepare_dataset.py --val-ratio 0.15 --test-ratio 0.05
python prepare_dataset.py --exclude openclaw   # 打包训练集但不含网络图片
```

### 3. 训练

```bash
python train.py
```

配置（模型大小、epoch、batch、设备）在 `train.py` 顶部改，输出权重在
`runs/detect/train/weights/best.pt`（`detect.py` 的默认权重路径）。

### 4. 命令行单独测试识别效果

```bash
python detect.py 照片.jpg --conf 0.25
```

输出 `<名称>_detected.jpg`（带框结果图）和 `<名称>_components.json`（器材列表）。

## 🛠️ 常见问题

**Q: 服务器上 `import cv2` 报 `libGL.so.1` 找不到？**
A: 精简版云服务器镜像常缺这个系统 OpenGL 库。项目里全程没用 cv2 的 GUI 函数，
`requirements.txt` 已经用 `opencv-python-headless` 代替完整版，装它就不需要这个系统库了。

**Q: 训练时出现 CUDA 错误？**
A: 改 `train.py` 里 `DEVICE = "cpu"`；GPU 架构和 CUDA 版本要求见 `requirements.txt`/`train.py` 顶部注释。

**Q: 检测精度很低（尤其 string / iron_block）？**
A: 这两类目前训练数据不够，`app.py` 的 Part 1 页面里可以直接在图上拖框手动补标注，
或者往 `data/video_frames/` 补更多样本重新训练。

**Q: GLM API 报 429 限速？**
A: `autolabel_glm.py` 内置指数退避重试，也支持断点续标——直接重跑同一条命令，
已经标注成功的图片会自动跳过，只补标失败的。

## 📝 版本记录

| 版本 | 日期       | 变更内容                                                                                                   | 类型      |
| ---- | ---------- | ------------------------------------------------------------------------------------------------------------ | --------- |
| P1b1 | 2026-07-14 | GLM 讲解的知识点清单改成"术语+简短展开"结构，不再是光秃秃的术语词条                                        | fix       |
| P1b  | 2026-07-14 | 所有耗时操作加黑板粉笔风格加载进度条；Part 2 计算结果改用指标卡片+徽章展示，不再用 st.json                     | feat      |
| P1a  | 2026-07-14 | 选项卡式导航替换侧边栏；黑板粉笔视觉主题；Part 1 支持拖框手动补标注（含 streamlit-drawable-canvas 兼容性修复） | feat      |
| P1   | 2026-07-14 | 打通 Part 1(8 类 YOLO 器材识别) + Part 2(意图识别/物理计算/动画/GLM讲解) 全链路，第三轮训练 mAP@0.5=0.577      | milestone |

## 📄 许可证

MIT License
