# Physics Pictorial Recognition 物理图像识别系统

一个基于 **YOLOv8 深度学习** 的物理实验图像识别与自动分析系统。该系统能够自动识别物理实验装置（弹簧、木块、弹簧测力计等），从图像中提取关键物理量，并生成可视化的力图动画。

## 📋 项目概述

本项目旨在将计算机视觉与物理学结合，实现物理实验的智能化识别和分析。系统能够：
- 🔍 自动检测物理实验中的各类物体
- 📐 计算物体之间的空间关系和物理量
- 📊 生成力图及动画演示
- 💾 导出结构化的物理分析报告

## 🎯 核心功能

### 1. 目标检测 (Object Detection)
支持识别 **5 类物体**：
- `wooden_block` - 木块
- `iron_block` - 铁块  
- `spring` - 弹簧
- `string` - 细绳
- `dynamometer` - 弹簧测力计

### 2. 物理量提取 (Physics Extraction)
- **像素标定** - 通过弹簧测力计的已知尺寸（16cm）推算缩放比例
- **弹簧形变计算** - 自动测量弹簧当前长度和形变量 (ΔL)
- **受力分析** - 推断物体所受的力（重力、弹力、拉力等）
- **胡克定律应用** - F = k·x 公式计算弹力大小

### 3. 可视化输出 (Visualization)
- **力图叠加** - 在原图上绘制彩色力箭头和标签
- **弹簧动画** - 生成 GIF 动画展示弹簧伸长过程

## 🚀 快速开始

### 环境要求
```bash
Python >= 3.8
```

### 安装依赖
```bash
pip install ultralytics opencv-python numpy matplotlib pillow
```

### 项目结构
```
Physics_Pictorial_Recognition/
├── README.md                      # 项目说明
├── train.py                       # 模型训练脚本
├── detect.py                      # 推理入口
├── physics.py                     # 物理量提取模块
├── animate.py                     # 可视化和动画模块
├── prepare_dataset.py             # 数据集准备脚本
└── data/
    ├── self_captured/             # 自拍数据
    │   ├── images/                # 图片文件
    │   └── labels/                # YOLO 格式标注
    ├── openclaw/                  # 公开数据
    │   ├── images/
    │   └── labels/
    └── dataset/                   # 合并后的训练集（自动生成）
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
```

## 📚 使用指南

### 步骤 1: 准备训练数据

将标注好的数据放入 `data/` 目录：

**YOLO 格式标注说明** (.txt 文件):
```
<class_id> <x_center> <y_center> <width> <height>
```
其中坐标均为相对值（0-1 之间）。

**类别编号**：
- 0: wooden_block
- 1: iron_block
- 2: spring
- 3: string
- 4: dynamometer

**文件放置**：
```
data/self_captured/
├── images/
│   ├── exp1.jpg
│   ├── exp2.jpg
│   └── ...
└── labels/
    ├── exp1.txt
    ├── exp2.txt
    └── ...
```

### 步骤 2: 数据集准备

合并自拍数据和公开数据集，分割为训练/验证/测试集：

```bash
python prepare_dataset.py --val-ratio 0.15 --test-ratio 0.05
```

**参数说明**：
- `--val-ratio`: 验证集比例（默认 15%）
- `--test-ratio`: 测试集比例（默认 5%）
- `--seed`: 随机种子（默认 42）

输出示例：
```
self_captured: 50 张有效图片
openclaw: 120 张有效图片

数据集已生成（共 170 张）:
  train: 138 张
  val  : 25 张
  test : 7 张
```

### 步骤 3: 训练模型

使用 YOLOv8 训练目标检测模型：

```bash
python train.py
```

**训练配置**（可在 train.py 中修改）：
- 模型: YOLOv8 Small (`yolov8s.pt`)
- 轮数: 100 epoch
- 批大小: 16
- 图像尺寸: 640×640
- 设备: GPU 0 (如无 GPU，改为 "cpu")

**输出**：
```
runs/detect/train/
├── weights/
│   ├── best.pt          # 最优权重 ⭐
│   └── last.pt
└── results.csv
```

训练完成后会输出 mAP@0.5 指标。

### 步骤 4: 推理与分析

对新图片进行检测和物理分析：

```bash
python detect.py test_image.jpg --conf 0.25
```

**参数说明**：
- `image`: 输入图片路径（必需）
- `--weights`: 模型权重路径（默认使用训练好的 best.pt）
- `--conf`: 置信度阈值，0.0-1.0（默认 0.25）
- `--save-dir`: 输出目录（默认 output/）

**输出文件**：
```
output/
├── test_image_detected.jpg        # 检测结果图（带 bounding box）
├── test_image_physics.json        # 物理量提取报告
├── test_image_force_overlay.jpg   # 力图叠加图
└── test_image_animation.gif       # 弹簧动画（如有弹簧）
```

## 📊 输出示例

### 检测结果图 (detected.jpg)
在原图上绘制彩色 bounding box，每个框标注类别和置信度。

### 物理分析报告 (physics.json)
```json
{
  "calibration": {
    "px_per_cm": 10.35,
    "note": "基于弹簧测力计尺寸估算"
  },
  "spring": {
    "deformation_cm": 2.45,
    "natural_length_cm": 5.0,
    "force_n": null,
    "spring_constant": null
  },
  "forces": [
    {
      "magnitude": 0.0,
      "angle_deg": 270.0,
      "origin_xy": [128, 320],
      "label": "重力 G"
    },
    {
      "magnitude": 0.0,
      "angle_deg": 90.0,
      "origin_xy": [128, 290],
      "label": "弹力 F=0.00N"
    }
  ],
  "detected_classes": ["wooden_block", "spring", "dynamometer"]
}
```

### 力图叠加 (force_overlay.jpg)
在原图上绘制彩色力箭头，箭头长度和方向代表力的大小和方向。

### 弹簧动画 (animation.gif)
展示弹簧从自然状态渐进式伸长的过程，同时显示胡克定律公式 F = k·x。

## 🔧 核心模块详解

### physics.py - 物理量提取

**主要函数**：

```python
def calibrate(detections, img_shape) -> float:
    """计算像素/厘米比例，基于弹簧测力计尺寸"""

def extract_spring_deformation(detections, px_per_cm) -> SpringInfo:
    """提取弹簧形变信息"""

def compute_force_diagram(detections, px_per_cm) -> list[Force]:
    """计算物体受力情况"""

def extract_physics(detections, img_shape) -> dict:
    """完整物理量提取入口"""
```

### animate.py - 可视化和动画

```python
def draw_force_overlay(image_path, physics_data, save_path):
    """在原图上叠加力箭头"""

def generate_spring_animation(spring_info, save_path):
    """生成弹簧伸长动画 GIF"""
```

### detect.py - 推理流程

整合完整的推理管线：
1. 加载 YOLOv8 模型
2. 执行目标检测
3. 提取物理量
4. 生成可视化输出

## 📈 训练建议

### 数据要求
- **最小训练集**: 50-100 张图片
- **推荐训练集**: 200+ 张图片
- **图片分辨率**: 640×480 以上
- **标注精度**: 确保 bounding box 准确标注

### 优化策略
如果训练精度不理想：
- 增加训练数据量
- 将模型改为 YOLOv8 Medium: `MODEL = "yolov8m.pt"`
- 增加训练轮数: `EPOCHS = 150`
- 调整数据增强参数

### 性能优化
- 使用 GPU 加速：确保 CUDA 环境配置正确
- 调整 batch_size: 若显存不足，改为 8；若显存充足，改为 32

## 🛠️ 常见问题

**Q: 训练时出现 CUDA 错误？**
A: 改为 CPU 训练。在 train.py 中设置 `DEVICE = "cpu"`。

**Q: 模型找不到权重文件？**
A: 确保已运行 `train.py` 完成训练，权重文件应在 `runs/detect/train/weights/best.pt`。

**Q: 检测精度很低？**
A: 检查训练数据标注质量，确保数据多样性；尝试增加训练数据或训练轮数。

**Q: 物理计算结果不正确？**
A: 物理模块需要精确的标定。确保弹簧测力计完整出现在图片中。

## 📝 项目状态

✅ **已完成**：
- 完整的检测管线
- 物理模块框架设计
- 可视化和动画生成

🚧 **进行中**：
- 收集和标注大规模训练数据
- 优化物理计算算法
- 测试和验证

## 🔮 未来计划

- [ ] 支持更多物体类型（重锤、滑轮等）
- [ ] 实现动态物体追踪和轨迹分析
- [ ] 增加摩擦力等复杂物理模型
- [ ] 开发 Web 界面
- [ ] 导出可编辑的物理图表

## 📄 许可证

MIT License

## 👤 作者

ChromiteCr

## 📧 反馈与建议

如有任何问题或建议，欢迎提交 Issue 或 Pull Request！
