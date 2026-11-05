"""本地路径约定。完整数据集（照片、视频帧、YOLO 标注）放在仓库的 datasets/ 下，已 gitignore，不上传。

可用环境变量 PPR_DATA_ROOT 指向别处。旧项目的 Proj/data 是指向这里的符号链接。
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PPR_DATA_ROOT", REPO / "datasets"))
RESULTS = REPO / "results"
WEIGHTS = REPO / "assets" / "weights"
