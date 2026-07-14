# 设计：视频抽帧 + GLM 辅助标注

日期：2026-07-13

## 背景

`data/video/` 下已有 2026-07-12 补拍的 6 段视频（共约 6000 帧，1280x720@30fps 一段 + 640x360@25fps 五段），用于扩充训练集，覆盖首轮训练中样本不足的 cart/track/ruler/string 类别。现有标注管线（`autolabel.py` 用 Grounding DINO 零样本检测 + `gen_label_gallery.py` 网页校对 + `prepare_dataset.py` 合并数据集）是围绕**静态图片**设计的，还没有从视频提取训练帧的能力。

本次设计新增视频抽帧 + 用 GLM 多模态 API 做辅助标注的能力，产出可供人工校对、最终合并进 `dataset/` 训练集的图片+YOLO标注。

不在本次范围内：不做去重复/去模糊帧的智能过滤（YAGNI，抽帧间隔本身就是稀疏采样手段）；不改动 `dataset.yaml` 类别定义；不在本机跑烟雾测试/GLM 调用验证（见项目记忆 [[no-local-smoke-tests]]，改完代码交付，验证留给用户在有网络/API Key 的机器上跑）。

## 架构

### 1. `extract_video_frames.py`（新文件）——纯抽帧，无 AI 依赖

- 输入：`data/video/*.mp4`（默认全部 6 个文件；`--videos` 可选参数限定处理某几个文件名，便于先小范围试跑）
- 核心参数：`--stride N`（默认 10；用户可传 25 或其他正整数，含义是"每 N 帧取 1 帧"）
- 用 OpenCV `cv2.VideoCapture` 顺序读帧，`frame_idx % stride == 0` 时落盘
- 输出路径：`data/video_frames/images/{video_stem}_f{frame_idx:06d}.jpg`
  - 文件名前缀带视频名，避免与 `self_captured`/`openclaw` 撞名，同时保留"来自哪段视频第几帧"的可追溯性
- 幂等：若目标 jpg 已存在则跳过该帧，不重复写盘，支持中断后重跑或换更小的 stride 补抽
- 运行结束打印汇总：每个视频抽了多少帧、总计多少张、输出目录

### 2. `autolabel_glm.py`（新文件）——GLM 辅助标注

- 定位：与 `autolabel.py`（Grounding DINO 路线）并列的另一条自动标注路线，复用同样的"自动出粗标注 + 人工校对"哲学，不是取代关系
- CLI：
  ```
  python autolabel_glm.py --images data/video_frames/images \
                          --out    data/video_frames/labels \
                          --model  glm-4.6v-flash
  ```
- `--model {glm-4.6v-flash,glm-4.6v}`：flash 为默认推荐（延续项目"选 GLM 是出于价格考量"的既定判断），glm-4.6v 更强但更贵
- 标注类别：沿用 `autolabel.py` 的 5 类映射（**不含 spring**，与已确认的"spring 误检率过高、不参与自动标注"结论保持一致）：
  ```python
  CLASS_NAMES = {0: "cart", 1: "track", 3: "string", 4: "ruler", 5: "dynamometer"}
  ```
  （编号沿用 `dataset.yaml` 的 6 类编号，2 号 spring 空缺不标）
- 单张图流程：
  1. base64 编码图片（复用 `explain_api.py._encode_image` 的思路，独立实现避免跨文件耦合）
  2. 调 GLM chat completions，vision 消息里带图 + 结构化 prompt：列出 5 个类别中文名+英文提示，要求返回 JSON 数组，每项含 `class`（类别名）+ `bbox`（归一化 `[cx, cy, w, h]`，0-1 之间），未检出任何目标则返回空数组
  3. 解析 JSON（复用 `explain_api.py._parse_json_response` 的宽松解析思路：取第一个 `[`/`{` 到最后一个 `]`/`}`），解析失败或某类别名不在映射表里的条目跳过并记日志，不中断整批
  4. 转成 YOLO 格式行写入 `{out}/{stem}.txt`
- 幂等：`{out}/{stem}.txt` 已存在则默认跳过该图（省 API 调用费用），`--overwrite` 可强制重跑全部
- 错误处理：借用 `explain_api.GLMAPIError`（缺 Key / 网络失败 / SDK 缺失）；单张图调用异常只打印警告并跳过，不让整批因为一张图失败而中断
- 顶部注释需明确写清：GLM 是通用视觉语言模型，不是专门训练的目标检测器，给出的框大概率比 Grounding DINO 粗糙，**这批标注必须人工校对，不能跳过校对直接拿去训练**——延续 `autolabel.py` 顶部注释的记录风格

### 3. 现有文件的两处小改动

- **`prepare_dataset.py`**：`SOURCES` 列表新增 `DATA_DIR / "video_frames"`，使其与 `self_captured`/`openclaw` 一样参与合并进 `dataset/{images,labels}/{train,val,test}`。其余合并逻辑（来源前缀防撞名、openclaw 占比提醒等）不变。
- **`gen_label_gallery.py`**：`IMG_DIR`/`LBL_DIR` 从硬编码常量改为可选命令行参数，默认值仍是 `data/self_captured/{images,labels}`（不破坏现有用法 `python gen_label_gallery.py`），新增 `--dir` 参数可指向 `data/video_frames` 等其他来源目录做校对。`NAMES`/`COLORS_HEX` 保持 6 类不变（仍显示 spring 选项，只是该批数据不会有 spring 框）。

## 端到端流程（人工执行顺序）

```
1. python extract_video_frames.py --stride 10
   → data/video_frames/images/ 出现若干 jpg

2. python autolabel_glm.py --images data/video_frames/images \
                           --out data/video_frames/labels \
                           --model glm-4.6v-flash
   → data/video_frames/labels/ 出现对应 txt（需要 GLM_API_KEY/ZHIPUAI_API_KEY 环境变量）

3. python gen_label_gallery.py --dir data/video_frames
   → 打开 output/label_gallery.html 人工校对/修正框，导出的 txt 手动放回 labels/

4. python prepare_dataset.py
   → data/video_frames 与 self_captured/openclaw 一起合并进 data/dataset/

5. python train.py（在 RTX 5090 上跑，不在本机）
```

## 追加（2026-07-14）：自定义类别描述 + 每图物体提示

实施时按用户要求追加了两个可配置输入，让 GLM 标注可以结合人工先验，而不是每张图都盲检全部类别：

- **`class_descriptions.json`**（项目根目录）：每个类别喂给 GLM 的文字描述，纯 JSON，key 是类别英文名、value 是中文描述。用户可以随时编辑这个文件调整措辞，不用碰 `autolabel_glm.py` 代码。文件不存在或某个类别没写时，代码内置默认描述兜底。
- **`data/video_frames/frame_hints.json`**：每张抽出的帧"已知包含哪些类别"的人工提示，格式 `{"文件名": ["cart", "track"]}`。`extract_video_frames.py` 每次跑完会自动给新抽出的帧补上空白条目（`[]`），不覆盖已有条目（保留用户已填的内容）。用户可以照着图片手动填几个类别名进去；填了的图片，`autolabel_glm.py` 的 prompt 会说"已知大概率包含 X、Y，优先检测这些"，缩小 GLM 的搜索空间、减少误检；留空的图片仍走全类别检测。

`autolabel_glm.py` 通过 `--class-desc-file`（默认指向 `class_descriptions.json`）和 `--hints-file`（默认不传，需要显式指定如 `data/video_frames/frame_hints.json`）接入这两个文件。

## 追加（2026-07-14）：类别调整——忽略 spring/ruler，base 合并进 track

用户人工描述抽出帧内容时，顺带确认了两处类别口径调整，已同步到 `data/dataset.yaml`、
`autolabel.py`、`autolabel_glm.py`、`class_descriptions.json`、`gen_label_gallery.py`、
`detect.py`（`EXCLUDED_FROM_DETECT`）：

- **spring(2)/ruler(4) 在所有标注/推理代码里忽略**：不再出现在 `autolabel.py` 的
  `PROMPT_TO_CLASS`、`autolabel_glm.py` 的 `CLASS_NAMES`、`gen_label_gallery.py` 的
  `NAMES`/`COLORS_HEX` 下拉框里，`detect.py` 的 `EXCLUDED_FROM_DETECT` 也加入
  `"spring"`（`"ruler"` 此前已在）。**编号不回收**：`dataset.yaml` 里 2 和 4 仍然
  保留占位，避免后来新增的类别把编号往前顶，导致跟已有历史标注数据错位。
- **base 合并进 track，不单独成类**：本次新增视频帧本来打算给"底座/支架"单独开
  一个 `base`(8) 类，用户随后决定这类物体直接按 `track` 标注即可。`data/video_frames/
  frame_hints.json` 里所有 `"base"` 提示词已批量替换成 `"track"`（去重），
  `dataset.yaml` 的 `nc` 也从 9 改回 8（8 号位直接删除，没有历史数据依赖这个新增
  没多久的编号，不需要占位）。

调整后的有效类别（`autolabel_glm.py`/`gen_label_gallery.py` 实际会用到的）：
`cart(0)`、`track(1)`、`string(3)`、`dynamometer(5)`、`wooden_block(6)`、
`iron_block(7)`——6 个；`spring(2)`、`ruler(4)` 仅在 `dataset.yaml` 里占位。

## 不做验证说明

代码交付后不在本机执行（无 GPU/内存有限，且抽帧+GLM调用都需要真实视频文件和 API Key）。用户需要在有 API Key 的环境里先跑一小批（比如 `--videos` 限定 1 个文件、`--stride 25`）验证 GLM 返回格式符合预期，再跑全量。
