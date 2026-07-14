"""
用 GLM 多模态 API 辅助标注（比如 data/video_frames/images 下抽出的视频帧）。

跟 autolabel.py（Grounding DINO 零样本检测）是并列的另一条自动标注路线，同样遵循
"自动出粗标注 + 人工用 gen_label_gallery.py 校对" 的流程，不是取代关系。

**重要**：GLM 是通用视觉语言模型，不是专门训练的目标检测器，给出的框大概率比
Grounding DINO 粗糙。这批标注必须人工校对，不能跳过校对直接拿去训练。

依赖：zhipuai（同 explain_api.py），`pip install zhipuai`
API Key：环境变量 GLM_API_KEY 或 ZHIPUAI_API_KEY（不硬编码在代码里）。

用法：
    python autolabel_glm.py --images data/video_frames/images \
                            --out    data/video_frames/labels \
                            --model  glm-4.6v-flash

可选：
  --class-desc-file class_descriptions.json
      每个类别喂给 GLM 的文字描述，自己改这个 JSON 文件的 value 即可，不用碰代码。
  --hints-file data/video_frames/frame_hints.json
      每张图"已知包含哪些类别"的人工提示（extract_video_frames.py 会生成空白模板），
      填了就只让 GLM 重点找这些类别，留空/没写的图片走全类别检测。
  --overwrite
      已经有标注文件的图片默认跳过（省 API 调用费用），加这个参数强制重新标注全部。
  --max-retries / --retry-delay
      单张图遇到限速（429）时的自动重试次数和退避基准秒数（指数退避），默认重试
      3 次、8 秒起步。重试用完仍失败的图片不会写出 .txt，直接重新执行整条命令即可
      只补标这些失败的图——已成功的图片默认跳过，不会重复消耗 API 调用。
  --sleep
      每次调用之间固定等待的秒数（默认 0），持续被限速时可以设成 1~2 主动降速。

**断点续标**：命令中途失败/中断（Ctrl+C、限速重试用完等）后，直接重新执行同一条
命令即可——已经成功标注的图片（已有 .txt）会自动跳过，只会继续标还没标过的。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Optional

PROJ = Path(__file__).parent

# class id 与 data/dataset.yaml 对齐。不含 spring(2)/ruler(4)：2026-07-14 起
# 按要求在所有标注代码里忽略这两类（spring 历史上误检率过高，ruler 是 track/ruler
# 混淆的重灾区），但编号保留占位，不挪作他用。也不含 base：同一天决定把 base 合并
# 进 track(1)，不再单独成类。
# wooden_block/iron_block（6/7）是 2026-07-14 新增，只走这条 GLM 标注路线，不在
# autolabel.py（Grounding DINO）的 PROMPT_TO_CLASS 里，改 dataset.yaml 时记得
# 同步这里和 gen_label_gallery.py 的 NAMES/COLORS_HEX。
CLASS_NAMES = {
    0: "cart", 1: "track", 3: "string", 5: "dynamometer",
    6: "wooden_block", 7: "iron_block",
}
NAME_TO_ID = {name: cid for cid, name in CLASS_NAMES.items()}

DEFAULT_CLASS_DESC_FILE = PROJ / "class_descriptions.json"
MODEL_CHOICES = ["glm-4.6v-flash", "glm-4.6v"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_FALLBACK_CLASS_DESC = {
    "cart": "实验小车，通常是带轮子的长方体木块或塑料车",
    "track": "导轨/轨道，细长的金属或塑料条，小车在上面滑动；也包括底座/支架一类的支撑结构",
    "string": "细线，摆动或牵引用",
    "dynamometer": "弹簧测力计，整机带外壳和刻度",
    "wooden_block": "木块，长方体木质物块，被细线牵引或直接放在轨道上",
    "iron_block": "钢球，金属球体",
}


class GLMAPIError(RuntimeError):
    """GLM API 调用失败（缺 Key / 网络错误 / 返回格式不对）时统一抛这个，
    方便调用方用一个 except 分支处理，给出清晰提示而不是堆栈。"""


def _get_api_key() -> str:
    key = os.environ.get("GLM_API_KEY") or os.environ.get("ZHIPUAI_API_KEY")
    if not key:
        raise GLMAPIError(
            "没有配置 GLM API Key。请设置环境变量 GLM_API_KEY（或 ZHIPUAI_API_KEY）"
            "后再试——出于安全考虑，Key 不写在代码里。"
        )
    return key


def _get_client():
    try:
        from zhipuai import ZhipuAI
    except ImportError as e:
        raise GLMAPIError(
            "缺少 zhipuai 库，请先 `pip install zhipuai`（见 requirements.txt）。"
        ) from e
    return ZhipuAI(api_key=_get_api_key())


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def load_class_descriptions(path: Path) -> dict[str, str]:
    """读取类别描述配置；文件不存在或某个类别没写就用内置默认描述兜底。"""
    desc = dict(_FALLBACK_CLASS_DESC)
    if path.exists():
        try:
            desc.update(json.loads(path.read_text()))
        except json.JSONDecodeError as e:
            raise GLMAPIError(f"{path} 不是合法 JSON: {e}") from e
    return desc


def load_frame_hints(path: Optional[Path]) -> dict[str, list[str]]:
    """读取每张图的物体提示清单；没传/文件不存在时返回空字典（所有图都走全类别检测）。"""
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise GLMAPIError(f"{path} 不是合法 JSON: {e}") from e


def build_prompt(class_desc: dict[str, str], hint_classes: Optional[list[str]]) -> str:
    candidates = hint_classes if hint_classes else list(CLASS_NAMES.values())
    desc_lines = "\n".join(f"- {name}: {class_desc.get(name, '')}" for name in candidates)

    hint_note = ""
    if hint_classes:
        hint_note = (f"\n已知这张图里大概率包含：{ '、'.join(hint_classes) }。"
                      f"请优先检测这些类别，其余类别不必强找。")

    return f"""你是一名物理实验图像标注员。请在下面这张图片里检测这些器材类别：

{desc_lines}
{hint_note}

对每个检测到的目标，给出类别名（必须是上面列表里的英文名）和边界框，边界框用归一化坐标
[cx, cy, w, h]（中心点x、中心点y、宽、高，均为 0-1 之间相对图片尺寸的小数）。

严格按下面的 JSON 数组格式回复，不要有多余文字，没检测到任何目标就返回空数组 []：
[
  {{"class": "cart", "bbox": [0.45, 0.6, 0.3, 0.2]}},
  {{"class": "track", "bbox": [0.5, 0.7, 0.8, 0.1]}}
]"""


def _parse_json_array(raw: str) -> list[dict]:
    """GLM 有时会在 JSON 前后加 ```json ... ``` 或说明文字，做宽松解析。"""
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise GLMAPIError(f"GLM 返回内容无法解析成 JSON 数组: {raw[:200]}")
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        raise GLMAPIError(f"GLM 返回的 JSON 格式有误: {e}\n原始内容: {raw[:200]}") from e


def _is_rate_limit_error(e: Exception) -> bool:
    """粗略识别限速错误：zhipuai 的具体异常类型不确定，从错误文字里找特征。
    实测出现过 HTTP 429 + 错误码 1305（"该模型当前访问量过大，请您稍后再试"）。"""
    msg = str(e)
    return "429" in msg or "1305" in msg or "访问量过大" in msg


def call_glm(client, model: str, image_path: Path, prompt: str,
             max_retries: int = 3, retry_delay: float = 8.0) -> list[dict]:
    content = [
        {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
        {"type": "text", "text": prompt},
    ]
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
            )
            return _parse_json_array(response.choices[0].message.content)
        except Exception as e:  # zhipuai 的具体异常类型不确定，统一收窄成 GLMAPIError
            last_err = e
            if _is_rate_limit_error(e) and attempt < max_retries:
                wait = retry_delay * (2 ** attempt)
                print(f"    [限速] 第 {attempt + 1} 次调用被限速，{wait:.0f} 秒后重试...")
                time.sleep(wait)
                continue
            break
    raise GLMAPIError(f"GLM API 调用失败: {last_err}") from last_err


def to_yolo_line(cls_id: int, bbox: list) -> Optional[str]:
    if len(bbox) != 4:
        return None
    cx, cy, w, h = bbox
    if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in (cx, cy, w, h)):
        return None
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def autolabel_dir(images_dir: Path, out_dir: Path, model: str,
                   class_desc: dict[str, str], hints: dict[str, list[str]],
                   overwrite: bool, max_retries: int = 3, retry_delay: float = 8.0,
                   sleep_between: float = 0.0):
    client = _get_client()
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    n_already = sum(1 for p in imgs if (out_dir / f"{p.stem}.txt").exists())
    print(f"共 {len(imgs)} 张，其中 {n_already} 张已有标注" +
          ("（默认跳过，用 --overwrite 强制重跑）" if not overwrite else "（--overwrite 已开启，全部重跑）"))
    print(f"模型={model}")

    n_ok = n_skip = n_fail = 0
    for img_path in imgs:
        out_path = out_dir / f"{img_path.stem}.txt"
        if out_path.exists() and not overwrite:
            n_skip += 1
            continue

        hint_classes = hints.get(img_path.name) or None
        prompt = build_prompt(class_desc, hint_classes)

        try:
            detections = call_glm(client, model, img_path, prompt,
                                   max_retries=max_retries, retry_delay=retry_delay)
        except GLMAPIError as e:
            print(f"  [失败] {img_path.name}: {e}")
            n_fail += 1
            continue
        finally:
            if sleep_between > 0:
                time.sleep(sleep_between)

        lines = []
        for det in detections:
            name = det.get("class") if isinstance(det, dict) else None
            bbox = det.get("bbox") if isinstance(det, dict) else None
            if name not in NAME_TO_ID or not isinstance(bbox, list):
                print(f"  [忽略无法识别的条目] {img_path.name}: {det}")
                continue
            line = to_yolo_line(NAME_TO_ID[name], bbox)
            if line:
                lines.append(line)

        out_path.write_text("\n".join(lines))
        print(f"  {img_path.name}: {len(lines)} 个框")
        n_ok += 1

    print(f"\n完成。成功 {n_ok} 张，跳过(已有标注) {n_skip} 张，失败 {n_fail} 张。")
    if n_fail:
        print("有失败的图片：直接重新执行同一条命令即可只补标这些失败的（已成功的会自动跳过）。")
    print(f"请人工校对: python gen_label_gallery.py --dir {images_dir.parent}")


def main():
    parser = argparse.ArgumentParser(description="用 GLM 多模态 API 辅助标注")
    parser.add_argument("--images", required=True, help="待标注图片目录")
    parser.add_argument("--out", required=True, help="YOLO 标注输出目录")
    parser.add_argument("--model", default="glm-4.6v-flash", choices=MODEL_CHOICES,
                         help="GLM 视觉模型，flash 更便宜更快（默认），glm-4.6v 更准但更贵")
    parser.add_argument("--class-desc-file", default=str(DEFAULT_CLASS_DESC_FILE),
                         help="类别描述 JSON 文件路径，编辑这个文件即可自定义 prompt 用词")
    parser.add_argument("--hints-file", default=None,
                         help="每张图物体提示 JSON 文件路径（如 data/video_frames/frame_hints.json）")
    parser.add_argument("--overwrite", action="store_true", help="已存在的标注也重新跑一遍")
    parser.add_argument("--max-retries", type=int, default=3,
                         help="单张图遇到限速(429)时的自动重试次数（默认3）")
    parser.add_argument("--retry-delay", type=float, default=8.0,
                         help="限速重试的退避基准秒数，指数退避（默认8秒起步）")
    parser.add_argument("--sleep", type=float, default=0.0,
                         help="每次调用之间固定等待的秒数（默认0，持续被限速可设成1~2主动降速）")
    args = parser.parse_args()

    class_desc = load_class_descriptions(Path(args.class_desc_file))
    hints = load_frame_hints(Path(args.hints_file) if args.hints_file else None)

    autolabel_dir(Path(args.images), Path(args.out), args.model, class_desc, hints,
                  args.overwrite, max_retries=args.max_retries, retry_delay=args.retry_delay,
                  sleep_between=args.sleep)


if __name__ == "__main__":
    main()
