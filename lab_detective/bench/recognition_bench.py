"""识别 bench（照片）：按拍摄时段切分、生成测试集、给预测打分、规则基线。

    python -m lab_detective.bench.recognition_bench split  [--gap-s 1800|auto] [--seed 0] [--src 旧 data 目录]
    python -m lab_detective.bench.recognition_bench build  [--review review.json]
    python -m lab_detective.bench.recognition_bench score  --pred 预测.jsonl [--out report.json]
    python -m lab_detective.bench.recognition_bench rules

这是工程脚手架，不是研究计划：bench 的切分间隔、"未知/Z 怎么计分"、主指标和 Δ 都由学生在冻结前亲笔定死
（设计文档 §16.5）。冻结（bench.freeze）永远由学生本人运行，本文件从不调用。

数据流
------
1. ``split``：照片 → 拍摄时间（EXIF 拍摄时间，读旧项目原图；回退到文件名里的时间戳，再回退到编号相邻）
   → 相邻照片间隔 ≤ gap_s 算同一时段 → 约 25% 的时段整段划入测试集（按"首张照片 id 的哈希"排序取前 k 个，
   与随机数顺序无关，新增照片不会打乱已有时段的归属）。写 ``results/bench/recognition/split.json``。
   gap_s 默认 1800（30 分钟）。**真实的 200 张照片全部拍于 29 分钟之内，30 分钟间隔只得到 1 个时段**，
   此时 ``split`` 拒绝输出并打印各间隔下的时段数，请学生用 ``--gap-s`` 定一个间隔（``auto`` 会取能凑够
   8 个时段的最大间隔并把结果记进 split.json）。测试集与训练集的最近照片间隔也写进摘要，
   间隔越小越可能出现"同一场景的相邻帧一边训练一边测试"的泄漏。
2. ``build``：对测试照片用 YOLO 检测（结果缓存到 detections.jsonl），用**检测框**（不是真值框，与推理一致）
   画 Set-of-Mark 图，按 vision/prompt.py 拼消息，写 ``test.jsonl`` + ``images/`` + ``meta.json``。
   gold = 学生核实过的标签（``--review``，格式 ``{"照片id": "字母"}``，由 review_gallery 导出）；
   没核实的保留规则的暂定标签并标记 ``meta.gold_status = "provisional"``。
   assistant 消息里只有 ``letter`` 是 gold，open_name / cues / apparatus 只是占位参考，不是标准答案。
3. ``score``：预测文件每行 ``{"id", "text", "letter_logprobs"?, "letter_probs"?, "latency_s"?}``，
   id 是 test.jsonl 的 id（或 meta.photo_id）。缺失的 id、JSON 解析不出字母的输出一律算错
   （设计文档 §16.5 选项①：所有 arm 都不用约束解码，非法输出照算错）。
4. ``rules``：YOLO + 规则流水线作参考组，分别在真值框和 YOLO 检测框上跑，结果与 VLM arm 分开报。

诚实度约定（写进报告的 caveats）
------------------------------
- 暂定标签就是 ``vision.cues.analyze(真值框)`` 的输出，所以"规则 + 真值框"对暂定 gold 的准确率
  按构造就是 100%，暂定 Z 也只表示"规则没有候选"。因此指标按 gold_status 分开算，
  只有"已核实"那一份是证据（headline），"暂定"那份只是流程自检。
- YOLO 权重训练时见过这批照片（yolo_split = train/val），"规则 + 检测框"在这些照片上偏乐观，
  所以报告按 yolo_split 分层；即使 yolo_split = test，YOLO 当年是逐帧随机划分的，相邻帧也在它的训练集里。
- 开放集：gold 为 Z 的照片是正类，P(Z) 取 logprobs/概率向量里 Z 的概率；没有概率时用 confidence
  （预测 Z 取 conf，否则取 1 − conf）。规则 arm 自带完整概率向量（由规则分数 softmax，Z 用虚拟分数）。
  只有一个类时 AUROC 为 null，不报错。
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image
from scipy import stats as sps

from ..training.format import Example, read_jsonl, sha256_file, write_jsonl
from ..vision import cues as _cues
from ..vision.letters import LETTERS, OTHER_LETTER, TABLE, entry, normalize_letter
from ..vision.prompt import PROMPT_VERSION, assistant_target, build_messages, prompt_hash
from ..vision.schema import RecognitionParseError, RecognitionResult, parse_model_output
from ..vision.som import draw_marks
from . import stats

REPO = Path(__file__).resolve().parents[2]
MANIFEST_DEFAULT = REPO / "results" / "recognition" / "manifest.jsonl"
# 原图目录（manifest 的 orig_path 相对它）。2026-11-05 21:06 旧项目的 data/ 被搬进了仓库的 datasets/，
# 旧路径只作兜底；EXIF 里的拍摄时间只能从原图读（results/recognition/images 里的图不带 EXIF）。
_SRC_CANDIDATES = (REPO / "datasets",
                   Path("/Users/billgao/Documents/Coding/SCPKU/Physics-Pictorial-Recognition/Proj/data"))
SRC_DEFAULT = next((p for p in _SRC_CANDIDATES if (p / "self_captured").exists()), _SRC_CANDIDATES[0])
BENCH_DIR = REPO / "results" / "bench" / "recognition"
RUNS_DIR = REPO / "results" / "bench" / "recognition_runs"

BENCH_VERSION = "rec-bench-v1"
SPLIT_FILE = "split.json"
TEST_FILE = "test.jsonl"
DETECTIONS_FILE = "detections.jsonl"
META_FILE = "meta.json"

DEFAULT_GAP_S = 1800.0  # 30 分钟：设计规定的"同一时段"
GAP_LADDER = (1800, 600, 180, 60, 30, 20, 15, 10, 5)  # auto 模式和报错提示用的候选间隔（秒）
MIN_SESSIONS = 4  # 少于这个数的时段没法切分，直接拒绝
AUTO_MIN_SESSIONS = 8  # auto 模式要凑够的时段数
TEST_FRAC = 0.25
UNDATED_SEQ_GAP = 10  # 没有时间的照片：文件编号相差 ≤ 此值算同一段

YOLO_VOCAB = frozenset(_cues.YOLO_CLASSES.values()) - _cues.EXCLUDED_FROM_DETECT  # 器材 F1 的词表
HELD_OUT_LETTERS = tuple(e.letter for e in TABLE if e.held_out)
TRAINED_LETTERS = tuple(e.letter for e in TABLE if e.experiment_id and not e.held_out)


class DegenerateSplit(ValueError):
    """按给定间隔切不出足够多的时段。"""


# ── 拍摄时间 ───────────────────────────────────────────────────────────

_EXIF_IFD = 0x8769
_EXIF_ORIGINAL, _EXIF_DIGITIZED, _EXIF_DATETIME = 36867, 36868, 306
_FILENAME_TS = re.compile(
    r"(?<!\d)(20\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})[-_ T.]?(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})(?!\d)")


def parse_exif_datetime(raw: Any) -> datetime | None:
    """'2026:07:12 11:51:32'（也容忍 '-' 分隔、bytes、尾部空字符）→ datetime；读不出返回 None。"""
    if isinstance(raw, bytes):
        raw = raw.decode("ascii", "ignore")
    if not isinstance(raw, str):
        return None
    s = raw.strip().strip("\x00")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    return None


def exif_time(path: str | Path) -> datetime | None:
    """EXIF 拍摄时间（DateTimeOriginal > Digitized > DateTime）；读不了返回 None。"""
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            try:
                sub = ex.get_ifd(_EXIF_IFD)
            except Exception:
                sub = {}
            raw = sub.get(_EXIF_ORIGINAL) or sub.get(_EXIF_DIGITIZED) or ex.get(_EXIF_DATETIME)
    except Exception:
        return None
    return parse_exif_datetime(raw)


def filename_time(name: str) -> datetime | None:
    """文件名里的时间戳：20260712_115132、2026-07-12 11.51.32 等；iPhone 的 IMG_9307 没有时间，返回 None。"""
    m = _FILENAME_TS.search(Path(name).stem)
    if not m:
        return None
    try:
        return datetime(*(int(g) for g in m.groups()))
    except ValueError:
        return None


def photo_time(row: dict, src_root: str | Path | None = None, image_root: str | Path | None = None
               ) -> tuple[datetime | None, str]:
    """一张照片的拍摄时间和来源标记。

    优先级：manifest 里的 taken_at → 旧项目原图 EXIF（src_root/orig_path）→ 已转存图片的 EXIF
    （build_dataset 写出的图不带 EXIF，通常读不到）→ 文件名时间戳。都没有返回 (None, "none")。
    """
    t = parse_exif_datetime(row.get("taken_at"))
    if t:
        return t, "manifest"
    if src_root is not None and row.get("orig_path"):
        t = exif_time(Path(src_root) / row["orig_path"])
        if t:
            return t, "exif_orig"
    if image_root is not None and row.get("file"):
        t = exif_time(Path(image_root) / row["file"])
        if t:
            return t, "exif_image"
    for name in (row.get("orig_path") or "", row.get("file") or "", row.get("id") or ""):
        t = filename_time(name)
        if t:
            return t, "filename"
    return None, "none"


def _secs(t: datetime) -> float:
    return (t.replace(tzinfo=None) - datetime(1970, 1, 1)).total_seconds()


def _trailing_int(s: str) -> int | None:
    m = re.search(r"(\d+)(?!.*\d)", s)
    return int(m.group(1)) if m else None


# ── 时段切分 ───────────────────────────────────────────────────────────


def group_sessions(dated: Sequence[tuple[float, str]], gap_s: float) -> list[list[str]]:
    """按 (时间, id) 排序，相邻间隔 > gap_s 就另起一个时段；返回按时间排好的 id 列表。"""
    out: list[list[str]] = []
    prev = None
    for t, i in sorted(dated):
        if prev is None or t - prev > gap_s:
            out.append([])
        out[-1].append(i)
        prev = t
    return out


def undated_sessions(ids: Iterable[str], seq_gap: int = UNDATED_SEQ_GAP) -> list[list[str]]:
    """没有时间的照片：文件编号连续（相差 ≤ seq_gap）的归一段，没编号的各自一段。"""
    keyed = sorted(ids, key=lambda i: (_trailing_int(i) is None, _trailing_int(i) or 0, i))
    out: list[list[str]] = []
    prev = None
    for i in keyed:
        n = _trailing_int(i)
        if n is None:
            out.append([i])
            prev = None
            continue
        if prev is not None and n - prev <= seq_gap:
            out[-1].append(i)
        else:
            out.append([i])
        prev = n
    return out


def gap_table(dated: Sequence[tuple[float, str]], gaps: Sequence[float] = GAP_LADDER) -> dict[str, int]:
    """各候选间隔下的时段数（报错提示和 split.json 里都带上，方便学生挑间隔）。"""
    return {str(g): len(group_sessions(dated, g)) for g in gaps}


def _hash_key(seed: int, first_id: str) -> str:
    return hashlib.sha1(f"{seed}|{first_id}".encode("utf-8")).hexdigest()


def session_split(
    rows: Sequence[dict],
    src_root: str | Path | None = None,
    image_root: str | Path | None = None,
    *,
    gap_s: float | str = DEFAULT_GAP_S,
    test_frac: float = TEST_FRAC,
    seed: int = 0,
    min_sessions: int = MIN_SESSIONS,
    auto_min_sessions: int = AUTO_MIN_SESSIONS,
    times: dict[str, str] | None = None,
) -> dict:
    """按拍摄时段把照片切成 train / test，整个时段只会在一边。

    gap_s：数字（秒）或 "auto"（取能凑够 auto_min_sessions 个时段的最大候选间隔，凑不够取最小的）。
    times：{id: ISO 时间} 缓存，给了就不再读 EXIF（原图不在本机时用）。
    时段数 < min_sessions 抛 DegenerateSplit（消息里带各间隔下的时段数）。
    测试时段 = 按 sha1(seed|首张照片 id) 排序的前 k 个，k = round(test_frac × 时段数)，夹在 [1, n−1]。
    """
    dated: list[tuple[float, str]] = []
    undated: list[str] = []
    taken: dict[str, tuple[datetime | None, str]] = {}
    for r in rows:
        if times and r["id"] in times:
            t, src = parse_exif_datetime(times[r["id"]]), "cache"
        else:
            t, src = photo_time(r, src_root, image_root)
        taken[r["id"]] = (t, src)
        if t is None:
            undated.append(r["id"])
        else:
            dated.append((_secs(t), r["id"]))
    table = gap_table(dated) if dated else {}

    if gap_s == "auto":
        gap_mode = "auto"
        gap = next((g for g in GAP_LADDER if table.get(str(g), 0) >= auto_min_sessions), GAP_LADDER[-1])
    else:
        gap_mode, gap = "fixed", float(gap_s)
    sessions = group_sessions(dated, gap) + undated_sessions(undated)
    n = len(sessions)
    if n < min_sessions:
        span = (max(t for t, _ in dated) - min(t for t, _ in dated)) if dated else 0.0
        raise DegenerateSplit(
            f"间隔 {gap:g} s 下只有 {n} 个时段（至少需要 {min_sessions} 个才能切出测试集）。"
            f"{len(dated)} 张有时间的照片跨度 {span / 60:.1f} 分钟；各间隔（秒）下的时段数：{table}。"
            "请在冻结前自己定一个间隔（--gap-s N 或 --gap-s auto），再重新运行。")

    k = min(max(1, int(math.floor(test_frac * n + 0.5))), n - 1)
    ranked = sorted(range(n), key=lambda s: _hash_key(seed, sessions[s][0]))
    test_idx = set(ranked[:k])

    photos: dict[str, dict] = {}
    for s, members in enumerate(sessions):
        for i in members:
            t, src = taken[i]
            photos[i] = {"session": s + 1, "split": "test" if s in test_idx else "train",
                         "taken_at": t.isoformat() if t else None, "time_source": src}
    n_test = sum(v["split"] == "test" for v in photos.values())
    warnings: list[str] = []
    if undated:
        warnings.append(f"{len(undated)} 张照片没有拍摄时间，按文件编号相邻归段，独立性无法按时间核对")
    frac_photos = n_test / len(photos) if photos else 0.0
    if not 0.1 <= frac_photos <= 0.5:
        warnings.append(f"测试集占照片 {frac_photos:.0%}（时段占 {k}/{n}），偏离 25% 较多；"
                        "时段大小悬殊时可换 --seed，但换之前不要看任何标签或结果")

    gaps_to_train = _nearest_train_gaps(photos, taken)
    return {
        "version": 1,
        "status": "draft",
        "params": {"gap_s": gap, "gap_mode": gap_mode, "test_frac": test_frac, "seed": seed,
                   "min_sessions": min_sessions, "auto_min_sessions": auto_min_sessions},
        "n_photos": len(photos),
        "n_sessions": n,
        "n_test_sessions": k,
        "n_test_photos": n_test,
        "session_fraction_test": round(k / n, 4),
        "photo_fraction_test": round(frac_photos, 4),
        "session_sizes": [len(m) for m in sessions],
        "gap_table": table,
        "n_undated": len(undated),
        "nearest_train_gap_s": gaps_to_train,
        "warnings": warnings,
        "photos": photos,
    }


def _nearest_train_gaps(photos: dict[str, dict], taken: dict[str, tuple[datetime | None, str]]) -> dict:
    """每张有时间的测试照片到最近训练照片的时间差（秒）：最小值和中位数。"""
    train = sorted(_secs(taken[i][0]) for i, v in photos.items() if v["split"] == "train" and taken[i][0])
    gaps = []
    for i, v in photos.items():
        t = taken[i][0]
        if v["split"] != "test" or t is None or not train:
            continue
        x = _secs(t)
        j = bisect.bisect_left(train, x)
        near = [abs(x - train[m]) for m in (j - 1, j) if 0 <= m < len(train)]
        gaps.append(min(near))
    if not gaps:
        return {"min": None, "median": None, "n": 0}
    return {"min": round(min(gaps), 1), "median": round(float(np.median(gaps)), 1), "n": len(gaps)}


def write_split(info: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


def load_split(path: str | Path = BENCH_DIR / SPLIT_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_ids(info: dict, which: str = "test") -> list[str]:
    """split.json 里某一边的照片 id（排序）。训练样本生成必须排除 'test' 这一边。"""
    return sorted(i for i, v in info["photos"].items() if v["split"] == which)


def check_no_overlap(train_photo_ids: Iterable[str], info: dict) -> dict:
    """训练样本用到的照片 id 与测试集的交集（应为空）。"""
    hits = sorted(set(train_photo_ids) & set(split_ids(info, "test")))
    return {"overlap": hits, "clean": not hits}


# ── 审核结果与 gold ────────────────────────────────────────────────────


def load_review(src: str | Path | dict, known_ids: Iterable[str] | None = None) -> dict[str, str]:
    """读 review.json（{id: 字母}），字母规范化为大写；非法字母或不在 known_ids 里的 id 直接报错。"""
    raw = src if isinstance(src, dict) else json.loads(Path(src).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("review.json 必须是 {照片id: 字母} 对象")
    bad = {k: v for k, v in raw.items() if normalize_letter(v) is None}
    if bad:
        raise ValueError(f"review.json 里有非法字母：{dict(list(bad.items())[:5])}")
    if known_ids is not None:
        unknown = sorted(set(raw) - set(known_ids))
        if unknown:
            raise ValueError(f"review.json 里有不在 manifest 的 id：{unknown[:5]}（共 {len(unknown)} 个）")
    return {k: normalize_letter(v) for k, v in raw.items()}


def gold_for(row: dict, review: dict[str, str] | None) -> tuple[str, str]:
    """(gold 字母, gold_status)：学生核实过的用核实结果，否则用规则暂定值并标记 provisional。"""
    if review and row["id"] in review:
        return review[row["id"]], "reviewed"
    return row["provisional_letter"], "provisional"


def gold_target(letter: str, gt_boxes: list[dict]) -> str:
    """assistant 消息：只有 letter 是 gold；open_name / cues / apparatus 是占位参考（来自标注框）。"""
    result = RecognitionResult(
        letter=letter, experiment_id=None,
        open_name=entry(letter).short_zh if letter != OTHER_LETTER else "未知",
        cues=_cues.configuration_cues(gt_boxes), apparatus=_cues.apparatus_from_detections(gt_boxes),
        confidence=1.0, source="manual")
    return assistant_target(result)


# ── 检测器与测试集生成 ─────────────────────────────────────────────────


class YoloDetector:
    """走 vision.cues.run_detector（与推理同一条路径）；懒加载，导入本模块不会加载 ultralytics。"""

    def __init__(self, weights: str | Path | None = None, conf: float = 0.25):
        self.weights, self.conf = weights, conf

    @property
    def info(self) -> dict:
        """权重路径与 SHA-256（默认权重的解析规则同 detect.DEFAULT_WEIGHTS，但不 import detect）。"""
        w = Path(self.weights) if self.weights else Path(
            os.environ.get("PPR_YOLO_WEIGHTS", REPO / "assets" / "weights" / "best.pt"))
        return {"weights": str(w), "weights_sha256": sha256_file(w) if w.exists() else None, "conf": self.conf}

    def __call__(self, image_path: str | Path) -> list[dict]:
        dets, _ = _cues.run_detector(image_path, self.weights, self.conf)
        return dets


def read_jsonl_dicts(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def example_id(photo_id: str) -> str:
    return f"recbench-{photo_id}"


def build_test_set(
    rows: Sequence[dict],
    split: dict,
    out_dir: str | Path = BENCH_DIR,
    *,
    image_root: str | Path,
    review: dict[str, str] | None = None,
    detector: Callable[[Path], list[dict]] | None = None,
    detector_info: dict | None = None,
    review_sha256: str | None = None,
    seed: int = 0,
) -> dict:
    """生成 test.jsonl、images/（带编号框）、detections.jsonl、meta.json；返回摘要。

    检测框：优先读 out_dir/detections.jsonl 缓存，缺的照片用 detector 补（缓存只增不改，
    所有 arm 共用同一份检测，规则基线也读它）。detector 与缓存都没有时报错。
    """
    out, image_root = Path(out_dir), Path(image_root)
    (out / "images").mkdir(parents=True, exist_ok=True)
    by_id = {r["id"]: r for r in rows}
    test = [by_id[i] for i in split_ids(split, "test")]
    if not test:
        raise ValueError("split 里没有测试照片")

    cache_path = out / DETECTIONS_FILE
    cache = {d["id"]: d for d in read_jsonl_dicts(cache_path)} if cache_path.exists() else {}
    examples: list[Example] = []
    for r in test:
        img = image_root / r["file"]
        if r["id"] not in cache:
            if detector is None:
                raise ValueError(f"{r['id']} 没有检测缓存，也没有给 detector")
            cache[r["id"]] = {"id": r["id"], "detections": detector(img), "detector": detector_info or {}}
        dets = cache[r["id"]]["detections"]
        rel = f"images/{r['id']}.jpg"
        _, legend = draw_marks(img, dets, out_path=out / rel)
        gold, status = gold_for(r, review)
        sp = split["photos"][r["id"]]
        meta = {
            "experiment_id": entry(gold).experiment_id,
            "split": "test", "seed": seed,
            "bench": "recognition", "bench_version": BENCH_VERSION,
            "prompt_hash": prompt_hash(), "prompt_version": PROMPT_VERSION,
            "photo_id": r["id"], "image_sha1": r.get("sha1"), "src_sha1": r.get("src_sha1"),
            "source": r.get("source"),
            "gold_letter": gold, "gold_status": status,
            "provisional_letter": r["provisional_letter"],
            "gold_fields": ["letter"], "placeholder_fields": ["open_name", "cues", "apparatus", "confidence"],
            "held_out_gold": gold in HELD_OUT_LETTERS,
            "session": sp["session"], "taken_at": sp["taken_at"],
            "yolo_split": r.get("yolo_split"),
            "yolo_seen": r.get("yolo_split") in ("train", "val"),
            "gt_boxes": r.get("gt_boxes", []),
            "gold_apparatus_partial": [a for a in _cues.apparatus_from_detections(r.get("gt_boxes", []))
                                       if a in YOLO_VOCAB],
            "detections": dets,
            "legend": legend,
        }
        ex = Example(example_id(r["id"]), "recognize", "photo",
                     [*build_messages(rel, legend), {"role": "assistant", "content": gold_target(gold, r.get("gt_boxes", []))}],
                     meta)
        ex.validate()
        examples.append(ex)

    keep = [cache[r["id"]] for r in test]  # 只留测试照片的检测，别的照片不进 bench 目录
    cache_path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in keep), encoding="utf-8")
    write_jsonl(out / TEST_FILE, examples)

    all_gold = Counter(gold_for(r, review)[0] for r in rows)
    test_gold = Counter(e.meta["gold_letter"] for e in examples)
    status_counts = Counter(e.meta["gold_status"] for e in examples)
    warnings = list(split.get("warnings", []))
    missing = sorted(set(all_gold) - set(test_gold))
    if missing:
        warnings.append(f"测试集里没有出现的字母（全集里有）：{'、'.join(missing)}；这些类别没有测试证据")
    if status_counts.get("reviewed", 0) == 0:
        warnings.append("测试集还没有任何学生核实过的标签：全部 gold 都是规则暂定值，指标不是证据")
    elif status_counts.get("provisional", 0):
        warnings.append(f"测试集里还有 {status_counts['provisional']} 张照片的标签没核实")
    meta = {
        "bench_version": BENCH_VERSION, "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash(),
        "n_test": len(examples),
        "gold_letter_counts": dict(sorted(test_gold.items())),
        "gold_status_counts": dict(status_counts),
        "n_reviewed_total": len(review or {}), "review_sha256": review_sha256,
        "split": {k: split[k] for k in ("params", "n_photos", "n_sessions", "n_test_sessions",
                                        "session_fraction_test", "photo_fraction_test", "nearest_train_gap_s")},
        "detector": detector_info or {},
        "warnings": warnings,
        "development_ban": "这些照片不得进入产品调试、提示词调整或 YOLO 重训：" + "、".join(e.meta["photo_id"] for e in examples),
    }
    (out / META_FILE).write_text(json.dumps(_clean(meta), ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


# ── 统计小工具 ─────────────────────────────────────────────────────────


def _clean(obj: Any) -> Any:
    """递归把 NaN / inf 换成 None，让 json.dumps 写出合法 JSON。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _rate(k: int, n: int) -> dict:
    lo, hi = stats.wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "wilson95": [lo, hi]}


def auroc(scores: Sequence[float], positive: Sequence[bool]) -> float:
    """Mann–Whitney 形式的 AUROC（并列取平均秩）；只有一个类时返回 NaN。"""
    s = np.asarray(scores, float)
    y = np.asarray(positive, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = sps.rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_ci(scores: Sequence[float], positive: Sequence[bool], n_boot: int = 1000, seed: int = 0) -> list[float]:
    """AUROC 的 bootstrap 95% 区间（按照片重抽样，跳过只剩一个类的重抽样）。"""
    s = np.asarray(scores, float)
    y = np.asarray(positive, bool)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(s), len(s))
        a = auroc(s[idx], y[idx])
        if not math.isnan(a):
            vals.append(a)
    if len(vals) < 20:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


# ── 预测文件与打分 ─────────────────────────────────────────────────────


@dataclass
class Pred:
    id: str
    letter: str | None  # None = 输出不合法 / 缺失
    valid: bool
    confidence: float | None
    p_other: float
    apparatus: list[str]
    latency_s: float | None = None
    confident: bool | None = None  # 只有规则基线填：规则是否"有把握"
    error: str | None = None


def read_predictions(path: str | Path) -> dict[str, dict]:
    """预测 JSONL → {id: 记录}；重复 id 报错（打分时不能悄悄挑一条）。"""
    out: dict[str, dict] = {}
    for rec in read_jsonl_dicts(path):
        if "id" not in rec:
            raise ValueError(f"预测记录缺 id：{str(rec)[:80]}")
        if rec["id"] in out:
            raise ValueError(f"预测文件里 id 重复：{rec['id']}")
        out[rec["id"]] = rec
    return out


def _letter_probs(rec: dict) -> dict[str, float] | None:
    """从 letter_probs 或 letter_logprobs 得到按字母归一的概率向量；都没有返回 None。"""
    raw = rec.get("letter_probs")
    if raw is None and rec.get("letter_logprobs") is not None:
        lp = rec["letter_logprobs"]
        top = max(lp.values()) if lp else 0.0
        raw = {k: math.exp(v - top) for k, v in lp.items()}
    if not isinstance(raw, dict) or not raw:
        return None
    probs: dict[str, float] = defaultdict(float)
    for k, v in raw.items():
        letter = normalize_letter(k)
        if letter is not None and isinstance(v, (int, float)) and v > 0:
            probs[letter] += float(v)
    tot = sum(probs.values())
    return {k: v / tot for k, v in probs.items()} if tot > 0 else None


def parse_prediction(pid: str, rec: dict | None) -> Pred:
    """一条预测记录 → Pred。缺失、解析不出字母都算 invalid（计分时算错）。"""
    if rec is None:
        return Pred(pid, None, False, None, 0.5, [], error="缺失")
    letter = conf = None
    apparatus: list[str] = []
    error = None
    if rec.get("text") is not None:
        try:
            r = parse_model_output(rec["text"])
            letter, conf, apparatus = r.letter, r.confidence, list(r.apparatus)
        except (RecognitionParseError, ValueError, TypeError) as e:
            error = str(e)[:80]
    elif rec.get("letter") is not None:  # 已解析好的记录（规则基线等）
        letter = normalize_letter(rec["letter"])
        conf = rec.get("confidence")
        apparatus = list(rec.get("apparatus") or [])
        if letter is None:
            error = f"字母不合法：{rec['letter']!r}"
    else:
        error = "既没有 text 也没有 letter"
    if conf is not None:
        conf = min(1.0, max(0.0, float(conf)))
    probs = _letter_probs(rec)
    if probs is not None:
        p_other = probs.get(OTHER_LETTER, 0.0)
    elif letter is None:
        p_other = 0.5
    elif conf is None:
        p_other = 1.0 if letter == OTHER_LETTER else 0.0
    else:
        p_other = conf if letter == OTHER_LETTER else 1.0 - conf
    lat = rec.get("latency_s")
    return Pred(pid, letter, letter is not None, conf, float(p_other), apparatus,
                float(lat) if isinstance(lat, (int, float)) else None, rec.get("confident"), error)


@dataclass
class ScoringPolicy:
    """学生在冻结前定死的计分规则。"""
    invalid_as: str = "wrong"  # "wrong"：非法输出算错（设计 §16.5 选项①）；"other"：当作选了 Z

    def __post_init__(self) -> None:
        if self.invalid_as not in ("wrong", "other"):
            raise ValueError("invalid_as 只能是 'wrong' 或 'other'")


def _apparatus_f1(items: list[dict]) -> dict:
    tp = fp = fn = 0
    for it in items:
        gold = set(it["gold_apparatus"]) & YOLO_VOCAB
        pred = set(it["apparatus"]) & YOLO_VOCAB
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "reference": "YOLO 标注框的器材（只含 6 类词表，标注可能漏标），不是学生亲写的参考答案"}


def _group_rates(items: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        groups[str(it[key])].append(it)
    return {g: _rate(sum(i["correct"] for i in v), len(v)) for g, v in sorted(groups.items())}


def _metrics(items: list[dict]) -> dict:
    n = len(items)
    out: dict[str, Any] = {"n": n}
    if not n:
        return out
    out["json_valid"] = _rate(sum(i["valid"] for i in items), n)
    out["top1"] = _rate(sum(i["correct"] for i in items), n)
    sub = {
        "closed_set_gold_not_Z": lambda i: i["gold"] != OTHER_LETTER,
        "trained_A_F": lambda i: i["gold"] in TRAINED_LETTERS,
        "held_out_G_L": lambda i: i["gold"] in HELD_OUT_LETTERS,
        "other_Z_recall": lambda i: i["gold"] == OTHER_LETTER,
    }
    for name, f in sub.items():
        grp = [i for i in items if f(i)]
        out[name] = _rate(sum(i["correct"] for i in grp), len(grp))
    z_pred = [i for i in items if i["pred"] == OTHER_LETTER]
    out["other_Z_precision"] = _rate(sum(i["gold"] == OTHER_LETTER for i in z_pred), len(z_pred))

    per: dict[str, dict] = {}
    conf: dict[str, dict[str, int]] = {}
    for letter in LETTERS:
        grp = [i for i in items if i["gold"] == letter]
        if grp:
            per[letter] = _rate(sum(i["correct"] for i in grp), len(grp))
            conf[letter] = dict(Counter(i["pred"] or "∅" for i in grp))
    out["per_letter_recall"] = per
    out["confusion"] = conf  # gold → {预测字母 或 "∅"(无效): 张数}
    out["by_source"] = _group_rates(items, "source")
    out["by_yolo_split"] = _group_rates(items, "yolo_split")

    pos = [i["gold"] == OTHER_LETTER for i in items]
    sc = [i["p_other"] for i in items]
    a = auroc(sc, pos)
    out["auroc_other"] = {"auroc": a, "n_pos": int(sum(pos)), "n_neg": int(n - sum(pos)),
                          "ci95": auroc_ci(sc, pos) if not math.isnan(a) else [None, None],
                          "note": "正类 = gold 为 Z；分数 = P(Z)"}
    valid = [i for i in items if i["valid"] and i["conf"] is not None]
    if valid:
        out["ece"] = stats.ece([i["conf"] for i in valid], [i["correct"] for i in valid], n_bins=10)["ece"]
    lat = [i["latency_s"] for i in items if i["latency_s"] is not None]
    if lat:
        out["latency_median_s"] = float(np.median(lat))
    out["apparatus_f1"] = _apparatus_f1(items)
    conf_items = [i for i in items if i["confident"] is not None]
    if conf_items:
        c = [i for i in conf_items if i["confident"]]
        out["confident_only"] = {"coverage": _rate(len(c), len(conf_items)),
                                 "accuracy": _rate(sum(i["correct"] for i in c), len(c))}
    return out


def score(examples: Sequence[Example] | str | Path, predictions: dict[str, dict] | str | Path,
          *, policy: ScoringPolicy | None = None, arm: str = "") -> dict:
    """给一组预测打分。缺失的 id 算无效（不丢弃）；predictions 里多出来的 id 只记录、不计分。"""
    policy = policy or ScoringPolicy()
    exs = list(read_jsonl(examples)) if isinstance(examples, (str, Path)) else list(examples)
    preds = read_predictions(predictions) if isinstance(predictions, (str, Path)) else dict(predictions)
    ids = {e.id for e in exs}
    by_photo = {e.meta.get("photo_id"): e.id for e in exs}
    keyed: dict[str, dict] = {}
    for k, rec in preds.items():
        keyed[k if k in ids else by_photo.get(k, k)] = rec  # 接受 example id 或 meta.photo_id
    items = []
    for e in exs:
        p = parse_prediction(e.id, keyed.get(e.id))
        gold = e.meta["gold_letter"]
        letter = p.letter if p.valid else (OTHER_LETTER if policy.invalid_as == "other" else None)
        items.append({
            "id": e.id, "gold": gold, "gold_status": e.meta["gold_status"], "source": e.meta.get("source"),
            "yolo_split": e.meta.get("yolo_split"), "pred": letter, "valid": p.valid,
            "correct": letter == gold, "conf": p.confidence, "p_other": p.p_other,
            "apparatus": p.apparatus, "gold_apparatus": e.meta.get("gold_apparatus_partial", []),
            "latency_s": p.latency_s, "confident": p.confident,
        })
    extra = sorted(k for k in keyed if k not in ids)
    reviewed = [i for i in items if i["gold_status"] == "reviewed"]
    provisional = [i for i in items if i["gold_status"] != "reviewed"]
    caveats = []
    if not reviewed:
        caveats.append("没有任何学生核实过的 gold：下面的数字全部基于规则暂定标签，不是证据，只能用来检查流程。")
    if provisional:
        caveats.append(f"{len(provisional)} 张照片的 gold 还是规则暂定值（'provisional' 一栏）；暂定标签就是规则对真值框的输出，"
                       "规则基线在这一栏按构造得满分，其他 arm 在这一栏的数字也不能当结果。")
    seen = [i for i in items if i["yolo_split"] in ("train", "val")]
    if seen:
        caveats.append(f"{len(seen)} 张照片进过 YOLO 训练或验证集，'规则 + YOLO 检测框'在其上偏乐观，请看 by_yolo_split。")
    return _clean({
        "bench_version": BENCH_VERSION, "prompt_hash": prompt_hash(), "arm": arm,
        "policy": asdict(policy),
        "n_test": len(exs),
        "n_missing": sum(1 for e in exs if e.id not in keyed),
        "extra_prediction_ids": extra,
        "headline_subset": "reviewed" if reviewed else None,
        "reviewed": _metrics(reviewed),
        "provisional": _metrics(provisional),
        "all": _metrics(items),
        "caveats": caveats,
    })


def brief(report: dict) -> str:
    """报告的几行中文摘要（命令行用）。"""
    lines = [f"arm={report.get('arm') or '-'}  测试照片 {report['n_test']}  缺失预测 {report['n_missing']}"]
    for name, label in (("reviewed", "已核实（证据）"), ("provisional", "暂定（非证据）")):
        m = report[name]
        if not m.get("n"):
            lines.append(f"  {label}：0 张")
            continue
        t = m["top1"]
        lo, hi = t["wilson95"]
        au = m["auroc_other"]["auroc"]
        lines.append(f"  {label}：{m['n']} 张  top-1 {t['k']}/{t['n']} = {t['rate']:.3f}"
                     f"（Wilson 95% [{lo:.3f}, {hi:.3f}]）  合法率 {m['json_valid']['rate']:.2f}  "
                     f"AUROC(其他) {'无' if au is None else f'{au:.3f}'}")
    lines += [f"  注意：{c}" for c in report["caveats"]]
    return "\n".join(lines)


# ── 规则基线 ───────────────────────────────────────────────────────────


def rule_probs(cands: list[dict]) -> dict[str, float]:
    """规则分数的 softmax 概率向量（Z 用虚拟分数 OTHER_SCORE，与 cues.rule_confidence 同一公式）。

    没有候选时 Z 的概率是 1。字母表里其余字母概率为 0（规则不认识它们）。
    """
    score_by = {c["letter"]: c["score"] for c in cands}
    score_by[OTHER_LETTER] = _cues.OTHER_SCORE
    m = max(score_by.values())
    w = {k: math.exp((v - m) / _cues.SOFTMAX_TAU) for k, v in score_by.items()}
    tot = sum(w.values())
    return {k: round(v / tot, 6) for k, v in sorted(w.items())}


def rules_prediction(dets: list[dict]) -> dict:
    """YOLO 加规则的预测：第一名候选（没有候选 → Z）；平局时取 cues 的排序结果（分数降序、字母升序）。"""
    a = _cues.analyze(dets)
    probs = rule_probs(a["candidates"])
    letter = a["candidates"][0]["letter"] if a["candidates"] else OTHER_LETTER
    return {"letter": letter, "confidence": probs[letter], "letter_probs": probs, "apparatus": a["apparatus"],
            "confident": bool(a["confident"]), "tied": a["tied"]}


def run_rules(examples: Iterable[Example], boxes: str = "detections") -> dict[str, dict]:
    """在测试样本上跑规则基线。boxes = 'gt'（真值框）或 'detections'（YOLO 检测框）。"""
    key = {"gt": "gt_boxes", "detections": "detections"}[boxes]
    return {e.id: {"id": e.id, **rules_prediction(e.meta.get(key, []))} for e in examples}


# ── 命令行 ─────────────────────────────────────────────────────────────


def _parse_gap(s: str) -> float | str:
    return "auto" if s == "auto" else float(s)


def _cmd_split(a: argparse.Namespace) -> int:
    rows = read_jsonl_dicts(a.manifest)
    old = a.out / SPLIT_FILE
    if old.exists() and not a.force:
        print(f"{old} 已存在（可能已经交给别人用了）。要重新切分请加 --force。", file=sys.stderr)
        return 2
    try:
        info = session_split(rows, a.src if a.src.exists() else None, a.manifest.parent, gap_s=a.gap_s,
                             test_frac=a.test_frac, seed=a.seed, min_sessions=a.min_sessions)
    except DegenerateSplit as e:
        print(f"拒绝切分：{e}", file=sys.stderr)
        return 2
    write_split(info, old)
    print(json.dumps({k: v for k, v in info.items() if k != "photos"}, ensure_ascii=False, indent=1))
    print(f"已写 {old}（status=draft：间隔和种子由学生在冻结前确认）")
    return 0


def _cmd_build(a: argparse.Namespace) -> int:
    rows = read_jsonl_dicts(a.manifest)
    split_path = a.out / SPLIT_FILE
    if not split_path.exists():
        print(f"没有 {split_path}，请先运行 split。", file=sys.stderr)
        return 2
    split = load_split(split_path)
    review = load_review(a.review, [r["id"] for r in rows]) if a.review else None
    det = YoloDetector(a.weights, a.conf)
    meta = build_test_set(rows, split, a.out, image_root=a.manifest.parent, review=review, detector=det,
                          detector_info=det.info, review_sha256=sha256_file(a.review) if a.review else None,
                          seed=split["params"]["seed"])
    print(json.dumps({k: v for k, v in meta.items() if k != "development_ban"}, ensure_ascii=False, indent=1))
    print("冻结由学生本人运行：python -m lab_detective.bench.freeze freeze", a.out, "--who <姓名>")
    return 0


def _cmd_score(a: argparse.Namespace) -> int:
    rep = score(a.out / TEST_FILE, a.pred, policy=ScoringPolicy(a.invalid_as), arm=a.arm or Path(a.pred).parent.name)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(brief(rep))
    return 0


def _cmd_rules(a: argparse.Namespace) -> int:
    exs = list(read_jsonl(a.out / TEST_FILE))
    for arm, boxes in (("rules_gt", "gt"), ("rules_yolo", "detections")):
        preds = run_rules(exs, boxes)
        d = a.runs / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / "predictions.jsonl").write_text(
            "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in preds.values()), encoding="utf-8")
        rep = score(exs, preds, arm=arm)
        (d / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(brief(rep))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="识别 bench：切分、生成、打分、规则基线")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    ap.add_argument("--out", type=Path, default=BENCH_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split", help="按拍摄时段切分，写 split.json")
    s.add_argument("--gap-s", type=_parse_gap, default=DEFAULT_GAP_S, help="时段间隔（秒）或 auto，默认 1800")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--test-frac", type=float, default=TEST_FRAC)
    s.add_argument("--min-sessions", type=int, default=MIN_SESSIONS)
    s.add_argument("--src", type=Path, default=SRC_DEFAULT, help="旧项目 data 目录（只读，用来读原图 EXIF）")
    s.add_argument("--force", action="store_true", help="覆盖已有 split.json")
    s.set_defaults(fn=_cmd_split)
    b = sub.add_parser("build", help="生成 test.jsonl（需要 split.json；会跑 YOLO）")
    b.add_argument("--review", type=Path, default=None, help="review_gallery 导出的 review.json")
    b.add_argument("--weights", type=Path, default=None)
    b.add_argument("--conf", type=float, default=0.25)
    b.set_defaults(fn=_cmd_build)
    c = sub.add_parser("score", help="给预测文件打分")
    c.add_argument("--pred", type=Path, required=True)
    c.add_argument("--report", type=Path, default=None)
    c.add_argument("--arm", default="")
    c.add_argument("--invalid-as", choices=("wrong", "other"), default="wrong")
    c.set_defaults(fn=_cmd_score)
    r = sub.add_parser("rules", help="YOLO 加规则基线（真值框 / 检测框）")
    r.add_argument("--runs", type=Path, default=RUNS_DIR)
    r.set_defaults(fn=_cmd_rules)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
