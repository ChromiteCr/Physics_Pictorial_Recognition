"""服务器端共用：训练 JSONL 读取与校验、对话模板渲染与"只对助手输出算损失"的掩码、
确定性打乱、泄漏检查、打乱标签对照、冒烟用的合成数据。

数据格式见 docs/ARCHITECTURE.md §2.5。本目录（server/）自成一体，不导入 lab_detective，
上传服务器时只需 server/ 和数据包。

约定：
1. 每条样本最后一条消息是助手回复（训练目标 / bench 的金标准）。只对这一条算损失：
   prompt = 模板渲染(messages[:-1], add_generation_prompt=True, enable_thinking=False)
   target = 助手内容 + 回合结束符（<|im_end|>）
   这样训练时的前缀和推理时逐字节一致；前面的助手回合只当上下文。
2. 关闭思考：渲染出的 prompt 必须以空思考块 "<think>\\n\\n</think>\\n\\n" 结尾，否则报错
   （处理器会把模板不认识的参数悄悄当作别的参数，只给一条警告）。
3. 超长样本直接丢弃，不截断（截断会切坏图像 token，和像素特征对不上）；正式训练时有超长样本就拒绝开训
   （train_sft.py --allow-truncation 才放行），免得目标最长的诊断样本被悄悄丢掉。
4. 数据顺序只由 (seed, epoch, id) 决定，与内容无关：S1 和打乱标签对照看到的批次完全相同。

本文件顶层不导入 torch / transformers / PIL（本机 CPU 测试只测纯 Python 逻辑）。
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

TASKS = ("diagnose", "design", "recognize", "guide", "explain", "card", "knowledge")
SOURCES = ("engine", "template", "distill-deepseek", "distill-hy4", "photo")
ROLES = ("system", "user", "assistant")
IGNORE_INDEX = -100
EMPTY_THINK = "<think>\n\n</think>\n\n"
END_MARKER = "<|im_end|>"
IMAGE_TOKEN = "<|image_pad|>"
# 和 input_ids 等长、需要按序列拼接的键；其余（pixel_values、image_grid_thw）按第 0 维拼接
SEQ_KEYS = ("attention_mask", "mm_token_type_ids", "token_type_ids")


# ---------------------------------------------------------------- 读取与校验

def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} 不是合法 JSON：{e}") from e
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def normalize_parts(content: Any) -> list[dict]:
    """消息内容统一成 [{"type": "text"|"image", ...}]。"""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append({"type": "text", "text": p})
            elif isinstance(p, dict) and p.get("type") == "text":
                parts.append({"type": "text", "text": str(p.get("text", ""))})
            elif isinstance(p, dict) and p.get("type") == "image":
                img = p.get("image") or p.get("path") or p.get("url")
                if not isinstance(img, str) or not img:
                    raise ValueError(f"图片片段缺少路径：{p}")
                parts.append({"type": "image", "image": img})
            else:
                raise ValueError(f"不认识的内容片段：{p!r}")
        return parts
    if isinstance(content, (dict, int, float)):
        return [{"type": "text", "text": json.dumps(content, ensure_ascii=False)}]
    raise ValueError(f"不认识的消息内容：{content!r}")


def text_of(content: Any) -> str:
    """只取文字部分（助手、系统消息用）。"""
    return "".join(p["text"] for p in normalize_parts(content) if p["type"] == "text")


def validate_record(rec: dict) -> list[str]:
    """返回错误列表（空 = 合法）。"""
    errs = []
    if not isinstance(rec, dict):
        return ["样本不是 JSON 对象"]
    rid = rec.get("id")
    if not isinstance(rid, str) or not rid:
        errs.append("缺少 id")
    if rec.get("task") not in TASKS:
        errs.append(f"task 不合法：{rec.get('task')!r}")
    src = rec.get("source")
    if not isinstance(src, str) or not (src in SOURCES or src.startswith("distill-")):
        errs.append(f"source 不合法：{src!r}")
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        errs.append("messages 至少要有 2 条")
        return errs
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            errs.append(f"messages[{i}] 角色不合法")
            continue
        try:
            parts = normalize_parts(m.get("content"))
        except ValueError as e:
            errs.append(f"messages[{i}]：{e}")
            continue
        if m["role"] != "user" and any(p["type"] == "image" for p in parts):
            errs.append(f"messages[{i}]：只有用户消息可以带图片")
    if msgs[-1].get("role") != "assistant":
        errs.append("最后一条消息必须是助手回复（训练目标 / 金标准）")
    elif not text_of(msgs[-1].get("content")).strip():
        errs.append("助手回复为空")
    if not any(m.get("role") == "user" for m in msgs if isinstance(m, dict)):
        errs.append("没有用户消息")
    meta = rec.get("meta", {})
    if meta is not None and not isinstance(meta, dict):
        errs.append("meta 必须是对象")
    return errs


def load_records(paths: Sequence[str | Path], strict: bool = True) -> list[dict]:
    """读多个 JSONL，校验并检查 id 重复。strict=False 时跳过坏样本（记到 stderr）。"""
    recs, seen = [], set()
    for path in paths:
        for i, rec in enumerate(read_jsonl(path)):
            errs = validate_record(rec)
            if not errs and rec["id"] in seen:
                errs = [f"id 重复：{rec['id']}"]
            if errs:
                msg = f"{path} 第 {i + 1} 条：" + "；".join(errs)
                if strict:
                    raise ValueError(msg)
                print("跳过 " + msg, file=sys.stderr)
                continue
            seen.add(rec["id"])
            recs.append(rec)
    return recs


def meta_of(rec: dict) -> dict:
    return rec.get("meta") or {}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 图片

def resolve_image(rel: str, root: str | Path | None) -> Path:
    """图片路径相对数据包根目录；不允许跳出根目录。"""
    p = Path(rel)
    if root is None:
        return p
    root = Path(root).resolve()
    full = (root / p).resolve()
    if root != full and root not in full.parents:
        raise ValueError(f"图片路径跳出了数据包根目录：{rel}")
    return full


def image_refs(messages: Sequence[dict]) -> list[str]:
    """按出现顺序列出图片的相对路径。"""
    return [p["image"] for m in messages for p in normalize_parts(m.get("content"))
            if p["type"] == "image"]


def unmarked_image_hits(records: Sequence[dict], root: str | Path | None) -> list[str]:
    """识别 bench 的图应是带编号框的版本；meta.image_sha1 记的是没画框的原图。
    解析出来的图片字节和 image_sha1 一样，说明打包时拿错了原图（提示里说有编号框，图上却没有）。返回撞上的 id。"""
    hits = []
    for rec in records:
        meta = meta_of(rec)
        sha1 = meta.get("image_sha1")
        if not sha1 or meta.get("legend") is None:
            continue
        for ref in image_refs(rec["messages"]):
            p = resolve_image(ref, root)
            if p.is_file() and hashlib.sha1(p.read_bytes()).hexdigest() == sha1:
                hits.append(rec["id"])
                break
    return hits


def load_pil(path: str | Path):
    from PIL import Image  # 只在服务器上用到

    with Image.open(path) as im:
        return im.convert("RGB")


# ---------------------------------------------------------------- 模板与掩码

def split_prompt_target(rec: dict) -> tuple[list[dict], str]:
    msgs = rec["messages"]
    if msgs[-1].get("role") != "assistant":
        raise ValueError(f"{rec.get('id')}: 最后一条不是助手回复")
    return list(msgs[:-1]), text_of(msgs[-1]["content"])


def to_template_messages(messages: Sequence[dict]) -> list[dict]:
    """转成 HF 处理器的对话格式：用户消息是片段列表（图片只放占位 {"type":"image"}），
    系统 / 助手消息是纯字符串。图片本身另行按顺序传给处理器。"""
    out = []
    for m in messages:
        parts = normalize_parts(m.get("content"))
        if m["role"] == "user":
            content = [{"type": "image"} if p["type"] == "image" else {"type": "text", "text": p["text"]}
                       for p in parts]
        else:
            content = "".join(p["text"] for p in parts if p["type"] == "text")
        out.append({"role": m["role"], "content": content})
    return out


@dataclass
class Codec:
    """模板渲染 + 分段编码。服务器上由 make_hf_codec 构造；测试里用假的渲染器和分词器。

    render(messages, add_generation_prompt) -> str
    tokenize(text, images) -> {"input_ids": [...], "seq": {键: [...]}, "mm": {键: 张量}}
    """

    render: Callable[[list[dict], bool], str]
    tokenize: Callable[[str, list], dict]
    image_token: str = IMAGE_TOKEN
    end_marker: str = END_MARKER
    empty_think: str = EMPTY_THINK


def render_prompt(codec: Codec, prompt_messages: Sequence[dict], require_no_think: bool = True) -> str:
    text = codec.render(to_template_messages(prompt_messages), True)
    if require_no_think and not text.endswith(codec.empty_think):
        raise ValueError(
            "渲染出的 prompt 没有以空思考块结尾：模板可能没认 enable_thinking=False。"
            f"结尾是：{text[-60:]!r}"
        )
    return text


def encode_example(rec: dict, codec: Codec, root: str | Path | None = None,
                   load_image: Callable[[Path], Any] | None = None,
                   require_no_think: bool = True) -> dict:
    """一条样本 → input_ids / labels（prompt 部分全是 IGNORE_INDEX）/ 序列附加键 / 图像张量。"""
    prompt_msgs, target = split_prompt_target(rec)
    prompt_text = render_prompt(codec, prompt_msgs, require_no_think)
    refs = image_refs(prompt_msgs)
    n_ph = prompt_text.count(codec.image_token)
    if n_ph != len(refs):
        raise ValueError(f"{rec['id']}: 模板里有 {n_ph} 个图片占位，消息里有 {len(refs)} 张图")
    load_image = load_image or load_pil
    images = [load_image(resolve_image(r, root)) for r in refs]
    p = codec.tokenize(prompt_text, images)
    t = codec.tokenize(target + codec.end_marker, [])
    n_p, n_t = len(p["input_ids"]), len(t["input_ids"])
    if n_t == 0:
        raise ValueError(f"{rec['id']}: 目标为空")
    seq = {}
    for k in set(p.get("seq", {})) | set(t.get("seq", {})):
        if k not in p.get("seq", {}) or k not in t.get("seq", {}):
            raise ValueError(f"序列附加键 {k} 只在一段里出现")
        seq[k] = list(p["seq"][k]) + list(t["seq"][k])
        if len(seq[k]) != n_p + n_t:
            raise ValueError(f"序列附加键 {k} 长度不对")
    if t.get("mm"):
        raise ValueError("目标段不应带图像张量")
    return {
        "id": rec["id"],
        "input_ids": list(p["input_ids"]) + list(t["input_ids"]),
        "labels": [IGNORE_INDEX] * n_p + list(t["input_ids"]),
        "seq": seq,
        "mm": p.get("mm", {}),
        "n_prompt": n_p,
        "n_target": n_t,
        "n_images": len(refs),
        "prompt_text": prompt_text,
    }


def check_full_render(rec: dict, codec: Codec) -> bool:
    """诊断用：完整对话渲染是否以 prompt + 目标 + 结束符开头（模板和我们的拼法一致）。"""
    prompt_msgs, target = split_prompt_target(rec)
    prompt_text = codec.render(to_template_messages(prompt_msgs), True)
    full = codec.render(to_template_messages(rec["messages"]), False)
    return full.startswith(prompt_text + target + codec.end_marker)


def make_hf_codec(processor, enable_thinking: bool = False) -> Codec:
    """服务器上用：基于 transformers 处理器的渲染和分段编码。"""
    tok = getattr(processor, "tokenizer", processor)
    image_token = getattr(processor, "image_token", None) or getattr(tok, "image_token", None) or IMAGE_TOKEN

    def render(messages, add_generation_prompt):
        msgs = copy.deepcopy(messages)
        kw = dict(tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)
        if getattr(processor, "chat_template", None):
            return processor.apply_chat_template(msgs, **kw)
        return tok.apply_chat_template(msgs, **kw)

    def tokenize(text, images):
        if images:
            out = processor(text=[text], images=images, return_tensors="pt")
        elif tok is not processor:
            out = processor(text=[text], return_tensors="pt")
        else:
            out = tok([text], return_tensors="pt", add_special_tokens=False)
        ids = out["input_ids"][0].tolist()
        seq, mm = {}, {}
        for k, v in out.items():
            if k == "input_ids":
                continue
            if k in SEQ_KEYS:
                seq[k] = v[0].tolist()
            else:
                mm[k] = v
        return {"input_ids": ids, "seq": seq, "mm": mm}

    return Codec(render=render, tokenize=tokenize, image_token=image_token)


def template_has_thinking_switch(processor) -> bool:
    tmpl = getattr(processor, "chat_template", None) or getattr(getattr(processor, "tokenizer", None), "chat_template", "")
    if isinstance(tmpl, dict):
        tmpl = tmpl.get("default", "")
    return "enable_thinking" in (tmpl or "")


# ---------------------------------------------------------------- 确定性顺序

def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def deterministic_order(ids: Sequence[str], seed: int, epoch: int = 0) -> list[int]:
    """按 sha256(seed:epoch:id) 排序：只依赖 id 集合和种子，与输入顺序、样本内容都无关。"""
    return sorted(range(len(ids)), key=lambda i: (_h(f"{seed}:{epoch}:{ids[i]}"), ids[i]))


def epoch_plan(ids: Sequence[str], seed: int, epochs: int) -> list[int]:
    plan = []
    for e in range(epochs):
        plan.extend(deterministic_order(ids, seed, e))
    return plan


# ---------------------------------------------------------------- 泄漏检查

def normalize_text(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).split())


def user_hash(rec: dict, root: str | Path | None = None) -> str:
    """用户回合内容哈希：文字做 NFKC + 合并空白；图片用文件字节的 sha256（找不到文件时退回路径）。
    不含系统提示（系统提示版本不同也能抓到同一份数据）。"""
    parts = []
    for m in rec["messages"]:
        if m.get("role") != "user":
            continue
        for p in normalize_parts(m.get("content")):
            if p["type"] == "text":
                parts.append(["t", normalize_text(p["text"])])
            else:
                full = resolve_image(p["image"], root) if root is not None else None
                if full is not None and full.is_file():
                    parts.append(["i", file_sha256(full)])
                else:
                    parts.append(["p", Path(p["image"]).as_posix()])
    return _h(json.dumps(parts, ensure_ascii=False, separators=(",", ":")))


_HEX64 = re.compile(r"^(sha256:)?[0-9a-f]{64}$")


def load_bench_index(paths: Sequence[str | Path], root: str | Path | None = None) -> dict:
    """bench 的 id 和用户回合哈希。支持 .jsonl（同格式样本，推荐，哈希算法保证一致）、
    .txt（每行一个 id 或 64 位十六进制哈希）、.json（列表或 {"ids": [...], "hashes": [...]}）。"""
    ids, hashes = set(), set()
    for path in paths:
        path = Path(path)
        if path.suffix == ".jsonl":
            for rec in read_jsonl(path):
                if "id" in rec:
                    ids.add(str(rec["id"]))
                if "messages" in rec:
                    hashes.add(user_hash(rec, root))
        elif path.suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                ids.update(map(str, obj.get("ids", [])))
                hashes.update(h.removeprefix("sha256:") for h in obj.get("hashes", []))
            else:
                for x in obj:
                    x = str(x)
                    (hashes.add(x.removeprefix("sha256:")) if _HEX64.match(x) else ids.add(x))
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                x = line.strip()
                if not x or x.startswith("#"):
                    continue
                (hashes.add(x.removeprefix("sha256:")) if _HEX64.match(x) else ids.add(x))
    return {"ids": ids, "hashes": hashes}


def leakage_report(records: Sequence[dict], bench: dict, root: str | Path | None = None) -> dict:
    id_hits, hash_hits = [], []
    for rec in records:
        if rec["id"] in bench["ids"]:
            id_hits.append(rec["id"])
        if bench["hashes"] and user_hash(rec, root) in bench["hashes"]:
            hash_hits.append(rec["id"])
    return {"n_checked": len(records), "n_bench_ids": len(bench["ids"]),
            "n_bench_hashes": len(bench["hashes"]), "id_hits": id_hits, "hash_hits": hash_hits,
            "clean": not id_hits and not hash_hits}


# ---------------------------------------------------------------- 打乱标签对照

def cause_domain(rec: dict) -> tuple[list[str], list[str | None]] | None:
    """样本自己的候选原因：优先 meta.causes（[{"id","name"}] 或 [id]），其次 meta.cause_ids / meta.candidates。"""
    meta = meta_of(rec)
    causes = meta.get("causes")
    if isinstance(causes, list) and causes:
        ids, names = [], []
        for c in causes:
            if isinstance(c, str):
                ids.append(c)
                names.append(None)
            elif isinstance(c, dict) and (c.get("id") or c.get("cause_id")):
                ids.append(str(c.get("id") or c.get("cause_id")))
                nm = c.get("name") or c.get("name_zh")
                names.append(str(nm) if nm else None)
        if ids:
            return ids, names
    for key in ("cause_ids", "candidates"):
        v = meta.get(key)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return list(v), [None] * len(v)
    return None


def _collect_key(obj: Any, key: str, out: list) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, str):
                out.append(v)
            else:
                _collect_key(v, key, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_key(v, key, out)


def target_cause_ids(text: str, key: str = "cause_id") -> list[str]:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    out: list[str] = []
    _collect_key(obj, key, out)
    return out


def experiment_domains(records: Sequence[dict], tasks: Sequence[str] = ("diagnose",),
                       key: str = "cause_id") -> dict[str, list[str]]:
    """兜底：没有 meta.causes 时，按实验汇总全数据集诊断目标里出现过的原因 id。"""
    dom: dict[str, set] = {}
    for rec in records:
        if rec.get("task") not in tasks:
            continue
        exp = str(meta_of(rec).get("experiment_id", ""))
        _, target = split_prompt_target(rec)
        dom.setdefault(exp, set()).update(target_cause_ids(target, key))
    return {k: sorted(v) for k, v in dom.items()}


def sample_permutation(sample_id: str, seed: int, n: int) -> list[int]:
    """每个样本独立的均匀随机置换（由 seed 和 id 决定）。不强制错排：
    均匀置换下标签与数据独立，这才是零假设；强制错排反而让标签和真相负相关。"""
    rng = random.Random(int(_h(f"shuffle:{seed}:{sample_id}")[:16], 16))
    perm = list(range(n))
    rng.shuffle(perm)
    return perm


def _token_pattern(tok: str) -> str:
    if re.fullmatch(r"[\x21-\x7e]+", tok):  # ASCII id：两侧加边界，避免改到更长的 id 里
        return r"(?<![A-Za-z0-9_.])" + re.escape(tok) + r"(?![A-Za-z0-9_]|\.[A-Za-z0-9_])"
    return re.escape(tok)


def relabel_text(text: str, mapping: dict[str, str]) -> str:
    """一次扫描同时替换所有标签（不会链式替换），长的优先。"""
    mapping = {k: v for k, v in mapping.items() if k and k != v}
    if not mapping:
        return text
    keys = sorted(mapping, key=len, reverse=True)
    pat = re.compile("|".join(_token_pattern(k) for k in keys))
    return pat.sub(lambda m: mapping[m.group(0)], text)


def shuffle_cause_labels(rec: dict, seed: int, fallback_domain: Sequence[str] | None = None) -> tuple[dict, dict]:
    """返回 (新样本, 信息)。只改最后一条助手回复；prompt 不变。
    原因 id 在目标里任何位置（值、键、解释文字）一起替换；meta.causes 给了名称时，名称（≥2 字）也一起换。"""
    dom = cause_domain(rec)
    how = "meta"
    if dom is None and fallback_domain:
        dom = (list(fallback_domain), [None] * len(fallback_domain))
        how = "experiment"
    if dom is None or len(dom[0]) < 2:
        return rec, {"how": "none", "changed": False, "top_changed": False}
    ids, names = dom
    perm = sample_permutation(rec["id"], seed, len(ids))
    mapping = {ids[i]: ids[perm[i]] for i in range(len(ids))}
    if all(n is not None for n in names):
        for i in range(len(ids)):
            if len(names[i]) >= 2:
                mapping[names[i]] = names[perm[i]]
    _, target = split_prompt_target(rec)
    new_target = relabel_text(target, mapping)
    new = copy.deepcopy(rec)
    new["messages"][-1] = {"role": "assistant", "content": new_target}
    new.setdefault("meta", {})["label_shuffle"] = {"seed": seed, "how": how, "perm": perm}
    # 目标里没有 cause_id 时（"自己提出原因"变体只写中文名）按第一个出现的 id，再按第一个出现的名称判断第一名
    named = [n for n in names if n and len(n) >= 2] if all(n is not None for n in names) else []
    old_ids = target_cause_ids(target) or _first_id(target, ids) or _first_id(target, named)
    new_ids = target_cause_ids(new_target) or _first_id(new_target, ids) or _first_id(new_target, named)
    top_changed = bool(old_ids and new_ids and old_ids[0] != new_ids[0])
    return new, {"how": how, "changed": new_target != target, "top_changed": top_changed}


def _first_id(text: str, ids: Sequence[str]) -> list[str]:
    best = None
    for c in ids:
        m = re.search(_token_pattern(c), text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), c)
    return [best[1]] if best else []


def apply_label_shuffle(records: Sequence[dict], seed: int, tasks: Sequence[str] = ("diagnose",)) -> tuple[list[dict], dict]:
    """打乱标签对照：对 tasks 里的样本逐个独立置换原因标签。返回 (新样本列表, 统计)。
    统计里 frac_changed 应接近 1，frac_top_changed 应接近 1 − 1/候选数；明显偏低说明对照没起作用。"""
    fallback = experiment_domains(records, tasks)
    out, stats = [], {"n_target_task": 0, "n_changed": 0, "n_top_changed": 0,
                      "by_how": {"meta": 0, "experiment": 0, "none": 0}}
    for rec in records:
        if rec.get("task") not in tasks:
            out.append(rec)
            continue
        stats["n_target_task"] += 1
        exp = str(meta_of(rec).get("experiment_id", ""))
        new, info = shuffle_cause_labels(rec, seed, fallback.get(exp))
        stats["by_how"][info["how"]] += 1
        stats["n_changed"] += int(info["changed"])
        stats["n_top_changed"] += int(info["top_changed"])
        out.append(new)
    n = max(stats["n_target_task"], 1)
    stats["frac_changed"] = stats["n_changed"] / n
    stats["frac_top_changed"] = stats["n_top_changed"] / n
    return out, stats


# ---------------------------------------------------------------- 统计

def dataset_summary(records: Sequence[dict]) -> dict:
    by = {"task": {}, "source": {}, "experiment_id": {}, "split": {}}
    n_img = 0
    for rec in records:
        meta = meta_of(rec)
        for k, v in (("task", rec.get("task")), ("source", rec.get("source")),
                     ("experiment_id", meta.get("experiment_id")), ("split", meta.get("split"))):
            by[k][str(v)] = by[k].get(str(v), 0) + 1
        n_img += len(image_refs(rec["messages"]))
    return {"n": len(records), "n_images": n_img, **by}


def length_stats(xs: Sequence[int]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)

    def q(p):
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]

    return {"n": len(s), "mean": sum(s) / len(s), "p50": q(0.5), "p90": q(0.9), "p95": q(0.95),
            "p99": q(0.99), "max": s[-1], "sum": sum(s)}


# ---------------------------------------------------------------- safetensors 头（只读键名和形状）

def safetensors_header(path: str | Path) -> dict:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def model_weight_shapes(model_dir: str | Path) -> dict[str, list[int]]:
    """模型目录（或单个 .safetensors / index.json）里所有权重的名称 → 形状（index.json 只给名称时形状为 None）。"""
    p = Path(model_dir)
    files: list[Path]
    if p.is_file() and p.suffix == ".safetensors":
        files = [p]
    elif p.is_file() and p.name.endswith(".json"):
        idx = json.loads(p.read_text(encoding="utf-8"))
        shards = sorted(set(idx["weight_map"].values()))
        files = [p.parent / s for s in shards if (p.parent / s).is_file()]
        if not files:
            return {k: None for k in idx["weight_map"]}
    else:
        files = sorted(p.glob("*.safetensors"))
        if not files and (p / "model.safetensors.index.json").is_file():
            return model_weight_shapes(p / "model.safetensors.index.json")
    out = {}
    for f in files:
        for k, v in safetensors_header(f).items():
            out[k] = v.get("shape")
    return out


# ---------------------------------------------------------------- 冒烟用合成数据

SYNTH_SYSTEM = "你是实验侦探，只输出 JSON。"


def make_synthetic(out_dir: str | Path, n: int = 24, n_bench: int = 4, seed: int = 0,
                   with_images: bool = True) -> dict:
    """生成一小份合成数据（诊断 JSON 目标 + 可选的看图识别），只用于冒烟测试，输出随后丢弃。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    causes = [{"id": "hooke.random_only", "name": "只有随机误差"},
              {"id": "hooke.hanger_omitted", "name": "漏计挂钩质量"},
              {"id": "hooke.zero_shift", "name": "原长读数偏差"},
              {"id": "hooke.over_limit", "name": "超出弹性限度"}]
    recs = []
    for i in range(n + n_bench):
        xs = [0.5 * (k + 1) for k in range(5)]
        icp = rng.choice([0.0, 1.5])
        ys = [round(4.0 * x + icp + rng.gauss(0, 0.05), 1) for x in xs]
        top = causes[1] if icp else causes[0]
        probs = sorted([rng.random() for _ in causes], reverse=True)
        ranked = [top] + [c for c in causes if c is not top]
        target = {"ranked": [{"cause_id": c["id"], "prob": round(p / sum(probs), 2)} for c, p in zip(ranked, probs)],
                  "explain": f"截距 {icp:.1f} cm，最可能是{top['name']}。"}
        user = f"样本 {i}。候选原因：" + "；".join(f"{c['id']}={c['name']}" for c in causes) + \
               "\n数据（F/N, x/cm）：" + json.dumps(list(zip(xs, ys)), ensure_ascii=False)
        recs.append({"id": f"synth-diag-{i:04d}", "task": "diagnose", "source": "engine",
                     "messages": [{"role": "system", "content": SYNTH_SYSTEM},
                                  {"role": "user", "content": [{"type": "text", "text": user}]},
                                  {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}],
                     "meta": {"experiment_id": "hooke", "split": "train", "seed": seed + i, "causes": causes}})
    bench_ids = {r["id"] for r in recs[n:n + n_bench]}
    if with_images:
        from PIL import Image, ImageDraw

        (out / "images").mkdir(exist_ok=True)
        for i in range(5):  # 前 4 张进训练，最后 1 张进 bench（冒烟评测也要走一遍带图的生成）
            im = Image.new("RGB", (320, 240), (240, 240, 235))
            d = ImageDraw.Draw(im)
            d.line([(160, 20), (160, 140)], fill=(60, 60, 60), width=3)
            d.rectangle([140, 140, 180, 180 + 10 * i], outline=(20, 20, 160), width=3)
            name = f"images/synth_{i}.jpg"
            im.save(out / name, quality=85)
            recs.append({"id": f"synth-rec-{i:04d}", "task": "recognize", "source": "photo",
                         "messages": [{"role": "system", "content": SYNTH_SYSTEM},
                                      {"role": "user", "content": [{"type": "image", "image": name},
                                                                   {"type": "text", "text": "这是什么实验？"}]},
                                      {"role": "assistant", "content": json.dumps(
                                          {"letter": "C", "experiment_id": "hooke"}, ensure_ascii=False)}],
                         "meta": {"experiment_id": "hooke", "split": "train", "seed": i}})
            if i == 4:
                bench_ids.add(recs[-1]["id"])
    train =[r for r in recs if r["id"] not in bench_ids]
    bench = [r for r in recs if r["id"] in bench_ids]
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "bench.jsonl", bench)
    return {"train": str(out / "train.jsonl"), "bench": str(out / "bench.jsonl"),
            "n_train": len(train), "n_bench": len(bench)}


# ---------------------------------------------------------------- 命令行

def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="训练数据工具：合成冒烟数据 / 校验 + 泄漏检查 + 统计")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="生成冒烟用合成数据")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--no-images", action="store_true")
    c = sub.add_parser("check", help="校验训练 JSONL，统计，并和 bench 做泄漏检查")
    c.add_argument("--train", nargs="+", required=True)
    c.add_argument("--data-root", default=None)
    c.add_argument("--bench", nargs="*", default=[])
    c.add_argument("--bench-root", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "synth":
        print(json.dumps(make_synthetic(a.out, a.n, with_images=not a.no_images), ensure_ascii=False))
        return 0
    recs = load_records(a.train)
    rep = {"summary": dataset_summary(recs)}
    if a.bench:
        bench = load_bench_index(a.bench, a.bench_root or a.data_root)
        rep["leakage"] = leakage_report(recs, bench, a.data_root)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0 if rep.get("leakage", {}).get("clean", True) else 2


if __name__ == "__main__":
    raise SystemExit(_main())
