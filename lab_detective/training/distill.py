"""离线蒸馏流水线（只在构建训练数据时运行，产品里从不调用任何外部模型）。

唯一借鉴格物论文的做法：
- 教师"盲做"：教师看到的提示和学生训练时的提示完全相同，看不到物理引擎的答案；
- 只保留和可核验标准一致的输出：我们的标准是物理引擎的判定（等价类内的最可能原因、
  信息增益前两名的追加测量）以及声明校验器；
- 蒸馏样本总数不超过 5,248 条。

教师有两种来源：
1. DeepSeek（API，自动）：key 放在 ~/.config/ppr/keys.env，写成一行 DEEPSEEK_API_KEY=...，不进仓库、不打印。
2. Hy4（没有 API）：export_prompt_pack() 导出提示词包，用户在别的工具里跑完、把结果文件放回来，
   import_responses() 用同一套过滤规则读入。
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .format import Example, write_jsonl

MAX_DISTILLED = 5248  # 不超过格物论文的蒸馏规模
KEY_FILE = Path.home() / ".config" / "ppr" / "keys.env"


@dataclass
class DistillJob:
    """一条待蒸馏的题：prompt 就是学生训练时看到的 messages（不含答案）。"""

    id: str
    task: str
    messages: list[dict]
    check: Callable[[str], tuple[bool, dict]]  # 教师输出 → (是否保留, 诊断信息)
    meta: dict = field(default_factory=dict)


@dataclass
class DistillResult:
    job_id: str
    teacher: str
    text: str
    kept: bool
    info: dict
    usage: dict = field(default_factory=dict)


def load_key(name: str = "DEEPSEEK_API_KEY") -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    if KEY_FILE.exists():
        for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip()
    return None


class DeepSeekTeacher:
    """最小 HTTP 客户端（OpenAI 兼容）。允许教师思考，但只取最终 content，思考内容不进训练数据。"""

    def __init__(self, model: str = "deepseek-flash", base_url: str = "https://api.deepseek.com",
                 thinking: bool = True, timeout: float = 180.0, log_path: str | Path | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.thinking = thinking
        self.timeout = timeout
        self.key = load_key()
        self.log_path = Path(log_path) if log_path else None
        if not self.key:
            raise RuntimeError(f"没有找到 DEEPSEEK_API_KEY（环境变量或 {KEY_FILE}）")

    @property
    def name(self) -> str:
        return f"deepseek:{self.model}:{'think' if self.thinking else 'nothink'}"

    def __call__(self, messages: list[dict], max_tokens: int | None = None) -> tuple[str, dict]:
        # 开思考时思考内容也占 max_tokens：2048 会被思考用光、最终答案为空，所以思考档给 12000
        if max_tokens is None:
            max_tokens = 12000 if self.thinking else 2048
        body = {"model": self.model, "messages": _text_only(messages), "max_tokens": max_tokens,
                "thinking": {"type": "enabled" if self.thinking else "disabled"}}
        if not self.thinking:
            body["temperature"] = 0.0
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        last = None
        for attempt in range(6):  # 限流（429）时退避 1、2、4、8、16 秒
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8"))
                msg = data["choices"][0]["message"]
                usage = data.get("usage", {})
                usage["model_returned"] = data.get("model")
                self._log(body, data)
                return msg.get("content") or "", usage
            except Exception as e:  # 网络或限流：退避重试
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"教师请求失败：{last}")

    def _log(self, body: dict, data: dict) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "request": body, "response": data}, ensure_ascii=False) + "\n")


def _text_only(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            c = "\n".join(p.get("text", "") for p in c if p.get("type") == "text")
        out.append({"role": m["role"], "content": c})
    return out


def run(jobs: Iterable[DistillJob], teacher: Callable[[list[dict]], tuple[str, dict]], teacher_name: str,
        out_dir: str | Path, max_kept: int = MAX_DISTILLED, workers: int = 4,
        already_kept: int = 0) -> dict:
    """并发调用教师、过滤、写出。达到上限即停。可重入：已有结果的 job 跳过。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.jsonl"
    done = set()
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["job_id"])
    kept_total = already_kept + _count_kept(raw_path)
    jobs = [j for j in jobs if j.id not in done]

    def work(job: DistillJob) -> DistillResult | None:
        try:
            text, usage = teacher(job.messages)
        except Exception as e:  # 单条失败（限流、网络）：不写结果，续跑时会重试
            stats["errors"] = stats.get("errors", 0) + 1
            stats["last_error"] = str(e)[:200]
            return None
        ok, info = job.check(text)
        return DistillResult(job.id, teacher_name, text, ok, info, usage)

    stats = {"submitted": 0, "kept": 0, "rejected": 0}
    with ThreadPoolExecutor(max_workers=workers) as ex, raw_path.open("a", encoding="utf-8") as f:
        for i in range(0, len(jobs), workers * 4):
            if kept_total >= max_kept:
                break
            batch = jobs[i:i + workers * 4]
            for res, job in zip(ex.map(work, batch), batch):
                if res is None:
                    continue
                stats["submitted"] += 1
                stats["kept" if res.kept else "rejected"] += 1
                kept_total += int(res.kept)
                f.write(json.dumps({"job_id": res.job_id, "task": job.task, "teacher": res.teacher,
                                    "text": res.text, "kept": res.kept, "info": res.info,
                                    "usage": res.usage, "meta": job.meta}, ensure_ascii=False) + "\n")
                f.flush()
    stats["kept_total"] = kept_total
    return stats


def _count_kept(raw_path: Path) -> int:
    if not raw_path.exists():
        return 0
    return sum(1 for line in raw_path.read_text(encoding="utf-8").splitlines()
               if line.strip() and json.loads(line)["kept"])


def to_examples(raw_path: str | Path, jobs_by_id: dict[str, DistillJob], source: str,
                max_kept: int = MAX_DISTILLED) -> list[Example]:
    """把保留下来的教师输出变成训练样本：prompt 原样（与教师所见一致），答案 = 教师最终输出。"""
    out: list[Example] = []
    for line in Path(raw_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r["kept"] or r["job_id"] not in jobs_by_id:
            continue
        job = jobs_by_id[r["job_id"]]
        msgs = list(job.messages) + [{"role": "assistant", "content": r["text"]}]
        out.append(Example(f"{source}-{job.id}", job.task, source, msgs,
                           dict(job.meta, teacher=r["teacher"], split="train")))
        if len(out) >= max_kept:
            break
    return out


# ── Hy4：导出提示词包 / 导入结果 ─────────────────────────────────────────

HY4_README = """# Hy4 蒸馏提示词包

1. prompts.jsonl 每行一个题：{"id": ..., "messages": [...]}。请按 messages 原样发给 Hy4
   （system 作为系统提示，user 作为用户消息；不要添加任何其他指令）。
2. 关闭"联网搜索"和任何工具；温度用默认值。
3. 把每个题的最终回答写成 responses.jsonl，每行 {"id": 同上, "text": "Hy4 的完整最终回答"}。
   不要包含思考过程。
4. 把 responses.jsonl 放回本文件夹，告诉开发者导入。
"""


def export_prompt_pack(jobs: Iterable[DistillJob], folder: str | Path) -> int:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    with (folder / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps({"id": j.id, "task": j.task, "messages": _text_only(j.messages)},
                               ensure_ascii=False) + "\n")
            n += 1
    (folder / "README.md").write_text(HY4_README, encoding="utf-8")
    return n


def import_responses(folder: str | Path, jobs_by_id: dict[str, DistillJob], out_dir: str | Path) -> dict:
    """读回 Hy4 的 responses.jsonl，用与 DeepSeek 相同的检查过滤，写成 raw_results.jsonl。"""
    folder, out_dir = Path(folder), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"read": 0, "kept": 0, "rejected": 0, "unknown_id": 0}
    with (out_dir / "raw_results.jsonl").open("a", encoding="utf-8") as f:
        for line in (folder / "responses.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            stats["read"] += 1
            job = jobs_by_id.get(r["id"])
            if job is None:
                stats["unknown_id"] += 1
                continue
            ok, info = job.check(r["text"])
            stats["kept" if ok else "rejected"] += 1
            f.write(json.dumps({"job_id": job.id, "task": job.task, "teacher": "hy4", "text": r["text"],
                                "kept": ok, "info": info, "usage": {}, "meta": job.meta},
                               ensure_ascii=False) + "\n")
    return stats


def write_examples(examples: list[Example], path: str | Path) -> int:
    return write_jsonl(path, examples)
