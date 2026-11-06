"""开训前核对数据包（Mac 上打包后、服务器上解包后各跑一次）。只用标准库 + common.py，
bootstrap 还没装好环境时用系统 python3 就能跑：

    python3 server/check_package.py <数据包根目录> [--json 报告.json]

核对内容：
1. 布局：<包>/train.jsonl（必须）、val.jsonl（可选）、bench/bench_*.jsonl（至少一个）、images/、server/。
2. 每个 JSONL 逐行解析并按 ARCHITECTURE §2.5 校验（错误全部收集，不在第一条就停）；文件内 id 不重复。
3. 图片：路径相对包根、不许跳出包根、文件存在且非空、文件头是 JPEG / PNG / WebP。
4. 各文件按 task、source 计数；字符长度分位数（本机没有 Qwen 分词器，token 长度看服务器上的 dry-run）。
5. 诊断样本带 meta.causes 的比例（打乱标签对照要用）。
6. 泄漏：训练 / 验证 对 bench 按 id 和用户回合哈希（图片按字节）；按照片（bench 的 meta.photo_id、image_sha1、
   src_sha1 对训练侧的图片字节 sha1 和 photo_id）；验证对训练只警告。
7. 识别 bench 的图必须是带编号框的版本（字节等于 meta.image_sha1 说明拿错了原图）。
8. 有 MANIFEST.json 时逐个核对 sha256：数据文件不一致是错误，server/ 下的脚本不一致只警告（脚本可能更新过）。
9. 蒸馏样本（source 以 distill- 开头）不超过上限（默认 5,248）。

有错误时退出码 1，只有警告时 0。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

DISTILL_CAP = 5248
SERVER_FILES = ("train_sft.py", "merge_and_eval.py", "common.py", "convert_adapter.py", "bootstrap.sh")
MAX_LISTED = 20  # 每类问题最多列出的条数
IMAGE_MAGIC = ((b"\xff\xd8\xff", "jpeg"), (b"\x89PNG\r\n\x1a\n", "png"), (b"RIFF", "webp"))


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: dict = {}

    def error(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)


def _listed(xs, n=5) -> str:
    xs = list(xs)
    return "、".join(map(str, xs[:n])) + (f" 等 {len(xs)} 条" if len(xs) > n else "")


def read_records(path: Path, rep: Report) -> list[dict]:
    """逐行解析 + 校验，坏行记错误后跳过。"""
    recs, seen, n_bad = [], set(), 0
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                n_bad += 1
                if n_bad <= MAX_LISTED:
                    rep.error(f"{path.name}:{ln} 不是合法 JSON：{e}")
                continue
            errs = C.validate_record(rec)
            if not errs and rec["id"] in seen:
                errs = [f"id 重复：{rec['id']}"]
            if errs:
                n_bad += 1
                if n_bad <= MAX_LISTED:
                    rep.error(f"{path.name}:{ln}：" + "；".join(errs))
                continue
            seen.add(rec["id"])
            recs.append(rec)
    if n_bad > MAX_LISTED:
        rep.error(f"{path.name}：共 {n_bad} 行有问题（只列了前 {MAX_LISTED} 行）")
    return recs


def check_images(name: str, recs: list[dict], root: Path, rep: Report, cache: dict) -> int:
    """返回引用的图片数。cache: 相对路径 → 问题描述（None = 正常）。"""
    bad: dict[str, list[str]] = {}
    n = 0
    for rec in recs:
        for ref in C.image_refs(rec["messages"]):
            n += 1
            if ref not in cache:
                cache[ref] = _image_problem(ref, root)
            if cache[ref]:
                bad.setdefault(cache[ref], []).append(f"{rec['id']}→{ref}")
    for problem, hits in bad.items():
        rep.error(f"{name}：{len(hits)} 处图片{problem}（{_listed(hits, 3)}）")
    return n


def _image_problem(ref: str, root: Path) -> str | None:
    try:
        p = C.resolve_image(ref, root)
    except ValueError:
        return "路径跳出了数据包根目录"
    if not p.is_file():
        return "不存在"
    if p.stat().st_size == 0:
        return "是空文件"
    with open(p, "rb") as f:
        head = f.read(12)
    if not any(head.startswith(m) for m, _ in IMAGE_MAGIC):
        return "文件头不是 JPEG / PNG / WebP"
    return None


def counts(recs: list[dict]) -> dict:
    by_task, by_source = {}, {}
    for r in recs:
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    return {"n": len(recs), "task": dict(sorted(by_task.items())), "source": dict(sorted(by_source.items()))}


def char_lengths(recs: list[dict]) -> dict:
    """按 task：整条对话字符数和答案字符数的分位数（中文大致 1 字 ≈ 1 token，只作粗估）。"""
    tot, ans = {}, {}
    for r in recs:
        n_all = sum(len(C.text_of(m.get("content"))) for m in r["messages"])
        tot.setdefault(r["task"], []).append(n_all)
        ans.setdefault(r["task"], []).append(len(C.text_of(r["messages"][-1]["content"])))
    pick = ("p50", "p90", "p99", "max")
    return {t: {"total": {k: C.length_stats(tot[t])[k] for k in pick},
                "answer": {k: C.length_stats(ans[t])[k] for k in pick}} for t in sorted(tot)}


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def photo_leaks(train_side: list[dict], bench: list[dict], root: Path) -> list[str]:
    """按照片查泄漏：bench 的 photo_id / image_sha1 / src_sha1 出现在训练侧（photo_id 或图片字节 sha1）。"""
    b_photo, b_sha1 = {}, {}
    for r in bench:
        meta = C.meta_of(r)
        if meta.get("photo_id"):
            b_photo[str(meta["photo_id"])] = r["id"]
        for k in ("image_sha1", "src_sha1"):
            if meta.get(k):
                b_sha1[str(meta[k])] = r["id"]
    if not b_photo and not b_sha1:
        return []
    hits, sha_cache = [], {}
    for r in train_side:
        meta = C.meta_of(r)
        if meta.get("photo_id") and str(meta["photo_id"]) in b_photo:
            hits.append(f"{r['id']}（photo_id {meta['photo_id']}）")
            continue
        for ref in C.image_refs(r["messages"]):
            try:
                p = C.resolve_image(ref, root)
            except ValueError:
                continue
            if not p.is_file():
                continue
            if ref not in sha_cache:
                sha_cache[ref] = _sha1(p)
            if sha_cache[ref] in b_sha1:
                hits.append(f"{r['id']}（图片 {ref} 与 bench {b_sha1[sha_cache[ref]]} 同一张原图）")
                break
    return hits


def check_manifest(pkg: Path, rep: Report) -> dict | None:
    mf = pkg / "MANIFEST.json"
    if not mf.is_file():
        rep.warn("没有 MANIFEST.json：跳过文件哈希核对（打包脚本会写这个文件）")
        return None
    try:
        files = json.loads(mf.read_text(encoding="utf-8")).get("files", [])
    except (json.JSONDecodeError, AttributeError) as e:
        rep.error(f"MANIFEST.json 读不了：{e}")
        return None
    bad_data, bad_server, missing = [], [], []
    for row in files:
        rel = row.get("file")
        if not rel:
            continue
        p = pkg / rel
        is_server = Path(rel).parts[0] == "server"
        if not p.is_file():
            (bad_server if is_server else missing).append(rel)
            continue
        if C.file_sha256(p) != row.get("sha256"):
            (bad_server if is_server else bad_data).append(rel)
    if missing:
        rep.error(f"MANIFEST 里的 {len(missing)} 个数据文件不见了：{_listed(missing)}")
    if bad_data:
        rep.error(f"{len(bad_data)} 个数据文件的 sha256 和 MANIFEST 不一致（上传损坏或被改过）：{_listed(bad_data)}")
    if bad_server:
        rep.warn(f"server/ 下 {len(bad_server)} 个脚本和 MANIFEST 不一致（脚本更新过就正常）：{_listed(bad_server)}")
    return {"n_files": len(files), "n_data_mismatch": len(bad_data), "n_missing": len(missing),
            "n_server_mismatch": len(bad_server)}


def check_package(pkg: str | Path, distill_cap: int = DISTILL_CAP, manifest: bool = True) -> Report:
    pkg = Path(pkg).resolve()
    rep = Report()
    rep.info["package"] = str(pkg)
    if not pkg.is_dir():
        rep.error(f"数据包目录不存在：{pkg}")
        return rep
    # 1. 布局
    train_p, val_p = pkg / "train.jsonl", pkg / "val.jsonl"
    bench_ps = sorted((pkg / "bench").glob("bench_*.jsonl"))
    if not train_p.is_file():
        rep.error("缺 train.jsonl（应在数据包根目录）")
    if not val_p.is_file():
        rep.warn("没有 val.jsonl：训练时不记验证损失")
    if not bench_ps:
        rep.error("bench/ 下没有 bench_*.jsonl：正式训练必须做泄漏检查")
    if not (pkg / "server").is_dir():
        rep.warn("数据包里没有 server/（脚本单独上传时正常）")
    else:
        lost = [f for f in SERVER_FILES if not (pkg / "server" / f).is_file()]
        if lost:
            rep.warn(f"server/ 缺 {lost}")
    # 2. 解析与校验
    files = {"train.jsonl": train_p, "val.jsonl": val_p, **{f"bench/{p.name}": p for p in bench_ps}}
    recs = {name: read_records(p, rep) for name, p in files.items() if p.is_file()}
    train, val = recs.get("train.jsonl", []), recs.get("val.jsonl", [])
    bench = [r for name, rs in recs.items() if name.startswith("bench/") for r in rs]
    # 3. 图片
    cache: dict = {}
    n_refs = {name: check_images(name, rs, pkg, rep, cache) for name, rs in recs.items()}
    if sum(n_refs.values()) and not (pkg / "images").is_dir():
        rep.error("样本引用了图片，但数据包里没有 images/")
    # 路径跳出包根的样本已记错误；后面的哈希 / 泄漏核对跳过它们（解析路径会抛错）
    escaped = {ref for ref, prob in cache.items() if prob == "路径跳出了数据包根目录"}
    if escaped:
        recs = {name: [r for r in rs if not set(C.image_refs(r["messages"])) & escaped] for name, rs in recs.items()}
        train, val = recs.get("train.jsonl", []), recs.get("val.jsonl", [])
        bench = [r for name, rs in recs.items() if name.startswith("bench/") for r in rs]
    # 4. 计数与长度
    rep.info["counts"] = {name: counts(rs) for name, rs in recs.items()}
    rep.info["n_image_refs"] = n_refs
    rep.info["n_unique_images"] = len(cache)
    rep.info["chars_train"] = char_lengths(train)
    # 5. meta.causes
    diag = [r for r in train if r["task"] == "diagnose"]
    with_causes = sum(1 for r in diag if C.cause_domain(r) is not None and len(C.cause_domain(r)[0]) >= 2)
    rep.info["diagnose_with_causes"] = {"n": len(diag), "with_meta_causes": with_causes}
    if diag and with_causes < len(diag):
        rep.warn(f"{len(diag) - with_causes}/{len(diag)} 条诊断训练样本没有 meta.causes（≥ 2 个候选）："
                 "打乱标签对照会退回按实验汇总的原因，偏弱")
    # 6. 泄漏
    ids_seen: dict[str, str] = {}
    for name, rs in recs.items():
        if not name.startswith("bench/"):
            continue
        for r in rs:
            if r["id"] in ids_seen:
                rep.error(f"bench 之间 id 重复：{r['id']}（{ids_seen[r['id']]} 和 {name}）")
            ids_seen[r["id"]] = name
    leak = {}
    if bench:
        idx = {"ids": {r["id"] for r in bench}, "hashes": {C.user_hash(r, pkg) for r in bench}}
        for name, rs in (("train.jsonl", train), ("val.jsonl", val)):
            lr = C.leakage_report(rs, idx, pkg)
            leak[name] = {"n_id_hits": len(lr["id_hits"]), "n_hash_hits": len(lr["hash_hits"])}
            if not lr["clean"]:
                rep.error(f"{name} 和 bench 泄漏：id 撞 {len(lr['id_hits'])} 条（{_listed(lr['id_hits'])}），"
                          f"内容撞 {len(lr['hash_hits'])} 条（{_listed(lr['hash_hits'])}）")
        ph = photo_leaks(train + val, bench, pkg)
        leak["photo_hits"] = len(ph)
        if ph:
            rep.error(f"训练 / 验证用到了 bench 的照片 {len(ph)} 条：{_listed(ph, 3)}")
    if train and val:
        tv = C.leakage_report(val, {"ids": {r["id"] for r in train},
                                    "hashes": {C.user_hash(r, pkg) for r in train}}, pkg)
        leak["val_vs_train"] = {"n_id_hits": len(tv["id_hits"]), "n_hash_hits": len(tv["hash_hits"])}
        if tv["id_hits"]:
            rep.error(f"val 和 train 的 id 重复 {len(tv['id_hits'])} 条：{_listed(tv['id_hits'])}")
        if tv["hash_hits"]:
            rep.warn(f"val 和 train 的用户回合相同 {len(tv['hash_hits'])} 条（验证损失会偏乐观）")
    rep.info["leakage"] = leak
    # 7. 识别 bench 的图必须带编号框
    unmarked = C.unmarked_image_hits(bench, pkg)
    if unmarked:
        rep.error(f"{len(unmarked)} 条 bench 的图片是没画编号框的原图（字节等于 meta.image_sha1）：{_listed(unmarked)}。"
                  "打包时 bench 图片要取自 bench 目录里带框的 images/")
    # 8. MANIFEST
    if manifest:
        rep.info["manifest"] = check_manifest(pkg, rep)
    # 9. 蒸馏上限
    n_distill = sum(1 for r in train + val if str(r.get("source", "")).startswith("distill-"))
    rep.info["n_distill"] = n_distill
    if distill_cap and n_distill > distill_cap:
        rep.error(f"蒸馏样本 {n_distill} 条，超过上限 {distill_cap}")
    return rep


def format_report(rep: Report) -> str:
    lines = [f"数据包：{rep.info.get('package')}"]
    for name, c in rep.info.get("counts", {}).items():
        lines.append(f"  {name:<36}{c['n']:>6} 条  task {c['task']}  source {c['source']}")
    if "n_unique_images" in rep.info:
        lines.append(f"  图片：引用 {sum(rep.info['n_image_refs'].values())} 处，{rep.info['n_unique_images']} 张不同的图")
    if rep.info.get("chars_train"):
        lines.append("  训练样本字符数（整条 / 答案，p50/p90/p99/max；token 长度以服务器 dry-run 为准）：")
        for t, s in rep.info["chars_train"].items():
            q = (lambda d: "/".join(str(d[k]) for k in ("p50", "p90", "p99", "max")))
            lines.append(f"    {t:<10}{q(s['total']):<26}{q(s['answer'])}")
    d = rep.info.get("diagnose_with_causes")
    if d and d["n"]:
        lines.append(f"  诊断样本带 meta.causes：{d['with_meta_causes']}/{d['n']}")
    if "leakage" in rep.info:
        lines.append(f"  泄漏核对：{json.dumps(rep.info['leakage'], ensure_ascii=False)}")
    if rep.info.get("manifest"):
        lines.append(f"  MANIFEST：{json.dumps(rep.info['manifest'], ensure_ascii=False)}")
    if "n_distill" in rep.info:
        lines.append(f"  蒸馏样本：{rep.info['n_distill']} 条")
    for w in rep.warnings:
        lines.append(f"警告：{w}")
    for e in rep.errors:
        lines.append(f"错误：{e}")
    lines.append(f"结论：{'不通过，' + str(len(rep.errors)) + ' 个错误' if rep.errors else '通过'}"
                 f"（{len(rep.warnings)} 个警告）")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="开训前核对数据包（布局、JSON、图片、计数、泄漏、MANIFEST 哈希）")
    ap.add_argument("package", help="数据包根目录（里面有 train.jsonl、bench/、images/）")
    ap.add_argument("--json", default=None, help="把完整报告写成 JSON")
    ap.add_argument("--distill-cap", type=int, default=DISTILL_CAP, help="蒸馏样本上限（0 = 不查）")
    ap.add_argument("--no-manifest", action="store_true", help="不核对 MANIFEST.json 的哈希")
    a = ap.parse_args(argv)
    rep = check_package(a.package, a.distill_cap, manifest=not a.no_manifest)
    print(format_report(rep), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps({"errors": rep.errors, "warnings": rep.warnings, **rep.info},
                                           ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
