"""server/check_package.py：在小的合成数据包上核对布局、JSON、图片、泄漏、MANIFEST 和蒸馏上限。"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
import check_package as CP  # noqa: E402
import common as C  # noqa: E402

pytest.importorskip("PIL")


def make_pkg(tmp_path: Path, manifest: bool = True) -> Path:
    """按打包布局摆好：train.jsonl、val.jsonl、bench/bench_synth.jsonl、images/、server/、MANIFEST.json。"""
    src = tmp_path / "synth"
    C.make_synthetic(src, n=12, n_bench=3)
    pkg = tmp_path / "pkg"
    (pkg / "bench").mkdir(parents=True)
    shutil.copytree(src / "images", pkg / "images")
    train = C.read_jsonl(src / "train.jsonl")
    C.write_jsonl(pkg / "train.jsonl", train[2:])
    val = [dict(r, id=r["id"] + "-val", messages=[*r["messages"][:-2],
                                                    {"role": "user", "content": f"验证 {r['id']}"},
                                                    r["messages"][-1]]) for r in train[:2]]
    C.write_jsonl(pkg / "val.jsonl", val)
    shutil.copy2(src / "bench.jsonl", pkg / "bench" / "bench_synth.jsonl")
    shutil.copytree(ROOT / "server", pkg / "server", ignore=shutil.ignore_patterns("__pycache__"))
    if manifest:
        files = sorted(p for p in pkg.rglob("*") if p.is_file())
        (pkg / "MANIFEST.json").write_text(json.dumps({"files": [
            {"file": str(p.relative_to(pkg)), "sha256": C.file_sha256(p)} for p in files]}), encoding="utf-8")
    return pkg


def test_clean_package_passes(tmp_path, capsys):
    pkg = make_pkg(tmp_path)
    rep = CP.check_package(pkg)
    assert rep.errors == [], rep.errors
    c = rep.info["counts"]
    assert c["train.jsonl"]["n"] == 14 and c["val.jsonl"]["n"] == 2 and c["train.jsonl"]["task"]["recognize"] == 4
    assert c["bench/bench_synth.jsonl"]["n"] == 4  # 3 条诊断 + 1 条带图识别
    assert rep.info["leakage"]["train.jsonl"] == {"n_id_hits": 0, "n_hash_hits": 0}
    assert rep.info["diagnose_with_causes"]["with_meta_causes"] == rep.info["diagnose_with_causes"]["n"]
    assert rep.info["manifest"]["n_data_mismatch"] == 0
    assert CP.main([str(pkg), "--json", str(tmp_path / "rep.json")]) == 0
    out = capsys.readouterr().out
    assert "结论：通过" in out and "diagnose" in out
    assert json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))["errors"] == []


def test_missing_image_and_bad_json_line(tmp_path):
    pkg = make_pkg(tmp_path, manifest=False)
    (pkg / "images" / "synth_0.jpg").unlink()
    with open(pkg / "train.jsonl", "a", encoding="utf-8") as f:
        f.write('{"id": "broken", "task": \n')
    rep = CP.check_package(pkg)
    assert any("不存在" in e and "synth_0.jpg" in e for e in rep.errors), rep.errors
    assert any("不是合法 JSON" in e for e in rep.errors)
    assert CP.main([str(pkg)]) == 1


def test_leaked_id_and_photo(tmp_path):
    pkg = make_pkg(tmp_path, manifest=False)
    bench = C.read_jsonl(pkg / "bench" / "bench_synth.jsonl")
    train = C.read_jsonl(pkg / "train.jsonl")
    # 1) bench 的一条原样混进训练集
    C.write_jsonl(pkg / "train.jsonl", train + [bench[0]])
    rep = CP.check_package(pkg)
    assert any("泄漏" in e and bench[0]["id"] in e for e in rep.errors), rep.errors
    # 2) 照片级泄漏：bench 记的原图 sha1 等于训练样本用的图
    C.write_jsonl(pkg / "train.jsonl", train)
    img_rec = next(r for r in train if C.image_refs(r["messages"]))
    sha1 = hashlib.sha1((pkg / C.image_refs(img_rec["messages"])[0]).read_bytes()).hexdigest()
    bench[0].setdefault("meta", {})["src_sha1"] = sha1
    C.write_jsonl(pkg / "bench" / "bench_synth.jsonl", bench)
    rep = CP.check_package(pkg)
    assert any("bench 的照片" in e and img_rec["id"] in e for e in rep.errors), rep.errors


def test_layout_unmarked_manifest_and_distill_cap(tmp_path):
    pkg = make_pkg(tmp_path)
    # 识别 bench 的图是没画框的原图（字节等于 meta.image_sha1）
    bench = C.read_jsonl(pkg / "bench" / "bench_synth.jsonl")
    rec = next(r for r in bench if C.image_refs(r["messages"]))
    img = pkg / C.image_refs(rec["messages"])[0]
    rec["meta"].update(image_sha1=hashlib.sha1(img.read_bytes()).hexdigest(), legend=[])
    C.write_jsonl(pkg / "bench" / "bench_synth.jsonl", bench)
    rep = CP.check_package(pkg)
    assert any("没画编号框" in e for e in rep.errors)
    # 数据文件和 MANIFEST 不一致 → 错误；server/ 脚本不一致只警告
    assert any("sha256 和 MANIFEST 不一致" in e and "bench_synth" in e for e in rep.errors)
    (pkg / "server" / "README.md").write_text("改过", encoding="utf-8")
    rep = CP.check_package(pkg)
    assert any("server/ 下" in w for w in rep.warnings)
    # 蒸馏上限
    train = C.read_jsonl(pkg / "train.jsonl")
    C.write_jsonl(pkg / "train.jsonl", [dict(r, source="distill-deepseek") for r in train])
    assert any("超过上限" in e for e in CP.check_package(pkg, distill_cap=3, manifest=False).errors)
    assert not any("超过上限" in e for e in CP.check_package(pkg, distill_cap=0, manifest=False).errors)
    # 缺 bench 目录
    shutil.rmtree(pkg / "bench")
    assert any("bench_*.jsonl" in e for e in CP.check_package(pkg, manifest=False).errors)


def test_image_path_escape_and_bad_header(tmp_path):
    pkg = make_pkg(tmp_path, manifest=False)
    train = C.read_jsonl(pkg / "train.jsonl")
    img_rec = next(r for r in train if C.image_refs(r["messages"]))
    (pkg / C.image_refs(img_rec["messages"])[0]).write_bytes(b"not an image at all")
    bad = json.loads(json.dumps(img_rec))
    bad["id"] = "escape-1"
    bad["messages"][1]["content"][0]["image"] = "../outside.jpg"
    C.write_jsonl(pkg / "train.jsonl", train + [bad])
    errs = CP.check_package(pkg).errors
    assert any("文件头不是" in e for e in errs) and any("跳出了数据包根目录" in e for e in errs)
