"""生成器 --exclude：输入与 bench 题相同的训练 / 验证样本删掉并计数。"""
from lab_detective.training import generate as gen
from lab_detective.training.format import content_hash, read_jsonl, write_jsonl


def test_exclude_drops_examples_matching_bench(tmp_path):
    counts = {"knowledge": 6}
    gen.generate(counts, tmp_path / "a", workers=1, chunk=3, quiet=True)
    train = list(read_jsonl(tmp_path / "a" / "train" / "knowledge.jsonl"))
    victim = train[0]
    bench = tmp_path / "bench_x.jsonl"
    write_jsonl(bench, [victim])
    side = tmp_path / "bench_x.gold.jsonl"
    side.write_text('{"not": "an example"}\n', encoding="utf-8")  # 旁注文件自动跳过
    hs = gen.exclude_hashes([bench, side])
    assert hs == {content_hash(victim)}
    rep = gen.generate(counts, tmp_path / "b", workers=1, chunk=3, quiet=True, exclude=hs)
    left = list(read_jsonl(tmp_path / "b" / "train" / "knowledge.jsonl"))
    assert len(left) == len(train) - 1
    assert content_hash(victim) not in {content_hash(e) for e in left}
    assert rep["tasks"]["knowledge"]["bench_overlap_removed"] == 1
