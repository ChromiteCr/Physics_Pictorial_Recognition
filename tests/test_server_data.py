"""服务器训练数据：校验、消息展开与只对助手算损失的掩码、图片分配、确定性顺序、泄漏检查、打乱标签对照。

不用真分词器：假模板模仿 Qwen 的 ChatML + 空思考块，假分词器按字符编码、每张图展开成 3 个 token。
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import common as C  # noqa: E402

VISION = "<|vision_start|><|image_pad|><|vision_end|>"
SP0 = 2_000_000  # 特殊记号的 id，高于所有 Unicode 码位
SPECIAL = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]


def fake_render(messages, add_generation_prompt, think_switch=True):
    s = ""
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            c = "".join(VISION if p["type"] == "image" else p["text"] for p in c)
        if m["role"] == "assistant":
            c = C.EMPTY_THINK + c
        s += f"<|im_start|>{m['role']}\n{c}<|im_end|>\n"
    if add_generation_prompt:
        s += "<|im_start|>assistant\n" + (C.EMPTY_THINK if think_switch else "<think>\n")
    return s


def fake_tokenize(text, images):
    """特殊记号算 1 个 token（id ≥ SP0），图片占位展开成 3 个 token；其余按字符。"""
    ids, mm_type, i, n_img = [], [], 0, 0
    while i < len(text):
        for k, sp in enumerate(SPECIAL):
            if text.startswith(sp, i):
                if sp == "<|image_pad|>":
                    ids += [SP0 + 4] * 3
                    mm_type += [1] * 3
                    n_img += 1
                else:
                    ids.append(SP0 + k)
                    mm_type.append(0)
                i += len(sp)
                break
        else:
            ids.append(ord(text[i]))
            mm_type.append(0)
            i += 1
    assert n_img == len(images)
    out = {"input_ids": ids, "seq": {"attention_mask": [1] * len(ids), "mm_token_type_ids": mm_type}, "mm": {}}
    if images:
        out["mm"] = {"pixel_values": list(images)}
    return out


def fake_detok(ids):
    return "".join(SPECIAL[i - SP0] if i >= SP0 else chr(i) for i in ids)


CODEC = C.Codec(render=fake_render, tokenize=fake_tokenize)


def rec(rid="d-1", task="diagnose", user="数据：1,2,3", target='{"ranked":[{"cause_id":"hooke.a"}]}',
        images=(), system="系统提示", meta=None):
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": user}]
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": content}, {"role": "assistant", "content": target}]
    return {"id": rid, "task": task, "source": "engine", "messages": msgs,
            "meta": meta if meta is not None else {"experiment_id": "hooke", "split": "train", "seed": 1}}


# ---------------------------------------------------------------- 掩码

def test_assistant_only_mask():
    r = rec()
    enc = C.encode_example(r, CODEC, load_image=lambda p: p)
    ids, labels = enc["input_ids"], enc["labels"]
    assert len(ids) == len(labels) == enc["n_prompt"] + enc["n_target"]
    assert all(x == C.IGNORE_INDEX for x in labels[: enc["n_prompt"]])
    target_ids = [x for x in labels if x != C.IGNORE_INDEX]
    assert fake_detok(target_ids) == r["messages"][-1]["content"] + "<|im_end|>"
    # prompt 以生成提示 + 空思考块结尾：和推理时逐字节一致
    assert fake_detok(ids[: enc["n_prompt"]]).endswith("<|im_start|>assistant\n" + C.EMPTY_THINK)
    assert enc["seq"]["attention_mask"] == [1] * len(ids)
    assert C.check_full_render(r, CODEC)


def test_multi_turn_only_last_assistant_is_target():
    r = rec(target="最后的回答")
    r["messages"][1:1] = [{"role": "user", "content": "先问一句"}, {"role": "assistant", "content": "先答一句"}]
    enc = C.encode_example(r, CODEC, load_image=lambda p: p)
    target = fake_detok([x for x in enc["labels"] if x != C.IGNORE_INDEX])
    assert target == "最后的回答<|im_end|>"
    assert "先答一句" in fake_detok(enc["input_ids"][: enc["n_prompt"]])


def test_images_go_to_prompt_in_order(tmp_path):
    for n in ("a.jpg", "b.jpg"):
        (tmp_path / n).write_bytes(n.encode())
    r = rec(images=("a.jpg", "b.jpg"))
    enc = C.encode_example(r, CODEC, root=tmp_path, load_image=lambda p: p.name)
    assert enc["mm"]["pixel_values"] == ["a.jpg", "b.jpg"]
    assert enc["n_images"] == 2
    mm = enc["seq"]["mm_token_type_ids"]
    assert sum(mm) == 6 and len(mm) == len(enc["input_ids"])
    assert all(t == 0 for t in mm[enc["n_prompt"]:])  # 目标段全是文字


def test_thinking_template_rejected():
    codec = C.Codec(render=lambda m, g: fake_render(m, g, think_switch=False), tokenize=fake_tokenize)
    with pytest.raises(ValueError, match="空思考块"):
        C.encode_example(rec(), codec, load_image=lambda p: p)


def test_placeholder_count_mismatch():
    codec = C.Codec(render=lambda m, g: fake_render(m, g).replace(VISION, ""), tokenize=fake_tokenize)
    with pytest.raises(ValueError, match="图片占位"):
        C.encode_example(rec(images=("a.jpg",)), codec, load_image=lambda p: p)


def test_template_messages_shape():
    msgs = C.to_template_messages(rec(images=("x.jpg",))["messages"])
    assert msgs[0] == {"role": "system", "content": "系统提示"}
    assert msgs[1]["content"][0] == {"type": "image"}  # 路径不进模板
    assert isinstance(msgs[2]["content"], str)


def test_image_path_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError):
        C.resolve_image("../secret.jpg", tmp_path)


# ---------------------------------------------------------------- 校验

def test_validate_record():
    assert C.validate_record(rec()) == []
    bad = rec(task="chat")
    assert any("task" in e for e in C.validate_record(bad))
    r = rec()
    r["messages"] = r["messages"][:-1]
    assert any("助手" in e for e in C.validate_record(r))
    r = rec()
    r["messages"][0]["content"] = [{"type": "image", "image": "x.jpg"}]
    assert any("图片" in e for e in C.validate_record(r))
    r = rec()
    r["source"] = ""
    assert C.validate_record(r)
    assert C.validate_record(dict(rec(), source="distill-qwen")) == []


def test_load_records_duplicates(tmp_path):
    p = tmp_path / "t.jsonl"
    C.write_jsonl(p, [rec("x"), rec("x")])
    with pytest.raises(ValueError, match="重复"):
        C.load_records([p])
    assert len(C.load_records([p], strict=False)) == 1


# ---------------------------------------------------------------- 顺序

def test_deterministic_order_depends_only_on_ids_and_seed():
    ids = [f"s{i}" for i in range(50)]
    o1 = [ids[i] for i in C.deterministic_order(ids, 7)]
    rev = ids[::-1]
    o2 = [rev[i] for i in C.deterministic_order(rev, 7)]
    assert o1 == o2
    assert o1 != [ids[i] for i in C.deterministic_order(ids, 8)]
    plan = C.epoch_plan(ids, 7, 2)
    assert len(plan) == 100 and plan[:50] != plan[50:]
    assert sorted(plan[:50]) == list(range(50))


# ---------------------------------------------------------------- 泄漏

def test_leakage_by_id_and_hash(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "p1.jpg").write_bytes(b"photo-1")
    (tmp_path / "img" / "copy.jpg").write_bytes(b"photo-1")  # 同一张图换了名字
    bench = [rec("b-1", user="数据：9,9,9"), rec("b-2", user="看图", images=("img/p1.jpg",))]
    C.write_jsonl(tmp_path / "bench.jsonl", bench)
    idx = C.load_bench_index([tmp_path / "bench.jsonl"], tmp_path)
    train = [
        rec("t-ok", user="数据：1,2,3"),
        rec("b-1", user="完全不同"),                          # id 撞了
        rec("t-dup", user="  数据：9,9,9 ", system="别的系统提示"),  # 内容一样（空白不同、系统提示不同）
        rec("t-img", user="看图", images=("img/copy.jpg",)),  # 图片字节一样
    ]
    rep = C.leakage_report(train, idx, tmp_path)
    assert rep["id_hits"] == ["b-1"]
    assert rep["hash_hits"] == ["t-dup", "t-img"]
    assert not rep["clean"]
    assert C.leakage_report(train[:1], idx, tmp_path)["clean"]


def test_bench_index_txt_and_json(tmp_path):
    h = C.user_hash(rec("z", user="泄漏的题"))
    (tmp_path / "list.txt").write_text(f"# 注释\nb-9\nsha256:{h}\n")
    idx = C.load_bench_index([tmp_path / "list.txt"])
    assert idx["ids"] == {"b-9"} and idx["hashes"] == {h}
    (tmp_path / "list.json").write_text(json.dumps({"ids": ["b-8"], "hashes": [h]}))
    idx2 = C.load_bench_index([tmp_path / "list.json"])
    rep = C.leakage_report([rec("b-8"), rec("q", user="泄漏的题")], idx2)
    assert rep["id_hits"] == ["b-8"] and rep["hash_hits"] == ["q"]


# ---------------------------------------------------------------- 打乱标签

CAUSES = [{"id": f"hooke.c{i}", "name": n} for i, n in enumerate(["只有随机误差", "漏计挂钩质量", "原长读数偏差", "超出弹性限度"])]


def diag(rid, top=1):
    order = [top] + [i for i in range(4) if i != top]
    target = {"ranked": [{"cause_id": CAUSES[i]["id"], "prob": p} for i, p in zip(order, (0.7, 0.2, 0.05, 0.05))],
              "fix": {"cause_id": CAUSES[top]["id"]}, "explain": f"最可能是{CAUSES[top]['name']}（{CAUSES[top]['id']}）"}
    return rec(rid, user="候选：" + "；".join(c["name"] for c in CAUSES),
               target=json.dumps(target, ensure_ascii=False),
               meta={"experiment_id": "hooke", "split": "train", "causes": CAUSES})


def test_relabel_is_simultaneous_and_respects_boundaries():
    m = {"hooke.a": "hooke.b", "hooke.b": "hooke.a"}
    # hooke.a.x 是另一个更长的 id，不动；句末的 "hooke.a. " 要换
    assert C.relabel_text("hooke.a>hooke.b hooke.ab hooke.a.x hooke.a. 完", m) == \
        "hooke.b>hooke.a hooke.ab hooke.a.x hooke.b. 完"
    assert C.relabel_text("原因A和原因B", {"原因A": "原因B", "原因B": "原因A"}) == "原因B和原因A"


def test_shuffle_labels_control():
    recs = [diag(f"d-{i:03d}", top=i % 4) for i in range(400)] + [rec("other", task="explain", target="不动")]
    new, st = C.apply_label_shuffle(recs, seed=11)
    assert st["n_target_task"] == 400 and st["by_how"]["meta"] == 400
    # 均匀置换：原第一名换掉的概率 1 − 1/4
    assert 0.68 < st["frac_top_changed"] < 0.82
    assert st["frac_changed"] > 0.9
    for a, b in zip(recs, new):
        assert a["messages"][:-1] == b["messages"][:-1]  # prompt 不变
    assert new[-1] is recs[-1]
    # 每个样本里一致替换：ranked 第一名、fix、解释里的 id 和名称指向同一个原因
    for b in new[:400]:
        t = json.loads(b["messages"][-1]["content"])
        top = t["ranked"][0]["cause_id"]
        assert t["fix"]["cause_id"] == top and top in t["explain"]
        assert next(c["name"] for c in CAUSES if c["id"] == top) in t["explain"]
        assert sorted(x["cause_id"] for x in t["ranked"]) == sorted(c["id"] for c in CAUSES)
        assert [x["prob"] for x in t["ranked"]] == [0.7, 0.2, 0.05, 0.05]
    # 置换因样本而异（不是每个实验一个固定的双射），且由种子决定
    perms = Counter(tuple(b["meta"]["label_shuffle"]["perm"]) for b in new[:400])
    assert len(perms) > 15
    again, _ = C.apply_label_shuffle(recs, seed=11)
    assert again == new
    other, _ = C.apply_label_shuffle(recs, seed=12)
    assert other != new


def test_shuffle_fallback_domain_from_experiment():
    recs = []
    for i in range(60):
        r = diag(f"f-{i}", top=i % 4)
        r["meta"] = {"experiment_id": "hooke"}  # 没有 meta.causes
        recs.append(r)
    _, st = C.apply_label_shuffle(recs, seed=3)
    assert st["by_how"]["experiment"] == 60 and st["frac_top_changed"] > 0.5


# ---------------------------------------------------------------- 合成数据与统计

def test_synthetic_dataset_valid(tmp_path):
    info = C.make_synthetic(tmp_path, n=10, n_bench=3, with_images=False)
    tr = C.load_records([info["train"]])
    be = C.load_records([info["bench"]])
    assert len(tr) == 10 and len(be) == 3
    assert C.leakage_report(tr, C.load_bench_index([info["bench"]]))["clean"]
    for r in tr:
        enc = C.encode_example(r, CODEC, load_image=lambda p: p)
        assert enc["n_target"] > 0
    _, st = C.apply_label_shuffle(tr, seed=0)
    assert st["by_how"]["meta"] == 10
    s = C.dataset_summary(tr)
    assert s["task"] == {"diagnose": 10}
    ls = C.length_stats([3, 1, 2, 10])
    assert ls["max"] == 10 and ls["sum"] == 16 and ls["p50"] in (2, 3)


def test_safetensors_header_shapes(tmp_path):
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
    import convert_adapter as ca

    ca.save_tensors(tmp_path / "m.safetensors", {"a.weight": np.zeros((2, 3), np.float32)})
    assert C.model_weight_shapes(tmp_path) == {"a.weight": [2, 3]}


# ---------------------------------------------------------------- train_sft 的纯 Python 部分

def _args(tmp_path, **kw):
    import train_sft as T

    argv = ["--model", "m", "--train", "t.jsonl", "--output", str(tmp_path), "--batch-size", "2",
            "--grad-accum", "2", "--max-len", str(kw.pop("max_len", 4096))]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    return T.parse_args(argv)


def test_s1_and_control_see_identical_batches(tmp_path):
    import train_sft as T

    recs = [diag(f"d-{i:03d}", top=i % 4) for i in range(40)]
    # 一条恰好卡在长度线上的样本：打乱后目标变长也不能改变去留
    a = _args(tmp_path)
    enc0 = C.encode_example(recs[0], CODEC, load_image=lambda p: p)
    a.max_len = len(enc0["input_ids"])
    orig = {r["id"]: r for r in recs}
    items1, st1 = T.encode_all(recs, orig, CODEC, a, lambda *x, **k: None)
    shuf, _ = C.apply_label_shuffle(recs, a.seed)
    items2, st2 = T.encode_all(shuf, orig, CODEC, a, lambda *x, **k: None)
    ids1 = [it["rec"]["id"] for it in items1]
    assert ids1 == [it["rec"]["id"] for it in items2]
    assert st1["n_dropped_too_long"] == st2["n_dropped_too_long"]
    m1, p1, s1 = T.build_plan(items1, a)
    m2, p2, s2 = T.build_plan(items2, a)
    assert m1 == m2 and (p1, s1) == (p2, s2) == (math_ceil(len(ids1) * 2 / 4), math_ceil(len(ids1) * 2 / 4))
    # 批次内容：prompt 部分逐 token 相同，只有目标不同
    i0 = m1[0][0]
    e1, e2 = items1[i0]["enc"], items2[i0]["enc"]
    n_p = e1["labels"].index(next(x for x in e1["labels"] if x != C.IGNORE_INDEX))
    assert e1["input_ids"][:n_p] == e2["input_ids"][:n_p]


def math_ceil(x):
    import math

    return math.ceil(x)


def test_match_steps_and_config_check(tmp_path):
    import train_sft as T

    s1 = tmp_path / "s1"
    s1.mkdir()
    a = _args(tmp_path)
    cfg = {k: getattr(a, k) for k in T.MATCH_KEYS if k != "targets"}
    cfg["targets"] = list(T.resolve_targets(a.targets))
    (s1 / "summary.json").write_text(json.dumps({"config": cfg, "planned_total_steps": 30, "final_step": 21,
                                                 "kept_ids_sha256": "abc"}))
    items = [{"rec": rec(f"r{i}"), "enc": None} for i in range(100)]
    b = _args(tmp_path, match_steps_from=s1)
    _, planned, stop = T.build_plan(items, b)
    assert (planned, stop) == (30, 21)  # 学习率调度按 S1 的计划总步数，停在 S1 的最终步数
    assert T.check_match(b, cfg, "abc", lambda *x, **k: None) == {}
    with pytest.raises(SystemExit):
        T.check_match(b, dict(cfg, lr=1e-3), "abc", lambda *x, **k: None)
    with pytest.raises(SystemExit):
        T.check_match(b, cfg, "other-ids", lambda *x, **k: None)


def test_lr_schedule_and_targets():
    import train_sft as T

    f = T.lr_lambda_factory(100, 0.03, 0.1)
    assert f(0) == pytest.approx(1 / 3) and f(2) == pytest.approx(1.0)
    assert f(99) == pytest.approx(0.1, abs=0.01) and f(200) == pytest.approx(0.1)
    assert T.resolve_targets("all-linear")[-1] == "out_proj"
    assert "q_proj" in T.resolve_targets("q_proj,down_proj")
    assert T.is_language_linear("model.language_model.layers.3.mlp.up_proj")
    assert not T.is_language_linear("model.visual.blocks.0.mlp.linear_fc1")
    assert not T.is_language_linear("mtp.layers.0.mlp.up_proj")
