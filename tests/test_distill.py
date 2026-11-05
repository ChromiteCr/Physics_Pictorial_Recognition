import json

from lab_detective.training import distill
from lab_detective.training.distill import DistillJob


def _jobs(n):
    def check(text):
        ok = text.strip().startswith("OK")
        return ok, {"reason": "starts with OK" if ok else "wrong"}
    return [DistillJob(f"j{i}", "diagnose", [{"role": "system", "content": "s"},
                                             {"role": "user", "content": f"题 {i}"}], check, {"i": i})
            for i in range(n)]


def test_run_filters_caps_and_resumes(tmp_path):
    calls = []

    def teacher(messages):
        calls.append(messages[-1]["content"])
        i = int(messages[-1]["content"].split()[-1])
        return ("OK 答案" if i % 2 == 0 else "错"), {"total_tokens": 10}

    jobs = _jobs(10)
    s = distill.run(jobs, teacher, "fake", tmp_path, max_kept=3, workers=2)
    assert s["kept_total"] >= 3
    exs = distill.to_examples(tmp_path / "raw_results.jsonl", {j.id: j for j in jobs}, "distill-deepseek", 3)
    assert len(exs) == 3
    assert exs[0].messages[:-1] == jobs[0].messages  # 训练提示与教师所见一致
    n_calls = len(calls)
    distill.run(jobs, teacher, "fake", tmp_path, max_kept=100, workers=2)  # 续跑不重复已完成的题
    assert len(calls) - n_calls == 10 - n_calls


def test_hy4_pack_roundtrip(tmp_path):
    jobs = _jobs(3)
    assert distill.export_prompt_pack(jobs, tmp_path / "pack") == 3
    lines = [json.loads(l) for l in (tmp_path / "pack" / "prompts.jsonl").read_text().splitlines()]
    assert lines[1]["messages"][1]["content"] == "题 1"
    (tmp_path / "pack" / "responses.jsonl").write_text(
        "\n".join(json.dumps({"id": f"j{i}", "text": "OK" if i else "no"}) for i in range(3)))
    st = distill.import_responses(tmp_path / "pack", {j.id: j for j in jobs}, tmp_path / "out")
    assert st == {"read": 3, "kept": 2, "rejected": 1, "unknown_id": 0}


def test_cap_constant():
    assert distill.MAX_DISTILLED == 5248
