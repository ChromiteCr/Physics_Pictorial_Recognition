"""server/README.md 和 bootstrap.sh 里写的命令行参数，必须是脚本真有的参数（防止手册和代码对不上）。"""
import argparse
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRV = ROOT / "server"
sys.path.insert(0, str(SRV))

FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")


class _Stop(Exception):
    pass


def _parser_of(main, monkeypatch) -> argparse.ArgumentParser:
    """调用脚本的 main，在第一次 parse_args 时截下顶层解析器（不用改脚本）。"""
    box = {}

    def grab(self, args=None, namespace=None):
        box["p"] = self
        raise _Stop

    with monkeypatch.context() as m:
        m.setattr(argparse.ArgumentParser, "parse_args", grab)
        with pytest.raises(_Stop):
            main([])
    return box["p"]


def _flags(p: argparse.ArgumentParser, sub: str | None = None) -> set[str]:
    if sub is not None:
        sp = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
        p = sp.choices[sub]
    return {s for s in p._option_string_actions if s.startswith("--")}


@pytest.fixture
def parsers(monkeypatch):
    import check_package
    import common
    import convert_adapter
    import merge_and_eval
    import train_sft

    me = merge_and_eval.build_parser()
    ca = _parser_of(convert_adapter._main, monkeypatch)
    cm = _parser_of(common._main, monkeypatch)
    return {
        "train_sft.py": _flags(_parser_of(train_sft.parse_args, monkeypatch)),
        "merge_and_eval.py merge": _flags(me, "merge"),
        "merge_and_eval.py eval": _flags(me, "eval"),
        "check_package.py": _flags(_parser_of(check_package.main, monkeypatch)),
        **{f"convert_adapter.py {s}": _flags(ca, s) for s in ("convert", "check-names", "inspect", "selftest")},
        **{f"common.py {s}": _flags(cm, s) for s in ("synth", "check")},
    }


def _logical_lines(text: str) -> list[str]:
    """把反斜杠续行接起来。"""
    out, cur = [], ""
    for line in text.splitlines():
        if line.rstrip().endswith("\\"):
            cur += line.rstrip()[:-1] + " "
            continue
        out.append(cur + line)
        cur = ""
    if cur:
        out.append(cur)
    return out


def _which(line: str, keys) -> str | None:
    if line.lstrip().startswith("$EV"):
        return "merge_and_eval.py eval"
    for k in sorted(keys, key=len, reverse=True):  # 先匹配带子命令的
        script, *sub = k.split(" ")
        if script in line and (not sub or re.search(re.escape(script) + r"\S*\s+" + re.escape(sub[0]) + r"\b", line)):
            return k
    return None


def _check(lines, parsers, where):
    bad, n = [], 0
    for line in lines:
        k = _which(line, parsers)
        if k is None:
            continue
        n += 1
        for f in FLAG.findall(line.split("#", 1)[0]):  # 行尾注释里的参数另算
            if f not in parsers[k]:
                bad.append(f"{where}: {k} 没有参数 {f}：{line.strip()[:120]}")
    return n, bad


def test_readme_commands_use_real_flags(parsers):
    text = (SRV / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", text, flags=re.S)
    n, bad = _check([ln for b in blocks for ln in _logical_lines(b)], parsers, "README 代码块")
    assert n >= 15, n  # 确实解析到了命令
    assert not bad, "\n".join(bad)
    # 正文和注释里提到的参数：至少要是某个脚本的参数
    known = set().union(*parsers.values())
    loose = {f for f in FLAG.findall(text)} - known - {"--no-build-isolation", "--version"}
    assert not loose, loose


def test_bootstrap_commands_use_real_flags(parsers):
    text = (SRV / "bootstrap.sh").read_text(encoding="utf-8")
    lines = _logical_lines(text)
    n, bad = _check(lines, parsers, "bootstrap.sh")
    m = re.search(r"COMMON_ARGS=\((.*?)\)", text, flags=re.S)  # 冒烟训练共用的参数（跨行数组）
    assert m
    bad += [f"bootstrap.sh COMMON_ARGS: train_sft.py 没有参数 {f}" for f in FLAG.findall(m.group(1))
            if f not in parsers["train_sft.py"]]
    assert n >= 6, n
    assert not bad, "\n".join(bad)


def test_readme_documents_current_defaults():
    import merge_and_eval as M
    import train_sft as T

    text = (SRV / "README.md").read_text(encoding="utf-8")
    a = T.parse_args(["--model", "m", "--train", "t", "--output", "o"])
    e = M.build_parser().parse_args(["eval", "--model", "m", "--bench", "b", "--out", "o", "--arm", "x"])
    assert f"max_len {a.max_len}" in text and f"微批次 {a.batch_size} × 累积 {a.grad_accum}" in text
    assert f"--max-new-tokens {e.max_new_tokens}" in text
    assert "max_len 2048" not in text and "微批次 2 × 累积 8" not in text
    assert text.index("前 30 分钟") < text.index("## 0.")  # 上机清单在最前面
