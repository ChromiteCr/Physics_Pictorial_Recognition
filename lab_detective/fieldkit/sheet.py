"""记录表（CSV）的读写：一次实验一个文件，一行一个读数。

格式（UTF-8，带 BOM 方便 Excel 打开）：
  #@ 键: 值        机器读的表头（运行编号、代码、实验、仪器配置、追加测量顺序），键只能是白名单里的
  # ……           说明文字，读入时忽略
  step,stage,item,kind,setting,value,unit
stage = base（基础方案）或追加测量的槽位（eig / best_fixed / repeat / twin_eig，同一项测量算几个槽位时用 + 连起来）；
item = 菜单项 id（基础方案写 standard，待定的写 ?）；kind = 观测种类（只有留出孪生认识的加 twin. 前缀）；
value = 读数，做不了的追加测量写 skip。表里不能有其他列，也不能有任何关于人的信息。
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

COLUMNS = ("step", "stage", "item", "kind", "setting", "value", "unit")
META_KEYS = ("fieldkit_version", "run_id", "code", "experiment", "session", "followup_order")
PENDING = "?"
SKIP = "skip"


@dataclass
class Row:
    step: int
    stage: str
    item: str
    kind: str
    setting: str
    value: str
    unit: str

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(s for s in self.stage.split("+") if s)

    def as_list(self) -> list[str]:
        return [str(self.step), self.stage, self.item, self.kind, self.setting, self.value, self.unit]


@dataclass
class Sheet:
    meta: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)  # 说明文字（# 开头的行），写回时原样保留
    rows: list = field(default_factory=list)

    @property
    def config(self) -> dict:
        out = {}
        for k, v in self.meta.items():
            if k.startswith("config."):
                out[k[len("config."):]] = parse_scalar(v)
        return out

    def base_rows(self) -> list[Row]:
        return [r for r in self.rows if r.stages == ("base",)]

    def followup_rows(self) -> list[Row]:
        return [r for r in self.rows if r.stages != ("base",)]


def parse_scalar(v: str):
    """配置值：数字转 float，true/false 转布尔，[a, b] 转数字列表，其余保留字符串。"""
    s = str(v).strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.startswith("[") and s.endswith("]"):
        return [float(x) for x in s[1:-1].replace(",", " ").split()]
    try:
        return float(s)
    except ValueError:
        return s


def fmt_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + " ".join(fmt_num(x) for x in v) + "]"  # 列表不用逗号，免得表格软件把这一行拆成几格
    if isinstance(v, (int, float)):
        return fmt_num(v)
    return str(v)


def fmt_num(x) -> str:
    x = float(x)
    return str(int(x)) if x == int(x) and abs(x) < 1e15 else repr(x)


class SheetError(ValueError):
    pass


def read_sheet(path: str | Path) -> Sheet:
    text = Path(path).read_text(encoding="utf-8-sig")
    return parse_sheet(text)


def parse_sheet(text: str) -> Sheet:
    sheet = Sheet()
    body = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith('"#'):
            s = s.rstrip(",").rstrip()  # 表格软件另存时会在说明行后面补逗号，
            if s.startswith('"') and s.endswith('"'):  # 也可能给带逗号的说明行加引号
                s = s[1:-1].replace('""', '"')
            line = s
        if s.startswith("#@"):
            if ":" not in s:
                raise SheetError(f"表头行格式不对：{line}")
            k, v = s[2:].split(":", 1)
            k, v = k.strip(), v.strip()
            if k not in META_KEYS and not k.startswith("config."):
                raise SheetError(f"表头里有不认识的键 {k!r}（只允许 {', '.join(META_KEYS)} 和 config.*）")
            sheet.meta[k] = v
        elif s.startswith("#"):
            sheet.notes.append(line.rstrip())
        elif s:
            body.append(line)
    if not body:
        raise SheetError("没有数据表头")
    reader = csv.reader(io.StringIO("\n".join(body)))
    header = [h.strip() for h in next(reader)]
    if tuple(header) != COLUMNS:
        raise SheetError(f"列必须正好是 {','.join(COLUMNS)}，现在是 {','.join(header)}")
    for i, rec in enumerate(reader, start=2):
        if not any(c.strip() for c in rec):
            continue
        if len(rec) != len(COLUMNS):
            raise SheetError(f"第 {i} 行有 {len(rec)} 列，应为 {len(COLUMNS)} 列")
        rec = [c.strip() for c in rec]
        try:
            step = int(float(rec[0]))
        except ValueError as e:
            raise SheetError(f"第 {i} 行 step 不是整数：{rec[0]!r}") from e
        sheet.rows.append(Row(step, *rec[1:]))
    return sheet


def render_sheet(sheet: Sheet) -> str:
    out = io.StringIO()
    for k in META_KEYS:
        if k in sheet.meta:
            out.write(f"#@ {k}: {sheet.meta[k]}\n")
    for k, v in sheet.meta.items():
        if k.startswith("config."):
            out.write(f"#@ {k}: {v}\n")
    for n in sheet.notes:
        out.write(n + "\n")
    w = csv.writer(out, lineterminator="\n")
    w.writerow(COLUMNS)
    for r in sheet.rows:
        w.writerow(r.as_list())
    return out.getvalue()


def write_sheet(sheet: Sheet, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_sheet(sheet), encoding="utf-8-sig")


def base_hash(sheet: Sheet) -> str:
    """基础方案读数的指纹（种类、控制量、读数），推荐下一次测量时记下，之后基础数据不能再改。"""
    h = hashlib.sha256()
    for r in sheet.base_rows():
        h.update(f"{r.kind}|{r.setting}|{r.value}\n".encode("utf-8"))
    return h.hexdigest()
