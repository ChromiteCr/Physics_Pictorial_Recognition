"""讲解声明校验器：把讲解文字里的数字、单位、方向词、原因名称、推荐测量逐一和事实单核对。

模型从不自己算物理：讲解里的每个数字都必须能在事实单里找到（按写出的位数四舍五入、允许换算单位）。
还要求讲解覆盖三项：主要原因、至少一个数值证据、下一次测量，防止空话过关。

宽严取舍：误杀正确讲解（逼着回退模板、还会错删训练数据）比放过可疑说法代价更大，
所以方向词只在指代明确、不是假设句时才检查；不带正负号的数字按绝对值匹配。
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction

# ── 文本规范化 ──────────────────────────────────────────────────────────

_SUP_CHARS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺"
_SUP = str.maketrans(_SUP_CHARS, "0123456789-+")
_SUP_RUN = re.compile(f"[{_SUP_CHARS}]+")


def normalize(text: str) -> str:
    """上标改成 ^ 记法（10⁻³ → 10^-3，m/s² → m/s^2），再 NFKC（全角数字和标点转半角），统一负号和 ±。"""
    text = _SUP_RUN.sub(lambda m: "^" + m.group(0).translate(_SUP), text)
    text = unicodedata.normalize("NFKC", text)
    for a in ("−", "﹣", "–", "—"):
        text = text.replace(a, "-")
    return text.replace("+/-", "±").replace("+-", "±")


# ── 单位：原子单位 + 组合（/ · 每 ^ √），量纲用基本量的指数表示 ───────────────

_L, _M, _T = {"L": 1}, {"M": 1}, {"T": 1}
_ATOMS: dict[str, tuple[dict, float]] = {}


def _atom(dims: dict, factor: float, *names: str) -> None:
    for n in names:
        _ATOMS[n] = (dims, factor)


_atom(_L, 1.0, "m", "米")
_atom(_L, 0.01, "cm", "厘米")
_atom(_L, 0.001, "mm", "毫米")
_atom(_L, 1000.0, "km", "千米", "公里")
_atom(_M, 1.0, "kg", "千克", "公斤")
_atom(_M, 1e-3, "g", "克")
_atom(_M, 1e-6, "mg", "毫克")
_atom(_T, 1.0, "s", "秒")
_atom(_T, 1e-3, "ms", "毫秒")
_atom(_T, 60.0, "min", "分钟")
_atom({"M": 1, "L": 1, "T": -2}, 1.0, "N", "牛", "牛顿")
_atom({"M": 1, "L": 1, "T": -2}, 1e-3, "mN", "毫牛")
_atom({"M": 1, "L": 2, "T": -2}, 1.0, "J", "焦", "焦耳")
_atom({"M": 1, "L": 2, "T": -2}, 1e-3, "mJ", "毫焦")
_atom({"M": 1, "L": 2, "T": -3}, 1.0, "W", "瓦", "瓦特")
_atom({"T": -1}, 1.0, "Hz", "赫兹")
_atom({"ang": 1}, 1.0, "rad", "弧度")
_atom({"ang": 1}, math.pi / 180.0, "°", "度")
_atom({}, 0.01, "%")
_atom({}, 1.0, "倍", "σ", "个标准差")
_atom({"bit": 1}, 1.0, "bit", "bits", "比特")
_SPECIAL = {"米每二次方秒": ({"L": 1, "T": -2}, 1.0), "米每平方秒": ({"L": 1, "T": -2}, 1.0),
            "厘米每二次方秒": ({"L": 1, "T": -2}, 0.01)}

_ATOM_ALT = "|".join(
    re.escape(u) + ("(?![A-Za-z])" if u[-1].isascii() and u[-1].isalpha() else "")
    for u in sorted(_ATOMS, key=len, reverse=True)
)
_EXP = r"(?:\^\(?-?\d+(?:/\d+)?\)?|(?<=[A-Za-z])\d(?![\d.]))?"
_ATOM_PAT = r"√?(?:" + _ATOM_ALT + ")" + _EXP
_COMPOUND = (r"(?:" + "|".join(map(re.escape, _SPECIAL)) + r"|" + _ATOM_PAT
             + r"(?:\s?[/·*]\s?" + _ATOM_PAT + r"|每" + _ATOM_PAT + r")*)")
_ATOM_RE = re.compile(r"(?P<sep>[/·*每]?)\s?(?P<sqrt>√?)(?P<name>" + _ATOM_ALT + r")(?P<exp>" + _EXP + ")")


def _dim_key(d: dict) -> str:
    items = sorted((k, v) for k, v in d.items() if v != 0)
    return "1" if not items else "·".join(f"{k}{v}" for k, v in items)


def parse_unit(unit: str | None) -> tuple[str, float]:
    """单位字符串 → (量纲键, 换算到国际单位的倍数)。空串 = 无量纲 "1"；不认识的单位按原字符串当量纲（只和自己相等）。"""
    u = normalize(unit or "").strip()
    if not u:
        return "1", 1.0
    if u in _SPECIAL:
        d, f = _SPECIAL[u]
        return _dim_key(d), f
    if not re.fullmatch(_COMPOUND, u):
        return "?" + u, 1.0
    dims: dict = {}
    factor = 1.0
    for m in _ATOM_RE.finditer(u):
        e = m.group("exp").strip("^()")
        exp = Fraction(e) if e else Fraction(1)
        if m.group("sqrt"):
            exp /= 2
        if m.group("sep") in ("/", "每"):
            exp = -exp
        d, f = _ATOMS[m.group("name")]
        for k, v in d.items():
            dims[k] = dims.get(k, 0) + v * exp
        factor *= f ** float(exp)
    return _dim_key(dims), factor


# ── 数字抽取 ────────────────────────────────────────────────────────────

_NUM_RE = re.compile(
    r"(?<![A-Za-z_\d.^])"
    r"(?P<sign>[+\-])?"
    r"(?P<mant>\d+(?:\.\d+)?|\.\d+)"
    r"(?:[eE](?P<e1>[+\-]?\d+)(?![A-Za-z])|\s*[×xX*·]\s*10\s*(?:\^|\*\*)\s*\(?(?P<e2>[+\-]?\d+)\)?)?"
)
_UNIT_RE = re.compile(r"\s*(?P<unit>" + _COMPOUND + ")")
_PM_RE = re.compile(r"\s*±\s*")
_COUNT_RE = re.compile(r"\s*(组|次|个|步|条|种|项|轮|位|名|行|列|张|段|遍|页|号|天|周|人)")
_LIST_RE = re.compile(r"(?:^|\n)\s*\(?$")


@dataclass
class Num:
    """讲解里的一个数字。tol = 按写出的位数推出的取整容差（与 value 同单位）。"""

    value: float
    tol: float
    sig: int
    signed: bool
    unit: str
    start: int
    end: int
    raw: str
    exempt: str = ""  # 非物理量（计数、序号、列表编号、公式系数）时写原因
    uend: int = -1  # 连同单位的结束位置


def _tol_and_sig(mant: str) -> tuple[float, int]:
    if "." in mant:
        dec = len(mant.split(".")[1])
        digits = mant.replace(".", "").lstrip("0")
        return 0.5 * 10.0 ** (-dec), max(len(digits), 1)
    stripped = mant.lstrip("0") or "0"
    tz = len(stripped) - len(stripped.rstrip("0")) if stripped != "0" else 0
    return 0.5 * 10.0**tz, max(len(stripped) - tz, 1)


def extract_numbers(text: str) -> list[Num]:
    """抽出所有数字（含科学计数法、± 对、单位）。text 应已 normalize。"""
    out: list[Num] = []
    pos = 0
    while True:
        m = _NUM_RE.search(text, pos)
        if not m:
            break
        mant = m.group("mant")
        exp = m.group("e1") or m.group("e2")
        tol, sig = _tol_and_sig(mant)
        val = float(mant)
        if exp is not None:
            scale = 10.0 ** int(exp)
            val *= scale
            tol *= scale
        if m.group("sign") == "-":
            val = -val
        end = m.end()
        nums = [Num(val, tol, sig, m.group("sign") is not None, "", m.start(), end, m.group(0))]
        # (a ± b) 单位：两个数共用单位
        pm = _PM_RE.match(text, end)
        if pm:
            m2 = _NUM_RE.match(text, pm.end())
            if m2:
                t2, s2 = _tol_and_sig(m2.group("mant"))
                v2 = float(m2.group("mant"))
                e2 = m2.group("e1") or m2.group("e2")
                if e2 is not None:
                    v2 *= 10.0 ** int(e2)
                    t2 *= 10.0 ** int(e2)
                end = m2.end()
                nums.append(Num(v2, t2, s2, False, "", m2.start(), end, m2.group(0)))
        close = re.match(r"\s*\)", text[end:])
        after = end + (close.end() if close else 0)
        um = _UNIT_RE.match(text, after)
        if um:
            for n in nums:
                n.unit = um.group("unit")
            end = um.end()
        for n in nums:
            n.uend = end if n.unit else n.end
            if not n.unit:
                n.exempt = _exemption(text, n)
        out.extend(nums)
        pos = max(end, m.end())
    return out


def _exemption(text: str, n: Num) -> str:
    before = text[: n.start]
    if re.search(r"第\s*$", before):
        return "ordinal"
    if _COUNT_RE.match(text, n.end):
        return "count"
    if not n.signed and re.fullmatch(r"\d{1,2}", n.raw) and re.match(r"[.、)]\s*(?!\d)", text[n.end:]) \
            and _LIST_RE.search(before):
        return "list"
    if re.match(r"\s?/\s?[A-Za-z一-鿿(]", text[n.end:]):
        return "formula"
    return ""


# ── 允许出现的数字（来自事实单）────────────────────────────────────────


@dataclass
class Allowed:
    name: str
    value: float
    unit: str
    dim: str
    factor: float
    sign: int | None  # 方向词可用的正负（None = 判断不了）
    evidence: bool  # 能否算作"数值证据"
    source: str


_DEVIATION_NAMES = ("intercept", "offset", "bias", "diff", "delta", "resid", "deviation", "excess",
                    "截距", "偏差", "偏移", "差")
_RATIO_NAMES = ("ratio", "比")


def _sgn(x: float) -> int | None:
    if x is None or not math.isfinite(x) or x == 0:
        return None
    return 1 if x > 0 else -1


def _entry_sign(e: dict) -> int | None:
    if e.get("sign") in (1, -1, "+", "-"):
        return 1 if e["sign"] in (1, "+") else -1
    try:
        v = float(e["value"])
    except (TypeError, ValueError, KeyError):
        return None
    if e.get("ref") is not None:
        return _sgn(v - float(e["ref"]))
    if e.get("z") is not None:
        return _sgn(float(e["z"]))
    name = f"{e.get('name', '')} {e.get('label', '')} {e.get('label_zh', '')}".lower()
    if any(k in name for k in _RATIO_NAMES):
        return _sgn(v - 1.0)
    if any(k in name for k in _DEVIATION_NAMES):
        return _sgn(v)
    return None


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def allowed_numbers(fs: dict, strict: bool = False) -> list[Allowed]:
    """事实单里允许引用的数字。strict=True 只认 numbers 列表；否则还认结构化字段和事实单字符串里的数字。"""
    out: list[Allowed] = []

    def add(name, value, unit, sign=None, evidence=True, source="numbers"):
        v = _num(value)
        if v is None:
            return
        dim, fac = parse_unit(unit)
        out.append(Allowed(str(name), v, unit or "", dim, fac, sign, evidence, source))

    for e in fs.get("numbers") or []:
        add(e.get("name", ""), e.get("value"), e.get("unit", ""), _entry_sign(e))
    if strict:
        return out
    ag = fs.get("agreement") or {}
    if ag:
        zs = _sgn(_num(ag.get("z")) or 0.0)
        q = ag.get("quantity", "agreement")
        add(q, ag.get("value"), ag.get("unit", ""), zs, True, "agreement")
        add(f"{q}_uncertainty", ag.get("uncertainty"), ag.get("unit", ""), None, True, "agreement")
        add("z", ag.get("z"), "", zs, True, "agreement")
        for k in ("predicted", "expected", "reference"):
            add(f"{q}_{k}", ag.get(k), ag.get("unit", ""), None, True, "agreement")
    dg = fs.get("diagnosis") or {}
    for r in dg.get("ranked") or []:
        add(f"p:{r.get('cause_id', '')}", r.get("prob"), "", None, False, "ranked")
    add("misfit_p", dg.get("misfit_p"), "", None, True, "diagnosis")
    nm = fs.get("next_measurement") or {}
    add("eig_bits", nm.get("eig_bits"), "bit", None, False, "next")
    for p in nm.get("predicted") or []:
        add(f"pred:{p.get('cause_id', '')}", p.get("value"), p.get("unit", ""), None, True, "next")
    add("n_points", (fs.get("data") or {}).get("n_points"), "", None, False, "data")
    # 事实单里的字符串（证据、测量名、修正方法……）本身含的数字
    strings = [(fs.get("experiment_name"), False), (nm.get("label"), False)]
    for r in dg.get("ranked") or []:
        strings += [(r.get("name"), False), (r.get("evidence"), True)]
    fx = fs.get("fix") or {}
    strings += [(fx.get("text"), False), (fx.get("predicted_rerun"), True)]
    for s, ev in strings:
        if not s:
            continue
        for n in extract_numbers(normalize(str(s))):
            if n.exempt:
                continue
            dim, fac = parse_unit(n.unit)
            out.append(Allowed("text", n.value, n.unit, dim, fac, None, ev, "string"))
    return out


def _close(t: float, tol: float, sig: int, v: float) -> bool:
    d = abs(t - v)
    if d > tol * (1 + 1e-9) + 1e-12 * max(1.0, abs(v)):
        return False
    if sig <= 1 and d > 0.25 * abs(v) + 1e-12:  # 只有一位有效数字时防止"1.52 写成 2"
        return False
    return True


def _match(n: Num, a: Allowed, ignore_unit: bool = False) -> bool:
    cands: list[tuple[float, float]] = []
    if not n.unit or ignore_unit:
        cands.append((n.value, n.tol))
        if a.dim == "1" and a.factor != 1.0:
            cands.append((n.value / a.factor, n.tol / a.factor))
    else:
        dim, fac = parse_unit(n.unit)
        if dim != a.dim:
            return False
        cands.append((n.value * fac / a.factor, n.tol * fac / a.factor))
    for t, tol in cands:
        if _close(t, tol, n.sig, a.value):
            return True
        if not n.signed and _close(t, tol, n.sig, abs(a.value)):
            return True
    return False


# ── 名称、目录 ─────────────────────────────────────────────────────────

_CATALOG_CACHE: dict | None = None


def default_catalog() -> dict:
    """从 labsim 注册表收集所有实验的原因名和测量菜单（懒加载；注册表导入失败就返回空目录）。"""
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE
    cat: dict = {"causes": [], "menus": {}}
    try:
        from ..labsim.registry import EXPERIMENTS

        for eid, exp in EXPERIMENTS.items():
            for c in exp.causes:
                cat["causes"].append({"id": c.id, "name_zh": c.name_zh, "name_en": c.name_en,
                                      "experiment_id": eid})
            cat["menus"][eid] = [{"id": m.id, "label_zh": m.label_zh, "label_en": m.label_en}
                                 for m in exp.menu]
    except Exception:  # noqa: BLE001  并行开发时某个实验文件可能还没写完
        cat = {"causes": [], "menus": {}}
    _CATALOG_CACHE = cat
    return cat


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", normalize(s or ""))


def _core_name(name: str) -> str:
    """去掉括号里的补充说明：'探究…（胡克定律）' → '探究…'。"""
    return re.sub(r"\(.*?\)", "", _squash(name)).strip("「」“”\"'")


def _bigrams(s: str) -> set[str]:
    s = re.sub(r"[^一-鿿A-Za-z0-9]", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def containment(label: str, text: str) -> float:
    """label 的二字组有多少比例出现在 text 里（中文改写容忍度）。"""
    b = _bigrams(_core_name(label))
    if not b:
        return 0.0
    return len(b & _bigrams(_squash(text))) / len(b)


_NEG_RE = re.compile(r"(不是|并非|而不是|排除|不太可能|可能性很小|不像|并不是|没有发现)")
_NOT_IN_LIST_RE = re.compile(r"(不在(清单|列表|名单|候选)|以上都不是|清单(之外|以外|外)|都对不上|没有考虑到|未建模|not in (the|my) list)",
                             re.I)
_CLAIM_TOP_RE = re.compile(
    r"(主要原因|最可能的原因|最大的嫌疑|嫌疑最大的|首要原因|罪魁祸首|元凶|最可能)(是|为|在于|可能是)?[:\s「“\"']*"
    r"(?P<x>[^,。;!?\n(「」“”\"']{2,24})")
_CLAIM_ANY_RE = re.compile(r"(?<!主要)(?<!首要)原因(是|可能是|在于|出在)[:\s「“\"']*(?P<x>[^,。;!?\n(「」“”\"']{2,24})")
_QUESTION_RE = re.compile(r"(什么|哪|吗|谁|怎么|为何)")
_RECOMMEND_RE = re.compile(r"(下一步|接下来|建议|推荐|下一次|应该先|最好先|next)", re.I)

# ── 方向词 ──────────────────────────────────────────────────────────────

_DIR_WORDS: dict[str, tuple[int, str]] = {
    "偏大": (1, "dev"), "偏小": (-1, "dev"), "偏高": (1, "dev"), "偏低": (-1, "dev"),
    "过大": (1, "dev"), "过小": (-1, "dev"), "为正": (1, "dev"), "为负": (-1, "dev"),
    "变大": (1, "trend"), "变小": (-1, "trend"), "增大": (1, "trend"), "减小": (-1, "trend"),
    "大于": (1, "cmp"), "小于": (-1, "cmp"), "高于": (1, "cmp"), "低于": (-1, "cmp"),
    "too large": (1, "dev"), "too high": (1, "dev"), "too small": (-1, "dev"), "too low": (-1, "dev"),
    "larger than": (1, "cmp"), "greater than": (1, "cmp"), "higher than": (1, "cmp"),
    "smaller than": (-1, "cmp"), "less than": (-1, "cmp"), "lower than": (-1, "cmp"),
    "increase": (1, "trend"), "decrease": (-1, "trend"),
}
_DIR_RE = re.compile("|".join(re.escape(w) for w in sorted(_DIR_WORDS, key=len, reverse=True)), re.I)
_HYPO_RE = re.compile(r"(如果|假如|假设|若|要是|万一|\bif\b)", re.I)
_TREND_SKIP_RE = re.compile(r"(会|将|随|当|后|再|修正|改正|\bwill\b|\bwhen\b|\bafter\b)", re.I)
_DIR_NEG_RE = re.compile(r"(不|没|未|并非|\bnot\b)")
_SENT_SPLIT = re.compile(r"[。!?\n]")
_CLAUSE_SPLIT = re.compile(r"[,;]")

QUANTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "intercept": ("截距", "intercept"),
    "slope": ("斜率", "slope"),
    "slope_ratio": ("斜率比",),
    "k": ("劲度系数", "k"),
    "g": ("重力加速度", "g"),
    "mu": ("动摩擦因数", "摩擦因数", "μ"),
    "a": ("加速度",),
    "z": ("z",),
    "zero_offset": ("零点偏差", "原长偏差"),
    "hanger_mass": ("挂钩质量",),
}


@dataclass
class _Ref:
    pos: int
    end: int
    sign: int | None


def _alias_refs(text: str, allowed: list[Allowed], fs: dict, unit_spans: list[tuple[int, int]]) -> list[_Ref]:
    names: dict[str, set] = {}

    def put(alias: str, sign):
        if alias:
            names.setdefault(alias, set()).add(sign)

    for a in allowed:
        if a.source == "string":
            continue
        for al in (a.name, *QUANTITY_ALIASES.get(a.name, ())):
            put(al, a.sign)
    for e in fs.get("numbers") or []:
        for k in ("label", "label_zh"):
            if e.get(k):
                put(normalize(str(e[k])), _entry_sign(e))
    ag = fs.get("agreement") or {}
    if ag.get("label"):
        put(normalize(str(ag["label"])), _sgn(_num(ag.get("z")) or 0.0))
    refs: list[_Ref] = []
    for alias, signs in names.items():
        sign = next(iter(signs)) if len(signs) == 1 else None
        if alias.isascii():
            if not re.fullmatch(r"[A-Za-zμ_]+", alias):
                continue
            pat = r"(?<![A-Za-z_\d])" + re.escape(alias) + r"(?![A-Za-z_\d])"
        else:
            pat = re.escape(alias)
        for m in re.finditer(pat, text):
            if any(s <= m.start() < e for s, e in unit_spans):
                continue
            refs.append(_Ref(m.start(), m.end(), sign))
    return refs


def _check_directions(text: str, nums: list[Num], matched: dict[int, list[Allowed]],
                      allowed: list[Allowed], fs: dict) -> list[dict]:
    problems: list[dict] = []
    unit_spans = [(n.end, n.uend) for n in nums if n.unit]
    alias_refs = _alias_refs(text, allowed, fs, unit_spans)
    num_refs: list[_Ref] = []
    for i, n in enumerate(nums):
        signs = {a.sign for a in matched.get(i, []) if a.sign is not None}
        if matched.get(i):
            num_refs.append(_Ref(n.start, n.end, next(iter(signs)) if len(signs) == 1 else None))
    refs = sorted(alias_refs + num_refs, key=lambda r: r.pos)

    s0 = 0
    for sent in _SENT_SPLIT.split(text):
        clauses, c0 = [], s0
        for cl in _CLAUSE_SPLIT.split(sent):
            clauses.append((c0, c0 + len(cl), cl))
            c0 += len(cl) + 1
        s0 += len(sent) + 1
        if _HYPO_RE.search(sent):
            continue
        for ci, (cs, ce, cl) in enumerate(clauses):
            for m in _DIR_RE.finditer(cl):
                word = m.group(0)
                d, kind = _DIR_WORDS[word.lower()]
                if kind == "trend" and _TREND_SKIP_RE.search(cl):
                    continue
                if _DIR_NEG_RE.search(cl[max(0, m.start() - 3): m.start()]):
                    d = -d
                ws, we = cs + m.start(), cs + m.end()
                if kind == "cmp":
                    left = [n for n in nums if cs <= n.start and n.end <= ws and not n.exempt]
                    if not left and ci > 0:  # "测得 24.1 N/m，小于预测的 22.0 N/m"：左边的数在上一小句
                        left = [n for n in nums if clauses[ci - 1][0] <= n.start and n.end <= ws and not n.exempt]
                    right = [n for n in nums if we <= n.start and n.end <= ce and not n.exempt]
                    if left and right:
                        a, b = left[-1], right[0]
                        da, fa = parse_unit(a.unit)
                        db, fb = parse_unit(b.unit or a.unit)
                        if da == db or not b.unit or not a.unit:
                            va, vb = a.value * fa, b.value * (fb if b.unit else fa)
                            if va != vb and (1 if va > vb else -1) != d:
                                problems.append(_p("direction", text[a.start:b.end],
                                                   f"「{word}」与两边的数字大小关系相反"))
                        continue
                before = [r for r in refs if cs <= r.pos and r.end <= ws]
                after = [r for r in refs if we <= r.pos and r.end <= ce]
                ref = before[-1] if before else (after[0] if after else None)
                if ref is None and ci > 0:
                    ps, pe, _ = clauses[ci - 1]
                    prev = [r for r in refs if ps <= r.pos and r.end <= pe]
                    ref = prev[-1] if prev else None
                if ref is None or ref.sign is None:
                    continue
                if ref.sign != d:
                    problems.append(_p("direction", text[min(ref.pos, ws):max(ref.end, we)],
                                       f"「{word}」与事实单里这个量的正负相反"))
    return problems


# ── 主入口 ──────────────────────────────────────────────────────────────


def _p(code: str, span: str, msg: str) -> dict:
    return {"code": code, "text": span, "msg": msg}


def verify(text: str, fact_sheet: dict, catalog: dict | None = None, strict: bool = False) -> dict:
    """核对一段讲解。返回 {ok, problems[{code,text,msg}], coverage{top_cause, number_evidence, next_measurement}}。

    catalog: {"causes": [{id,name_zh,name_en,experiment_id}], "menus": {exp_id: [{id,label_zh,label_en}]}}，
    用来认出"别的原因 / 别的测量"的名字；不给就从 labsim 注册表懒加载。
    """
    fs = fact_sheet or {}
    cat = default_catalog() if catalog is None else catalog
    t = normalize(text or "")
    problems: list[dict] = []

    # 1. 数字与单位
    allowed = allowed_numbers(fs, strict)
    nums = extract_numbers(t)
    matched: dict[int, list[Allowed]] = {}
    evidence_hit = False
    for i, n in enumerate(nums):
        if n.exempt:
            continue
        hits = [a for a in allowed if _match(n, a)]
        if hits:
            matched[i] = hits
            evidence_hit |= any(a.evidence for a in hits)
            continue
        span = t[n.start:n.uend].strip()
        if n.unit and any(a.dim != "1" and _match(n, a, ignore_unit=True) for a in allowed):
            problems.append(_p("unit_mismatch", span, "数值和事实单一致，但单位不对"))
        else:
            problems.append(_p("number_mismatch", span, "事实单里没有这个数字"))

    # 2. 方向词
    problems += _check_directions(t, nums, matched, allowed, fs)

    # 3. 原因名称
    dg = fs.get("diagnosis") or {}
    ranked = [r for r in dg.get("ranked") or [] if r.get("name") or r.get("cause_id")]
    # 下一次测量里给了预测值的原因也算事实单里的原因（只用于"名字是否合法"，不参与排名判断）
    extra = [p for p in (fs.get("next_measurement") or {}).get("predicted") or []
             if p.get("cause_id") not in {r.get("cause_id") for r in ranked} and (p.get("name") or p.get("cause_id"))]
    by_id = {c["id"]: c for c in cat.get("causes", [])}
    rnames: list[list[str]] = []
    for r in ranked + extra:
        names = [_core_name(r.get("name", ""))]
        c = by_id.get(r.get("cause_id", ""))
        if c:
            names += [_core_name(c.get("name_zh", "")), c.get("name_en", "")]
        if str(r.get("cause_id", "")).endswith(".random_only"):
            names.append("随机误差")
        rnames.append([n for n in names if n])
    ranked_ids = {r.get("cause_id") for r in ranked + extra}
    flat = sorted({n for ns in rnames for n in ns}, key=len, reverse=True)
    masked = t
    for n in flat:
        masked = re.sub(re.escape(n), lambda m: "□" * len(m.group(0)), masked, flags=re.I)
    for c in cat.get("causes", []):
        if c["id"] in ranked_ids:
            continue
        for nm in (_core_name(c.get("name_zh", "")), c.get("name_en", "")):
            if len(nm) < 3:
                continue
            for m in re.finditer(re.escape(nm), masked, re.I):
                clause = _clause_at(t, m.start())
                if _NEG_RE.search(clause):
                    continue
                problems.append(_p("unknown_cause", nm, "这个原因不在事实单的嫌疑人名单里"))
    for rx, top_only in ((_CLAIM_TOP_RE, True), (_CLAIM_ANY_RE, False)):
        for m in rx.finditer(t):
            x = m.group("x").strip()
            if _QUESTION_RE.search(x) or not ranked:
                continue
            if dg.get("not_in_list") and _NOT_IN_LIST_RE.search(x):
                continue
            if _NEG_RE.search(t[max(0, m.start() - 4):m.start()]):
                continue
            scores = [max(_name_score(n, x) for n in ns) if ns else 0.0 for ns in rnames]
            best = max(range(len(scores)), key=lambda j: scores[j])
            if scores[best] < 0.5:
                problems.append(_p("unknown_cause", x, "说的原因不在事实单的嫌疑人名单里"))
            elif top_only and best != 0 and scores[0] < scores[best] and not dg.get("not_in_list"):
                problems.append(_p("wrong_top_cause", x, "事实单里排第一的不是这个原因"))

    # 4. 推荐测量
    nm = fs.get("next_measurement") or {}
    exp_id = fs.get("experiment_id")
    for item in (cat.get("menus", {}).get(exp_id) or []):
        if item.get("id") == nm.get("id"):
            continue
        for lab in (item.get("label_zh", ""), item.get("label_en", "")):
            lab_s = _squash(lab)
            if len(lab_s) < 4 or lab_s == _squash(nm.get("label", "")):
                continue
            for m in re.finditer(re.escape(lab_s), _squash(t), re.I):
                ctx = _squash(t)[max(0, m.start() - 12): m.start()]
                if _RECOMMEND_RE.search(ctx) and not _NEG_RE.search(ctx):
                    problems.append(_p("wrong_next_measurement", lab, "推荐的下一次测量和事实单不一致"))

    # 5. 覆盖（先抹掉测量名，免得"用天平称挂钩质量"被当成"漏算挂钩质量"的改写）
    labels = [nm.get("label", "")] + [i.get("label_zh", "") for i in cat.get("menus", {}).get(exp_id) or []]
    t_nolabel = t
    for lab in sorted({normalize(x) for x in labels if x}, key=len, reverse=True):
        t_nolabel = t_nolabel.replace(lab, "□")
    coverage = {
        "top_cause": _covers_top(t_nolabel, ranked, rnames, dg),
        "number_evidence": evidence_hit,
        "next_measurement": _covers_next(t, nm),
    }
    msgs = {"top_cause": "没有说出最可能的原因（请照抄事实单里的原因名称）",
            "number_evidence": "没有引用任何数值证据",
            "next_measurement": "没有说出推荐的下一次测量"}
    for k, ok in coverage.items():
        if not ok:
            problems.append(_p(f"missing_{k}", "", msgs[k]))
    seen, uniq = set(), []
    for p in problems:
        key = (p["code"], p["text"])
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return {"ok": not uniq, "problems": uniq, "coverage": coverage}


def _name_score(name: str, x: str) -> float:
    """说法 x 和原因名 name 有多像：x 是名字的一段（名字里带逗号时会被截断）算 1，否则看二字组覆盖率。"""
    if name.isascii():
        return float(name.lower() in x.lower())
    xs = _squash(x)
    if len(xs) >= 3 and xs in _squash(name):
        return 1.0
    return containment(name, x)


def _clause_at(t: str, pos: int) -> str:
    s = max(t.rfind(ch, 0, pos) for ch in ",;。!?\n")
    ends = [i for i in (t.find(ch, pos) for ch in ",;。!?\n") if i >= 0]
    return t[s + 1: min(ends) if ends else len(t)]


def _covers_top(t: str, ranked: list[dict], rnames: list[list[str]], dg: dict) -> bool:
    if not ranked:
        return True
    if dg.get("not_in_list") and _NOT_IN_LIST_RE.search(t):
        return True
    ts = _squash(t).lower()
    for n in rnames[0] + [str(ranked[0].get("cause_id", ""))]:
        if n and _squash(n).lower() in ts:
            return True
    zh = [n for n in rnames[0] if not n.isascii()]
    return any(containment(n, cl) >= 0.6 for n in zh for cl in re.split(r"[,;。!?\n]", t))


def _covers_next(t: str, nm: dict) -> bool:
    if not nm or not (nm.get("label") or nm.get("id")):
        return True
    ts = _squash(t)
    lab = _squash(nm.get("label", ""))
    if (lab and lab in ts) or (nm.get("id") and nm["id"] in t):
        return True
    return bool(lab) and any(containment(lab, cl) >= 0.6 for cl in re.split(r"[。!?\n]", t))


def feedback_zh(result: dict, limit: int = 6) -> str:
    """把校验问题写成给模型改写用的中文反馈。"""
    lines = []
    for p in result.get("problems", [])[:limit]:
        lines.append(f"- {p['msg']}" + (f"：{p['text']}" if p.get("text") else ""))
    return "你的讲解没有通过事实核对，请只用事实单里的数字和名称重写：\n" + "\n".join(lines)
