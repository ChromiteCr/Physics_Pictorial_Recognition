"""探究式实验会话的状态机（可 JSON 往返）。

阶段：identify → plan → predict → measure → check → diagnose → explain → reflect。
- predict 可跳过（plan → measure）；每测一行数据就 check 一次，测完再 diagnose；
- diagnose 之后按 EIG 推荐的追加测量回到 predict / measure（最多 3 轮），或进入 explain；
- reflect 之后可以改正方法重测（→ plan），或换一个实验（→ identify）。
会话只存精简的轮次摘要，不存完整对话（上下文组装见 context.py）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

PHASES: tuple[str, ...] = ("identify", "plan", "predict", "measure", "check", "diagnose", "explain", "reflect")

TRANSITIONS: dict[str, tuple[str, ...]] = {
    "identify": ("plan",),
    "plan": ("predict", "measure", "identify"),
    "predict": ("measure",),
    "measure": ("check",),
    "check": ("measure", "diagnose"),
    "diagnose": ("predict", "measure", "explain"),
    "explain": ("reflect",),
    "reflect": ("plan", "identify"),
}

MAX_HINT = 3
MAX_TURNS = 40  # 只留最近这么多条摘要
SUMMARY_CHARS = 60
MAX_EIG_ROUNDS = 3


class IllegalTransition(ValueError):
    pass


def _clip(text: str, n: int = SUMMARY_CHARS) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _plain(v):
    """转成 JSON 原生类型（numpy 标量、元组等）。"""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if hasattr(v, "item") and callable(v.item):  # numpy 标量
        return v.item()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


@dataclass
class LabSession:
    experiment_id: str | None = None
    protocol: str = "standard"
    phase: str = "identify"
    card: dict | None = None
    observations: list[dict] = field(default_factory=list)  # Observation.to_dict() 形状
    checks: list[dict] = field(default_factory=list)  # 每行的实时核对（由 twin.py 算好后存进来）
    student_predictions: list[dict] = field(default_factory=list)
    fact_sheet: dict | None = None
    hint_level: int = 0  # 当前问题的提示阶梯 0..3，换阶段清零
    hints_used: int = 0
    kp_touched: list[str] = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)
    eig_round: int = 0  # 已经做了几轮"追加测量"
    pending_measurement: dict | None = None  # {"id", "label"}：推荐了、还没做完的追加测量
    version: int = 1

    def __post_init__(self):
        if self.phase not in PHASES:
            raise ValueError(f"unknown phase {self.phase}")

    # ---- 阶段 ----

    def can_advance(self, to: str) -> bool:
        return to in TRANSITIONS.get(self.phase, ())

    def advance(self, to: str) -> "LabSession":
        if to not in PHASES:
            raise ValueError(f"unknown phase {to}")
        if not self.can_advance(to):
            raise IllegalTransition(f"{self.phase} → {to} 不合法；可去：{TRANSITIONS[self.phase]}")
        if self.phase == "diagnose" and to in ("predict", "measure"):
            if self.eig_round >= MAX_EIG_ROUNDS:
                raise IllegalTransition(f"追加测量最多 {MAX_EIG_ROUNDS} 轮")
            self.eig_round += 1
        if self.phase == "reflect" and to in ("plan", "identify"):
            self.eig_round = 0
            self.pending_measurement = None
        self.phase = to
        self.hint_level = 0
        return self

    def restart(self) -> "LabSession":
        """换实验：清空数据和预测，保留已接触过的知识点。"""
        kp = list(self.kp_touched)
        self.__init__()  # type: ignore[misc]
        self.kp_touched = kp
        return self

    # ---- 数据 ----

    def add_observation(self, kind, setting: float | None = None, values=()) -> int:
        """记一行数据，返回它的序号。也可直接传 labsim 的 Observation 或其 dict。"""
        if hasattr(kind, "to_dict"):
            d = kind.to_dict()
        elif isinstance(kind, dict):
            d = dict(kind)
        else:
            vals = values if isinstance(values, (list, tuple)) else [values]
            d = {"kind": kind, "setting": setting, "values": vals}
        d = {"kind": str(d["kind"]), "setting": float(d["setting"] or 0.0),
             "values": [float(v) for v in d["values"]]}
        self.observations.append(d)
        return len(self.observations) - 1

    def record_check(self, index: int, check: dict) -> None:
        """存一行的实时核对结果，例如 {"predicted": 4.4, "uncertainty": 0.2, "value": 4.5, "unit": "cm", "z": 0.5, "band": "consistent"}。"""
        if not 0 <= index < len(self.observations):
            raise IndexError(index)
        self.checks = [c for c in self.checks if c.get("index") != index]
        self.checks.append(_plain({"index": index, **check}))

    def check_for(self, index: int) -> dict | None:
        for c in self.checks:
            if c.get("index") == index:
                return c
        return None

    def record_prediction(self, target: str, value, unit: str = "", kind: str = "number") -> dict:
        """学生的预测。kind: number（数值）| choice（选项，如嫌疑原因）| text。target 例：'main@200'、'cause@1'。"""
        if kind == "number":
            value = float(value)
        p = {"target": str(target), "value": _plain(value), "unit": unit, "kind": kind,
             "phase": self.phase, "round": self.eig_round}
        self.student_predictions.append(p)
        return p

    def prediction_for(self, target: str) -> dict | None:
        for p in reversed(self.student_predictions):
            if p["target"] == target:
                return p
        return None

    # ---- 提示阶梯、知识点、轮次 ----

    def request_hint(self) -> int:
        if self.hint_level < MAX_HINT:
            self.hint_level += 1
            self.hints_used += 1
        return self.hint_level

    def touch_kp(self, *kp_ids: str) -> None:
        for k in kp_ids:
            if k and k not in self.kp_touched:
                self.kp_touched.append(k)

    def add_turn(self, tutor: str = "", student: str = "") -> dict:
        t = {"i": (self.turns[-1]["i"] + 1) if self.turns else 0, "phase": self.phase,
             "tutor": _clip(tutor), "student": _clip(student)}
        self.turns.append(t)
        del self.turns[:-MAX_TURNS]
        return t

    def last_turns(self, n: int = 2) -> list[dict]:
        return self.turns[-n:] if n > 0 else []

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "LabSession":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "LabSession":
        return cls.from_dict(json.loads(s))
