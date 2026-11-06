"""引导（guide）任务的训练样本：用脚本导师模拟整节实验课，每条样本取其中一轮导师发言作目标。

- 实验：只用训练实验（splits.TRAIN_EXPERIMENTS），真实原因从 allowed_causes 里抽（扣下的原因不出现）；
  事实单也只用 allowed_causes 作候选，免得扣下的原因从事实单 / 诊断选项里漏进来。
  物理组件层的镜像组合暂不使用（摆的镜像是留出实验；组合实验的原因 id 与扣下清单对不上）。
- 学生输入是合成的：预测有时对、有时错（×0.5 / ×2），有时要提示（提示阶梯 1..3）；读数来自 labsim 模拟；
  约 12% 的会话有一行抄写错误（×10），实时核对会报警，学生重读后改正；嫌疑原因有时选对、有时选错。
- 导师发言 = ScriptedTutor.next_message()["text"]（探究式：先预测再测量、一次一个问题、学生选之前不揭示原因）；
  上下文 = tutor.context.build_context(会话, 事实单, 实验卡, phase=当前阶段, student_text=学生刚说的话)，
  与推理时同一个函数，只放精简材料和最近 2 轮摘要，从不放完整对话。
- 先按权重选目标阶段，只模拟到这一阶段为止；事实单（带 EIG）只在诊断及以后的阶段才算。
- 过滤：诊断揭示、讲解两类发言必须通过 tutor/verifier.py；学生给出嫌疑原因之前的发言不得出现原因名称；
  一句话里多个问号先合并（guide_problems / one_question，蒸馏的引导检查也用同一套）。

种子：splits.TASK_CODES 里没有 guide，这里按同样的 SeedSequence 做法用任务码 5（训练 / 验证 / bench 三个命名空间）。
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np

from ..eig import protocol_plan
from ..facts import band_of, build_fact_sheet, dec_for_res
from ..labsim.core import Dataset, Observation
from ..labsim.registry import get_experiment
from ..tutor.context import build_context
from ..tutor.policy import ScriptedTutor
from ..tutor.session import LabSession
from ..tutor.verifier import verify
from ..twin import expected_readings
from .format import Example
from .splits import BASE_SEED, SPLIT_CODES, TRAIN_EXPERIMENTS, allowed_causes
from .tasks_diagnose import sample_cause

TASK_CODE = 5
BUDGET_CHARS = 3000
EIG_SAMPLES = 32
P_SLIP = 0.12
P_HINT = 0.15
P_EXTRA = 0.5

# 目标阶段（类别）权重：识别 / 规划只依赖实验卡，样本高度雷同，压低比例
# 预测 / 测量两类发言要说出下一行的设定值，而 context.py 这两个阶段的材料里没有它（见 guide_problems 的数字检查），
# 多数会被丢掉，所以权重压低，免得白算
PHASE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("identify", 0.03), ("plan", 0.07), ("predict", 0.04), ("measure", 0.04), ("check", 0.16),
    ("diagnose_ask", 0.17), ("diagnose_reveal", 0.17), ("predict_next", 0.08), ("explain", 0.06),
    ("reflect", 0.12),
)
STAGE = {"identify": 0, "plan": 1, "predict": 2, "measure": 3, "check": 3, "diagnose_ask": 4,
         "diagnose_reveal": 4, "predict_next": 5, "explain": 6, "reflect": 7}
FACT_PHASES = ("diagnose_reveal", "explain")  # 这两类发言引用事实单里的数字，必须过校验器


def guide_seed(split: str, index: int, base: int = BASE_SEED) -> int:
    """与 splits.example_seed 同一做法（任务码 5）；split 可为 train / val / bench。"""
    ss = np.random.SeedSequence([int(base), TASK_CODE, SPLIT_CODES[split], int(index)])
    return int(ss.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


_KP_NAMES: dict | None = None


def kp_names() -> dict:
    global _KP_NAMES
    if _KP_NAMES is None:
        try:
            from .. import knowledge

            _KP_NAMES = {k["id"]: k.get("name_zh", k["id"]) for k in knowledge.all_kps()}
        except Exception:  # noqa: BLE001  知识库不可用时不带知识点名称
            _KP_NAMES = {}
    return _KP_NAMES


def load_card(eid: str) -> dict:
    try:
        from .. import knowledge

        return knowledge.load_card(eid) or {}
    except Exception:  # noqa: BLE001
        return {}


# ── 问号、原因名称检查（蒸馏的引导检查也用）────────────────────────────

_Q = re.compile(r"[？?]")


def one_question(text: str) -> str:
    """一段话里多于一个问号时，只留最后一个：行内的换成"，"，行尾的换成"。"。"""
    pos = [m.start() for m in _Q.finditer(text)]
    if len(pos) <= 1:
        return text
    chars = list(text)
    for p in pos[:-1]:
        nxt = text[p + 1] if p + 1 < len(text) else "\n"
        chars[p] = "。" if nxt == "\n" else "，"
    return "".join(chars)


def n_questions(text: str) -> int:
    return len(_Q.findall(text or ""))


_NUM = re.compile(r"\d+(?:\.\d+)?")


def alien_numbers(text: str, prompt_text: str) -> list[str]:
    """发言里 ≥ 10 的数字没有在提示里出现过的（小整数多是序号，不查）。"""
    return [n for n in _NUM.findall(text or "") if float(n) >= 10 and n not in prompt_text]


def guide_problems(text: str, category: str, fs: dict | None, cause_names, revealed: bool,
                   prompt_text: str | None = None) -> list[str]:
    """一轮导师发言的问题清单（空 = 通过）。

    - 最多一个问题；
    - 学生还没给出嫌疑原因（revealed=False）时不得出现任何原因名称；
    - 诊断揭示 / 讲解要通过声明校验器（对照事实单）；
    - 其他阶段（给了 prompt_text 时）：≥ 10 的数字必须在提示里出现过，不教模型编数字。"""
    probs = []
    if not (text or "").strip():
        return ["空发言"]
    if n_questions(text) > 1:
        probs.append(f"问题多于一个（{n_questions(text)}）")
    if not revealed:
        hit = [n for n in cause_names if n and n in text]
        if hit:
            probs.append(f"学生选之前说出了原因：{hit[:2]}")
    if category in FACT_PHASES and fs:
        v = verify(text, fs)
        if not v["ok"]:
            probs.append("校验器不通过：" + "；".join(str(p) for p in v.get("problems", [])[:2]))
    if prompt_text is not None and category not in FACT_PHASES:
        alien = alien_numbers(text, prompt_text)
        if alien:
            probs.append(f"提示里没有的数字：{alien[:3]}")
    return probs


# ── 会话模拟 ──────────────────────────────────────────────────────────


@dataclass
class Turn:
    category: str
    messages: list[dict]
    msg: dict
    fs: dict | None
    revealed: bool
    hint_level: int
    student_text: str


class _Stop(Exception):
    pass


def _reading_text(v: float, res) -> str:
    dec = dec_for_res(float(res)) if res else 3
    return f"{v:.{dec}f}"


def _parse_target(t: str) -> tuple[str, float] | None:
    kind, _, s = str(t).partition("@")
    try:
        return kind, float(s)
    except ValueError:
        return None


class GuideSim:
    """一节模拟实验课。run(target) 模拟到目标阶段结束为止，返回收集到的轮次。"""

    def __init__(self, seed: int, n_samples: int = EIG_SAMPLES):
        self.rng = np.random.default_rng(seed)
        self.n_samples = n_samples
        rng = self.rng
        self.eid = TRAIN_EXPERIMENTS[int(rng.integers(len(TRAIN_EXPERIMENTS)))]
        self.exp = get_experiment(self.eid)
        self.allowed = allowed_causes(self.exp)
        self.cause = sample_cause(rng, self.exp, self.allowed)
        self.plan = list(protocol_plan(self.exp, "standard"))
        self.ds = self.exp.simulate(rng, self.cause, self.plan)
        self.card = load_card(self.eid)
        self.s = LabSession(experiment_id=self.eid, card=self.card)
        self.tutor = ScriptedTutor(self.s, {}, self.card, experiment=self.exp, kp_names=kp_names())
        self.names = [self.exp.cause(c).name_zh for c in self.allowed]
        self.turns: list[Turn] = []
        self.revealed = False
        self.target_stage = 99

    # ---- 工具 ----

    def _gate(self, category: str) -> None:
        if STAGE[category] > self.target_stage:
            raise _Stop

    def emit(self, category: str, student_text: str) -> dict:
        self._gate(category)
        msg = self.tutor.next_message()
        fs = self.tutor.fs or None
        ctx = build_context(self.s, fs or {}, self.card, BUDGET_CHARS, phase=self.s.phase,
                            student_text=student_text, kp_names=kp_names())
        self.turns.append(Turn(category, ctx, msg, fs, self.revealed, self.s.hint_level, student_text))
        return msg

    def maybe_hints(self, category: str, last: str, max_hints: int = 2) -> tuple[dict, str]:
        msg = self.emit(category, last)
        k = 0
        while k < max_hints and self.rng.random() < P_HINT:
            self.tutor.receive("提示")
            last = "提示"
            msg = self.emit(category, last)
            k += 1
        return msg, last

    def fact_sheet(self) -> dict:
        ds = Dataset(self.exp.id, [Observation.from_dict(o) for o in self.s.observations], self.ds.truth,
                     {"protocol": "standard"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fs = build_fact_sheet(self.exp, ds, np.random.default_rng(int(self.rng.integers(0, 2**62))),
                                  n_samples=self.n_samples, cause_ids=self.allowed)
        self.tutor.fs = fs
        self.s.fact_sheet = fs
        return fs

    # ---- 各阶段 ----

    def measure_rows(self, obs_list: list[Observation], last: str, start_order: int) -> str:
        exp, rng = self.exp, self.rng
        plan = [(o.kind, o.setting) for o in obs_list]
        twin = expected_readings(exp, exp.random_only_id, None, plan, start_order=start_order)
        slip_row = int(rng.integers(len(obs_list))) if rng.random() < P_SLIP else -1
        for i, (o, tw) in enumerate(zip(obs_list, twin)):
            mmsg = self.emit("measure", last)
            rd = exp.obs_kinds[o.kind].readings[0]
            true_vals = [float(v) for v in o.values]
            vals = list(true_vals)
            if i == slip_row:
                vals[0] = vals[0] * 10.0
            text = _reading_text(vals[0], rd.resolution)
            idx = self.s.add_observation({"kind": o.kind, "setting": o.setting, "values": vals})
            self.s.add_turn(mmsg["text"], text)
            self.s.advance("check")
            r0 = tw["readings"][0]

            def record(v):
                z = (v - r0["mean"]) / r0["sd"] if r0["sd"] > 0 else 0.0
                self.s.record_check(idx, {"value": float(_reading_text(v, rd.resolution)), "unit": rd.unit,
                                          "predicted": float(f"{r0['mean']:.4g}"), "uncertainty": float(f"{r0['sd']:.2g}"),
                                          "z": round(float(z), 2), "band": band_of(z)})
            record(vals[0])
            msg = self.emit("check", text)
            if msg.get("options") and "重读这一行" in msg["options"]:
                if i == slip_row:
                    ans = "重读这一行"
                    self.s.observations[idx]["values"] = true_vals
                    record(true_vals[0])
                    self.s.add_turn(msg["text"], ans)
                    msg = self.emit("check", ans)
                    ans = "继续"
                else:
                    ans = "保留，继续"
            else:
                ans = "继续"
            self.tutor.receive(ans)
            last = ans
            nxt = self.tutor.suggest_next_phase()
            self.s.advance(nxt)
        return last

    def diagnose_round(self, last: str) -> str:
        self._gate("diagnose_ask")
        fs = self.fact_sheet()
        msg, last = self.maybe_hints("diagnose_ask", last)
        if msg.get("target"):  # 还在问学生的猜测
            ranked = (fs.get("diagnosis") or {}).get("ranked") or []
            top = ranked[0].get("name") if ranked else None
            opts = msg.get("options") or self.names
            guess = top if (top and self.rng.random() < 0.5) else opts[int(self.rng.integers(len(opts)))]
            self.tutor.receive(guess)
            last = guess
            self.revealed = True
            msg = self.emit("diagnose_reveal", last)
        else:  # 提示到第 3 级时导师直接揭示
            self.revealed = True
            self.turns[-1].category = "diagnose_reveal"
            self.turns[-1].revealed = True
        return last

    def run(self, target: str) -> list[Turn]:
        self.target_stage = STAGE[target]
        s, tutor, rng = self.s, self.tutor, self.rng
        try:
            self.emit("identify", "")
            tutor.receive("对")
            s.advance("plan")
            msg, last = self.maybe_hints("plan", "对")
            opts = msg.get("options") or []
            ans = msg.get("answer") if (msg.get("answer") and rng.random() < 0.7) else (
                opts[int(rng.integers(len(opts)))] if opts else "不知道")
            tutor.receive(ans)
            s.advance("predict")
            msg, last = self.maybe_hints("predict", ans)
            pt = _parse_target(msg.get("target", ""))
            if pt and pt[0] in self.exp.obs_kinds:
                mean = expected_readings(self.exp, self.exp.random_only_id, None, [pt])[0]["readings"][0]["mean"]
                u = rng.random()
                f = (1.0 + 0.1 * rng.standard_normal()) if u < 0.6 else (0.5 if u < 0.8 else 2.0)
                ans = f"{mean * f:.3g}"
            else:
                ans = "不知道"
            tutor.receive(ans)
            s.advance("measure")
            last = self.measure_rows(self.ds.observations, ans, 0)
            last = self.diagnose_round(last)
            fs = self.tutor.fs
            nm = fs.get("next_measurement") or {}
            if nm.get("label") and (target == "predict_next" or rng.random() < P_EXTRA):
                tutor.receive("做这个测量")
                tutor.start_next_measurement()
                s.advance("predict")
                msg = self.emit("predict_next", "做这个测量")
                opts = msg.get("options") or ["说不准"]
                ans = opts[int(rng.integers(len(opts)))]
                tutor.receive(ans)
                s.advance("measure")
                plan = tutor.plan_rows()
                ext = self.exp.extend(rng, Dataset(self.exp.id, list(self.ds.observations), self.ds.truth, {}), plan)
                new = ext.observations[len(self.ds.observations):]
                last = self.measure_rows(new, ans, len(self.ds.observations))
                s.pending_measurement = None
                self._gate("diagnose_ask")
                self.revealed = False
                last = self.diagnose_round(last)
                self.revealed = True
                tutor.receive("看讲解")
                last = "看讲解"
            else:
                last = "直接看讲解" if nm.get("label") else "看讲解"
                tutor.receive(last)
            s.advance("explain")
            self.emit("explain", last)
            tutor.receive("明白了，去写结论")
            s.advance("reflect")
            self.maybe_hints("reflect", "明白了，去写结论")
        except _Stop:
            pass
        return self.turns


def _pick_category(rng: np.random.Generator) -> str:
    names = [c for c, _ in PHASE_WEIGHTS]
    w = np.array([p for _, p in PHASE_WEIGHTS], float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def pick_turn(seed: int, n_samples: int = EIG_SAMPLES) -> tuple[GuideSim, Turn, str] | None:
    """一个种子 → (模拟的会话, 选中的一轮, 规范化后的导师发言)；蒸馏任务也用它取提示和参照。"""
    pick_rng = np.random.default_rng([seed, 1])
    target = _pick_category(pick_rng)
    sim = GuideSim(seed, n_samples)
    turns = sim.run(target)
    pool = [t for t in turns if t.category == target] or turns
    if not pool:
        return None
    t = pool[int(pick_rng.integers(len(pool)))]
    return sim, t, one_question(t.msg["text"])


def prompt_text(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages if isinstance(m["content"], str))


def build_from_seed(seed: int, split: str, index: int, n_samples: int = EIG_SAMPLES) -> Example | None:
    """一个种子 → 一条引导样本（过不了检查返回 None，由调用方计入丢弃数）。"""
    got = pick_turn(seed, n_samples)
    if got is None:
        return None
    sim, t, text = got
    if guide_problems(text, t.category, t.fs, sim.names, t.revealed, prompt_text(t.messages)):
        return None
    meta = {
        "experiment_id": sim.eid, "split": split, "seed": seed, "category": t.category,
        "phase": t.msg.get("phase"), "expects": t.msg.get("expects"), "options": t.msg.get("options"),
        "hint_level": t.hint_level, "truth_cause": sim.cause, "revealed": t.revealed,
        "student_text": t.student_text, "n_turns_simulated": len(sim.turns),
        "causes": [{"id": c, "name": n} for c, n in zip(sim.allowed, sim.names)],
    }
    ex = Example(f"guide-{sim.eid}-{split}-{index:06d}", "guide", "template",
                 t.messages + [{"role": "assistant", "content": text}], meta)
    ex.validate()
    return ex


def build_example(index: int, split: str, base_seed: int | None = None,
                  n_samples: int = EIG_SAMPLES) -> Example | None:
    seed = guide_seed(split, index, BASE_SEED if base_seed is None else base_seed)
    return build_from_seed(seed, split, index, n_samples)
