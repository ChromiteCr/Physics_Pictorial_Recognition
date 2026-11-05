"""脚本化导师：不用模型、按探究式教学法确定性地给出下一句引导（也用来生成 guide 训练轨迹）。

教学法：先预测再测量 → 每测一行就和孪生对照 → 找原因时先让学生自己选嫌疑人，再揭示引擎排名和证据
→ 推荐最有信息量的下一次测量并给出各原因的预测值 → 讲解关联知识点 → 学生自己写结论。
提示阶梯 0..3：1 = 概念提醒，2 = 指向证据，3 = 接近答案。
另有 guarded_explanation：模型讲解 → 校验 → 改写一次 → 仍不通过就换模板（设计 §5.8）。
"""

from __future__ import annotations

from . import templates as tpl
from .context import build_context
from .session import MAX_EIG_ROUNDS, LabSession
from .templates import T, fmt
from .verifier import feedback_zh, verify


class ScriptedTutor:
    """ScriptedTutor(session, fact_sheet, card).next_message() → {phase, text, expects, options?, ...}。

    experiment: labsim 的 Experiment（或有同样属性的对象），用来读标准方案的控制量和读数分辨率；
    不给就按 session.experiment_id 从注册表懒加载，失败时只用实验卡。
    """

    def __init__(self, session: LabSession, fact_sheet: dict | None = None, card: dict | None = None,
                 experiment=None, kp_names: dict | None = None):
        self.s = session
        self.fs = fact_sheet if fact_sheet is not None else (session.fact_sheet or {})
        self.card = card if card is not None else (session.card or {})
        self._exp = experiment
        self.kp_names = kp_names

    # ---- 实验信息 ----

    def experiment(self):
        if self._exp is None and self.s.experiment_id:
            try:
                from ..labsim.registry import get_experiment

                self._exp = get_experiment(self.s.experiment_id)
            except Exception:  # noqa: BLE001  注册表不可用时只用实验卡
                self._exp = False
        return self._exp or None

    def plan_rows(self) -> list[tuple[str, float]]:
        exp = self.experiment()
        pm = self.s.pending_measurement
        try:
            if pm and pm.get("id"):
                return list(exp.menu_item(pm["id"]).plan) if exp else []
            return list(exp.protocols[self.s.protocol].plan) if exp else []
        except (KeyError, AttributeError):
            return []

    def kind_info(self, kind: str) -> dict:
        exp = self.experiment()
        ok = getattr(exp, "obs_kinds", {}).get(kind) if exp else None
        if ok is not None and ok.readings:
            r = ok.readings[0]
            return {"setting_label": ok.setting_label_zh, "setting_unit": ok.setting_unit,
                    "reading_label": r.label_zh, "reading_unit": r.unit, "resolution": r.resolution,
                    "obs_label": ok.label_zh}
        cols = (self.card.get("data_table") or {}).get("columns") or []
        return {"setting_label": str(cols[0]) if cols else "控制量", "setting_unit": "",
                "reading_label": str(cols[1]) if len(cols) > 1 else "读数", "reading_unit": "",
                "resolution": None, "obs_label": ""}

    def name(self) -> str:
        exp = self.experiment()
        return self.card.get("name_zh") or self.fs.get("experiment_name") or (exp.name_zh if exp else "") or ""

    def ranked(self) -> list[dict]:
        return list((self.fs.get("diagnosis") or {}).get("ranked") or [])

    def _hint(self, phase: str, **kw) -> str:
        lv = self.s.hint_level
        return ("\n" + tpl.hint(phase, lv, **kw)) if lv > 0 and tpl.hint(phase, lv, **kw) else ""

    # ---- 主入口 ----

    def next_message(self) -> dict:
        return getattr(self, f"_{self.s.phase}")()

    def _identify(self) -> dict:
        name = self.name()
        if self.s.experiment_id and name:
            return {"phase": "identify", "text": T["identify_known"].format(name=name),
                    "expects": "choice", "options": ["对", "不是"]}
        return {"phase": "identify", "text": T["identify_unknown"], "expects": "free"}

    def _plan(self) -> dict:
        variables = self.card.get("variables") or []
        text = T["plan"].format(name=self.name(), goal=self.card.get("goal", "按实验卡完成测量。"))
        control = next((v for v in variables if v.get("role") == "control"), None)
        options = [f"{v.get('name', '')} {v.get('symbol', '')}".strip() for v in variables if v.get("role") != "derived"]
        text += self._hint("plan", control=(control or {}).get("name", ""))
        msg = {"phase": "plan", "text": text, "expects": "choice" if options else "free"}
        if options:
            msg["options"] = options
            if control:
                msg["answer"] = f"{control.get('name', '')} {control.get('symbol', '')}".strip()
        return msg

    def _predict(self) -> dict:
        pm = self.s.pending_measurement
        nm = self.fs.get("next_measurement") or {}
        if pm and nm.get("predicted") and nm.get("id") == pm.get("id"):
            names = {r.get("cause_id"): r.get("name") for r in self.ranked()}
            preds = sorted(nm["predicted"], key=lambda p: (float(p.get("value", 0)), p.get("cause_id", "")))
            opts = [f"约 {tpl.with_unit(p['value'], p.get('unit', ''))}（{names.get(p.get('cause_id'), p.get('cause_id'))}）"
                    for p in preds]
            text = T["predict_next"].format(label=nm.get("label", pm.get("label", "")), options="；".join(opts))
            return {"phase": "predict", "text": text, "expects": "choice", "options": opts + ["说不准"],
                    "target": f"{pm.get('id')}@{self.s.eig_round}"}
        rows = [r for r in self.plan_rows() if r[0] == "main"] or self.plan_rows()
        if not rows:
            return {"phase": "predict", "text": "动手之前先预测：你觉得测量结果会是什么样？写下你的估计。",
                    "expects": "prediction", "target": "result"}
        kind, setting = rows[-1]
        info = self.kind_info(kind)
        unit_hint = f"（单位 {info['reading_unit']}）" if info["reading_unit"] else ""
        text = T["predict_first"].format(setting_label=info["setting_label"] or "控制量",
                                         setting=tpl.with_unit(setting, info["setting_unit"]),
                                         reading_label=info["reading_label"], unit_hint=unit_hint)
        kp = self.card.get("knowledge_points") or []
        text += self._hint("predict", kp=tpl.kp_name(kp[0], self.kp_names) if kp else "相关定律",
                           formula=self.card.get("formula", ""))
        return {"phase": "predict", "text": text, "expects": "prediction",
                "target": f"{kind}@{fmt(setting)}", "unit": info["reading_unit"]}

    def _measure(self) -> dict:
        rows = self.plan_rows()
        pm = self.s.pending_measurement
        done = len(self.s.observations) - (pm.get("start", 0) if pm else 0)
        if 0 <= done < len(rows):
            kind, setting = rows[done]
            info = self.kind_info(kind)
            res = info["resolution"]
            unit_hint = ""
            if info["reading_unit"]:
                unit_hint = f"（单位 {info['reading_unit']}" + (
                    f"，记到 {tpl.with_unit(res, info['reading_unit'])}" if res else "") + "）"
            if kind == "main" or info["setting_label"]:
                text = T["measure_row"].format(i=done + 1, setting_label=info["setting_label"] or "控制量",
                                               setting=tpl.with_unit(setting, info["setting_unit"]),
                                               reading_label=info["reading_label"], unit_hint=unit_hint)
            else:
                text = T["measure_extra"].format(label=(pm or {}).get("label") or info["obs_label"])
                text = text.rstrip("。") + f"：{info['reading_label']}{unit_hint}。"
            return {"phase": "measure", "text": text, "expects": "reading",
                    "target": {"kind": kind, "setting": float(setting)}, "unit": info["reading_unit"]}
        return {"phase": "measure", "text": T["measure_done"], "expects": "choice", "options": ["开始分析"]}

    def _check(self) -> dict:
        i = len(self.s.observations) - 1
        if i < 0:
            return {"phase": "check", "text": "还没有数据，先测一行吧。", "expects": "choice", "options": ["去测量"]}
        c = self.s.check_for(i)
        obs = self.s.observations[i]
        unit = (c or {}).get("unit", "")
        val = tpl.with_unit((c or {}).get("value", obs["values"][0] if obs["values"] else 0.0), unit)
        if not c or c.get("predicted") is None:
            return {"phase": "check", "text": T["check_recorded"].format(i=i + 1, value=val),
                    "expects": "choice", "options": ["继续"]}
        pred = fmt(c["predicted"])
        if c.get("uncertainty") is not None:
            pred += f" ± {fmt(c['uncertainty'])}"
        pred = f"{pred} {unit}".strip()
        band = c.get("band", "consistent")
        kw = dict(i=i + 1, value=val, pred=pred, z=fmt(c.get("z", 0.0), 2), band=tpl.BAND_ZH.get(band, band))
        if band == "consistent":
            return {"phase": "check", "text": T["check_ok"].format(**kw), "expects": "choice", "options": ["继续"]}
        return {"phase": "check", "text": T["check_warn"].format(**kw), "expects": "choice",
                "options": ["重读这一行", "保留，继续"]}

    def _diagnose(self) -> dict:
        ranked = self.ranked()
        if not ranked:
            return {"phase": "diagnose", "text": "数据还不够，先多测几组再分析。", "expects": "choice",
                    "options": ["继续测量"]}
        top = ranked[0]
        target = f"cause@{self.s.eig_round}"
        guess = self.s.prediction_for(target)
        sig = tpl.signatures_zh(self.fs)
        if guess is None and self.s.hint_level < 3:
            # 选项按 cause_id 排序，不按概率，免得泄露排名
            opts = [r.get("name") for r in sorted(ranked, key=lambda r: r.get("cause_id", ""))]
            text = T["diagnose_ask"].format(signatures=sig)
            ev = top.get("evidence") or f"数据特征是{sig}"
            text += self._hint("diagnose", signatures=sig, evidence=ev, top=top.get("name"))
            return {"phase": "diagnose", "text": text, "expects": "choice", "options": opts, "target": target,
                    "answer": top.get("name")}
        parts = []
        if guess is not None:
            g = guess["value"]
            parts.append(f"你选了「{g}」。" + ("和引擎的判断一致。" if g == top.get("name")
                                              else f"引擎认为更可能的是「{top.get('name')}」。"))
        parts += [tpl.block_cause(self.fs), tpl.block_evidence(self.fs), tpl.block_next(self.fs)]
        nm = self.fs.get("next_measurement")
        if nm and nm.get("label") and self.s.eig_round < MAX_EIG_ROUNDS:
            opts = ["做这个测量", "直接看讲解"]
        else:
            opts = ["看讲解"]
        return {"phase": "diagnose", "text": "\n".join(p for p in parts if p), "expects": "choice", "options": opts}

    def _explain(self) -> dict:
        return {"phase": "explain", "text": tpl.explanation_text(self.fs, self.kp_names), "expects": "choice",
                "options": ["明白了，去写结论", "还有疑问"],
                "knowledge_points": list(self.fs.get("knowledge_points") or [])}

    def _reflect(self) -> dict:
        text = T["reflect"]
        for p in reversed(self.s.student_predictions):
            if p.get("kind") != "number" or "@" not in p["target"]:
                continue
            kind, _, setting = p["target"].partition("@")
            hit = next((o for o in self.s.observations if o["kind"] == kind and fmt(o["setting"]) == setting), None)
            if hit and hit["values"]:
                unit = p.get("unit", "")
                text += "\n" + T["reflect_pred"].format(pred=tpl.with_unit(p["value"], unit),
                                                        actual=tpl.with_unit(hit["values"][0], unit))
                break
        text += self._hint("reflect")
        return {"phase": "reflect", "text": text, "expects": "free"}

    # ---- 学生作答 ----

    def receive(self, answer) -> None:
        """记下学生对当前这句引导的回答（不自动换阶段；换阶段用 suggest_next_phase + session.advance）。"""
        msg = self.next_message()
        a = str(answer).strip()
        if a in ("提示", "hint", "?", "？"):
            self.s.request_hint()
            self.s.add_turn(msg["text"], "（要提示）")
            return
        if msg["expects"] == "prediction":
            try:
                self.s.record_prediction(msg["target"], float(a), msg.get("unit", ""))
            except ValueError:
                self.s.record_prediction(msg["target"], a, kind="text")
        elif msg["expects"] == "choice" and msg.get("target"):
            self.s.record_prediction(msg["target"], a, kind="choice")
        elif msg["expects"] == "reading":
            t = msg["target"]
            self.s.add_observation(t["kind"], t["setting"], [float(a)])
        if msg["phase"] == "explain":
            self.s.touch_kp(*[k for k in self.fs.get("knowledge_points") or [] if isinstance(k, str)])
        self.s.add_turn(msg["text"], a)

    def start_next_measurement(self) -> dict | None:
        """学生同意做推荐的追加测量：记下待做项（start = 当前数据行数），返回它。"""
        nm = self.fs.get("next_measurement")
        if not nm or not nm.get("id"):
            return None
        self.s.pending_measurement = {"id": nm["id"], "label": nm.get("label", ""),
                                      "start": len(self.s.observations)}
        return self.s.pending_measurement

    def suggest_next_phase(self) -> str:
        p = self.s.phase
        if p == "check":
            pm = self.s.pending_measurement
            done = len(self.s.observations) - (pm.get("start", 0) if pm else 0)
            return "measure" if done < len(self.plan_rows()) else "diagnose"
        if p == "diagnose":
            return "predict" if self.s.pending_measurement and self.s.eig_round < MAX_EIG_ROUNDS else "explain"
        return {"identify": "plan", "plan": "predict", "predict": "measure", "measure": "check",
                "explain": "reflect", "reflect": "plan"}[p]


def guarded_explanation(model, session: LabSession, fact_sheet: dict, card: dict | None = None,
                        kp_names: dict | None = None, catalog: dict | None = None,
                        max_rewrites: int = 1, budget_chars: int = 3000) -> dict:
    """模型写讲解 → 校验 → 不通过就带着问题改写一次 → 还不通过就用模板。

    返回 {text, source: model|model_rewrite|template, verify, attempts, model_problems?, error?}。
    """
    template = tpl.explanation_text(fact_sheet, kp_names)
    out: dict = {"attempts": 0}
    if model is not None:
        messages = build_context(session, fact_sheet, card, budget_chars, phase="explain", kp_names=kp_names)
        try:
            for attempt in range(max_rewrites + 1):
                r = model.chat(messages)
                text = (r.get("text") or "").strip()
                v = verify(text, fact_sheet, catalog)
                out["attempts"] = attempt + 1
                if v["ok"]:
                    return {**out, "text": text, "source": "model" if attempt == 0 else "model_rewrite",
                            "verify": v}
                out["model_problems"] = v["problems"]
                messages = messages + [{"role": "assistant", "content": text},
                                       {"role": "user", "content": feedback_zh(v)}]
        except Exception as e:  # noqa: BLE001  模型出错一律回退模板
            out["error"] = f"{type(e).__name__}: {e}"
    return {**out, "text": template, "source": "template", "verify": verify(template, fact_sheet, catalog)}
