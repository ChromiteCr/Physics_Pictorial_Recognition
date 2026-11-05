"""② 实验设计：实验卡 → 先写下预测 → 孪生预测表、不确定度预算、本方案分不开的原因、建议的测量范围。

孪生的预测数字在学生写下自己的预测（或明确跳过）之前不显示，"先预测再测量"才有意义。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lab_detective.tutor import ScriptedTutor

from . import catalog, lab, state
from .theme import card as draw_card, esc, formula


@st.cache_data(show_spinner=False, max_entries=64)
def _budget(key: str, protocol: str) -> dict:
    from lab_detective.uncertainty import budget

    import numpy as np

    return budget(catalog.get_experiment(key), protocol, n=200, rng=np.random.default_rng(0))


@st.cache_data(show_spinner=False, max_entries=64)
def _separability(key: str, protocol: str) -> list[dict]:
    return lab.separability(catalog.get_experiment(key), protocol)


@st.cache_data(show_spinner=False, max_entries=64)
def _range(key: str, protocol: str) -> dict:
    return lab.range_advice(catalog.get_experiment(key), protocol)


def _card_section(exp, card: dict, key: str) -> None:
    body = []
    if card.get("goal"):
        body.append(f"<p><b>目的</b>：{esc(card['goal'])}</p>")
    if card.get("principle"):
        body.append(f"<p><b>原理</b>：{esc(card['principle'])}</p>")
    if catalog.is_other(key):
        body.append(f"<p><b>孪生来源</b>：{esc(catalog.backend_name(key))}</p>")
    draw_card(card.get("name_zh") or exp.name_zh, "".join(body) or "<p>（没有实验卡）</p>")
    if card.get("formula"):
        formula(card["formula"])
    with st.expander("器材与课本步骤"):
        if card.get("apparatus"):
            st.markdown("**器材**：" + "、".join(card["apparatus"]))
        for i, s in enumerate(card.get("steps") or [], 1):
            st.markdown(f"{i}. {s}")
        for s in card.get("safety") or []:
            st.caption(f"安全：{s}")


def _protocol_section(exp) -> str:
    pids = list(exp.protocols)
    cur = st.session_state.get("protocol")
    pick = st.selectbox("测量方案 / 仪器配置", pids, index=pids.index(cur) if cur in pids else 0,
                        format_func=lambda p: exp.protocols[p].label_zh, key="protocol_pick")
    if pick != cur:
        state.set_protocol(pick)
    return st.session_state["protocol"]


def _prediction_section(exp, card: dict) -> None:
    sess = state.lab()
    sess.phase = "predict"
    tutor = ScriptedTutor(sess, None, card, experiment=exp)
    msg = tutor.next_message()
    st.markdown("#### 先预测，再测量")
    st.markdown(esc(msg["text"]))
    target, unit = msg.get("target", "result"), msg.get("unit", "")
    done = sess.prediction_for(target)
    if done is not None:
        st.success(f"你的预测：{done['value']} {done.get('unit', '')}".strip())
        return
    c1, c2, c3 = st.columns([2, 1, 1])
    val = c1.number_input(f"你的预测{f'（{unit}）' if unit else ''}", value=None, format="%g", key="pred_value")

    def _save() -> None:
        v = st.session_state.get("pred_value")
        if v is not None:
            sess.record_prediction(target, float(v), unit)
            sess.add_turn(msg["text"], str(v))

    c2.button("记下预测", key="pred_save", disabled=val is None, on_click=_save)
    c3.button("跳过预测", key="pred_skip", on_click=skip_prediction)


def skip_prediction() -> None:
    st.session_state.pred_skipped = True


def _twin_table(exp, protocol: str) -> None:
    from lab_detective.twin import instantiate

    rows = instantiate(exp).predict_table(protocol)
    out = []
    for r in rows:
        for rd in r["readings"]:
            u = rd["unit"]
            out.append({
                "序号": r["order"] + 1,
                r["setting_label"] or "设定": f"{lab.fmt(r['setting'])} {r['setting_unit']}".strip(),
                "读数": rd["label"],
                "孪生预期": f"{lab.fmt(rd['mean'])} ± {lab.fmt(rd['sd'], 2)} {u}".strip(),
                "其中仪器读数误差": f"{lab.fmt(rd['sd_reading'], 2)} {u}".strip(),
                "其中参数未知": f"{lab.fmt(rd['sd_param'], 2)} {u}".strip(),
            })
    st.markdown("#### 孪生的预测表（实验前）")
    st.caption("“参数未知”是装置本身还不知道的量（例如弹簧的劲度系数）带来的宽度；测了前几组之后，"
               "③ 里的实时对照会用你的数据把它定下来，预测会变窄。")
    st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)


def _budget_section(key: str, protocol: str) -> None:
    b = _budget(key, protocol)
    st.markdown("#### 不确定度预算")
    st.markdown(esc(b.get("summary_zh", "")))
    s = b.get("slope")
    if s and s.get("sources"):
        df = pd.DataFrame([{"来源": it["label_zh"], "占斜率方差": f"{it['share'] * 100:.0f}%"}
                           for it in s["sources"] if it["share"] >= 0.005])
        st.dataframe(df, hide_index=True, use_container_width=True)
    for n in b.get("notes", []):
        st.caption(n)


def _separability_section(key: str, protocol: str) -> None:
    rows = _separability(key, protocol)
    st.markdown("#### 本方案分不开的原因")
    if not rows:
        st.success("按这个方案模拟，各候选原因两两都能分开。")
        return
    st.caption("每对原因各模拟 60 份数据，只看这一对时判对的比例；50% 等于瞎猜。")
    for r in rows[:6]:
        word = "几乎分不开" if r["status"] == "confused" else "只能部分分开"
        line = f"- 「{r['name_a']}」和「{r['name_b']}」：判对 {r['acc'] * 100:.0f}%，{word}"
        if r.get("best_label"):
            line += f"；追加「{r['best_label']}」后可达 {r['best_item_acc'] * 100:.0f}%"
        st.markdown(line)


def _range_section(key: str, protocol: str) -> None:
    a = _range(key, protocol)
    st.markdown("#### 建议的测量范围")
    su, ru = a["setting_unit"], a["reading_unit"]
    lines = [f"- {a['setting_label'] or '控制量'}：{lab.fmt(a['lo'])} ~ {lab.fmt(a['hi'])} {su}，共 {a['n']} 组。"]
    if a["pred_lo"] is not None:
        lines.append(f"- 预计{a['reading_label']}：{lab.fmt(a['pred_lo'])} ~ {lab.fmt(a['pred_hi'])} {ru}"
                     f"（仪器分辨率 {lab.fmt(a['resolution'])} {ru}）。")
    if a["min_ratio"]:
        lines.append(f"- 最小的读数约是分辨率的 {a['min_ratio']:.0f} 倍，取整带来的相对误差约 "
                     f"{a['rel_round_pct']:.1f}%" + ("；可以适当加大最小的设定值。" if a["min_ratio"] < 20 else "。"))
    if a["bend_causes"]:
        lines.append(f"- 不要超过方案的最大值：超出后可能出现「{'」「'.join(a['bend_causes'])}」。")
    st.markdown("\n".join(lines))
    with st.expander("可以追加的测量"):
        for label, minutes in a["menu"]:
            st.markdown(f"- {label}（约 {lab.fmt(minutes)} 分钟）")


def render() -> None:
    st.subheader("② 实验设计")
    cur = state.current()
    if not cur:
        st.info("先在「① 拍照识别」里选定一个实验。")
        return
    key, exp, card_ = cur
    left, right = st.columns([1, 1], gap="large")
    with left:
        _card_section(exp, card_, key)
        protocol = _protocol_section(exp)
        _prediction_section(exp, card_)
    with right:
        if state.predicted_or_skipped():
            sp = catalog.spec_of(key) if catalog.is_other(key) else {}
            if sp and sp.get("ideal_slope") is None and "slope_guess" not in sp:
                st.caption("斜率未知、也没有填估计：下表的数值只是占位，测了前两组以后 ③ 会按你的数据定下斜率。")
            _twin_table(exp, protocol)
        else:
            st.info("先写下你的预测（或点“跳过预测”），再看孪生的预测表。")
        _range_section(key, protocol)
    _budget_section(key, protocol)
    _separability_section(key, protocol)
