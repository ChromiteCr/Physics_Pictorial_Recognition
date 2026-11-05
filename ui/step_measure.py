"""③ 测量：多行数据表（按方案/仪器配置预填控制量），填完一行就和孪生实时对照（z 值和提示）。

实时对照用"按前面各行校准过的孪生"：第一行是实验前的预测，之后每一行的预测都用前面的数据更新过参数。
"""

from __future__ import annotations

import streamlit as st

from lab_detective.tutor import ScriptedTutor
from lab_detective.tutor.templates import fmt as tfmt

from . import catalog, charts, lab, state
from .step_design import skip_prediction
from .theme import band_badge, esc

RANDOM_PICK = "__random__"


def working_experiment(key: str, observations):
    """其他实验斜率未知时，用已测数据定斜率先验后的孪生；整理好的实验原样返回。"""
    ckey = catalog.calibrate_other(key, observations)
    return ckey, catalog.get_experiment(ckey)


def _column_config(exp, kind: str) -> dict:
    labels = lab.column_labels(exp, kind)
    cfg = {}
    if "setting" in labels:
        cfg["setting"] = st.column_config.NumberColumn(labels["setting"], format="%g")
    else:
        cfg["setting"] = None  # 附加测量没有控制量，隐藏
    for r in exp.obs_kinds[kind].readings:
        cfg[r.key] = st.column_config.NumberColumn(labels[r.key], format="%g")
    return cfg


def _sources(exp, protocol: str) -> dict:
    src = st.session_state.get("meas_src")
    if src is None:
        plan = lab.plan_for(exp, protocol)
        src = {k: lab.template_df(exp, k, [s for kk, s in plan if kk == k]) for k in lab.kinds_in(plan)}
        st.session_state.meas_src = src
    return src


def _simulate_cb(key: str, protocol: str) -> None:
    """回调：按选中的原因模拟一整张表（演示用），并清掉下游的诊断、追加测量和复测。"""
    exp = catalog.get_experiment(key)
    cause = st.session_state.get("sim_cause", RANDOM_PICK)
    seed = int(st.session_state.get("sim_seed", 7))
    if cause not in exp.cause_ids():
        import numpy as np

        cause = str(np.random.default_rng(seed).choice(exp.cause_ids()))
    ds = lab.simulate(exp, cause, protocol, seed)
    plan = lab.plan_for(exp, protocol)
    state.new_data()
    st.session_state.meas_src = {k: lab.obs_to_df(exp, k, ds.observations) for k in lab.kinds_in(plan)}
    st.session_state.sim = {"truth": ds.truth, "cause": cause, "seed": seed}


def _reset_cb() -> None:
    state.new_data()
    st.session_state.meas_src = None


def _demo_controls(key: str, exp, protocol: str) -> None:
    with st.expander("演示：用模拟数据填表（不用真实器材）"):
        ids = [RANDOM_PICK] + list(exp.cause_ids())
        names = {RANDOM_PICK: "随机挑一个原因（先不告诉我）", **{c.id: c.name_zh for c in exp.causes}}
        c1, c2 = st.columns([3, 1])
        c1.selectbox("模拟时注入的原因", ids, format_func=names.get, key="sim_cause")
        c2.number_input("随机种子", min_value=0, value=7, step=1, key="sim_seed")
        st.button("用模拟数据填表", key="simulate", on_click=_simulate_cb, args=(key, protocol))


def _prediction_vs_actual(observations, checks) -> None:
    sess = state.lab()
    for p in sess.student_predictions:
        if p.get("kind") != "number" or "@" not in str(p.get("target")):
            continue
        kind, _, setting = p["target"].partition("@")
        hit = next((i for i, o in enumerate(observations) if o.kind == kind and tfmt(o.setting) == setting), None)
        if hit is None:
            continue
        c = checks[hit] if hit < len(checks) else {}
        unit = p.get("unit", "")
        st.info(f"你实验前预测 {p['value']} {unit}，实测 {lab.fmt(observations[hit].values[0])} {unit}。"
                f"想一想差别从哪里来。".replace("  ", " "))
        if c.get("predicted") is not None and c.get("index") == hit:
            st.caption(f"（孪生用前面各行校准后预测这一行为 {lab.fmt(c['predicted'])} ± "
                       f"{lab.fmt(c['uncertainty'], 2)} {c.get('unit', '')}）")
        break


def render() -> list:
    """返回当前已填完整的观测（诊断步骤用）。"""
    st.subheader("③ 测量")
    cur = state.current()
    if not cur:
        st.info("先在「① 拍照识别」里选定一个实验。")
        return []
    key, exp, card = cur
    protocol = st.session_state["protocol"]
    if not state.predicted_or_skipped():
        st.info("先在「② 实验设计」里写下你的预测，再动手测量。")
        st.button("跳过预测，直接测量", key="pred_skip_measure", on_click=skip_prediction)
        return []

    st.caption(f"方案：{exp.protocols[protocol].label_zh}。每填完一行，右边就和孪生对照一次。"
               "可以在表格底部加行（例如重复测量或更多设定值）。")
    _demo_controls(key, exp, protocol)
    src = _sources(exp, protocol)
    observations, partial = [], 0
    left, right = st.columns([1, 1], gap="large")
    with left:
        for kind, df in src.items():
            if len(src) > 1:
                st.markdown(f"**{esc(exp.obs_kinds[kind].label_zh)}**")
            edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True,
                                    column_config=_column_config(exp, kind),
                                    key=f"meas_{kind}_{st.session_state.meas_ver}")
            obs, part = lab.df_to_observations(exp, kind, edited)
            observations += obs
            partial += part
        st.button("清空表格", key="reset_table", on_click=_reset_cb)
    st.session_state.obs = observations

    sess = state.lab()
    sess.observations = [o.to_dict() for o in observations]
    sess.checks = []
    ckey, wexp = working_experiment(key, observations)
    checks = lab.live_checks(wexp, observations)
    for c in checks:
        sess.record_check(c["index"], {k: c[k] for k in ("value", "unit", "predicted", "uncertainty", "z", "band")
                                       if k in c})
    with right:
        sess.phase = "measure"
        st.markdown(esc(ScriptedTutor(sess, None, card, experiment=exp).next_message()["text"]))
        if partial:
            st.caption(f"有 {partial} 行还没填完整，先不参与对照。")
        if checks:
            last = checks[-1]
            st.markdown(band_badge(last.get("band"), f"第 {last['index'] + 1} 行："), unsafe_allow_html=True)
            st.markdown(esc(last.get("hint", "")))
        else:
            st.info("填完第一行就能看到孪生的对照。")
        _prediction_vs_actual(observations, checks)
    if checks:
        st.markdown("#### 逐行对照孪生")
        st.dataframe(lab.checks_table(checks), hide_index=True, use_container_width=True,
                     column_config={"提示": st.column_config.TextColumn("提示", width="large")})
    if ckey != key:
        st.caption("其他实验：斜率先验已按你前几组数据的粗拟合定下（只影响先验中心，不影响诊断用的数据）。")
    if len(observations) >= 2:
        try:
            ds = lab.dataset(wexp, observations, protocol)
            st.altair_chart(charts.data_chart(lab.data_points(wexp, ds), "你的数据（面板坐标）"),
                            use_container_width=True)
        except Exception as e:  # noqa: BLE001  个别观测组合无法线性化时不画图
            st.caption(f"暂时画不出图：{e}")
    if len(observations) >= 3:
        st.success("数据够了，去「④ 诊断」看看数据说明了什么。")
    return observations
