"""④ 诊断：一致性 → 先猜原因 → 嫌疑人看板（概率条 + 证据）→ 幽灵曲线 → "下一次测量"卡（各原因的预测值）
→ 修正后复测验证。所有数字来自事实单（facts.build_fact_sheet）。
"""

from __future__ import annotations

import json

import streamlit as st

from lab_detective.labsim.core import Observation
from lab_detective.tutor.session import MAX_EIG_ROUNDS
from lab_detective.tutor.templates import signatures_zh

from . import catalog, charts, lab, state
from .step_measure import working_experiment
from .theme import band_badge, card, esc, suspect_rows

NO_SEPARATION_BITS = 0.01


def _obs_json(observations) -> str:
    return json.dumps([o.to_dict() for o in observations], sort_keys=True)


def _obs_from_json(s: str) -> list[Observation]:
    return [Observation.from_dict(d) for d in json.loads(s)]


@st.cache_data(show_spinner=False, max_entries=64)
def _fact_sheet(key: str, obs_json: str, protocol: str) -> dict:
    exp = catalog.get_experiment(key)
    return lab.fact_sheet(exp, lab.dataset(exp, _obs_from_json(obs_json), protocol))


@st.cache_data(show_spinner=False, max_entries=64)
def _ghosts(key: str, obs_json: str, protocol: str) -> tuple[dict, list[dict]]:
    exp = catalog.get_experiment(key)
    ds = lab.dataset(exp, _obs_from_json(obs_json), protocol)
    return lab.data_points(exp, ds), lab.ghosts(exp, ds)


def _agreement(fs: dict) -> None:
    a = fs.get("agreement") or {}
    if a.get("value") is None:
        return
    unc = f" ± {lab.fmt(a['uncertainty'], 2)}" if a.get("uncertainty") is not None else ""
    z = f"，z = {a['z']:.1f}" if a.get("z") is not None else ""
    st.markdown(f"**一致性**：{esc(a.get('quantity', ''))} = {lab.fmt(a['value'])}{unc} {esc(a.get('unit', ''))}{z}",
                unsafe_allow_html=False)
    st.markdown(band_badge(a.get("band"), "和理想规律相比："), unsafe_allow_html=True)


def _guess_and_reveal(exp, fs: dict) -> None:
    sess = state.lab()
    target = f"cause@{len(st.session_state.extras)}"
    st.markdown(f"数据特征：{esc(signatures_zh(fs))}。")
    names = [c.name_zh for c in sorted(exp.causes, key=lambda c: c.id)]
    rkey = f"guess_{target}"
    st.radio("先猜一猜：你觉得最可能是哪个原因？", names, index=None, key=rkey)

    def _reveal() -> None:
        pick = st.session_state.get(rkey)
        if pick:
            sess.record_prediction(target, pick, kind="choice")
        st.session_state.revealed = True

    st.button("揭晓嫌疑人看板", type="primary", key="reveal", on_click=_reveal)


def _board(fs: dict) -> None:
    dg = fs.get("diagnosis") or {}
    ranked = dg.get("ranked") or []
    sess = state.lab()
    guess = sess.prediction_for(f"cause@{len(st.session_state.extras)}")
    if guess is not None and ranked:
        top = ranked[0]["name"]
        st.markdown(f"你猜的是「{esc(guess['value'])}」，" +
                    ("和引擎的判断一致。" if guess["value"] == top else f"引擎认为更可能的是「{esc(top)}」。"))
    st.markdown("#### 嫌疑人看板")
    st.markdown(suspect_rows(ranked), unsafe_allow_html=True)
    if dg.get("not_in_list"):
        st.warning("注意：数据和清单里所有原因都对不上，可能存在清单以外的原因（例如读错单位、换了器材）。")


def _ghost_section(ckey: str, obs_json: str, protocol: str) -> None:
    try:
        pts, gh = _ghosts(ckey, obs_json, protocol)
    except Exception as e:  # noqa: BLE001
        st.caption(f"幽灵曲线暂时画不出来：{e}")
        return
    st.markdown("#### 幽灵曲线")
    st.caption("每条虚线是一个嫌疑原因在它最可能的参数下“应该”画出的曲线；白点是你的数据。哪条线穿过你的点，嫌疑就大。")
    st.altair_chart(charts.ghost_chart(pts, gh), use_container_width=True)


def _extra_editor(exp, plan, prefix: str, src_key: str, ver_key: str) -> list[Observation]:
    from .step_measure import _column_config

    src = st.session_state.get(src_key)
    obs = []
    for kind in lab.kinds_in(plan):
        df = (src or {}).get(kind)
        if df is None:
            df = lab.template_df(exp, kind, [s for k, s in plan if k == kind])
        edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True,
                                column_config=_column_config(exp, kind),
                                key=f"{prefix}_{kind}_{st.session_state[ver_key]}")
        obs += lab.df_to_observations(exp, kind, edited)[0]
    return obs


def _next_card(exp, sim_exp, fs: dict, all_obs) -> None:
    nm = fs.get("next_measurement")
    rounds = len(st.session_state.extras)
    st.markdown("#### 下一次测量")
    if not nm:
        st.info("菜单里没有能进一步区分这些原因的测量。")
        return
    if (nm.get("eig_bits") or 0) < NO_SEPARATION_BITS:
        st.info(f"菜单里最好的一项是「{nm['label']}」，但它几乎分不开剩下的嫌疑人（信息增益 "
                f"{nm.get('eig_bits', 0):.2f} bit）。可以直接看修正方法和讲解。")
        return
    if rounds >= MAX_EIG_ROUNDS:
        st.info(f"已经做了 {MAX_EIG_ROUNDS} 轮追加测量，去看修正方法和讲解吧。")
        return
    rows = "".join(f"<li>如果是「{esc(p.get('name', p['cause_id']))}」：预计 {lab.fmt(p['value'])} {esc(p.get('unit', ''))}</li>"
                   for p in nm.get("predicted", []))
    mins = f"约 {lab.fmt(nm['minutes'])} 分钟，" if nm.get("minutes") is not None else ""
    card(f"推荐：{nm['label']}",
         f"<p>{mins}期望信息增益 {nm['eig_bits']:.2f} bit（越大越能把嫌疑人分开）。动手之前先看各原因的预测：</p>"
         f"<ul>{rows}</ul>")
    opts = [f"约 {lab.fmt(p['value'])} {p.get('unit', '')}（{p.get('name', p['cause_id'])}）".replace(" （", "（")
            for p in nm.get("predicted", [])] + ["说不准"]
    st.radio("你猜结果会是哪一种？", opts, index=None, key=f"nm_guess_{rounds}")
    try:
        item = exp.menu_item(nm["id"])
    except KeyError:
        st.caption("这项测量不在当前孪生的菜单里。")
        return
    plan = [(str(k), float(s)) for k, s in item.plan]
    new_obs = _extra_editor(exp, plan, "extra", "extra_sim", "extra_ver")
    c1, c2 = st.columns(2)
    if st.session_state.sim:
        c1.button("演示：模拟这次测量", key="simulate_extra", on_click=_simulate_extra_cb,
                  args=(sim_exp, plan, [o.to_dict() for o in all_obs], rounds))
    c2.button("加入数据并重新诊断", type="primary", key="add_extra", disabled=not new_obs, on_click=_add_extra_cb,
              args=(item.id, item.label_zh, [o.to_dict() for o in new_obs]))


def _simulate_extra_cb(sim_exp, plan, obs_dicts, rounds: int) -> None:
    sim = st.session_state.sim
    if not sim:
        return
    prev = [Observation.from_dict(d) for d in obs_dicts]
    new = lab.simulate_more(sim_exp, sim["truth"], prev, plan, sim["seed"] + 101 + rounds)
    st.session_state.extra_sim = {k: lab.obs_to_df(sim_exp, k, new) for k in lab.kinds_in(plan)}
    st.session_state.extra_ver += 1


def _add_extra_cb(item_id: str, label: str, obs_dicts: list) -> None:
    """把追加测量并进数据，回到"先猜再揭晓"（新数据下重新诊断）。"""
    if not obs_dicts:
        return
    st.session_state.extras = st.session_state.extras + [{"item_id": item_id, "label": label, "obs": obs_dicts}]
    st.session_state.extra_sim = None
    st.session_state.extra_ver += 1
    st.session_state.revealed = False
    st.session_state.explain_result = None


def _simulate_rerun_cb(sim_exp, protocol: str, plan) -> None:
    sim = st.session_state.sim
    if not sim:
        return
    ds = lab.simulate_fixed(sim_exp, sim["truth"], protocol, sim["seed"] + 500)
    st.session_state.rerun_src = {k: lab.obs_to_df(sim_exp, k, ds.observations) for k in lab.kinds_in(plan)}
    st.session_state.rerun_ver += 1


def _fix_and_rerun(key: str, exp, sim_exp, fs: dict, protocol: str) -> None:
    fx = fs.get("fix") or {}
    st.markdown("#### 修正，再复测验证")
    if fx.get("cause_id") == exp.random_only_id:
        st.success("引擎认为只有随机误差：方法没有问题，多测几组取平均即可。")
        return
    if fx.get("text"):
        st.markdown(f"**修正方法**：{esc(fx['text'])}")
    if fx.get("predicted_rerun"):
        st.markdown(f"**预测**：{esc(fx['predicted_rerun'])}")
    with st.expander("按修正方法重做一遍，填复测数据"):
        plan = lab.plan_for(exp, protocol)
        rerun_obs = _extra_editor(exp, plan, "rerun", "rerun_src", "rerun_ver")
        if st.session_state.sim:
            st.button("演示：模拟修正后的复测", key="simulate_rerun", on_click=_simulate_rerun_cb,
                      args=(sim_exp, protocol, plan))
        if len(rerun_obs) < 3:
            st.caption("填满至少 3 行复测数据后，这里会对照修正前的预测。")
            return
        ckey, wexp = working_experiment(key, rerun_obs)
        fs2 = _fact_sheet(ckey, _obs_json(rerun_obs), protocol)
        cmp = lab.rerun_compare(fs, fs2, wexp.random_only_id)
        if cmp["intercept"] is not None and cmp["pred_intercept"] is not None:
            unit = (fs2.get("agreement") or {}).get("unit", "")
            st.markdown(f"预计截距 {lab.fmt(cmp['pred_intercept'])} ± {lab.fmt(cmp['pred_se'], 2)}，"
                        f"复测得到 {lab.fmt(cmp['intercept'])} ± {lab.fmt(cmp['intercept_se'], 2)} {esc(unit)}"
                        + (f"（z = {cmp['z']:.1f}）" if cmp["z"] is not None else ""))
            st.markdown(band_badge(cmp["band"], "复测与预测："), unsafe_allow_html=True)
        top = cmp["top"]
        if top:
            st.markdown(f"复测后最可能的原因：「{esc(top['name'])}」（{round(top['prob'] * 100)}%）。")
        if cmp["verdict"] == "fixed":
            st.success("修正有效：复测数据最像只剩随机误差，并且和修正前的预测相符。")
        elif cmp["verdict"] == "likely":
            st.info("复测数据最像只剩随机误差，但把握还不到一半：再多测几组会更确定。")
        else:
            st.warning("复测数据里仍有系统偏离，可能还有别的原因，或者修正没有做到位。")


def render(observations) -> dict | None:
    """返回事实单（讲解步骤用）；数据不够时返回 None。"""
    st.subheader("④ 诊断")
    cur = state.current()
    if not cur:
        st.info("先在「① 拍照识别」里选定一个实验。")
        return None
    key, exp, _card = cur
    protocol = st.session_state["protocol"]
    extra_obs = [Observation.from_dict(d) for e in st.session_state.extras for d in e["obs"]]
    all_obs = list(observations) + extra_obs
    if len(observations) < 3:
        st.info("至少要 3 行完整的数据才能诊断：先到「③ 测量」填表（也可以用演示数据）。")
        return None
    ckey, wexp = working_experiment(key, all_obs)
    oj = _obs_json(all_obs)
    fs = _fact_sheet(ckey, oj, protocol)
    sess = state.lab()
    sess.fact_sheet = fs
    sess.phase = "diagnose"
    if st.session_state.extras:
        st.caption("已加入的追加测量：" + "；".join(e["label"] for e in st.session_state.extras))
    _agreement(fs)
    if not st.session_state.revealed:
        _guess_and_reveal(wexp, fs)
        return fs
    left, right = st.columns([1, 1], gap="large")
    with left:
        _board(fs)
    with right:
        _ghost_section(ckey, oj, protocol)
    _next_card(wexp, exp, fs, all_obs)
    _fix_and_rerun(key, wexp, exp, fs, protocol)
    sim = st.session_state.sim
    if sim:
        with st.expander("演示答案（模拟时注入的原因）"):
            try:
                st.markdown(f"注入的是「{esc(exp.cause(sim['cause']).name_zh)}」。")
            except KeyError:
                st.markdown(f"注入的是 {esc(sim['cause'])}。")
    return fs
