"""会话状态：只在 st.session_state 里存字符串、数据表和 LabSession，不存 Experiment 对象。

换实验或换方案时，下游的数据、预测、诊断、追加测量、复测全部清空（实验键变了，旧数据就没有意义了）。
"""

from __future__ import annotations

import streamlit as st

from lab_detective.tutor.session import LabSession

from . import catalog

DEFAULTS = {
    # ① 拍照识别（P1 的检测 + 画布修正）
    "components": None, "annotated_image_rgb": None, "init_image_path": None, "canvas_key_version": 0,
    "recog": None,
    # 当前实验
    "exp_key": None, "protocol": "standard",
    # ②③ 预测与测量
    "pred_skipped": False, "meas_src": None, "meas_ver": 0, "obs": [], "sim": None,
    # ④ 诊断、追加测量、复测
    "revealed": False, "extras": [], "extra_ver": 0, "extra_sim": None,
    "rerun_src": None, "rerun_ver": 0, "rerun_obs": [],
    # 模型模式
    "model_mode": False, "llm_base": None, "llm_model": "", "llm_probe": None,
    # ⑤ 讲解
    "explain_result": None, "reflection": "",
}

_DOWNSTREAM = ("pred_skipped", "meas_src", "meas_ver", "obs", "sim", "revealed", "extras", "extra_ver",
               "extra_sim", "rerun_src", "rerun_ver", "rerun_obs", "explain_result", "reflection")


def init() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v if not isinstance(v, (list, dict)) else type(v)(v)
    if "lab" not in st.session_state:
        st.session_state["lab"] = LabSession()


def _reset_downstream() -> None:
    ver = {k: st.session_state.get(k, 0) + 1 for k in ("meas_ver", "extra_ver", "rerun_ver")}
    for k in _DOWNSTREAM:
        v = DEFAULTS[k]
        st.session_state[k] = v if not isinstance(v, (list, dict)) else type(v)(v)
    st.session_state.update(ver)  # 换 key，数据编辑器从空表重新开始


def new_data() -> None:
    """测量表换了一批数据（模拟或清空）：诊断、追加测量、复测、讲解都作废，预测保留。"""
    keep = st.session_state.get("pred_skipped", False)
    _reset_downstream()
    st.session_state["pred_skipped"] = keep
    lab_ = st.session_state.get("lab")
    if lab_ is not None:  # 对旧数据的"猜原因"作废，实验前的数值预测保留
        lab_.student_predictions = [p for p in lab_.student_predictions if p.get("kind") == "number"]


def set_experiment(key: str, protocol: str | None = None) -> None:
    exp = catalog.get_experiment(key)
    if protocol not in exp.protocols:
        protocol = "standard" if "standard" in exp.protocols else next(iter(exp.protocols))
    st.session_state["exp_key"] = key
    st.session_state["protocol"] = protocol
    _reset_downstream()
    card = card_of(key, exp)
    lab = st.session_state.get("lab") or LabSession()
    lab.restart()
    lab.experiment_id, lab.card, lab.protocol, lab.phase = exp.id, card, protocol, "plan"
    st.session_state["lab"] = lab


def set_protocol(protocol: str) -> None:
    if protocol == st.session_state.get("protocol"):
        return
    key = st.session_state.get("exp_key")
    if key:
        set_experiment(key, protocol)


def card_of(key: str, exp=None) -> dict:
    exp = exp or catalog.get_experiment(key)
    if catalog.is_other(key):
        return catalog.pseudo_card(exp, catalog.spec_of(key))
    return catalog.card_for(exp.id) or {"id": exp.id, "name_zh": exp.name_zh, "variables": [], "steps": []}


def current():
    """(实验键, Experiment, 实验卡)；还没选实验时返回 None。"""
    key = st.session_state.get("exp_key")
    if not key:
        return None
    try:
        exp = catalog.get_experiment(key)
    except Exception:  # noqa: BLE001  实验键失效（例如组件层改了）时当作没选
        st.session_state["exp_key"] = None
        return None
    return key, exp, card_of(key, exp)


def lab() -> LabSession:
    return st.session_state["lab"]


def predicted_or_skipped() -> bool:
    return bool(st.session_state.get("pred_skipped")) or bool(lab().student_predictions)
