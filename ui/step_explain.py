"""⑤ 讲解与知识点：按事实单讲解（脚本模式用模板；模型模式由本地模型写、校验器把关，不通过就换模板），
显示校验器结果、相关知识点和前置知识学习路径，最后学生自己写结论。
"""

from __future__ import annotations

import hashlib
import json

import streamlit as st

from lab_detective.facts import to_markdown_zh
from lab_detective.tutor import ScriptedTutor, explanation_text, guarded_explanation, verify

from . import lab, state
from .theme import chalk_loading, esc

SOURCE_ZH = {"template": "模板（脚本模式）", "model": "本地模型", "model_rewrite": "本地模型（改写一次后通过）"}
COVER_ZH = {"top_cause": "主要原因", "number_evidence": "数值证据", "next_measurement": "下一次测量"}


@st.cache_data(show_spinner=False)
def _kp_names() -> dict:
    return lab.kp_names()


def _fs_id(fs: dict) -> str:
    return hashlib.sha1(json.dumps(fs, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _explanation(fs: dict, card: dict, model) -> dict:
    """默认模板讲解；模型模式下按钮触发本地模型（结果按事实单缓存，事实单变了就作废）。"""
    names = _kp_names()
    fid = _fs_id(fs)
    res = st.session_state.get("explain_result")
    if model is not None and st.button("让本地模型写讲解（校验不通过就用模板）", key="model_explain"):
        with chalk_loading("本地模型正在写讲解……"):
            r = guarded_explanation(model, state.lab(), fs, card, kp_names=names)
        res = {**r, "fs_id": fid}
        st.session_state.explain_result = res
    if res and res.get("fs_id") == fid:
        return res
    text = explanation_text(fs, names)
    return {"text": text, "source": "template", "verify": verify(text, fs), "fs_id": fid}


def _verifier(v: dict) -> None:
    cov = v.get("coverage") or {}
    parts = [f"{COVER_ZH.get(k, k)}{'✓' if ok else '✗'}" for k, ok in cov.items()]
    if v.get("ok"):
        st.success("事实核对通过：每个数字和名称都能在事实单里找到。" + ("（覆盖：" + "，".join(parts) + "）" if parts else ""))
    else:
        st.error("事实核对没有通过：")
        for p in v.get("problems", []):
            st.markdown(f"- {esc(p.get('msg', ''))}" + (f"：{esc(p['text'])}" if p.get("text") else ""))


def _knowledge(exp, fs: dict) -> None:
    ranked = (fs.get("diagnosis") or {}).get("ranked") or []
    top = ranked[0]["cause_id"] if ranked else None
    ids = lab.related_kps(exp, top)
    names = _kp_names()
    st.markdown("#### 相关知识点")
    if not ids:
        st.caption("暂时没有关联的知识点。")
        return
    sess = state.lab()
    sess.touch_kp(*ids)
    for kid in ids:
        d = lab.kp_detail(kid)
        if not d:
            continue
        with st.expander(d.get("name_zh", kid), expanded=kid == ids[0]):
            st.markdown(esc(d.get("statement", "")))
            if d.get("formula"):
                st.code(d["formula"], language=None)
            for m in d.get("misconceptions") or []:
                st.caption(f"常见错误：{m.get('text', '')}（表现：{m.get('symptom', '')}）")
            path = d.get("path") or []
            if path:
                chain = " → ".join(f"<b>{esc(names.get(p, p))}</b>" for p in path[-6:])
                more = "…… → " if len(path) > 6 else ""
                st.markdown(f'<div class="kp-path">学习路径（先学的在前）：{more}{chain} → '
                            f'<b>{esc(d.get("name_zh", kid))}</b></div>', unsafe_allow_html=True)
            else:
                st.caption("这是基础知识点，没有前置知识。")


def render(fs: dict | None, model=None) -> None:
    st.subheader("⑤ 讲解与知识点")
    cur = state.current()
    if not cur or not fs:
        st.info("先完成「④ 诊断」，这里会按事实单讲解。")
        return
    key, exp, card = cur
    if not st.session_state.get("revealed"):
        st.info("先在「④ 诊断」里猜一猜原因、揭晓嫌疑人看板，再来看讲解。")
        return
    res = _explanation(fs, card, model)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### 讲解")
        st.caption(f"来源：{SOURCE_ZH.get(res.get('source'), res.get('source'))}。所有数字都来自物理引擎的事实单。")
        for para in (res.get("text") or "").split("\n"):
            if para.strip():
                st.markdown(esc(para))
        _verifier(res.get("verify") or {})
        with st.expander("事实单原文（引擎输出）"):
            st.markdown(to_markdown_zh(fs))
    with right:
        _knowledge(exp, fs)
    st.markdown("#### 写结论")
    sess = state.lab()
    sess.phase = "reflect"
    st.markdown(esc(ScriptedTutor(sess, fs, card, experiment=exp).next_message()["text"]))
    st.text_area("你的结论（只保存在这次会话里）", key="reflection", height=120)
