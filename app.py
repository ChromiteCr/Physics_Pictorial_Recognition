"""实验侦探 Lab Detective：Streamlit 网页入口（完全离线，默认脚本模式）。

五步：① 拍照识别（YOLO 框 + 画布修正 + 识别实验 + 一个澄清问题，手动选择永远可用，含"其他实验"）
     ② 实验设计（先预测；孪生预测表、不确定度预算、本方案分不开的原因、建议的测量范围）
     ③ 测量（多行数据表，填一行就和孪生实时对照）
     ④ 诊断（嫌疑人看板、幽灵曲线、"下一次测量"卡、修正后复测验证）
     ⑤ 讲解与知识点（事实单讲解 + 校验器结果 + 知识点和学习路径）
所有数字由物理代码（lab_detective）算出；侧栏可选"模型模式"连本机的 mlx_vlm.server，模型只认图、写讲解，从不算物理。
界面代码在 ui/ 里，这个文件只负责页面骨架。

运行：
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 离线：YOLO / Hugging Face 都不联网（检测模块在点"开始检测"时才懒加载）
for _k, _v in (("YOLO_OFFLINE", "true"), ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1")):
    os.environ.setdefault(_k, _v)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="实验侦探", layout="wide")

from ui import llm, state, step_design, step_diagnose, step_explain, step_identify, step_measure, theme  # noqa: E402


def _restart() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def _sidebar():
    """模式切换与进度。返回本地模型客户端（脚本模式或探测不到服务时为 None）。"""
    with st.sidebar:
        st.markdown("### 模式")
        on = st.toggle("模型模式（本机模型，可选）", key="model_mode")
        st.caption("默认是脚本模式：讲解用模板，数字全部来自物理引擎，不需要任何模型、不联网。")
        client = None
        if on:
            base = st.text_input("本机服务地址（mlx_vlm.server）", llm.DEFAULT_BASE, key="llm_base_input")
            pr = st.session_state.llm_probe
            if pr is None or pr.get("base") != base or st.button("重新检测", key="llm_reprobe"):
                pr = {"base": base, **llm.probe(base)}
                st.session_state.llm_probe = pr
            if pr["ok"]:
                models = pr["models"] or ["local"]
                name = st.selectbox("模型", models, key="llm_model_pick")
                client = llm.make_client(base, name)
                st.success("已连上本机模型：识别拿不准时由它看图，讲解由它写、校验器把关。")
            else:
                st.warning("没有检测到本机模型服务，继续使用脚本模式。")
                st.caption(pr.get("error", ""))
    return client


def _sidebar_progress() -> None:
    """进度写在页面最后（这时本轮的测量表已经解析完，行数是最新的）。"""
    with st.sidebar:
        st.markdown("### 进度")
        cur = state.current()
        if cur:
            st.markdown(f"实验：{theme.esc(cur[1].name_zh)}")
            st.markdown(f"已测：{len(st.session_state.obs)} 行；追加测量：{len(st.session_state.extras)} 轮")
        else:
            st.markdown("还没有选实验。")
        st.button("重新开始", key="restart_all", on_click=_restart)


theme.inject()
state.init()
st.markdown(
    "<h1>实验侦探 Lab Detective</h1>"
    f'<p style="color:{theme.CHALK["muted"]};margin-top:-0.6rem;">拍照认出实验 → 物理孪生先预测 → 边测边对照 → '
    "贝叶斯找出系统误差来源 → 推荐下一次测量 → 按事实单讲解</p>",
    unsafe_allow_html=True,
)
st.caption("整理好的实验只是起点，其他装置可以由物理组件组合。所有数字由物理代码算出，完全离线运行。")

model = _sidebar()
tabs = st.tabs(["① 拍照识别", "② 实验设计", "③ 测量", "④ 诊断", "⑤ 讲解与知识点"])
with tabs[0]:
    step_identify.render(model)
with tabs[1]:
    step_design.render()
with tabs[2]:
    observations = step_measure.render()
with tabs[3]:
    fact_sheet = step_diagnose.render(observations)
with tabs[4]:
    step_explain.render(fact_sheet, model)
_sidebar_progress()
