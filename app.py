"""
Streamlit 网页入口。

两个页面（左侧切换）：
  Part 1：上传初状态照片(+可选视频) → YOLO 检测器材 → 网页上人工核对/修正
  Part 2：确认后的器材 → 意图识别(规则+GLM兜底，可手动改) → 按实验类型动态生成
          输入表单，填实测数据 → 算物理量 + 生成动画 + 调 GLM 生成讲解/公式/知识点清单

运行：
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from detect import detect_components, extract_reference_track, DEFAULT_WEIGHTS
from physics import EXPERIMENT_TYPES, compute_experiment, list_experiment_types
from intent_classifier import classify_intent_with_fallback
from animate import generate_experiment_animation
from explain_api import generate_explanation, GLMAPIError

st.set_page_config(page_title="力学实验识别与讲解", layout="wide")

CLASS_OPTIONS = ["cart", "track", "spring", "string", "ruler", "dynamometer"]

# ── 会话状态初始化 ──────────────────────────────────────────────────
for key, default in [
    ("components", None), ("annotated_image_rgb", None), ("confirmed", False),
    ("track_result", None), ("init_image_path", None), ("intent_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.getvalue())
        return f.name


page = st.sidebar.radio("步骤", ["Part 1: 识别初状态", "Part 2: 实验分析"])

# ══════════════════════════════════════════════════════════════════
# Part 1
# ══════════════════════════════════════════════════════════════════
if page == "Part 1: 识别初状态":
    st.title("Part 1：识别实验初状态")
    st.caption("上传一张实验初状态照片（必需），可选加一段实验过程视频"
               "（只用于动画参考路径，不做任何标定/测量）。")

    uploaded_image = st.file_uploader("初状态照片", type=["jpg", "jpeg", "png", "webp"])
    uploaded_video = st.file_uploader("实验过程视频（可选）", type=["mp4", "mov", "avi"])

    with st.expander("检测参数（一般不用改）"):
        weights_path = st.text_input("模型权重路径", str(DEFAULT_WEIGHTS))
        conf = st.slider("置信度阈值", 0.05, 0.9, 0.25, 0.05)
        marker_color = st.selectbox("视频里运动物体的标记色块颜色", ["red", "green", "blue"])

    if uploaded_image and st.button("开始检测", type="primary"):
        img_path = _save_uploaded_file(uploaded_image)
        st.session_state.init_image_path = img_path

        try:
            result = detect_components(img_path, weights_path, conf)
        except FileNotFoundError as e:
            st.error(str(e))
        else:
            st.session_state.components = result["detections"]
            st.session_state.annotated_image_rgb = _bgr_to_rgb(result["annotated_image"])
            st.session_state.confirmed = False
            st.session_state.intent_result = None  # 器材变了，意图识别要重跑

        if uploaded_video:
            video_path = _save_uploaded_file(uploaded_video)
            with st.spinner("提取视频参考轨迹中..."):
                try:
                    st.session_state.track_result = extract_reference_track(
                        video_path, color=marker_color
                    )
                except Exception as e:
                    st.warning(f"视频轨迹提取失败（不影响后续流程，可以跳过）：{e}")

    if st.session_state.components is not None:
        st.image(st.session_state.annotated_image_rgb, caption="检测结果", use_container_width=True)
        if st.session_state.track_result is not None:
            n_pts = len(st.session_state.track_result["track"])
            st.caption(f"视频参考轨迹：{n_pts} 个点（仅供 Part 2 动画参考）")

        st.subheader("检测到的器材（可编辑：改类别、加/删行）")
        st.caption("识别可能有误差，置信度低的行建议重点核对；bbox/置信度不可编辑，"
                   "手动新增的行留空即可。")
        df = pd.DataFrame(st.session_state.components)
        if df.empty:
            df = pd.DataFrame(columns=["class_name", "confidence", "bbox"])

        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            column_config={
                "class_name": st.column_config.SelectboxColumn("类别", options=CLASS_OPTIONS, required=True),
                "confidence": st.column_config.NumberColumn("置信度", disabled=True),
                "bbox": st.column_config.Column("边界框(只读)", disabled=True),
                "class_id": None,  # 隐藏，内部字段不需要人工看
            },
            use_container_width=True,
            key="component_editor",
        )

        if st.button("确认器材列表，进入 Part 2", type="primary"):
            records = edited_df.dropna(subset=["class_name"]).to_dict("records")
            for r in records:
                r.setdefault("bbox", None)
            st.session_state.components = records
            st.session_state.confirmed = True
            st.session_state.intent_result = None
            st.success("已确认。请点击左侧切换到「Part 2: 实验分析」")

# ══════════════════════════════════════════════════════════════════
# Part 2
# ══════════════════════════════════════════════════════════════════
else:
    st.title("Part 2：识别实验意图 → 填实测数据 → 动画 + 讲解")

    if not st.session_state.confirmed:
        st.warning("请先在「Part 1: 识别初状态」完成检测并点击「确认器材列表」。")
        st.stop()

    components = st.session_state.components
    st.write("已确认的器材：", ", ".join(c["class_name"] for c in components) or "（无）")

    if st.session_state.intent_result is None:
        with st.spinner("识别实验意图中..."):
            st.session_state.intent_result = classify_intent_with_fallback(
                components, image_path=st.session_state.init_image_path
            )
    intent_result = st.session_state.intent_result

    if intent_result.get("warning"):
        st.info(intent_result["warning"])
    if intent_result.get("candidates"):
        with st.expander("规则表候选详情"):
            for c in intent_result["candidates"]:
                st.write(f"- {c['name']}（{c['level']}）分数={c['score']}：{c['reason']}")

    all_types = list_experiment_types()
    type_ids = [t["id"] for t in all_types]
    labels = {t["id"]: f"{t['name']}（{t['level']}）" for t in all_types}

    picked = intent_result.get("picked")
    picked_id = picked["experiment_id"] if picked and picked.get("experiment_id") else None
    default_index = type_ids.index(picked_id) if picked_id in type_ids else 0
    if picked and not picked.get("experiment_id"):
        st.info(f"GLM 判断这可能是「{picked['name']}」，但不在已支持的实验类型里，"
                f"请从下拉框手动选择最接近的一项。")

    selected_id = st.selectbox(
        "识别到的实验类型（可手动修改）",
        options=type_ids,
        format_func=lambda k: labels[k],
        index=default_index,
    )
    exp = EXPERIMENT_TYPES[selected_id]

    st.subheader(f"请填写实测数据 — {exp['name']}（{exp['level']}）")
    st.caption("这些数值不需要能被照片识别出来，照实物读数/自己判断填写即可。")

    inputs: dict = {}
    with st.form("input_form"):
        for fld in exp["input_fields"]:
            label = f"{fld.label}" + (f" ({fld.unit})" if fld.unit else "")
            if not fld.required:
                label += " [可选]"
            if fld.type == "select":
                inputs[fld.key] = st.selectbox(label, fld.options)
            else:
                default_val = float(fld.default) if isinstance(fld.default, (int, float)) else 0.0
                inputs[fld.key] = st.number_input(label, value=default_val)
        submitted = st.form_submit_button("提交，生成结果", type="primary")

    if submitted:
        # 可选字段留空(0.0)时不强行传给计算函数——除非该字段就是合法取 0
        clean_inputs = {
            k: v for k, v in inputs.items()
            if not (v == 0.0 and not next(f.required for f in exp["input_fields"] if f.key == k))
        }
        try:
            result = compute_experiment(selected_id, clean_inputs)
        except (KeyError, ValueError) as e:
            st.error(f"计算失败，请检查必填项：{e}")
            st.stop()

        st.subheader("计算结果")
        st.json(result)

        st.subheader("示意动画")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as f:
            anim_path = f.name
        try:
            track = st.session_state.track_result["track"] if st.session_state.track_result else None
            generate_experiment_animation(selected_id, result, clean_inputs, anim_path, track=track)
            st.image(anim_path)
        except Exception as e:
            st.warning(f"动画生成失败：{e}")

        st.subheader("讲解 / 公式 / 知识点清单")
        try:
            explanation = generate_explanation(
                exp["name"], exp["level"], clean_inputs, result,
                image_path=st.session_state.init_image_path,
            )
        except GLMAPIError as e:
            st.warning(f"讲解生成暂不可用（{e}）。计算结果和动画不受影响。")
        else:
            st.markdown(explanation["explanation"])
            st.markdown("**涉及公式：**")
            for f_ in explanation["formulas"]:
                st.markdown(f"- {f_}")
            st.markdown("**知识点清单：**")
            for kp in explanation["knowledge_points"]:
                st.markdown(f"- {kp}")
