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

import contextlib
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

# streamlit-drawable-canvas (0.9.3) 内部还在调用旧版 streamlit 的
# streamlit.elements.image.image_to_url，这个函数在新版 streamlit 里挪到了
# streamlit.elements.lib.image_utils，旧位置已经删除，直接 import 会在真正画
# 背景图时抛 AttributeError。这里把新位置的同签名函数补回旧模块上，不用等
# 上游库更新或降级 streamlit。
import streamlit.elements.image as _st_image_compat
if not hasattr(_st_image_compat, "image_to_url"):
    from streamlit.elements.lib.image_utils import image_to_url as _image_to_url_compat
    _st_image_compat.image_to_url = _image_to_url_compat

from streamlit_drawable_canvas import st_canvas

from detect import detect_components, extract_reference_track, DEFAULT_WEIGHTS
from physics import EXPERIMENT_TYPES, compute_experiment, list_experiment_types
from intent_classifier import classify_intent_with_fallback
from animate import generate_experiment_animation
from explain_api import generate_explanation, GLMAPIError

st.set_page_config(page_title="力学实验识别与讲解", layout="wide")

CLASS_OPTIONS = ["cart", "track", "spring", "string", "ruler", "dynamometer",
                  "wooden_block", "iron_block"]

CANVAS_MAX_WIDTH = 760  # 画框画布的显示宽度上限（像素），超过这个宽度的图会按比例缩小


def _inject_chalkboard_theme():
    """黑板粉笔实验记录本视觉——中文标题用手写粉笔字体，数值用等宽字体，
    选项卡下划线做成虚线粉笔笔触。配色主体交给 .streamlit/config.toml 的
    [theme]，这里只补字体和几处签名细节（data_editor 是内部 canvas 渲染，
    CSS 管不到，不强求）。"""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Ma+Shan+Zheng&family=Noto+Sans+SC:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] { font-family: 'Noto Sans SC', sans-serif; }

        h1, h2, h3 {
            font-family: 'Ma Shan Zheng', 'Noto Sans SC', cursive;
            letter-spacing: 0.04em;
        }
        h1 {
            color: #eec25f !important;
            font-size: 2.5rem !important;
            padding-bottom: 0.2rem;
        }
        h1::after {
            content: "";
            display: block;
            width: 6rem;
            margin-top: 0.5rem;
            border-bottom: 3px dashed #f0836a;
            opacity: 0.75;
        }

        code, [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 1.6rem;
            border-bottom: 1px solid rgba(243,239,224,0.18);
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Ma Shan Zheng', 'Noto Sans SC', cursive;
            font-size: 1.1rem;
            color: #a9b6ac;
        }
        .stTabs [aria-selected="true"] {
            color: #eec25f !important;
            border-bottom: 3px dashed #eec25f !important;
        }

        .stButton>button, .stFormSubmitButton>button {
            border: 1.5px solid #f3efe0;
            border-radius: 4px;
            transition: all 0.15s ease;
        }
        .stButton>button:hover, .stFormSubmitButton>button:hover {
            border-color: #eec25f;
            color: #eec25f;
        }

        .stCaption, [data-testid="stCaptionContainer"] { color: #a9b6ac !important; }

        /* 加载进度条：轨道画成带刻度的"尺子"，呼应力学实验的测量主题；
           这些操作(YOLO推理/GLM调用)时长不定，做不出真实百分比，用来回滑动的
           不定进度条表示"正在量"，纯 CSS 动画，Python 侧阻塞时浏览器也照常播放。 */
        .chalk-loading-wrap { margin: 0.7rem 0 1.1rem 0; }
        .chalk-loading-label {
            font-size: 0.92rem;
            color: #eec25f;
            margin-bottom: 0.45rem;
            letter-spacing: 0.02em;
        }
        .chalk-loading-track {
            position: relative;
            height: 10px;
            border-radius: 5px;
            background:
                repeating-linear-gradient(90deg,
                    rgba(243,239,224,0.14) 0px, rgba(243,239,224,0.14) 1px,
                    transparent 1px, transparent 9px),
                rgba(243,239,224,0.06);
            border: 1px solid rgba(243,239,224,0.25);
            overflow: hidden;
        }
        .chalk-loading-fill {
            position: absolute;
            top: 0; left: -40%;
            height: 100%;
            width: 40%;
            border-radius: 5px;
            background: linear-gradient(90deg, transparent, #eec25f 35%, #f0836a 65%, transparent);
            box-shadow: 0 0 8px rgba(238,194,95,0.55);
            animation: chalk-slide 1.3s ease-in-out infinite;
        }
        @keyframes chalk-slide {
            0%   { left: -40%; }
            100% { left: 100%; }
        }
        @media (prefers-reduced-motion: reduce) {
            .chalk-loading-fill { animation: none; left: 0; width: 100%; opacity: 0.5; }
        }

        /* Part 2 计算结果卡片，代替 st.json——数值结果不是代码，别让它长得像代码 */
        .chalk-result-title {
            font-family: 'Ma Shan Zheng', 'Noto Sans SC', cursive;
            font-size: 1.3rem;
            color: #f3efe0;
            margin-bottom: 0.6rem;
        }
        .chalk-badge {
            display: inline-block;
            padding: 0.25rem 0.8rem;
            border-radius: 999px;
            font-size: 0.85rem;
            margin-bottom: 0.9rem;
            border: 1px solid;
        }
        .chalk-badge-ok { color: #8fd19e; border-color: #8fd19e; background: rgba(143,209,158,0.08); }
        .chalk-badge-warn { color: #f0836a; border-color: #f0836a; background: rgba(240,131,106,0.08); }
        [data-testid="stMetric"] {
            background: rgba(243,239,224,0.04);
            border: 1px solid rgba(243,239,224,0.16);
            border-radius: 8px;
            padding: 0.7rem 0.9rem 0.5rem 0.9rem;
        }
        [data-testid="stMetricLabel"] { color: #a9b6ac !important; }
        [data-testid="stMetricValue"] { color: #eec25f !important; }
        .chalk-formula {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1rem;
            color: #7fb8c9;
            background: rgba(127,184,201,0.08);
            border-left: 3px dashed #7fb8c9;
            padding: 0.5rem 0.9rem;
            margin-top: 0.7rem;
            border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@contextlib.contextmanager
def chalk_loading(message: str):
    """黑板粉笔风格的加载指示，替代 st.spinner，包住所有耗时等待
    （YOLO 推理、GLM 调用、动画生成）统一视觉。"""
    placeholder = st.empty()
    placeholder.markdown(
        f'<div class="chalk-loading-wrap">'
        f'<div class="chalk-loading-label">{message}</div>'
        f'<div class="chalk-loading-track"><div class="chalk-loading-fill"></div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        placeholder.empty()


_inject_chalkboard_theme()
st.markdown(
    '<h1>力学实验智能助教</h1>'
    '<p style="color:#a9b6ac;margin-top:-0.8rem;">拍照识别器材 → 画框补标注 → '
    '填测量数据 → 生成动画与讲解</p>',
    unsafe_allow_html=True,
)

# ── 会话状态初始化 ──────────────────────────────────────────────────
for key, default in [
    ("components", None), ("annotated_image_rgb", None), ("confirmed", False),
    ("track_result", None), ("init_image_path", None), ("intent_result", None),
    ("canvas_key_version", 0),
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


# compute_experiment() 的返回字段是英文 snake_case 的物理量 key，直接摊平成
# 卡片的话不好认；这里给 physics.py 六种实验会用到的字段配中文标签+单位，没配
# 到的 key（比如 surface_material）走兜底显示，不会因为漏配就消失。
RESULT_LABELS: dict[str, tuple[str, str]] = {
    "friction_force_N": ("摩擦力", "N"),
    "normal_force_N": ("正压力", "N"),
    "mu_estimated": ("动摩擦因数 μ", ""),
    "average_speed_m_s": ("平均速度", "m/s"),
    "distance_m": ("运动距离", "m"),
    "time_s": ("运动时间", "s"),
    "spring_constant_N_per_m": ("劲度系数 k", "N/m"),
    "intercept_N": ("拟合截距", "N"),
    "n_measurements": ("测量组数", ""),
    "deformation_cm": ("形变量", "cm"),
    "force_n": ("力值", "N"),
    "mass_kg": ("质量", "kg"),
    "force_N": ("拉力", "N"),
    "measured_accel_m_s2": ("实测加速度", "m/s²"),
    "predicted_accel_m_s2": ("预测加速度", "m/s²"),
    "accel_m_s2": ("加速度", "m/s²"),
    "error_ratio": ("误差比例", ""),
    "momentum_before_kg_m_s": ("碰撞前总动量", "kg·m/s"),
    "momentum_after_kg_m_s": ("碰撞后总动量", "kg·m/s"),
    "conservation_error_ratio": ("守恒误差比例", ""),
    "initial_potential_energy_J": ("初始势能", "J"),
    "final_kinetic_energy_J": ("末动能", "J"),
}


def _render_result(result: dict):
    """把 compute_experiment() 的结果 dict 渲染成卡片，而不是 st.json——
    这是给学生看的计算结果，不是代码，长得像 JSON 容易让人误会。"""
    skip_keys = {"formula", "verified", "experiment_id", "experiment_name"}

    if "verified" in result:
        ok = bool(result["verified"])
        badge = "✓ 结论：守恒 / 验证通过" if ok else "✗ 结论：误差偏大，建议检查测量或操作"
        cls = "chalk-badge-ok" if ok else "chalk-badge-warn"
        st.markdown(f'<div class="chalk-badge {cls}">{badge}</div>', unsafe_allow_html=True)

    numeric_items, text_items = [], []
    for k, v in result.items():
        if k in skip_keys or v is None:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            numeric_items.append((k, v))
        else:
            text_items.append((k, v))

    if numeric_items:
        cols = st.columns(min(3, len(numeric_items)))
        for i, (k, v) in enumerate(numeric_items):
            label, unit = RESULT_LABELS.get(k, (k.replace("_", " "), ""))
            if k.endswith("_ratio"):
                display_val = f"{v * 100:.2f}%"
            else:
                display_val = f"{v:g}" if isinstance(v, float) else str(v)
                if unit:
                    display_val += f" {unit}"
            with cols[i % len(cols)]:
                st.metric(label, display_val)

    for k, v in text_items:
        label, _ = RESULT_LABELS.get(k, (k.replace("_", " "), ""))
        st.caption(f"{label}：{v}")

    if result.get("formula"):
        st.markdown(f'<div class="chalk-formula">{result["formula"]}</div>', unsafe_allow_html=True)


def _rect_to_bbox(obj: dict, inv_scale: float, img_w: int, img_h: int) -> list[float]:
    """把画布上的矩形对象（fabric.js 坐标）换算回原图像素坐标 [x1,y1,x2,y2]。"""
    eff_w = obj["width"] * obj.get("scaleX", 1)
    eff_h = obj["height"] * obj.get("scaleY", 1)
    x1, x2 = sorted((obj["left"] * inv_scale, (obj["left"] + eff_w) * inv_scale))
    y1, y2 = sorted((obj["top"] * inv_scale, (obj["top"] + eff_h) * inv_scale))
    x1, x2 = max(0.0, x1), min(float(img_w), x2)
    y1, y2 = max(0.0, y1), min(float(img_h), y2)
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


tab_part1, tab_part2 = st.tabs(["① 识别初状态", "② 实验分析"])

# ══════════════════════════════════════════════════════════════════
# Part 1
# ══════════════════════════════════════════════════════════════════
with tab_part1:
    st.subheader("识别实验初状态")
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
            with chalk_loading("正在识别器材（YOLO 推理中）..."):
                result = detect_components(img_path, weights_path, conf)
        except FileNotFoundError as e:
            st.error(str(e))
        else:
            st.session_state.components = result["detections"]
            st.session_state.annotated_image_rgb = _bgr_to_rgb(result["annotated_image"])
            st.session_state.confirmed = False
            st.session_state.intent_result = None  # 器材变了，意图识别要重跑
            st.session_state.canvas_key_version += 1  # 换了张图，画布也要清空重来

        if uploaded_video:
            video_path = _save_uploaded_file(uploaded_video)
            with chalk_loading("正在提取视频参考轨迹..."):
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

        st.subheader("手动补框：把模型漏检的器材圈出来")
        st.caption("在下面的图上拖框，圈出模型没检测到的器材，点「同步新框到下表」后，"
                   "去下面的表格里给新框选类别。已有检测框只是画在图上供参考，画新框不会"
                   "动到它们；调整/删除已有框请直接在下面的表格里操作。")

        orig_img = Image.open(st.session_state.init_image_path).convert("RGB")
        img_w, img_h = orig_img.size
        canvas_w = min(CANVAS_MAX_WIDTH, img_w)
        scale = canvas_w / img_w
        canvas_h = round(img_h * scale)
        bg_for_canvas = orig_img.resize((canvas_w, canvas_h))

        canvas_result = st_canvas(
            fill_color="rgba(240, 131, 106, 0.15)",
            stroke_width=3,
            stroke_color="#f0836a",
            background_image=bg_for_canvas,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="rect",
            display_toolbar=True,
            key=f"canvas_{st.session_state.canvas_key_version}",
        )

        if st.button("同步新框到下表"):
            new_boxes = []
            if canvas_result.json_data is not None:
                for obj in canvas_result.json_data["objects"]:
                    if obj.get("type") != "rect":
                        continue
                    bbox = _rect_to_bbox(obj, 1 / scale, img_w, img_h)
                    new_boxes.append({"class_id": None, "class_name": "",
                                       "confidence": None, "bbox": bbox})
            if new_boxes:
                st.session_state.components = list(st.session_state.components) + new_boxes
                st.session_state.canvas_key_version += 1  # 换个 key，画布重新变空白
                st.toast(f"已添加 {len(new_boxes)} 个新框，请到下表选择类别。")
                st.rerun()
            else:
                st.warning("画布里没有矩形框，先在图上拖一个框再同步。")

        st.subheader("检测到的器材（可编辑：改类别、加/删行）")
        st.caption("识别可能有误差，置信度低的行建议重点核对；新框/手动新增的行"
                   "在这里选类别即可，bbox/置信度不可编辑。")
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
            st.success("已确认。请点击上方「② 实验分析」")

# ══════════════════════════════════════════════════════════════════
# Part 2
# ══════════════════════════════════════════════════════════════════
with tab_part2:
    st.subheader("识别实验意图 → 填实测数据 → 动画 + 讲解")

    if not st.session_state.confirmed:
        st.warning("请先在「① 识别初状态」完成检测并点击「确认器材列表」。")
        st.stop()

    components = st.session_state.components
    st.write("已确认的器材：", ", ".join(c["class_name"] for c in components) or "（无）")

    if st.session_state.intent_result is None:
        with chalk_loading("正在识别实验意图..."):
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

        res_tab, anim_tab, explain_tab = st.tabs(["计算结果", "示意动画", "讲解 / 公式 / 知识点"])

        with res_tab:
            _render_result(result)

        with anim_tab:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as f:
                anim_path = f.name
            try:
                with chalk_loading("正在生成示意动画..."):
                    track = st.session_state.track_result["track"] if st.session_state.track_result else None
                    generate_experiment_animation(selected_id, result, clean_inputs, anim_path, track=track)
                st.image(anim_path)
            except Exception as e:
                st.warning(f"动画生成失败：{e}")

        with explain_tab:
            try:
                with chalk_loading("正在生成 GLM 讲解..."):
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
                    if isinstance(kp, dict):
                        st.markdown(f"- **{kp.get('term', '')}**：{kp.get('detail', '')}")
                    else:
                        st.markdown(f"- {kp}")  # 兜底：GLM 没按新格式返回时至少不崩
