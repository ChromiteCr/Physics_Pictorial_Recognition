"""① 拍照识别：上传照片 → YOLO 框 + 画布修正（P1 功能保留）→ 识别实验 → 一个澄清问题 → 手动选择永远可用。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from . import catalog, state
from .theme import chalk_loading, esc

CLASS_OPTIONS = ["cart", "track", "spring", "string", "ruler", "dynamometer", "wooden_block", "iron_block"]
CLASS_ZH = {"cart": "小车", "track": "轨道", "spring": "弹簧", "string": "细线", "ruler": "刻度尺",
            "dynamometer": "弹簧测力计", "wooden_block": "木块", "iron_block": "钢球/铁块"}
CANVAS_MAX_WIDTH = 760  # 画框画布的显示宽度上限（像素）
SOURCE_ZH = {"rule": "器材规则", "model": "本地模型", "rule_fallback": "器材规则（模型失败后退回）",
             "question": "你的回答", "manual": "你的手动选择"}


def _offline_env() -> None:
    for k, v in (("YOLO_OFFLINE", "true"), ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1")):
        os.environ.setdefault(k, v)


def _canvas():
    """streamlit-drawable-canvas 0.9.3 还在调旧位置的 image_to_url，补回去再导入（懒加载）。"""
    import streamlit.elements.image as _img

    if not hasattr(_img, "image_to_url"):
        from streamlit.elements.lib.image_utils import image_to_url

        _img.image_to_url = image_to_url
    from streamlit_drawable_canvas import st_canvas

    return st_canvas


def _save_upload(f) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
        tmp.write(f.getvalue())
        return tmp.name


def _rect_to_bbox(obj: dict, inv_scale: float, img_w: int, img_h: int) -> list[float]:
    """画布上的矩形（fabric.js 坐标）→ 原图像素坐标 [x1, y1, x2, y2]。"""
    eff_w = obj["width"] * obj.get("scaleX", 1)
    eff_h = obj["height"] * obj.get("scaleY", 1)
    x1, x2 = sorted((obj["left"] * inv_scale, (obj["left"] + eff_w) * inv_scale))
    y1, y2 = sorted((obj["top"] * inv_scale, (obj["top"] + eff_h) * inv_scale))
    x1, x2 = max(0.0, x1), min(float(img_w), x2)
    y1, y2 = max(0.0, y1), min(float(img_h), y2)
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


# ── 照片：检测 + 画布修正 ──────────────────────────────────────────────


def _photo_section(model) -> None:
    st.markdown("#### 拍一张实验装置的照片")
    st.caption("照片只在本机处理。YOLO 先找出器材，你可以在图上补框、在表里改类别，再让它判断是哪个实验。")
    up = st.file_uploader("实验装置照片", type=["jpg", "jpeg", "png", "webp"], key="photo")
    with st.expander("检测参数（一般不用改）"):
        weights = st.text_input("YOLO 权重路径", os.environ.get("PPR_YOLO_WEIGHTS", "assets/weights/best.pt"),
                                key="yolo_weights")
        conf = st.slider("置信度阈值", 0.05, 0.9, 0.25, 0.05, key="yolo_conf")

    if up is not None and st.button("开始检测器材", type="primary", key="detect_btn"):
        path = _save_upload(up)
        st.session_state.init_image_path = path
        _offline_env()
        try:
            with chalk_loading("正在识别器材（YOLO 推理）……"):
                import cv2

                from detect import detect_components  # 懒加载：导入时才加载 ultralytics / torch

                res = detect_components(path, weights, conf)
        except Exception as e:  # noqa: BLE001  权重缺失、图片读不了：手动选择仍然可用
            st.error(f"器材检测失败：{e}。可以直接在下面手动选择实验。")
        else:
            st.session_state.components = res["detections"]
            st.session_state.annotated_image_rgb = cv2.cvtColor(res["annotated_image"], cv2.COLOR_BGR2RGB)
            st.session_state.recog = None
            st.session_state.canvas_key_version += 1

    if st.session_state.components is None:
        return

    st.image(st.session_state.annotated_image_rgb, caption="检测结果", use_container_width=True)
    st.markdown("##### 手动补框：把漏检的器材圈出来")
    st.caption("在图上拖框，点「同步新框到下表」，再到表里给新框选类别。改类别、删框请直接在表里操作。")
    try:
        from PIL import Image

        img = Image.open(st.session_state.init_image_path).convert("RGB")
        w, h = img.size
        cw = min(CANVAS_MAX_WIDTH, w)
        scale = cw / w
        ch = round(h * scale)
        canvas = _canvas()(
            fill_color="rgba(240, 131, 106, 0.15)", stroke_width=3, stroke_color="#f0836a",
            background_image=img.resize((cw, ch)), height=ch, width=cw, drawing_mode="rect",
            display_toolbar=True, key=f"canvas_{st.session_state.canvas_key_version}",
        )
    except Exception as e:  # noqa: BLE001  画布组件不可用时只保留表格修正
        st.warning(f"画布不可用（{e}），请直接在下表里增删器材。")
        canvas, scale, w, h = None, 1.0, 0, 0

    if canvas is not None and st.button("同步新框到下表", key="sync_boxes"):
        new = []
        for obj in (canvas.json_data or {}).get("objects", []):
            if obj.get("type") == "rect":
                new.append({"class_id": None, "class_name": "", "confidence": None,
                            "bbox": _rect_to_bbox(obj, 1 / scale, w, h)})
        if new:
            st.session_state.components = list(st.session_state.components) + new
            st.session_state.canvas_key_version += 1
            st.toast(f"已添加 {len(new)} 个新框，请在下表选择类别。")
            st.rerun()
        else:
            st.warning("画布里没有矩形框，先在图上拖一个框再同步。")

    st.markdown("##### 器材清单（可编辑：改类别、加/删行）")
    df = pd.DataFrame(st.session_state.components)
    if df.empty:
        df = pd.DataFrame(columns=["class_id", "class_name", "confidence", "bbox"])
    edited = st.data_editor(
        df, num_rows="dynamic", use_container_width=True, key=f"components_{st.session_state.canvas_key_version}",
        column_config={
            "class_name": st.column_config.SelectboxColumn("类别", options=CLASS_OPTIONS, required=True),
            "confidence": st.column_config.NumberColumn("置信度", disabled=True),
            "bbox": st.column_config.Column("边界框（只读）", disabled=True),
            "class_id": None,
        },
    )
    if st.button("确认器材，识别实验", type="primary", key="recognize_btn"):
        records = edited.dropna(subset=["class_name"]).to_dict("records")
        dets = [{"class_id": r.get("class_id"), "class_name": r["class_name"], "confidence": r.get("confidence"),
                 "bbox": r.get("bbox")} for r in records if r.get("class_name")]
        from lab_detective.vision import recognize

        with chalk_loading("正在判断是哪个实验……"):
            res = recognize(st.session_state.init_image_path, model, detections=dets)
        st.session_state.recog = res.to_dict()


# ── 识别结果 + 澄清问题 ───────────────────────────────────────────────


def _recognition_section() -> None:
    from lab_detective.vision import RecognitionResult, answer

    d = st.session_state.recog
    if not d:
        return
    res = RecognitionResult.from_dict(d)
    st.markdown("#### 识别结果")
    cols = st.columns([2, 1, 1])
    cols[0].metric("可能是", res.open_name)
    cols[1].metric("把握", f"{res.confidence * 100:.0f}%")
    cols[2].metric("依据", SOURCE_ZH.get(res.source, res.source))
    if res.cues:
        st.caption("装置线索：" + "；".join(res.cues))
    if res.apparatus:
        st.caption("器材：" + "、".join(CLASS_ZH.get(a, a) for a in res.apparatus))
    for w in res.warnings:
        st.info(w)
    if res.needs_question and res.question:
        q = res.question
        st.markdown(f"**还差一个问题：{esc(q['text'])}**")
        labels = [o["label"] for o in q["options"]]
        rkey = f"clarify_{q['id']}"
        pick = st.radio("选一个最符合的", labels, key=rkey, index=None)

        def _answer() -> None:  # 回调里更新结果，下一轮直接画新状态（不用 st.rerun）
            chosen = st.session_state.get(rkey)
            if chosen in labels:
                st.session_state.recog = answer(res, labels.index(chosen)).to_dict()

        st.button("回答", key="clarify_btn", disabled=pick is None, on_click=_answer)
    elif res.experiment_id:
        if st.button(f"就做「{res.open_name}」", type="primary", key="use_recognized",
                     on_click=state.set_experiment, args=(res.experiment_id,)):
            st.success("已选定实验，去「② 实验设计」。")
    else:
        st.info("这台装置不在整理好的实验里：在下面选「其他实验」，用物理组件组合出它的孪生。")


# ── 手动选择（永远可用）与其他实验 ─────────────────────────────────────


def _other_form() -> None:
    st.markdown("##### 其他实验：用物理组件组合")
    st.caption("整理好的实验只是起点。其他装置可以按“控制一个量、读另一个量、两者满足某种关系”组合出孪生："
               "原因清单是通用的误差类型（起点/零点偏差、比例不准、大范围弯曲、读数散布、漂移），"
               "不是针对这台装置推导的专用孪生。")
    with st.form("other_form"):
        name = st.text_input("实验名称", "探究同种物质的质量与体积的关系", key="o_name")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**自变量 x（你主动改变的量）**")
            xn = st.text_input("名称", "体积", key="o_xn")
            xs = st.text_input("符号", "V", key="o_xs")
            xu = st.text_input("单位", "cm³", key="o_xu")
            xt = st.text_input("用什么量", "量筒", key="o_xt")
            xr = st.number_input("仪器分辨率", min_value=0.0, value=1.0, format="%g", key="o_xr")
        with c2:
            st.markdown("**因变量 y（你读出来的量）**")
            yn = st.text_input("名称", "质量", key="o_yn")
            ys = st.text_input("符号", "m", key="o_ys")
            yu = st.text_input("单位", "g", key="o_yu")
            yt = st.text_input("用什么读", "电子天平", key="o_yt")
            yr = st.number_input("仪器分辨率", min_value=0.0, value=0.1, format="%g", key="o_yr")
        fam = st.selectbox("y 与 x 的关系", list(catalog.FAMILIES), format_func=catalog.FAMILIES.get, key="o_fam")
        expo = st.number_input("幂关系的指数 n（只在选“幂”时用）", value=2.0, format="%g", key="o_exp")
        settings = st.text_input("x 打算取哪些值（逗号或空格分隔）", "10, 20, 30, 40, 50", key="o_set")
        known = st.number_input("理论斜率 a（知道就填，例如已知密度；不知道填 0）", value=0.0, format="%g", key="o_known")
        guess = st.number_input("斜率 a 的粗略估计（可不填；不填时等数据测出来后再定）", value=0.0, format="%g",
                                key="o_guess")
        ok = st.form_submit_button("组合出孪生", type="primary")
    if ok:
        try:
            spec = catalog.make_spec(
                name, {"name": xn, "symbol": xs, "unit": xu, "tool": xt, "resolution": xr},
                {"name": yn, "symbol": ys, "unit": yu, "tool": yt, "resolution": yr},
                fam, catalog.parse_settings(settings), exponent=expo,
                slope_known=known or None, slope_guess=guess or None)
            key = catalog.other_key(spec)
            catalog.get_experiment(key)  # 先构造一次，参数不合理时在这里报错
        except (ValueError, KeyError) as e:
            st.error(f"还不能组合：{e}")
        else:
            state.set_experiment(key)
            st.success(f"已组合「{spec['name']}」（{catalog.backend_name(key)}），去「② 实验设计」。")


def _manual_section() -> None:
    st.markdown("#### 手动选择实验（随时可用）")
    items = catalog.curated()
    ids = [i["id"] for i in items] + [catalog.OTHER]
    names = {i["id"]: (f"{i['name']}（{i['grade']}）" if i["grade"] else i["name"]) for i in items}
    names[catalog.OTHER] = "其他实验（用物理组件组合）"
    cur = st.session_state.get("exp_key")
    idx = ids.index(cur) if cur in ids else (len(ids) - 1 if catalog.is_other(cur) else 0)
    pick = st.selectbox("整理好的实验是起点；不在列表里的装置选“其他实验”", ids, index=idx,
                        format_func=names.get, key="manual_pick")
    if pick == catalog.OTHER:
        _other_form()
    elif st.button("开始这个实验", type="primary", key="pick_manual", on_click=_pick_cb):
        st.success(f"已选定「{names[pick]}」，去「② 实验设计」。")


def _pick_cb() -> None:
    pick = st.session_state.get("manual_pick")  # 回调里读下拉框的当前值，不用渲染时绑定的旧值
    if pick and pick != catalog.OTHER:
        state.set_experiment(pick)


def render(model=None) -> None:
    st.subheader("① 拍照识别")
    left, right = st.columns([3, 2], gap="large")
    with left:
        _photo_section(model)
        _recognition_section()
    with right:
        _manual_section()
        cur = state.current()
        if cur:
            st.markdown(f"**当前实验**：{esc(cur[1].name_zh)}")
