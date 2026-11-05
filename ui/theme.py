"""黑板粉笔主题（完全离线）：配色在 .streamlit/config.toml 的 [theme]，这里补字体和几处签名细节。

不加载任何在线字体：中文用系统自带字体（PingFang SC / Hiragino Sans GB / Noto Sans CJK），
标题用系统楷体营造粉笔手写感，数字用系统等宽字体。
"""

from __future__ import annotations

import contextlib
import html

import streamlit as st

SANS = "'PingFang SC', 'Hiragino Sans GB', 'Noto Sans CJK SC', 'Noto Sans CJK', 'Microsoft YaHei', sans-serif"
KAI = "'Kaiti SC', 'STKaiti', 'KaiTi', " + SANS
MONO = "'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

CHALK = {
    "yellow": "#eec25f",
    "coral": "#f0836a",
    "cream": "#f3efe0",
    "muted": "#a9b6ac",
    "green": "#8fd19e",
    "blue": "#7fb8c9",
    "board": "#1c2b23",
}
BAND_COLOR = {"consistent": CHALK["green"], "tension": CHALK["yellow"], "inconsistent": CHALK["coral"]}
BAND_ZH = {"consistent": "一致", "tension": "有张力", "inconsistent": "不一致"}

CSS = f"""
<style>
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{ font-family: {SANS}; }}
h1, h2, h3 {{ font-family: {KAI}; letter-spacing: 0.04em; }}
h1 {{ color: {CHALK['yellow']} !important; font-size: 2.3rem !important; padding-bottom: 0.2rem; }}
h1::after {{ content: ""; display: block; width: 6rem; margin-top: 0.5rem;
            border-bottom: 3px dashed {CHALK['coral']}; opacity: 0.75; }}
code, [data-testid="stMetricValue"] {{ font-family: {MONO} !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.2rem; border-bottom: 1px solid rgba(243,239,224,0.18); }}
.stTabs [data-baseweb="tab"] {{ font-family: {KAI}; font-size: 1.08rem; color: {CHALK['muted']}; }}
.stTabs [aria-selected="true"] {{ color: {CHALK['yellow']} !important; border-bottom: 3px dashed {CHALK['yellow']} !important; }}
.stButton>button, .stFormSubmitButton>button {{ border: 1.5px solid {CHALK['cream']}; border-radius: 4px;
            transition: border-color 0.15s ease, color 0.15s ease; }}
.stButton>button:hover, .stFormSubmitButton>button:hover {{ border-color: {CHALK['yellow']}; color: {CHALK['yellow']}; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: {CHALK['muted']} !important; }}
[data-testid="stMetric"] {{ background: rgba(243,239,224,0.04); border: 1px solid rgba(243,239,224,0.16);
            border-radius: 8px; padding: 0.7rem 0.9rem 0.5rem 0.9rem; }}
[data-testid="stMetricLabel"] {{ color: {CHALK['muted']} !important; }}
[data-testid="stMetricValue"] {{ color: {CHALK['yellow']} !important; }}

/* 加载条：轨道画成带刻度的尺子，时长不定所以来回滑动；纯 CSS，Python 阻塞时也照常播放 */
.chalk-loading-wrap {{ margin: 0.7rem 0 1.1rem 0; }}
.chalk-loading-label {{ font-size: 0.92rem; color: {CHALK['yellow']}; margin-bottom: 0.45rem; }}
.chalk-loading-track {{ position: relative; height: 10px; border-radius: 5px; overflow: hidden;
    background: repeating-linear-gradient(90deg, rgba(243,239,224,0.14) 0px, rgba(243,239,224,0.14) 1px,
                transparent 1px, transparent 9px), rgba(243,239,224,0.06);
    border: 1px solid rgba(243,239,224,0.25); }}
.chalk-loading-fill {{ position: absolute; top: 0; left: -40%; height: 100%; width: 40%; border-radius: 5px;
    background: linear-gradient(90deg, transparent, {CHALK['yellow']} 35%, {CHALK['coral']} 65%, transparent);
    animation: chalk-slide 1.3s ease-in-out infinite; }}
@keyframes chalk-slide {{ 0% {{ left: -40%; }} 100% {{ left: 100%; }} }}
@media (prefers-reduced-motion: reduce) {{ .chalk-loading-fill {{ animation: none; left: 0; width: 100%; opacity: 0.5; }} }}

.chalk-badge {{ display: inline-block; padding: 0.2rem 0.75rem; border-radius: 999px; font-size: 0.88rem;
    margin: 0.1rem 0.3rem 0.5rem 0; border: 1px solid; }}
.chalk-card {{ border: 1px dashed rgba(243,239,224,0.35); border-radius: 8px; padding: 0.8rem 1rem;
    margin: 0.4rem 0 0.9rem 0; background: rgba(243,239,224,0.03); }}
.chalk-card h4 {{ font-family: {KAI}; margin: 0 0 0.4rem 0; color: {CHALK['yellow']}; }}
.chalk-formula {{ font-family: {MONO}; font-size: 0.98rem; color: {CHALK['blue']}; background: rgba(127,184,201,0.08);
    border-left: 3px dashed {CHALK['blue']}; padding: 0.45rem 0.9rem; margin: 0.5rem 0; border-radius: 4px; }}
.suspect {{ margin: 0.35rem 0 0.75rem 0; }}
.suspect-head {{ display: flex; justify-content: space-between; font-size: 0.98rem; }}
.suspect-name {{ color: {CHALK['cream']}; }}
.suspect-pct {{ font-family: {MONO}; color: {CHALK['yellow']}; }}
.suspect-bar {{ height: 9px; border-radius: 5px; background: rgba(243,239,224,0.10); margin: 0.25rem 0; overflow: hidden; }}
.suspect-fill {{ height: 100%; border-radius: 5px; background: linear-gradient(90deg, {CHALK['yellow']}, {CHALK['coral']}); }}
.suspect-ev {{ font-size: 0.86rem; color: {CHALK['muted']}; }}
.kp-path {{ font-size: 0.9rem; color: {CHALK['muted']}; }}
.kp-path b {{ color: {CHALK['cream']}; font-weight: 500; }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text))


@contextlib.contextmanager
def chalk_loading(message: str):
    """黑板风格的加载条，包住耗时操作（YOLO 推理、模型请求）。"""
    ph = st.empty()
    ph.markdown(
        f'<div class="chalk-loading-wrap"><div class="chalk-loading-label">{esc(message)}</div>'
        f'<div class="chalk-loading-track"><div class="chalk-loading-fill"></div></div></div>',
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        ph.empty()


def badge(text: str, color: str) -> str:
    return (f'<span class="chalk-badge" style="color:{color};border-color:{color};'
            f'background:rgba(243,239,224,0.04)">{esc(text)}</span>')


def band_badge(band: str | None, prefix: str = "") -> str:
    if not band:
        return badge(f"{prefix}数据点太少，暂不判断", CHALK["muted"])
    return badge(f"{prefix}{BAND_ZH.get(band, band)}", BAND_COLOR.get(band, CHALK["muted"]))


def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="chalk-card"><h4>{esc(title)}</h4>{body_html}</div>', unsafe_allow_html=True)


def formula(text: str) -> None:
    st.markdown(f'<div class="chalk-formula">{esc(text)}</div>', unsafe_allow_html=True)


def suspect_rows(ranked: list[dict]) -> str:
    """嫌疑人看板：每个原因一行，名字 + 概率条 + 证据。"""
    out = []
    for r in ranked:
        pct = max(0.0, min(1.0, float(r.get("prob") or 0.0))) * 100
        out.append(
            f'<div class="suspect"><div class="suspect-head"><span class="suspect-name">{esc(r.get("name", ""))}</span>'
            f'<span class="suspect-pct">{pct:.0f}%</span></div>'
            f'<div class="suspect-bar"><div class="suspect-fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="suspect-ev">证据：{esc(r.get("evidence", ""))}</div></div>'
        )
    return "".join(out)
