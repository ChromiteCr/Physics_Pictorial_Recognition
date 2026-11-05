"""Altair 图（Streamlit 前端自带 Vega-Lite，离线可用）。配色与黑板主题一致。"""

from __future__ import annotations

import altair as alt
import pandas as pd

from .theme import CHALK, SANS

GHOST_COLORS = [CHALK["yellow"], CHALK["coral"], CHALK["blue"], CHALK["green"]]


def _style(chart: alt.Chart) -> alt.Chart:
    font = SANS.replace("'", "")
    return (chart.configure(font=font, background="transparent")
            .configure_axis(labelColor=CHALK["muted"], titleColor=CHALK["cream"], gridColor="#2e4337",
                            domainColor=CHALK["muted"], tickColor=CHALK["muted"])
            .configure_legend(labelColor=CHALK["cream"], titleColor=CHALK["muted"], orient="bottom",
                              direction="vertical", columns=1, labelLimit=360)
            .configure_view(strokeWidth=0))


def data_chart(points: dict, title: str = "") -> alt.Chart:
    """学生数据（面板坐标）的散点 + 误差棒。"""
    df = pd.DataFrame({"x": points["x"], "y": points["y"], "sy": points["sy"]})
    df["lo"], df["hi"] = df["y"] - df["sy"], df["y"] + df["sy"]
    base = alt.Chart(df)
    pts = base.mark_circle(size=70, color=CHALK["cream"]).encode(
        x=alt.X("x:Q", title=points["x_label"]), y=alt.Y("y:Q", title=points["y_label"]),
        tooltip=[alt.Tooltip("x:Q", title=points["x_label"]), alt.Tooltip("y:Q", title=points["y_label"])])
    bars = base.mark_rule(color=CHALK["cream"], opacity=0.6).encode(x="x:Q", y="lo:Q", y2="hi:Q")
    return _style((bars + pts).properties(height=300, title=title))


def ghost_chart(points: dict, ghosts: list[dict]) -> alt.Chart:
    """幽灵曲线：每个高概率原因在它最可能的参数下"应该"画出的曲线，和学生数据叠在一起。"""
    rows = []
    for g in ghosts:
        label = f"{g['name']}（{round(g['prob'] * 100)}%）"
        rows += [{"x": x, "y": y, "原因": label} for x, y in zip(g["x"], g["y"])]
    lines_df = pd.DataFrame(rows, columns=["x", "y", "原因"])
    order = list(dict.fromkeys(lines_df["原因"])) if len(lines_df) else []
    color = alt.Color("原因:N", title="幽灵曲线", sort=order,
                      scale=alt.Scale(domain=order, range=GHOST_COLORS[:max(1, len(order))]))
    lines = alt.Chart(lines_df).mark_line(strokeWidth=2.2, strokeDash=[6, 4], opacity=0.9).encode(
        x=alt.X("x:Q", title=points["x_label"]), y=alt.Y("y:Q", title=points["y_label"]), color=color)
    df = pd.DataFrame({"x": points["x"], "y": points["y"], "sy": points["sy"]})
    df["lo"], df["hi"] = df["y"] - df["sy"], df["y"] + df["sy"]
    pts = alt.Chart(df).mark_circle(size=80, color=CHALK["cream"]).encode(
        x="x:Q", y="y:Q", tooltip=[alt.Tooltip("x:Q", title=points["x_label"]),
                                   alt.Tooltip("y:Q", title=points["y_label"])])
    bars = alt.Chart(df).mark_rule(color=CHALK["cream"], opacity=0.6).encode(x="x:Q", y="lo:Q", y2="hi:Q")
    return _style((lines + bars + pts).properties(height=340))
