"""Render LLM chart specs using the SmartAnalytics visual theme."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SKY = "#0ea5e9"
SKY_LIGHT = "#7dd3fc"
SKY_DEEP = "#0369a1"
PIE_COLORS = [SKY, SKY_LIGHT, SKY_DEEP, "#38bdf8", "#0284c7", "#bae6fd"]


def bare(fig, height: int):
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=6, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(size=11, color="#64748b"),
    )
    return fig


def _pick_column(df: pd.DataFrame, name: str | None, fallback_idx: int = 0) -> str:
    if name and name in df.columns:
        return name
    return df.columns[fallback_idx] if len(df.columns) > fallback_idx else df.columns[0]


def _chart_key(spec: dict[str, Any]) -> str:
    if spec.get("id"):
        return str(spec["id"])
    raw = f"{spec.get('title', '')}|{spec.get('sql', '')}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _spec_with_id(spec: dict[str, Any], suffix: str) -> dict[str, Any]:
    base_id = spec.get("id") or _chart_key(spec)
    return {**spec, "id": f"{base_id}{suffix}"}


def render_insight(spec: dict[str, Any], height: int = 280) -> None:
    chart_type = spec.get("chart_type", "table")
    df: pd.DataFrame = spec.get("data")
    title = spec.get("title", "Insight")
    key = _chart_key(spec)

    if df is None or df.empty:
        st.warning(f"No data for: {title}", icon="⚠️")
        return

    st.markdown(f"#### {title}")

    if chart_type == "kpi":
        row = df.iloc[0]
        value = row.get("value", row.iloc[0])
        label = row.get("label", "")
        label_html = f'<div class="metric-label">{label}</div>' if label else '<div class="metric-label">KPI</div>'
        st.markdown(
            f'{label_html}<div class="metric-value">{value}</div>',
            unsafe_allow_html=True,
        )
        if len(df.columns) >= 2 and len(df) > 1:
            y_col = _pick_column(df, spec.get("y"), 1)
            x_col = _pick_column(df, spec.get("x"), 0)
            spark = go.Figure(
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode="lines",
                    line=dict(color=SKY, width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(14,165,233,.18)",
                )
            )
            spark.update_xaxes(visible=False)
            spark.update_yaxes(visible=False)
            st.plotly_chart(bare(spark, 90), use_container_width=True, key=f"kpi_spark_{key}")
        return

    if chart_type == "table":
        st.dataframe(df, use_container_width=True, key=f"table_{key}")
        return

    x_col = _pick_column(df, spec.get("x"), 0)
    y_col = _pick_column(df, spec.get("y"), 1)
    group_col = spec.get("groupby")

    if chart_type == "bar":
        fig = px.bar(df, x=x_col, y=y_col)
        fig.update_traces(marker_color=SKY_LIGHT)
        fig.update_xaxes(gridcolor="#e2e8f0", title=None)
        fig.update_yaxes(gridcolor="#e2e8f0", title=None)
        st.plotly_chart(bare(fig, height), use_container_width=True, key=f"bar_{key}")
        return

    if chart_type == "grouped_bar":
        color = group_col if group_col and group_col in df.columns else None
        fig = px.bar(df, x=x_col, y=y_col, color=color, barmode="group")
        fig.update_xaxes(gridcolor="#e2e8f0", title=None)
        fig.update_yaxes(gridcolor="#e2e8f0", title=None)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.12))
        st.plotly_chart(bare(fig, height), use_container_width=True, key=f"grouped_bar_{key}")
        return

    if chart_type == "line":
        fig = go.Figure()
        if group_col and group_col in df.columns:
            for i, (name, group) in enumerate(df.groupby(group_col)):
                color = PIE_COLORS[i % len(PIE_COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=group[x_col],
                        y=group[y_col],
                        name=str(name),
                        mode="lines",
                        line=dict(color=color, width=2.5),
                    )
                )
            fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.12))
        else:
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode="lines",
                    line=dict(color=SKY, width=3),
                    fill="tozeroy",
                    fillcolor="rgba(14,165,233,.15)",
                )
            )
        fig.update_xaxes(gridcolor="#e2e8f0", title=None)
        fig.update_yaxes(gridcolor="#e2e8f0", title=None)
        st.plotly_chart(bare(fig, height), use_container_width=True, key=f"line_{key}")
        return

    if chart_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col)
        fig.update_traces(marker=dict(color=SKY, size=8))
        fig.update_xaxes(gridcolor="#e2e8f0", title=None)
        fig.update_yaxes(gridcolor="#e2e8f0", title=None)
        st.plotly_chart(bare(fig, height), use_container_width=True, key=f"scatter_{key}")
        return

    if chart_type == "pie":
        label_col = _pick_column(df, spec.get("x"), 0)
        value_col = _pick_column(df, spec.get("y"), 1)
        fig = go.Figure(
            go.Pie(
                labels=df[label_col],
                values=df[value_col],
                hole=0.6,
                marker=dict(colors=PIE_COLORS),
            )
        )
        fig.update_layout(showlegend=True)
        st.plotly_chart(bare(fig, height), use_container_width=True, key=f"pie_{key}")
        return

    st.dataframe(df, use_container_width=True, key=f"fallback_table_{key}")


def render_dashboard_grid(insights: list[dict[str, Any]]) -> None:
    if not insights:
        st.info("No insights generated yet. Upload data or click **Generate more insights**.")
        return

    kpis = [i for i in insights if i.get("chart_type") == "kpi"]
    others = [i for i in insights if i.get("chart_type") != "kpi"]

    if kpis:
        cols = st.columns(min(len(kpis), 2))
        for idx, spec in enumerate(kpis[:2]):
            with cols[idx]:
                render_insight(_spec_with_id(spec, f"_kpi{idx}"), height=90)

    wide = [i for i in others if i.get("chart_type") in ("line", "table")]
    half = [i for i in others if i.get("chart_type") not in ("line", "table")]

    for idx, spec in enumerate(wide[:2]):
        render_insight(
            _spec_with_id(spec, f"_wide{idx}"),
            height=300 if spec.get("chart_type") == "line" else 240,
        )

    for i in range(0, len(half), 2):
        c1, c2 = st.columns(2)
        with c1:
            render_insight(_spec_with_id(half[i], f"_half{i}"), height=280)
        if i + 1 < len(half):
            with c2:
                render_insight(_spec_with_id(half[i + 1], f"_half{i + 1}"), height=280)

    remaining_wide = wide[2:]
    remaining_kpis = kpis[2:]
    for idx, spec in enumerate(remaining_wide + remaining_kpis):
        render_insight(_spec_with_id(spec, f"_rem{idx}"), height=280)
