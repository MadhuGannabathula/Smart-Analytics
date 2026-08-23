"""SmartAnalytics — AI-driven analytics for uploaded spreadsheets.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import copy
import uuid

import streamlit as st
from dotenv import load_dotenv

from charts import render_dashboard_grid, render_insight
from data_layer import (
    get_all_schema_profiles,
    get_or_create_connection,
    list_tables,
    parse_upload,
    register_table,
)
from llm import answer_question, get_client, sanitize_chat_input, suggest_insights

load_dotenv()

# ---------------------------------------------------------------- config

st.set_page_config(
    page_title="SmartAnalytics — AI analytics for your spreadsheets",
    page_icon="📊",
    layout="wide",
)

SKY = "#0ea5e9"
SKY_LIGHT = "#7dd3fc"
SKY_DEEP = "#0369a1"

st.markdown(
    """
    <style>
      .stApp { background: #f6fafd; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      .card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px;
        padding: 20px 24px; margin-bottom: 18px;
      }
      .metric-label {
        font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
        color: #64748b; font-weight: 600;
      }
      .metric-value { font-size: 34px; font-weight: 500; letter-spacing: -0.03em; }
      .pill {
        background: rgba(14,165,233,.1); color: #0284c7; border-radius: 999px;
        padding: 2px 10px; font-size: 12px;
      }
      .pill-warn {
        background: rgba(245,158,11,.12); color: #b45309; border-radius: 999px;
        padding: 2px 10px; font-size: 11px; margin-left: 6px;
      }
      .bubble-ai {
        background: #eef2f6; border-radius: 16px 16px 16px 4px;
        padding: 10px 14px; font-size: 14px; margin-bottom: 4px;
      }
      .bubble-user {
        background: #0ea5e9; color: white; border-radius: 16px 16px 4px 16px;
        padding: 10px 14px; font-size: 14px; margin-bottom: 4px;
      }
      .file-chip {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 8px 12px; margin-bottom: 8px; font-size: 13px;
      }
      .ts { font-size: 10px; color: #94a3b8; }
      [data-testid="stSidebar"] { min-width: 340px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    defaults = {
        "conn": None,
        "file_registry": [],
        "table_names": set(),
        "schema_profiles": [],
        "insights": [],
        "messages": [],
        "has_data": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _refresh_schema() -> None:
    conn = st.session_state.conn
    if conn is None:
        st.session_state.schema_profiles = []
        return
    st.session_state.schema_profiles = get_all_schema_profiles(conn)


def _process_uploads(uploaded_files) -> None:
    if not uploaded_files:
        return

    conn = get_or_create_connection(st.session_state.conn)
    st.session_state.conn = conn
    existing = set(st.session_state.table_names)
    new_files: list[dict] = []

    for uploaded in uploaded_files:
        already = any(f["name"] == uploaded.name for f in st.session_state.file_registry)
        if already:
            continue
        try:
            tables = parse_upload(uploaded.name, uploaded.getvalue(), existing)
        except Exception as exc:
            new_files.append({"name": uploaded.name, "tables": [], "warnings": [str(exc)]})
            continue

        for entry in tables:
            register_table(conn, entry["table_name"], entry["df"])
            st.session_state.table_names.add(entry["table_name"])

        new_files.append(
            {
                "name": uploaded.name,
                "tables": [t["table_name"] for t in tables],
                "warnings": [w for t in tables for w in t["warnings"]],
            }
        )

    if new_files:
        st.session_state.file_registry.extend(new_files)
        st.session_state.has_data = bool(st.session_state.table_names)
        _refresh_schema()


def _generate_initial_insights() -> None:
    if not st.session_state.has_data or st.session_state.insights:
        return
    _generate_more_insights(initial=True)


def _ensure_insight_ids() -> None:
    for insight in st.session_state.insights:
        if "id" not in insight:
            insight["id"] = uuid.uuid4().hex[:10]


def _generate_more_insights(initial: bool = False) -> None:
    if not st.session_state.schema_profiles:
        return
    try:
        client = get_client()
        shown = [i["title"] for i in st.session_state.insights]
        new_insights = suggest_insights(
            client,
            st.session_state.conn,
            st.session_state.schema_profiles,
            already_shown_titles=shown,
            count=6 if initial else 4,
        )
        for insight in new_insights:
            insight["id"] = uuid.uuid4().hex[:10]
        st.session_state.insights.extend(new_insights)
    except Exception as exc:
        st.error(f"Could not generate insights: {exc}")


def _add_chart_to_dashboard(chart: dict, msg: dict) -> None:
    insight = copy.deepcopy(chart)
    insight["id"] = uuid.uuid4().hex[:10]
    if not insight.get("title") or insight["title"] == "Answer":
        insight["title"] = "Chat insight"
    st.session_state.insights.append(insight)
    msg["on_dashboard"] = True


def _render_file_chips() -> None:
    if not st.session_state.file_registry:
        return
    st.markdown("**Uploaded files**")
    for file_info in st.session_state.file_registry:
        warnings = file_info.get("warnings", [])
        warn_html = "".join(f'<span class="pill-warn">{w}</span>' for w in warnings)
        tables = ", ".join(file_info.get("tables", [])) or "—"
        st.markdown(
            f'<div class="file-chip">📄 {file_info["name"]}<br>'
            f'<span style="color:#64748b;font-size:11px;">Tables: {tables}</span>'
            f"{warn_html}</div>",
            unsafe_allow_html=True,
        )


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("#### 📁 Upload data")
        uploaded = st.file_uploader(
            "CSV or Excel — up to 50MB",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            _process_uploads(uploaded)
            if st.session_state.has_data and not st.session_state.insights:
                with st.spinner("Generating insights…"):
                    _generate_initial_insights()

        _render_file_chips()

        st.divider()
        st.markdown("#### 🟢 Analyst Online")

        for msg_idx, msg in enumerate(st.session_state.messages):
            role = msg.get("role", "ai")
            css = "bubble-ai" if role == "ai" else "bubble-user"
            st.markdown(f'<div class="{css}">{msg["text"]}</div>', unsafe_allow_html=True)
            chart = msg.get("chart")
            if chart:
                chart_spec = {**chart, "id": f"{chart.get('id', f'chat_{msg_idx}')}_sidebar{msg_idx}"}
                render_insight(chart_spec, height=130)
                if msg.get("on_dashboard"):
                    st.caption("✓ Added to dashboard")
                elif st.button(
                    "➕ Add to dashboard",
                    key=f"add_dash_{msg_idx}",
                    use_container_width=True,
                ):
                    _add_chart_to_dashboard(chart, msg)
                    st.rerun()

        prompt = st.chat_input("Ask about your data…")
        if prompt:
            if not st.session_state.has_data:
                st.session_state.messages.append(
                    {"role": "ai", "text": "Please upload a CSV or Excel file first."}
                )
            else:
                cleaned, input_error = sanitize_chat_input(prompt)
                st.session_state.messages.append({"role": "user", "text": cleaned or prompt})
                if input_error:
                    st.session_state.messages.append({"role": "ai", "text": input_error})
                else:
                    try:
                        client = get_client()
                        history = [(m["role"], m["text"]) for m in st.session_state.messages[:-1]]
                        with st.spinner("Thinking…"):
                            result = answer_question(
                                client,
                                st.session_state.conn,
                                st.session_state.schema_profiles,
                                cleaned or prompt,
                                history,
                            )
                        ai_msg = {"role": "ai", "text": result["answer_text"]}
                        if result.get("chart_data"):
                            chart = result["chart_data"]
                            chart["id"] = uuid.uuid4().hex[:10]
                            ai_msg["chart"] = chart
                        st.session_state.messages.append(ai_msg)
                    except Exception as exc:
                        st.session_state.messages.append(
                            {"role": "ai", "text": f"Sorry, something went wrong: {exc}"}
                        )
            st.rerun()


def _render_empty_state() -> None:
    st.markdown("## Welcome to SmartAnalytics")
    st.markdown(
        '<div class="card">'
        "<p>Upload a <strong>CSV</strong> or <strong>Excel</strong> file in the sidebar to get started. "
        "We'll profile your data, generate AI-driven dashboard insights, and let you ask questions in chat.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption("No data loaded yet — use the sidebar uploader above the chat panel.")


def _render_dashboard() -> None:
    source_names = [f["name"] for f in st.session_state.file_registry]
    subtitle = ", ".join(source_names) if source_names else "Uploaded datasets"

    st.markdown("## Data Insights")
    st.caption(f"Synthesized from {subtitle}")

    tab_dash, tab_raw = st.tabs(["Dashboard", "Raw Data"])

    with tab_dash:
        if st.button("Generate more insights", type="primary"):
            with st.spinner("Generating…"):
                _generate_more_insights()
            st.rerun()

        _ensure_insight_ids()
        render_dashboard_grid(st.session_state.insights)

    with tab_raw:
        conn = st.session_state.conn
        if conn is None:
            st.info("No tables available.")
            return
        for table_idx, table in enumerate(list_tables(conn)):
            st.markdown(f"**{table}**")
            df = conn.execute(f'SELECT * FROM "{table}" LIMIT 500').fetchdf()
            st.dataframe(df, use_container_width=True, key=f"raw_{table}_{table_idx}")


# ---------------------------------------------------------------- main

_init_state()
_render_sidebar()

nav_l, nav_r = st.columns([3, 1])
with nav_l:
    st.markdown("### 📊 SmartAnalytics")
    st.caption("AI-powered analytics for your spreadsheets")
with nav_r:
    if st.button("＋ New Analysis", use_container_width=True):
        for key in ("conn", "file_registry", "table_names", "schema_profiles", "insights", "messages", "has_data"):
            if key == "table_names":
                st.session_state[key] = set()
            elif key in ("file_registry", "schema_profiles", "insights", "messages"):
                st.session_state[key] = []
            elif key == "has_data":
                st.session_state[key] = False
            else:
                st.session_state[key] = None
        st.rerun()

st.divider()

if st.session_state.has_data:
    _render_dashboard()
else:
    _render_empty_state()

st.divider()
st.caption("© 2026 SmartAnalytics · System Nominal")
