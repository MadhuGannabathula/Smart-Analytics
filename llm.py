"""LLM integration for insight suggestions and chat Q&A."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from config import get_setting
from data_layer import execute_query

CHART_TYPES = {"bar", "line", "pie", "grouped_bar", "scatter", "kpi", "table"}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

INSIGHT_SYSTEM_PROMPT = """
You are analyzing uploaded data to build a dashboard. Given the schema profiles below,
generate insights following this priority order — only skip a category if the data
genuinely doesn't support it:

1. TOTAL/AVERAGE (kpi) — one clear aggregate metric, the single most important number
   in this dataset (e.g. total revenue, average order value)
2. TREND (line) — if any date/time column exists, one insight showing change over time
3. COMPARISON (bar or grouped_bar) — one insight comparing a numeric measure across
   the most meaningful categorical column
4. COMPOSITION (pie) — if a categorical column has 2-6 distinct values, show its share
   of a relevant total
5. CROSS-FILE (bar, line, or table) — if two or more tables share a plausible join key
   (matching column names or semantic overlap like customer_id / region), one insight
   that joins across files

Generate at most one insight per category, skip a category only if truly not
applicable, and order the output list in the priority order above.

Return ONLY a JSON array of: {"title", "chart_type", "sql", "x", "y", "groupby"}
chart_type must be one of: bar, line, pie, grouped_bar, scatter, kpi, table

sql must be DuckDB-compatible SELECT-only queries referencing exact table and column names.
For kpi charts, sql should return a single row with columns: value and optional label.
For pie charts, sql should return label and value columns.
Use x and y for axis column names; groupby for grouped_bar color/series column.
"""


def get_client() -> OpenAI:
    api_key = get_setting("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env locally or Streamlit Cloud secrets."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def _model() -> str:
    return get_setting("GROQ_MODEL", DEFAULT_GROQ_MODEL) or DEFAULT_GROQ_MODEL


def is_select_only(sql: str) -> bool:
    stripped = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL).strip()
    if not stripped:
        return False
    first = stripped.split(";")[0].strip()
    if not re.match(r"^SELECT\b", first, re.IGNORECASE):
        return False
    forbidden = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|COPY|ATTACH|DETACH|PRAGMA)\b",
        re.IGNORECASE,
    )
    return forbidden.search(first) is None


def _extract_json(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def _schema_context(schema_profiles: list[dict[str, Any]]) -> str:
    ordered = sorted(schema_profiles, key=lambda p: p.get("table_name", ""))
    return json.dumps(ordered, indent=2, default=str, sort_keys=True)


def _call_llm(client: OpenAI, system: str, user: str) -> str:
    kwargs: dict[str, Any] = {
        "model": _model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": float(get_setting("LLM_TEMPERATURE", "0") or "0"),
    }
    seed = get_setting("LLM_SEED")
    if seed is not None:
        kwargs["seed"] = int(seed)
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _normalize_insight(item: dict[str, Any]) -> dict[str, Any]:
    chart_type = str(item.get("chart_type", "table")).lower()
    if chart_type not in CHART_TYPES:
        chart_type = "table"
    return {
        "title": str(item.get("title", "Insight")),
        "chart_type": chart_type,
        "sql": str(item.get("sql", "")).strip(),
        "x": item.get("x"),
        "y": item.get("y"),
        "groupby": item.get("groupby"),
    }


def suggest_insights(
    client: OpenAI,
    conn,
    schema_profiles: list[dict[str, Any]],
    already_shown_titles: list[str] | None = None,
    count: int = 6,
) -> list[dict[str, Any]]:
    already_shown_titles = already_shown_titles or []
    exclude = ", ".join(already_shown_titles) if already_shown_titles else "none"
    system = INSIGHT_SYSTEM_PROMPT.strip()
    if already_shown_titles:
        user = (
            f"Schema profiles:\n{_schema_context(schema_profiles)}\n\n"
            f"These dashboard titles are already shown — do NOT repeat them: {exclude}\n"
            "Generate additional insights for any remaining applicable categories from the "
            f"priority list that are not yet covered. Return at most {count} new insights."
        )
    else:
        user = f"Schema profiles:\n{_schema_context(schema_profiles)}"
    raw = _call_llm(client, system, user)
    items = _extract_json(raw)
    if isinstance(items, dict):
        items = items.get("insights", items.get("items", [items]))
    if not isinstance(items, list):
        raise ValueError("LLM did not return a JSON array of insights")

    results: list[dict[str, Any]] = []
    for item in items:
        insight = _normalize_insight(item)
        if not insight["sql"]:
            continue
        df, err = execute_sql_with_retry(client, conn, schema_profiles, insight["sql"])
        if err:
            continue
        insight["data"] = df
        results.append(insight)
    return results


def answer_question(
    client: OpenAI,
    conn,
    schema_profiles: list[dict[str, Any]],
    question: str,
    chat_history: list[tuple[str, str]],
) -> dict[str, Any]:
    history_lines = []
    for role, text in chat_history[-3:]:
        prefix = "User" if role == "user" else "Assistant"
        history_lines.append(f"{prefix}: {text}")
    history_block = "\n".join(history_lines) if history_lines else "No prior turns."

    system = (
        "You are a data analyst assistant. Answer questions using DuckDB SQL over the provided tables. "
        "Return ONLY valid JSON with keys: sql, answer_text, chart_spec. "
        "sql must be SELECT-only DuckDB SQL. "
        "chart_spec is null or an object with keys: title, chart_type, x, y, groupby "
        "(same rules as dashboard insights). "
        "When the user asks for a chart, visualization, or dashboard widget, always return a chart_spec "
        "with a short descriptive title. "
        "Keep answer_text concise and grounded in query results."
    )
    user = (
        f"Schema profiles:\n{_schema_context(schema_profiles)}\n\n"
        f"Recent chat (last 3 turns):\n{history_block}\n\n"
        f"Question: {question}"
    )
    raw = _call_llm(client, system, user)
    payload = _extract_json(raw)
    sql = str(payload.get("sql", "")).strip()
    answer_text = str(payload.get("answer_text", "I couldn't generate an answer."))
    chart_spec = payload.get("chart_spec")

    chart_data = None
    if sql:
        df, err = execute_sql_with_retry(client, conn, schema_profiles, sql)
        if err:
            return {"sql": sql, "answer_text": f"{answer_text}\n\n(Query failed: {err})", "chart_spec": None, "chart_data": None}
        if chart_spec and isinstance(chart_spec, dict):
            chart_spec = _normalize_insight({**chart_spec, "sql": sql, "title": chart_spec.get("title", "Answer")})
            chart_spec["data"] = df
            chart_data = chart_spec
        return {"sql": sql, "answer_text": answer_text, "chart_spec": chart_spec, "chart_data": chart_data}

    return {"sql": "", "answer_text": answer_text, "chart_spec": None, "chart_data": None}


def execute_sql_with_retry(
    client: OpenAI,
    conn,
    schema_profiles: list[dict[str, Any]],
    sql: str,
    max_retries: int = 2,
) -> tuple[Any, str | None]:
    current_sql = sql
    last_error: str | None = None

    for attempt in range(max_retries + 1):
        if not is_select_only(current_sql):
            return None, "Only SELECT queries are allowed."
        try:
            df = execute_query(conn, current_sql)
            return df, None
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_retries:
                break
            fix_system = (
                "Fix the DuckDB SQL query. Return ONLY valid JSON: {\"sql\": \"...\"}. "
                "The query must be SELECT-only and reference valid tables/columns."
            )
            fix_user = (
                f"Schema profiles:\n{_schema_context(schema_profiles)}\n\n"
                f"Failed SQL:\n{current_sql}\n\nError:\n{last_error}"
            )
            raw = _call_llm(client, fix_system, fix_user)
            try:
                fixed = _extract_json(raw)
                current_sql = str(fixed.get("sql", current_sql)).strip()
            except (json.JSONDecodeError, ValueError):
                break

    return None, last_error
