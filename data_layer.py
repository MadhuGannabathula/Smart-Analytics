"""Parse uploads, validate, and register tables in an in-memory DuckDB connection."""

from __future__ import annotations

import io
import re
from typing import Any

import duckdb
import pandas as pd

MAX_SAMPLE_ROWS = 5


def get_or_create_connection(existing: duckdb.DuckDBPyConnection | None) -> duckdb.DuckDBPyConnection:
    if existing is not None:
        return existing
    return duckdb.connect(database=":memory:")


def _sanitize_table_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "table"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned.lower()


def _unique_table_name(base: str, existing: set[str]) -> str:
    name = _sanitize_table_name(base)
    if name not in existing:
        return name
    i = 2
    while f"{name}_{i}" in existing:
        i += 1
    return f"{name}_{i}"


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if df.empty or len(df.columns) == 0:
        warnings.append("Empty file")
        return warnings

    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        warnings.append(f"Duplicate columns: {', '.join(map(str, dup_cols))}")

    for col in df.columns:
        series = df[col]
        if series.dtype == object:
            non_null = series.dropna()
            if len(non_null) == 0:
                continue
            type_names = {type(v).__name__ for v in non_null.head(100)}
            if len(type_names) > 1:
                warnings.append(f"Inconsistent types in '{col}'")
    return warnings


def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def _read_excel_sheets(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, engine="openpyxl")


def parse_upload(
    filename: str,
    file_bytes: bytes,
    existing_tables: set[str],
) -> list[dict[str, Any]]:
    """Return a list of {table_name, df, warnings, source_file} dicts."""
    stem = filename.rsplit(".", 1)[0]
    ext = filename.rsplit(".", 1)[-1].lower()
    results: list[dict[str, Any]] = []

    if ext == "csv":
        df = _read_csv(file_bytes)
        table_name = _unique_table_name(stem, existing_tables)
        existing_tables.add(table_name)
        results.append(
            {
                "table_name": table_name,
                "df": df,
                "warnings": validate_dataframe(df),
                "source_file": filename,
            }
        )
    elif ext in ("xlsx", "xls"):
        sheets = _read_excel_sheets(file_bytes)
        for sheet_name, df in sheets.items():
            base = f"{stem}_{sheet_name}" if len(sheets) > 1 else stem
            table_name = _unique_table_name(base, existing_tables)
            existing_tables.add(table_name)
            results.append(
                {
                    "table_name": table_name,
                    "df": df,
                    "warnings": validate_dataframe(df),
                    "source_file": filename,
                }
            )
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return results


def register_table(conn: duckdb.DuckDBPyConnection, table_name: str, df: pd.DataFrame) -> None:
    conn.register("_upload_df", df)
    conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM _upload_df')
    conn.unregister("_upload_df")


def build_schema_profile(conn: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, Any]:
    row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
    columns_info = conn.execute(
        f'DESCRIBE "{table_name}"'
    ).fetchdf()
    columns = [
        {"name": row["column_name"], "dtype": row["column_type"]}
        for _, row in columns_info.iterrows()
    ]
    sample_df = conn.execute(f'SELECT * FROM "{table_name}" LIMIT {MAX_SAMPLE_ROWS}').fetchdf()
    sample_rows = sample_df.where(sample_df.notna(), None).to_dict(orient="records")
    return {
        "table_name": table_name,
        "row_count": int(row_count),
        "columns": columns,
        "sample_rows": sample_rows,
    }


def get_all_schema_profiles(conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchdf()
    return [build_schema_profile(conn, row["table_name"]) for _, row in tables.iterrows()]


def execute_query(conn: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return conn.execute(sql).fetchdf()


def list_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    df = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchdf()
    return df["table_name"].tolist()
