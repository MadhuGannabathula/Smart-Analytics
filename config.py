"""Read settings from Streamlit secrets (cloud) or environment variables (local)."""

from __future__ import annotations

import os


def get_setting(name: str, default: str | None = None) -> str | None:
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)
