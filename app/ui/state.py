from __future__ import annotations

import streamlit as st

from ui.constants import SESSION_DEFAULTS


def ensure_session_defaults() -> None:
    for key, value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def reset_reports() -> None:
    st.session_state["performance_report"] = None
    st.session_state["overall_report"] = None
    st.session_state["overall_exports"] = None


def set_flash_notice(level: str, message: str) -> None:
    st.session_state["flash_notice"] = {"level": level, "message": message}


def render_flash_notice() -> None:
    notice = st.session_state.pop("flash_notice", None)
    if not notice:
        return
    level = str(notice.get("level", "info"))
    message = str(notice.get("message", "")).strip()
    if not message:
        return
    getattr(st, level, st.info)(message)


def clear_upload_selection() -> None:
    st.session_state["upload_widget_nonce"] = int(st.session_state.get("upload_widget_nonce", 0)) + 1
    for key in list(st.session_state.keys()):
        if key.startswith("transcript::"):
            del st.session_state[key]
