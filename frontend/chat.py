"""
Chat display and message rendering components.
"""

import streamlit as st


def render_message(msg: dict):
    """Renders a single message with role avatar."""
    role = msg.get("role", "assistant")
    content = msg.get("content", "")

    with st.chat_message(role):
        st.markdown(content)


def render_sources(sources: list[dict]):
    """Renders sources pill cards under an assistant response."""
    if not sources:
        return

    st.divider()
    st.caption("📄 Sources")
    for source in sources:
        src = source.get("source", "Unknown")
        page = source.get("page")
        page_str = f" (Page {page})" if page else ""
        st.markdown(
            f'<div class="source-card">📄 <b>{src}</b>{page_str}</div>',
            unsafe_allow_html=True,
        )

