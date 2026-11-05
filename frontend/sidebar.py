"""
Sidebar rendering components for document management.
"""

import streamlit as st
from frontend.api import delete_document, fetch_documents


def render_document_list():
    """Renders the list of currently uploaded documents with delete buttons."""
    st.subheader("📄 Uploaded Documents")

    try:
        docs = fetch_documents()

        if not docs:
            st.info("No documents uploaded.")
            return

        for doc in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(doc)
            with col2:
                if st.button("🗑", key=f"del_{doc}"):
                    try:
                        delete_document(doc)
                        st.toast(f"Deleted {doc}", icon="✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete: {e}")
    except Exception:
        st.warning("Backend Offline (cannot reach /documents)")

