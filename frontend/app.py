from pathlib import Path
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/v1"
HEALTH_URL = "http://127.0.0.1:8000/health"

st.set_page_config(
    page_title="RAG-Pro 2.0 | Industrial Intelligence Assistant",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- STYLES ---------------- #
css_file = Path(__file__).parent / "styles.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def check_backend_status() -> bool:
    """Check if the backend FastAPI server is healthy."""
    try:
        response = requests.get(HEALTH_URL, timeout=1.2)
        return response.status_code == 200
    except Exception:
        return False


# Check live health
is_online = check_backend_status()
status_text = "ONLINE" if is_online else "OFFLINE"
status_class = "online" if is_online else "offline"
blinker_class = "blinker-online" if is_online else "blinker-offline"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None


# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.markdown("### 🏭 RAG-Pro 2.0")
    st.caption("Enterprise AI Knowledge & Industrial Intelligence")
    st.markdown(
        """
        <div class="developer-badge">
            <span>👨‍💻</span>
            <span>Developed by <b>Tarun</b> · B.Tech CS (DS & AI)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### 📂 Upload & Ingest")
    SUPPORTED_TYPES = [
        "pdf", "docx", "pptx", "txt", "md",
        "csv", "xlsx", "png", "jpg", "jpeg",
        "tiff", "bmp", "webp",
    ]

    uploaded_files = st.file_uploader(
        "Upload Technical Files or Manuals",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="Supports PDF, DOCX, CSV, Excel, TXT, and technical images.",
    )

    if st.button("📤 Upload & Index", key="upload_index_btn", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please choose one or more documents.")
        elif not is_online:
            st.error("Cannot upload: Backend is currently OFFLINE.")
        else:
            with st.spinner("Uploading, chunking & embedding into Pinecone..."):
                files = []
                for uploaded_file in uploaded_files:
                    files.append(
                        (
                            "files",
                            (
                                uploaded_file.name,
                                uploaded_file,
                                uploaded_file.type or "application/octet-stream",
                            ),
                        )
                    )

                try:
                    response = requests.post(
                        f"{API_URL}/upload",
                        files=files,
                        timeout=120,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        disciplines = result.get("disciplines", [])
                        disc_str = f"\n\n**Disciplines:** {', '.join(disciplines)}" if disciplines else ""

                        st.success(f"✅ Indexed {len(result.get('files', []))} file(s)")
                        st.info(
                            f"**Documents:** {result.get('documents', 0)}  \n"
                            f"**Chunks:** {result.get('chunks', 0)}{disc_str}"
                        )
                    else:
                        st.error(response.text)
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend server.")
                except Exception as e:
                    st.error(f"Backend Error: {e}")

    st.divider()

    st.markdown("#### 📄 Knowledge Base Documents")

    if is_online:
        try:
            response = requests.get(f"{API_URL}/documents", timeout=5)
            if response.status_code == 200:
                data = response.json()
                docs = data.get("documents", [])

                if not docs:
                    st.info("No documents uploaded yet.")
                else:
                    st.caption(f"📚 {len(docs)} document(s) available")
                    for doc in docs:
                        col_doc, col_del = st.columns([4, 1])
                        with col_doc:
                            st.markdown(
                                f'<div class="document-name" title="{doc}">📄 {doc}</div>',
                                unsafe_allow_html=True,
                            )
                        with col_del:
                            if st.button("🗑", key=f"del_{doc}", help=f"Delete {doc}"):
                                try:
                                    requests.delete(f"{API_URL}/documents/{doc}", timeout=10)
                                    st.toast(f"Removed {doc}", icon="✅")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to delete: {e}")
                    if docs:
                        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                        if st.button("🗑 Wipe All Documents", use_container_width=True, help="Removes all uploaded documents and resets Pinecone index"):
                            try:
                                requests.delete(f"{API_URL}/documents", timeout=10)
                                st.toast("Knowledge base cleared!", icon="✅")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to clear knowledge base: {e}")
            else:
                st.warning("Could not fetch document list.")
        except Exception:
            st.warning("Backend Offline (unable to load documents)")
    else:
        st.warning("⚠️ Backend Offline")

    st.divider()

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.quick_prompt = None
        st.rerun()


# ---------------- MAIN CONTENT ---------------- #

col_header, col_badge = st.columns([3.8, 1.4])
with col_header:
    st.markdown("## 🤖 RAG-Pro 2.0 Industrial Intelligence")
    st.caption("Grounded Document Q&A · Industrial Root Cause Analysis · Multi-Disciplinary Engineering")
with col_badge:
    st.markdown(
        f"""
        <div class="inline-status-badge {status_class}">
            <span class="{blinker_class}"></span>
            <span>{status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not is_online:
    st.error(
        "⚠️ **Backend Offline:** The FastAPI server is currently unreachable on `127.0.0.1:8000`. "
        "Please verify the backend server is running (`python run.py`)."
    )

# Hero Section when chat is empty
if not st.session_state.messages:
    st.markdown(
        """
        <div class="hero-box">
            <h3 style="margin-top:0;">👋 Welcome to RAG-Pro 2.0</h3>
            <p style="color:#94a3b8; font-size: 0.95rem;">
                Ask questions about your uploaded documents,
                solve engineering equations, diagnose equipment failures, or test common conversational queries.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### 💡 Try asking:")
    sug_col1, sug_col2 = st.columns(2)
    with sug_col1:
        if st.button("📄 Summarize the uploaded documents", use_container_width=True):
            st.session_state.quick_prompt = "Summarize the uploaded documents"
            st.rerun()
        if st.button("⚙️ Why is my PC heating 98 degree celcius?", use_container_width=True):
            st.session_state.quick_prompt = "Why is my PC heating 98 degree celcius"
            st.rerun()
    with sug_col2:
        if st.button("👨‍💻 Who developed you and what is your purpose?", use_container_width=True):
            st.session_state.quick_prompt = "Who developed you and what is your purpose?"
            st.rerun()
        if st.button("🔧 Explain turbocharger failure root causes & safety", use_container_width=True):
            st.session_state.quick_prompt = "Explain turbocharger failure root causes & safety"
            st.rerun()

# Render message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown('<div class="source-pill-container">', unsafe_allow_html=True)
            for source in msg["sources"]:
                src = source.get("source", "Document")
                page = source.get("page")
                page_str = f" · Page {int(page)}" if page is not None else ""
                st.markdown(
                    f'<div class="source-card">📄 {src}{page_str}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

# Determine active prompt (from input bar or quick suggestion button)
input_prompt = st.chat_input("Ask a question about your documents, engineering problems, or calculations...")
prompt = input_prompt or st.session_state.quick_prompt

if prompt:
    # Reset quick prompt
    st.session_state.quick_prompt = None

    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not is_online:
            offline_msg = "⚠️ Cannot generate response: The FastAPI backend is currently OFFLINE. Please start the server (`python run.py`)."
            st.error(offline_msg)
            st.session_state.messages.append({"role": "assistant", "content": offline_msg})
        else:
            with st.spinner("Analyzing and retrieving relevant knowledge..."):
                try:
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "question": prompt,
                            "history": st.session_state.messages[:-1][-4:],
                        },
                        timeout=120,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "No response received.")
                        sources = data.get("sources", [])

                        st.markdown(answer)

                        if sources:
                            st.markdown('<div class="source-pill-container">', unsafe_allow_html=True)
                            for source in sources:
                                src = source.get("source", "Document")
                                page = source.get("page")
                                page_str = f" · Page {int(page)}" if page is not None else ""
                                st.markdown(
                                    f'<div class="source-card">📄 {src}{page_str}</div>',
                                    unsafe_allow_html=True,
                                )
                            st.markdown('</div>', unsafe_allow_html=True)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                            }
                        )
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    err_msg = "⚠️ Cannot connect to backend server. Is FastAPI running on port 8000?"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
                except Exception as e:
                    err_msg = f"Backend Error: {e}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
