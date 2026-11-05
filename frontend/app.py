from pathlib import Path
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# Load custom styles
css_file = Path(__file__).parent / "styles.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("📚 RAG Assistant")

    st.divider()

    SUPPORTED_TYPES = [
        "pdf",
        "docx",
        "pptx",
        "txt",
        "md",
        "csv",
        "xlsx",
        "png",
        "jpg",
        "jpeg",
        "tiff",
        "bmp",
        "webp",
    ]

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )

    if st.button("📤 Upload & Index"):

        if not uploaded_files:
            st.warning("Please choose one or more documents.")

        else:

            with st.spinner("Uploading & Indexing..."):

                files = []

                for uploaded_file in uploaded_files:
                    files.append(
                        (
                            "files",
                            (
                                uploaded_file.name,
                                uploaded_file,
                                uploaded_file.type
                                or "application/octet-stream",
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

                        st.success(
                            f"✅ Indexed {len(result.get('files', []))} file(s)"
                        )

                        st.info(
                            f"""
**Documents:** {result.get('documents', 0)}

**Chunks:** {result.get('chunks', 0)}
"""
                        )

                    else:
                        st.error(response.text)

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is the server running?")
                except Exception as e:
                    st.error(f"Backend Error: {e}")

    st.divider()

    st.subheader("📄 Uploaded Documents")

    try:

        response = requests.get(f"{API_URL}/documents", timeout=5)

        if response.status_code == 200:

            data = response.json()
            docs = data.get("documents", [])

            if len(docs) == 0:
                st.info("No documents uploaded.")

            else:

                for doc in docs:

                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.write(doc)

                    with col2:

                        if st.button("🗑", key=doc):

                            requests.delete(
                                f"{API_URL}/documents/{doc}"
                            )

                            st.rerun()

        else:
            st.warning("Couldn't fetch documents.")

    except requests.exceptions.ConnectionError:
        st.warning("Backend Offline")
    except Exception:
        st.warning("Backend Offline")

    st.divider()

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ---------------- MAIN ---------------- #

st.title("🤖 Production RAG Assistant")

st.caption("FastAPI + LangChain + Pinecone + OpenRouter")

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask anything...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": prompt,
                        "history": st.session_state.messages[:-1][-6:],
                    },
                    timeout=60,
                )

                if response.status_code == 200:

                    data = response.json()

                    answer = data.get("answer", "No response received.")

                    st.markdown(answer)

                    if data.get("sources"):

                        st.divider()
                        st.caption("📄 Sources")

                        for source in data["sources"]:
                            page = source.get("page")
                            src = source.get("source", "Unknown")
                            page_str = f" (Page {page})" if page else ""
                            st.write(f"📄 {src}{page_str}")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                else:
                    st.error(response.text)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Is the server running?")
            except Exception as e:
                st.error(f"Backend Error: {e}")
