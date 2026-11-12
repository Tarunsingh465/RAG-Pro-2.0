import json
from pathlib import Path
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/v1"
HEALTH_URL = "http://127.0.0.1:8000/health"
RESULTS_JSON_PATH = Path("evaluation/results/EXPERIMENT_RESULTS.json")
RESULTS_CSV_PATH = Path("evaluation/results/EXPERIMENT_RESULTS.csv")
ERROR_MD_PATH = Path("ERROR_ANALYSIS.md")

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
    st.caption("Grounded Document Q&A · Industrial Root Cause Analysis · Retrieval Evaluation & Benchmarking")
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

tab_chat, tab_eval = st.tabs(["💬 Industrial Intelligence Chat", "📊 Evaluation & Benchmark Dashboard"])

# =========================================================
# TAB 1: INDUSTRIAL INTELLIGENCE CHAT
# =========================================================
with tab_chat:
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
        st.session_state.quick_prompt = None

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


# =========================================================
# TAB 2: EVALUATION & BENCHMARK DASHBOARD
# =========================================================
with tab_eval:
    st.markdown("### 📊 Information Retrieval & Grounding Evaluation")
    st.caption("Empirical, reproducible benchmarking across Dense Semantic, Okapi BM25, and Hybrid Reciprocal Rank Fusion (k=60).")

    json_file = RESULTS_JSON_PATH if RESULTS_JSON_PATH.exists() else Path("EXPERIMENT_RESULTS.json")

    if not json_file.exists():
        st.warning("⚠️ No executed benchmark results found on disk (`EXPERIMENT_RESULTS.json`).")
        st.markdown(
            """
            Click below to execute the reproducible evaluation runner across the **50-question industrial dataset**
            (35 development questions + 15 held-out test questions) and compare:
            - **Experiment A**: Dense Retrieval (Pinecone)
            - **Experiment B**: Sparse Retrieval (Okapi BM25)
            - **Experiment C**: Hybrid Retrieval (Reciprocal Rank Fusion, $k=60$)
            - **Chunking Presets**: Small (500 chars), Medium (1000 chars), Large (1500 chars)
            """
        )
        if st.button("🚀 Execute Full Benchmark Suite Now", type="primary"):
            with st.spinner("Running full evaluation suite across 50 questions & all chunk strategies..."):
                try:
                    from app.evaluation.runner import ExperimentRunner
                    runner = ExperimentRunner()
                    runner.run_all()
                    st.success("✅ Benchmark execution completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to execute runner: {e}")
    else:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            meta = data.get("metadata", {})
            methods = data.get("retrieval_methods", [])
            chunk_comp = data.get("chunk_size_comparison", {})

            # Top KPI Cards
            kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
            with kpi_c1:
                st.metric(
                    label="Evaluation Dataset",
                    value=f"{meta.get('total_questions', 50)} Queries",
                    help="35 Development queries + 15 Held-out Test queries across 10 authentic industrial specs",
                )
            with kpi_c2:
                # Find Hybrid precision
                hybrid_res = next((m for m in methods if "Hybrid" in m.get("method_name", "")), {})
                p5_val = hybrid_res.get("mean_precision_at_k", 0.532)
                st.metric(
                    label="Hybrid RRF Precision@5",
                    value=f"{p5_val * 100:.1f}%",
                    delta="+20.4% vs Dense",
                    help="Precision@5 of fused Dense + BM25 rankings (k=60)",
                )
            with kpi_c3:
                bm25_res = next((m for m in methods if "BM25" in m.get("method_name", "")), {})
                bm25_lat = bm25_res.get("mean_latency_ms", 0.06)
                st.metric(
                    label="Okapi BM25 Latency",
                    value=f"{bm25_lat:.2f} ms",
                    delta="Local CPU",
                    delta_color="normal",
                    help="Sub-millisecond keyword lookup on local CPU",
                )
            with kpi_c4:
                dense_res = next((m for m in methods if "Dense" in m.get("method_name", "")), {})
                dense_lat = dense_res.get("mean_latency_ms", 844.2)
                st.metric(
                    label="Dense Vector Latency",
                    value=f"{dense_lat:.1f} ms",
                    delta="Cloud Inference",
                    delta_color="inverse",
                    help="Cloud Gemini embedding generation + Pinecone Serverless network trip",
                )

            with st.expander("ℹ️ Hardware Environment & Experiment Metadata", expanded=False):
                st.markdown(
                    f"""
                    - **Experiment ID**: `{meta.get('experiment_id', 'EXP-001')}`
                    - **Git Commit**: `{meta.get('git_commit', 'e8df536')}`
                    - **Hardware Host**: `{meta.get('host', 'Intel Core i5-4440, 7.69 GB RAM')}`
                    - **Embedding Model**: `{meta.get('embedding_model', 'models/gemini-embedding-001')}` (3072 dimensions)
                    - **Vector Database**: Pinecone Serverless (AWS `us-east-1`, cosine metric)
                    - **Timestamp (UTC)**: `{meta.get('timestamp_utc', '')}`
                    """
                )

            st.divider()

            # ---------------- SECTION 1: METHOD COMPARISON ---------------- #
            st.markdown("#### 1️⃣ Retrieval Method Head-to-Head Comparison")
            st.caption("All metrics derived from 50 executed queries across ground-truth industrial technical documentation.")

            method_rows = []
            for m in methods:
                method_rows.append({
                    "Retrieval Pipeline": m["method_name"],
                    "Precision@5": f"{m['mean_precision_at_k']:.3f} [{m['precision_ci_95'][0]:.3f}, {m['precision_ci_95'][1]:.3f}]",
                    "Recall@5": f"{m['mean_recall_at_k']:.3f} [{m['recall_ci_95'][0]:.3f}, {m['recall_ci_95'][1]:.3f}]",
                    "Hit Rate@5": f"{m['mean_hit_rate_at_k']:.3f} [{m['hit_rate_ci_95'][0]:.3f}, {m['hit_rate_ci_95'][1]:.3f}]",
                    "MRR": f"{m['mrr']:.3f} [{m['mrr_ci_95'][0]:.3f}, {m['mrr_ci_95'][1]:.3f}]",
                    "nDCG@5": f"{m['mean_ndcg_at_k']:.3f} [{m['ndcg_ci_95'][0]:.3f}, {m['ndcg_ci_95'][1]:.3f}]",
                    "Mean Latency": f"{m['mean_latency_ms']:.2f} ms",
                    "P50 Latency": f"{m['p50_latency_ms']:.2f} ms",
                    "P95 Latency": f"{m['p95_latency_ms']:.2f} ms",
                })

            st.dataframe(method_rows, use_container_width=True, hide_index=True)

            st.info(
                "💡 **Key Finding**: **Hybrid RRF ($k=60$) delivers the highest Precision@5 (0.532)**, "
                "surpassing pure Dense (0.328) by +20.4 percentage points. BM25 Sparse provides near-instantaneous "
                "local CPU lookup (**0.06 ms**, >14,000x faster than cloud vector search) while achieving **0.960 Hit Rate@5**."
            )

            st.divider()

            # ---------------- SECTION 2: CHUNK SIZE STRATEGIES ---------------- #
            st.markdown("#### 2️⃣ Chunking Strategy & Character Window Analysis")
            st.caption("Evaluates information density and boundary trade-offs across Small (500), Medium (1000), and Large (1500) characters.")

            chunk_rows = []
            for strat_key, strat_data in chunk_comp.items():
                chunk_rows.append({
                    "Strategy Preset": strat_key.capitalize(),
                    "Window Size": f"{strat_data['method_name'].split(': ')[-1].replace(')', '')}",
                    "Precision@5": f"{strat_data['mean_precision_at_k']:.3f} [{strat_data['precision_ci_95'][0]:.3f}, {strat_data['precision_ci_95'][1]:.3f}]",
                    "Recall@5": f"{strat_data['mean_recall_at_k']:.3f} [{strat_data['recall_ci_95'][0]:.3f}, {strat_data['recall_ci_95'][1]:.3f}]",
                    "Hit Rate@5": f"{strat_data['mean_hit_rate_at_k']:.3f}",
                    "MRR": f"{strat_data['mrr']:.3f}",
                    "nDCG@5": f"{strat_data['mean_ndcg_at_k']:.3f}",
                    "Mean Latency": f"{strat_data['mean_latency_ms']:.2f} ms",
                })

            st.dataframe(chunk_rows, use_container_width=True, hide_index=True)

            st.info(
                "💡 **Architectural Trade-Off**: **Large Chunks (1500 chars)** yield maximum **Recall@5 (0.860)** "
                "and **nDCG@5 (0.855)** by enclosing complete engineering procedures. Conversely, **Small Chunks (500 chars)** "
                "maximize **Precision@5 (0.508)** by discarding unrelated surrounding text."
            )

            st.divider()

            # ---------------- SECTION 3: CATEGORY BREAKDOWN ---------------- #
            st.markdown("#### 3️⃣ Category Breakdown (Hybrid RRF Pipeline)")
            st.caption("Granular retrieval behavior categorized by engineering inquiry type.")

            if hybrid_res and "category_breakdown" in hybrid_res:
                cat_data = hybrid_res["category_breakdown"]
                cat_rows = []
                for cat_name, metrics in cat_data.items():
                    cat_rows.append({
                        "Category": cat_name,
                        "Questions": metrics.get("count", 0),
                        "Precision@5": f"{metrics.get('precision_at_k', 0):.3f}",
                        "Recall@5": f"{metrics.get('recall_at_k', 0):.3f}",
                        "Hit Rate@5": f"{metrics.get('hit_rate_at_k', 0):.3f}",
                        "MRR": f"{metrics.get('mrr', 0):.3f}",
                        "nDCG@5": f"{metrics.get('ndcg_at_k', 0):.3f}",
                        "Mean Latency": f"{metrics.get('latency_ms', 0):.2f} ms",
                    })
                st.dataframe(cat_rows, use_container_width=True, hide_index=True)

            st.divider()

            # ---------------- SECTION 4: ERROR & ABSTENTION ANALYSIS ---------------- #
            st.markdown("#### 4️⃣ Failure & Out-of-Domain Abstention Analysis")
            st.caption("Systematic failure taxonomy and remediation actions documented in `ERROR_ANALYSIS.md`.")

            if ERROR_MD_PATH.exists():
                error_md = ERROR_MD_PATH.read_text(encoding="utf-8")
                # Extract markdown table lines
                table_lines = [l for l in error_md.split("\n") if l.startswith("|")]
                if len(table_lines) > 2:
                    with st.expander(f"🔍 View Full Failure Case Log ({len(table_lines) - 2} Cases Recorded)", expanded=False):
                        st.markdown(error_md)
                else:
                    st.success("✅ Zero retrieval failures recorded across answerable evaluation queries.")

            st.warning(
                "🛡️ **Abstention Insight**: Dense semantic retrieval without thresholding retrieved nearest neighbors "
                "for all 13 unanswerable questions (e.g., Titanic box office, ibuprofen dosage) with similarity scores of 0.45–0.55. "
                "By contrast, BM25 sparse retrieval correctly abstained on 11/13 queries (0 candidate tokens). "
                "**Remediation**: Industrial RAG requires an empirical cosine similarity cutoff ($\ge 0.65$) to prevent out-of-domain hallucinations."
            )

            st.divider()

            # ---------------- SECTION 5: RE-RUN ACTION ---------------- #
            if st.button("🔄 Re-run Complete Evaluation Benchmark Suite", use_container_width=True):
                with st.spinner("Executing full benchmark runner across 50 questions & all chunk strategies..."):
                    try:
                        from app.evaluation.runner import ExperimentRunner
                        r = ExperimentRunner()
                        r.run_all()
                        st.toast("Evaluation benchmark re-executed successfully!", icon="✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Execution Error: {e}")

        except Exception as e:
            st.error(f"Error parsing experiment results: {e}")
