from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import time
from pathlib import Path

import streamlit as st

from src.config import cfg
from src.logger import get_logger
from src.pipeline import run_ingestion
from src.query import answer_question
from src.vectorstore import delete_index, describe_index, get_or_create_index

logger = get_logger(__name__)

st.set_page_config(
    page_title="DocMind — AI Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0; font-size: 1rem; }
    .source-chip {
        display: inline-block;
        background: #e8f4fd;
        color: #1a73e8;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px 4px 2px 0;
        border: 1px solid #b8d8f5;
    }
    .similarity-bar {
        height: 6px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 3px;
    }
    .chunk-card {
        background: #f8f9ff;
        border: 1px solid #e0e4ff;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
    }
    .stButton>button { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

st.markdown(
    """
    <div class="main-header">
        <h1>🧠 DocMind</h1>
        <p>AI-powered Q&amp;A chatbot — ask anything about your documents</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configuration")

    with st.expander("🗄️ Endee Settings", expanded=False):
        st.code(f"URL:   {cfg.endee_base_url}")
        st.code(f"Index: {cfg.endee_index_name}")
        st.code(f"Dim:   {cfg.embedding_dimension}")

        if st.button("🔍 Check index status"):
            with st.spinner("Connecting to Endee…"):
                try:
                    get_or_create_index()
                    st.success("✅ Endee connected")
                    st.session_state.index_ready = True
                except Exception as exc:
                    st.error(f"❌ Cannot reach Endee: {exc}")

    st.divider()

    st.header("📄 Ingest Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or text files",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    chunk_size = col1.number_input("Chunk size", 128, 2048, cfg.chunk_size, 64)
    chunk_overlap = col2.number_input("Overlap", 0, 256, cfg.chunk_overlap, 16)

    if st.button("🚀 Ingest documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document first.")
        else:
            doc_dir = Path(cfg.documents_dir)
            doc_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            for uf in uploaded_files:
                dest = doc_dir / uf.name
                dest.write_bytes(uf.read())
                saved.append(uf.name)

            with st.spinner(f"Ingesting {len(saved)} document(s)…"):
                try:
                    result = run_ingestion(
                        source=doc_dir,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                    )
                    st.session_state.ingested_docs = saved
                    st.session_state.index_ready = True
                    st.success(
                        f"✅ Done!  "
                        f"{result['document_count']} doc(s) · "
                        f"{result['chunk_count']} chunks · "
                        f"{result['vector_count']} vectors"
                    )
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    st.divider()
    st.subheader("📁 Or ingest from disk")
    if st.button("Ingest data/documents/", use_container_width=True):
        with st.spinner("Ingesting from data/documents/…"):
            try:
                result = run_ingestion(
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap)
                )
                st.session_state.index_ready = True
                st.success(
                    f"✅ {result['document_count']} doc(s), "
                    f"{result['chunk_count']} chunks, "
                    f"{result['vector_count']} vectors"
                )
            except FileNotFoundError:
                st.error("data/documents/ not found — create it and add files.")
            except Exception as exc:
                st.error(f"Error: {exc}")

    st.divider()

    st.header("🔎 Query Settings")
    top_k = st.slider("Top-K chunks to retrieve", 1, 10, cfg.top_k)
    show_sources = st.toggle("Show source chunks", value=True)

    st.divider()
    with st.expander("⚠️ Danger zone"):
        if st.button("🗑️ Clear vector index", type="secondary"):
            try:
                delete_index()
                st.session_state.index_ready = False
                st.session_state.ingested_docs = []
                st.warning("Index deleted.")
            except Exception as exc:
                st.error(f"Error: {exc}")

        if st.button("🧹 Clear chat history"):
            st.session_state.messages = []
            st.rerun()

    if st.session_state.ingested_docs:
        st.divider()
        st.caption("📚 Ingested documents")
        for doc in st.session_state.ingested_docs:
            st.markdown(f'<span class="source-chip">📄 {doc}</span>', unsafe_allow_html=True)

if not st.session_state.index_ready:
    st.info(
        "👈 Get started: Upload documents in the sidebar and click Ingest documents, "
        "then ask questions here."
    )

for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role, avatar="🧠" if role == "assistant" else "👤"):
        st.markdown(msg["content"])

        if role == "assistant" and show_sources and msg.get("chunks"):
            with st.expander(f"📚 {len(msg['chunks'])} source chunk(s) used"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    sim_pct = int(chunk["similarity"] * 100)
                    st.markdown(
                        f"""
                        <div class="chunk-card">
                          <strong>#{i}</strong>
                          <span class="source-chip">{chunk['source']}</span>
                          <span style="color:#888;font-size:0.8rem">
                            similarity: {chunk['similarity']:.3f}
                          </span>
                          <div class="similarity-bar" style="width:{sim_pct}%;margin:4px 0 8px"></div>
                          {chunk['text'][:400]}{'…' if len(chunk['text']) > 400 else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

if prompt := st.chat_input(
    "Ask a question about your documents…",
    disabled=not st.session_state.index_ready,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Searching documents and generating answer…"):
            try:
                t0 = time.perf_counter()
                result = answer_question(question=prompt, top_k=int(top_k))
                elapsed = time.perf_counter() - t0

                st.markdown(result.answer)
                st.caption(
                    f"⏱ {elapsed:.2f}s · "
                    f"🔎 {len(result.chunks)} chunk(s) retrieved · "
                    f"🪙 {result.tokens_used} tokens"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "chunks": [
                            {
                                "text": c.text,
                                "source": c.source,
                                "similarity": c.similarity,
                            }
                            for c in result.chunks
                        ],
                    }
                )

                if show_sources and result.chunks:
                    with st.expander(f"📚 {len(result.chunks)} source chunk(s) used"):
                        for i, chunk in enumerate(result.chunks, 1):
                            sim_pct = int(chunk.similarity * 100)
                            st.markdown(
                                f"""
                                <div class="chunk-card">
                                  <strong>#{i}</strong>
                                  <span class="source-chip">{chunk.source}</span>
                                  <span style="color:#888;font-size:0.8rem">
                                    similarity: {chunk.similarity:.3f}
                                  </span>
                                  <div class="similarity-bar" style="width:{sim_pct}%;margin:4px 0 8px"></div>
                                  {chunk.text[:400]}{'…' if len(chunk.text) > 400 else ''}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            except Exception as exc:
                err_msg = f"❌ Error generating answer: {exc}"
                st.error(err_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err_msg}
                )