"""Streamlit frontend. Run with:  streamlit run app.py"""

import json

import streamlit as st

from bm25_retriever import BM25Retriever
from config import (
    CHUNKS_JSON_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LLM,
    DEFAULT_TOP_K,
    INDEX_PATH,
    RETRIEVER_BM25,
    RETRIEVER_COSINE,
)
from cosine_similarity import cosine_search
from document_processor import make_signature, process_documents
from embeddings import embed_query, embed_texts
from evaluation import overlap, precision_at_k, reciprocal_rank
from fact_checker import fact_check_chunks, skip_fact_check
from llm import build_context, stream_answer
from vector_store import VectorStore

st.set_page_config(page_title="PDF Chatbot + Fact Checker", page_icon="📄", layout="wide")


# ------------------------------------------------------------------ helpers
def render_reports(reports):
    """Show each retrieved chunk with its fact-check result."""
    for i, r in enumerate(reports, 1):
        c, j, v = r["chunk"], r["judge"], r["verification"]
        with st.expander(f"Chunk {i} · {c['source']} p.{c['page']} · score {c['score']:.2f}"):
            st.write(c["text"])
            if "error" in j:
                st.warning(f"Fact-check error: {j['error']}")
            elif j.get("verdict") == "doubtful":
                st.warning(f"Suspect claim: {j.get('suspect_claim')}")
                if v:
                    verdict = v.get("verdict", "inconclusive")
                    msg = f"Web verdict: **{verdict}**. {v.get('explanation', '')}"
                    if verdict == "contradicted":
                        st.error(msg + f"\n\nCorrect info: {v.get('correct_information', '')}")
                    elif verdict == "supported":
                        st.success(msg)
                    else:
                        st.info(msg)
                    for url in v.get("source_urls", []):
                        st.markdown(f"- {url}")
            else:
                st.success("No doubtful claims found.")


def render_comparison(cmp, key_prefix):
    """Side-by-side Cosine vs BM25 results, with relevance marking and accuracy metrics."""
    cos, bm = cmp["cosine"], cmp["bm25"]
    cos_ids, bm_ids = [c["id"] for c in cos], [c["id"] for c in bm]
    shared = overlap(cos_ids, bm_ids)

    with st.expander("Retrieval comparison: Cosine vs BM25", expanded=True):
        st.caption(
            f"{len(shared)} chunk(s) were returned by both methods (marked 🔗). "
            "Scores are on different scales (cosine is -1 to 1, BM25 is unbounded), "
            "so compare the rankings, not the raw numbers."
        )

        col_cos, col_bm = st.columns(2)
        for col, title, results in (
            (col_cos, "Cosine similarity (embeddings)", cos),
            (col_bm, "BM25 (keywords)", bm),
        ):
            with col:
                st.markdown(f"**{title}**")
                if not results:
                    st.caption("No chunks shared any keyword with the question.")
                for rank, c in enumerate(results, 1):
                    link = " 🔗" if c["id"] in shared else ""
                    st.markdown(f"{rank}. chunk {c['id']} · p.{c['page']} · score `{c['score']:.3f}`{link}")
                    st.caption(c["text"][:180] + "...")

        # Union of both result lists, so the user can mark which chunks truly answer the question
        union = {c["id"]: c for c in cos + bm}
        if not union:
            return

        st.markdown("**Measure accuracy:** tick the chunks that actually help answer the question.")
        relevant = set()
        for cid, c in union.items():
            label = f"Chunk {cid} (p.{c['page']}): {c['text'][:90]}..."
            if st.checkbox(label, key=f"{key_prefix}_rel_{cid}"):
                relevant.add(cid)

        if relevant:
            m_cos, m_bm = st.columns(2)
            for col, name, ids in ((m_cos, "Cosine", cos_ids), (m_bm, "BM25", bm_ids)):
                with col:
                    st.metric(
                        f"{name} Precision@{len(ids) or 0}",
                        f"{precision_at_k(ids, relevant):.0%}",
                        help="Share of retrieved chunks that you marked relevant.",
                    )
                    st.metric(
                        f"{name} MRR",
                        f"{reciprocal_rank(ids, relevant):.2f}",
                        help="1 / rank of the first relevant chunk (1.0 = relevant chunk ranked first).",
                    )
        else:
            st.caption("Tick at least one chunk to see Precision@k and MRR for both methods.")


def index_documents(files, chunk_size, overlap_chars, signature):
    """PDF -> chunks -> embeddings -> FAISS + JSON on disk."""
    with st.spinner("Chunking documents..."):
        chunks = process_documents(files, chunk_size, overlap_chars)
    if not chunks:
        return None
    with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
        vectors = embed_texts([c["text"] for c in chunks])
    new_store = VectorStore()
    new_store.build(chunks, vectors, signature)
    new_store.save()  # writes data/faiss.index and data/chunks.json
    return new_store


def get_bm25(vector_store: VectorStore) -> BM25Retriever:
    """BM25 index is cheap to build from the chunks, so rebuild it whenever the store changes."""
    cached = st.session_state.get("bm25")
    if cached is None or cached[0] != vector_store.signature:
        cached = (vector_store.signature, BM25Retriever(vector_store.chunks))
        st.session_state.bm25 = cached
    return cached[1]


# ------------------------------------------------------------------ state
if "store" not in st.session_state:
    loaded = VectorStore()
    loaded.load()  # reuse a previously saved index if one exists on disk
    st.session_state.store = loaded
    st.session_state.messages = []

store: VectorStore = st.session_state.store

# ------------------------------------------------------------------ sidebar
st.title("📄 PDF Chatbot with Fact-Checking")

with st.sidebar:
    st.header("Settings")
    llm_model = st.text_input("Ollama model", DEFAULT_LLM)
    retriever_choice = st.radio(
        "Retriever used to answer",
        [RETRIEVER_COSINE, RETRIEVER_BM25],
        help="Both retrievers always run for the comparison; this picks which one feeds the LLM.",
    )
    show_comparison = st.toggle("Show Cosine vs BM25 comparison", value=True)
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, DEFAULT_TOP_K)
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, DEFAULT_CHUNK_SIZE, step=100)
    overlap_chars = st.slider("Chunk overlap (characters)", 0, 500, DEFAULT_CHUNK_OVERLAP, step=50)
    do_fact_check = st.toggle("Fact-check retrieved chunks", value=True)
    uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

    if uploaded:
        files = [(f.name, f.getvalue()) for f in uploaded]
        signature = make_signature(files, chunk_size, overlap_chars)
        if store.signature != signature:
            new_store = index_documents(files, chunk_size, overlap_chars, signature)
            if new_store is None:
                st.error("No extractable text found. The PDF may be scanned images (needs OCR).")
            else:
                st.session_state.store = store = new_store
                st.session_state.messages = []

    if store.is_ready:
        st.success(f"{len(store.chunks)} chunks indexed")
        with st.expander("Stored data (FAISS + JSON)"):
            st.caption(f"FAISS index: `{INDEX_PATH}`  \nJSON chunks: `{CHUNKS_JSON_PATH}`")
            st.write(f"Vectors in FAISS: **{store.index.ntotal}** · dimension: **{store.index.d}**")
            st.json(json.loads(CHUNKS_JSON_PATH.read_text(encoding="utf-8"))["chunks"][:3])
            st.caption("Showing the first 3 chunks from chunks.json")

# ------------------------------------------------------------------ chat
if not store.is_ready:
    st.info("Upload a PDF in the sidebar to start.")

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("reports"):
            render_reports(msg["reports"])
        if msg.get("comparison"):
            render_comparison(msg["comparison"], key_prefix=f"msg{idx}")

question = st.chat_input("Ask something about your document...", disabled=not store.is_ready)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    assistant_idx = len(st.session_state.messages)  # index this reply will get in the history

    with st.chat_message("assistant"):
        # Run BOTH retrievers so they can be compared
        cosine_results = cosine_search(store, embed_query(question), top_k)
        bm25_results = get_bm25(store).search(question, top_k)
        comparison = {"cosine": cosine_results, "bm25": bm25_results}

        # The chosen retriever feeds the LLM
        retrieved = cosine_results if retriever_choice == RETRIEVER_COSINE else bm25_results

        reports = []
        if not retrieved:
            answer = (
                "BM25 found no chunks that share a keyword with your question. "
                "Try rephrasing, or switch to the cosine retriever in the sidebar."
            )
            st.warning(answer)
        else:
            if do_fact_check:
                with st.status("Retrieving and fact-checking...", expanded=True) as status:
                    reports = fact_check_chunks(llm_model, retrieved, on_status=status.write)
                    status.update(label="Fact-check complete", state="complete", expanded=False)
            else:
                reports = skip_fact_check(retrieved)

            try:
                answer = st.write_stream(stream_answer(llm_model, question, build_context(reports)))
            except Exception as e:
                answer = f"Could not reach Ollama model `{llm_model}`: {e}"
                st.error(answer)

            render_reports(reports)

        if show_comparison:
            render_comparison(comparison, key_prefix=f"msg{assistant_idx}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "reports": reports,
            "comparison": comparison if show_comparison else None,
        }
    )