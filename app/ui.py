"""Streamlit web UI for RepoGuide.

Run from the project root:  python -m streamlit run app/ui.py

This file only handles the page. All the real work (indexing, search,
answering, citation checks) is done by the same functions the CLI uses.
"""

import time
from pathlib import Path

import streamlit as st

from app.citations import citation_summary, is_supported, validate_citations
from app.embeddings import EmbeddingModel
from app.generator import generate_answer
from app.retriever import build_index, retrieve

SAMPLE_REPO = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_repo"
TOP_K = 5


@st.cache_resource
def get_embedding_model() -> EmbeddingModel:
    # Streamlit reruns this whole script on every click. cache_resource keeps
    # one model in memory instead of reloading it each time.
    return EmbeddingModel()


def index_repository(repo_path: str) -> None:
    """Build the index and store it in session_state so it survives reruns."""
    start = time.perf_counter()
    index = build_index(repo_path, get_embedding_model())
    st.session_state.index = index
    st.session_state.indexed_path = repo_path
    st.session_state.index_stats = (
        len(index.chunks),
        len({c.file_path for c in index.chunks}),
        time.perf_counter() - start,
    )


def show_sidebar() -> None:
    st.sidebar.header("Repository")
    repo_path = st.sidebar.text_input("Path to a Python project", value=str(SAMPLE_REPO))

    if st.sidebar.button("Index repository", type="primary"):
        if not Path(repo_path).is_dir():
            st.sidebar.error("That folder doesn't exist.")
        else:
            with st.sidebar.status("Indexing...") as status:
                try:
                    index_repository(repo_path)
                    status.update(label="Indexed", state="complete")
                except ValueError as e:
                    status.update(label="Indexing failed", state="error")
                    st.sidebar.error(str(e))

    if "index" in st.session_state:
        chunks, files, seconds = st.session_state.index_stats
        st.sidebar.success(f"{chunks} chunks from {files} files ({seconds:.1f}s)")
        st.sidebar.caption(f"Indexed: `{st.session_state.indexed_path}`")
    else:
        st.sidebar.info("Index a repository to start asking questions.")


def show_sources(results, answer: str) -> None:
    valid_citations = validate_citations(answer, [r.chunk for r in results]).valid

    st.subheader("Sources")
    for r in results:
        c = r.chunk
        is_cited = any(is_supported(v, [c]) for v in valid_citations)
        label = f"{'✅ cited · ' if is_cited else ''}{c.citation}  {c.symbol or ''}  (score {r.score:.2f})"
        with st.expander(label, expanded=is_cited):
            language = "python" if c.file_path.endswith(".py") else "markdown"
            st.code(c.text, language=language)


def main() -> None:
    st.set_page_config(page_title="RepoGuide", page_icon="🔎", layout="wide")
    st.title("RepoGuide")
    st.caption("Ask questions about a Python codebase. Answers cite the exact files and lines.")

    show_sidebar()
    if "index" not in st.session_state:
        return

    with st.form("question_form"):
        question = st.text_input(
            "Question", placeholder="e.g. Where is the database connection initialized?"
        )
        asked = st.form_submit_button("Ask")

    if not (asked and question.strip()):
        return

    results = retrieve(question, get_embedding_model(), st.session_state.index, k=TOP_K)
    with st.spinner("Llama 3.2 is thinking..."):
        try:
            answer = generate_answer(question, results)
        except RuntimeError as e:
            st.error(str(e))
            return

    st.subheader("Answer")
    st.markdown(answer)

    level, message = citation_summary(answer, [r.chunk for r in results])
    {"ok": st.success, "warning": st.warning, "info": st.info}[level](message)

    show_sources(results, answer)


main()
