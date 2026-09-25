"""Streamlit web UI for RepoGuide.

Run from the project root:  python -m streamlit run app/ui.py

This file only handles the page. All the real work (indexing, search,
answering, citation checks) is done by the same functions the CLI uses.
"""

import os
import threading
import time
from pathlib import Path

import streamlit as st

from app.citations import BRACKETS, CITATION, citation_summary, is_supported, validate_citations
from app.embeddings import EmbeddingModel
from app.generator import CODE_SPANS, generate_answer
from app.github import canonical_name, resolve_repository
from app.index import RepositoryIndex
from app.retriever import build_index, retrieve

SAMPLE_REPO = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_repo"
TOP_K = 5

# On the public website (REPOGUIDE_PUBLIC=1) only GitHub links are allowed,
# so visitors can't make the server read its own files by typing a path.
PUBLIC_MODE = os.environ.get("REPOGUIDE_PUBLIC") == "1"

AUTHOR_NAME = "Anurag Veluri"
LINKEDIN_URL = "https://www.linkedin.com/in/anurag-veluri-bb7069308/"

EXAMPLE_REPOS = {
    "🌐 psf/requests": "https://github.com/psf/requests",
    "🧠 karpathy/micrograd": "https://github.com/karpathy/micrograd",
    "🧪 Sample bookstore app": str(SAMPLE_REPO),
}
EXAMPLE_QUESTIONS = [
    "What does this project do?",
    "Where is the main logic?",
    "How are errors handled?",
]


@st.cache_resource
def get_embedding_model() -> EmbeddingModel:
    # Streamlit reruns this whole script on every click. cache_resource keeps
    # one model in memory instead of reloading it each time.
    return EmbeddingModel()


# --- Indexing ------------------------------------------------------------------


@st.cache_resource
def get_index_lock() -> threading.Lock:
    # One lock shared by every visitor, so only one repo is indexed at a time
    # on a small server. (A plain module-level Lock would be re-created on
    # every rerun and never block anything.)
    return threading.Lock()


@st.cache_resource(max_entries=5, show_spinner=False)
def load_index(name: str) -> tuple[RepositoryIndex, float]:
    """Build an index once and share it: if two visitors ask about the same
    repo, it's only downloaded and embedded the first time."""
    allow_local = not PUBLIC_MODE or name == str(SAMPLE_REPO)
    with get_index_lock():
        start = time.perf_counter()
        index = build_index(str(resolve_repository(name, allow_local)), get_embedding_model())
        return index, time.perf_counter() - start


def index_repository(source: str) -> None:
    """Load the index for a repo into this visitor's session and start a fresh chat."""
    name = canonical_name(source)
    index, seconds = load_index(name)
    st.session_state.index = index
    st.session_state.indexed_path = name
    st.session_state.index_stats = (
        len({c.file_path for c in index.chunks}),
        len(index.chunks),
        seconds,
    )
    st.session_state.messages = []


def repo_display_name(name: str) -> str:
    if name == str(SAMPLE_REPO):
        return "Sample bookstore app"
    if "github.com" in name:
        name = canonical_name(name)
    return name if name.count("/") == 1 and not name.startswith("/") else Path(name).name


def show_sidebar() -> None:
    st.sidebar.markdown("##### 📦&nbsp; Repository")

    # A form means pressing Enter in the box submits it, same as the button.
    with st.sidebar.form("repo_form"):
        source = st.text_input(
            "Paste a public GitHub repo link" if PUBLIC_MODE else "Paste a GitHub repo link (or a local folder path)",
            placeholder="https://github.com/psf/requests",
        )
        submitted = st.form_submit_button("Index repository", type="primary", use_container_width=True)

    st.sidebar.caption("Or try one:")
    for name, url in EXAMPLE_REPOS.items():
        if st.sidebar.button(name, use_container_width=True):
            source, submitted = url, True

    if submitted and not source.strip():
        st.sidebar.error("Paste a GitHub link or a folder path first.")
    elif submitted:
        with st.sidebar.status("Downloading and indexing...", expanded=False) as status:
            try:
                index_repository(source)
                status.update(label="Ready", state="complete")
            except (ValueError, RuntimeError) as e:
                status.update(label="Indexing failed", state="error")
                st.sidebar.error(str(e))

    if "index" in st.session_state:
        files, chunks, seconds = st.session_state.index_stats
        st.sidebar.divider()
        st.sidebar.markdown(f"**{repo_display_name(st.session_state.indexed_path)}**")
        c1, c2, c3 = st.sidebar.columns(3)
        c1.metric("Files", files)
        c2.metric("Chunks", f"{chunks:,}")
        c3.metric("Seconds", f"{seconds:.0f}")
        if st.session_state.messages and st.sidebar.button("🧹 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("Open models only · BGE embeddings + FAISS · Llama 3.2 via Ollama")


# --- Answers -------------------------------------------------------------------


def highlight_citations(answer: str) -> str:
    """Show citations like [app/db.py:10-15] as code so they stand out."""
    parts = CODE_SPANS.split(answer)
    for i in range(0, len(parts), 2):  # skip text that is already code
        parts[i] = BRACKETS.sub(
            lambda m: f"`{m.group(1)}`" if CITATION.search(m.group(1)) else m.group(0),
            parts[i],
        )
    return "".join(parts)


def show_sources(results, answer: str) -> None:
    valid = validate_citations(answer, [r.chunk for r in results]).valid
    labels, cited_flags = [], []
    for r in results:
        cited = any(is_supported(v, [r.chunk]) for v in valid)
        cited_flags.append(cited)
        name = Path(r.chunk.file_path).name
        labels.append(f"{'✅ ' if cited else ''}{name}:{r.chunk.start_line}-{r.chunk.end_line}")

    for tab, r, cited in zip(st.tabs(labels), results, cited_flags):
        with tab:
            c = r.chunk
            details = f"`{c.citation}`"
            if c.symbol:
                details += f" · **{c.symbol}**"
            details += f" · {c.chunk_type} · relevance {r.score:.2f}"
            if cited:
                details += " · ✅ cited in the answer"
            st.markdown(details)
            language = "python" if c.file_path.endswith(".py") else "markdown"
            st.code(c.text, language=language)


def show_assistant_message(message: dict) -> None:
    if "error" in message:
        st.error(message["error"])
        return
    st.markdown(highlight_citations(message["answer"]))
    show_level = {"ok": st.success, "warning": st.warning, "info": st.info}[message["level"]]
    show_level(message["check"], icon={"ok": "✅", "warning": "⚠️", "info": "ℹ️"}[message["level"]])
    st.caption(f"Answered in {message['seconds']:.1f}s using {len(message['results'])} sources")
    with st.expander("📚 Sources", expanded=False):
        show_sources(message["results"], message["answer"])


def answer_question(question: str) -> dict:
    start = time.perf_counter()
    results = retrieve(question, get_embedding_model(), st.session_state.index, k=TOP_K)
    try:
        answer = generate_answer(question, results)
    except RuntimeError as e:
        return {"role": "assistant", "error": str(e)}
    level, check = citation_summary(answer, [r.chunk for r in results])
    return {
        "role": "assistant",
        "answer": answer,
        "level": level,
        "check": check,
        "results": results,
        "seconds": time.perf_counter() - start,
    }


# --- Page ----------------------------------------------------------------------


ABOUT_TEXT = """
**RepoGuide answers questions about a Python codebase and shows exactly which
files and lines each answer came from.** Paste a public GitHub link, ask
something like *"Where is the database connection set up?"*, and get a short
answer with citations like `app/db.py:10-15` that you can open and check.
It only uses open models (no paid AI APIs), so it can run entirely on a laptop.

#### How it works
1. **Download and split the code.** RepoGuide clones the repo and uses
   Python's `ast` module to split each file into chunks along functions and
   classes, keeping the file path and line numbers for every chunk.
2. **Turn chunks into embeddings.** A Hugging Face BGE model turns each chunk
   into 384 numbers that capture its meaning, and FAISS stores them for search.
3. **Find the relevant code.** Your question is embedded the same way, and
   FAISS returns the 5 closest chunks, even if they use different words.
4. **Write the answer.** Llama 3.2 (running locally in Ollama) answers using
   only those 5 chunks and cites them by number. My code swaps the numbers for
   the real file and lines, so the model never writes a line number itself.
5. **Check the citations.** Rule-based code confirms every citation points to
   code the model was actually shown, and flags anything that doesn't.

#### Things I learned building it
- **PyTorch and FAISS crashed together on my Mac.** I traced it to both
  bundling their own copy of OpenMP, and switched to ONNX Runtime after
  checking it gives the same embeddings. It also cut ~500 MB of installs.
- **Small models copy examples.** Asking Llama to copy file paths gave broken
  citations. Letting it cite `[1]`, `[2]` instead took citations on the
  requests repo from 10 valid / 5 broken to 20 valid / 0 broken.
- **Sorting chunks by length made indexing 2x faster**, because batches waste
  less work on padding.

#### Tech
Python · Hugging Face BGE (ONNX Runtime) · FAISS · Llama 3.2 via Ollama ·
Streamlit · pytest (100+ tests)

#### Limits
Works best for "where is X / how does Y work" questions. The citation check
confirms answers point to real code, not that every sentence is correct.
Public repos up to about 3,000 chunks.
"""


@st.dialog("About RepoGuide", width="large")
def show_about() -> None:
    st.markdown(ABOUT_TEXT)
    st.caption(f"Built by {AUTHOR_NAME}")
    if LINKEDIN_URL:
        st.link_button("Connect on LinkedIn", LINKEDIN_URL)


def show_top_bar() -> None:
    """Author credit, LinkedIn, and About in the top-right corner."""
    credit = f"Built by <b>{AUTHOR_NAME}</b>"
    if LINKEDIN_URL:
        credit += f" · <a href='{LINKEDIN_URL}' target='_blank'>LinkedIn</a>"

    _, credit_column, about_column = st.columns([4, 3, 1.4], vertical_alignment="center")
    credit_column.markdown(
        f"<div style='text-align: right; opacity: 0.85'>{credit}</div>",
        unsafe_allow_html=True,
    )
    if about_column.button("About", icon=":material/info:", use_container_width=True):
        show_about()


# Styling that Streamlit's theme settings can't do (widths, gradients, hover
# effects). The theme colors themselves live in .streamlit/config.toml.
CUSTOM_CSS = """
<style>
[data-testid="stMainBlockContainer"] { max-width: 1120px; padding-top: 2rem; }
.rg-hero { margin: 0.5rem 0 0.25rem; white-space: nowrap; }
.rg-hero .rg-emoji { font-size: 3rem; vertical-align: middle; margin-right: 0.6rem; }
.rg-hero .rg-name {
    font-size: 3.6rem; font-weight: 800; letter-spacing: -0.02em; vertical-align: middle;
    background: linear-gradient(90deg, #C7D2FE, #7C83FF 45%, #38BDF8);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.rg-tagline { font-size: 1.3rem; opacity: 0.85; margin: 0.25rem 0 0.75rem; }
@media (max-width: 640px) {  /* phones: smaller title so it fits on one line */
    .rg-hero .rg-emoji { font-size: 2rem; }
    .rg-hero .rg-name { font-size: 2.5rem; }
    .rg-tagline { font-size: 1.1rem; }
}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlockBorderWrapper"] {
    transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-3px); border-color: #7C83FF;
    box-shadow: 0 8px 24px rgba(124, 131, 255, 0.15);
}
[data-testid="stChatMessage"] {
    background: #121932; border: 1px solid #2A3150; border-radius: 1rem; padding: 1rem 1.25rem;
}
</style>
"""


def show_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    show_top_bar()
    st.markdown(
        "<div class='rg-hero'><span class='rg-emoji'>🔎</span> <span class='rg-name'>RepoGuide</span></div>"
        "<div class='rg-tagline'>Ask questions about any Python codebase and get answers "
        "that cite the exact files and lines.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        ":green-badge[No paid APIs] :violet-badge[Llama 3.2] "
        ":blue-badge[BGE + FAISS] :orange-badge[Checked citations]"
    )


def show_welcome() -> None:
    st.write("")
    steps = [
        ("1️⃣", "Paste a repo", "Drop a public GitHub link in the sidebar, or pick an example."),
        ("2️⃣", "Ask anything", "Where is X? How does Y work? Ask in plain English."),
        ("3️⃣", "Check the sources", "Every answer cites file and line numbers you can open."),
    ]
    for column, (icon, title, text) in zip(st.columns(3), steps):
        with column.container(border=True, height="stretch"):
            st.markdown(f"### {icon}\n**{title}**")
            st.caption(text)

    st.write("")
    st.markdown("#### What an answer looks like")
    with st.container(border=True):
        st.markdown("🧑‍💻&nbsp; **Where is the database connection initialized?**")
        st.markdown(
            "🔎&nbsp; The database connection is initialized in the `get_connection` function, "
            "which opens a SQLite connection with `sqlite3.connect(DB_PATH)` and caches it "
            "so it's reused `bookstore/db.py:10-15`."
        )
        st.success("Citations checked: 1 of 1 match the retrieved code.", icon="✅")
        st.caption("A real answer from the sample bookstore app. Try it from the sidebar.")


def show_chat() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🔎"):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                show_assistant_message(message)

    question = None
    if not st.session_state.messages:
        st.caption("Try asking:")
        for column, example in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
            if column.button(example, use_container_width=True):
                question = example

    name = repo_display_name(st.session_state.indexed_path)
    question = st.chat_input(f"Ask about {name}...") or question
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🔎"):
        with st.spinner("Searching the code and asking Llama 3.2..."):
            message = answer_question(question)
        show_assistant_message(message)
    st.session_state.messages.append(message)
    if len(st.session_state.messages) == 2:
        st.rerun()  # hide the example-question buttons after the first answer


def main() -> None:
    st.set_page_config(page_title="RepoGuide", page_icon="🔎", layout="wide")
    st.session_state.setdefault("messages", [])

    show_header()
    show_sidebar()
    if "index" in st.session_state:
        show_chat()
    else:
        show_welcome()


if __name__ == "__main__":  # Streamlit runs the file as __main__
    main()
