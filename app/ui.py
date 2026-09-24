"""Streamlit web UI for RepoGuide.

Run from the project root:  python -m streamlit run app/ui.py

This file only handles the page. All the real work (indexing, search,
answering, citation checks) is done by the same functions the CLI uses.
"""

import time
from pathlib import Path

import streamlit as st

from app.citations import BRACKETS, CITATION, citation_summary, is_supported, validate_citations
from app.embeddings import EmbeddingModel
from app.generator import CODE_SPANS, generate_answer
from app.github import resolve_repository
from app.retriever import build_index, retrieve

SAMPLE_REPO = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_repo"
TOP_K = 5

AUTHOR_NAME = "Anurag Veluri"
LINKEDIN_URL = ""  # TODO: paste your LinkedIn profile URL here

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


def index_repository(source: str) -> None:
    """Download (if it's a GitHub URL) and index a repo, keeping the result
    in session_state so it survives reruns. Starts a fresh chat."""
    start = time.perf_counter()
    index = build_index(str(resolve_repository(source)), get_embedding_model())
    st.session_state.index = index
    st.session_state.indexed_path = source
    st.session_state.index_stats = (
        len({c.file_path for c in index.chunks}),
        len(index.chunks),
        time.perf_counter() - start,
    )
    st.session_state.messages = []


def repo_display_name(source: str) -> str:
    if source == str(SAMPLE_REPO):
        return "Sample bookstore app"
    if "github.com" in source:
        return source.split("github.com")[-1].strip("/:").split("?")[0].split("#")[0]
    return Path(source).name


def show_sidebar() -> None:
    st.sidebar.markdown("### 📦 Repository")

    # A form means pressing Enter in the box submits it, same as the button.
    with st.sidebar.form("repo_form"):
        source = st.text_input(
            "Paste a GitHub repo link (or a local folder path)",
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
    st.sidebar.caption("Runs locally · BGE embeddings + FAISS · Llama 3.2 via Ollama")


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
Everything runs locally on a laptop, with no paid APIs.

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


def show_header() -> None:
    show_top_bar()
    st.title("🔎 RepoGuide")
    st.markdown(
        "Ask questions about any Python codebase and get answers that cite the exact files and lines.  \n"
        ":green-badge[Runs locally] :violet-badge[Llama 3.2] "
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
    st.set_page_config(page_title="RepoGuide", page_icon="🔎", layout="centered")
    st.session_state.setdefault("messages", [])

    show_header()
    show_sidebar()
    if "index" in st.session_state:
        show_chat()
    else:
        show_welcome()


if __name__ == "__main__":  # Streamlit runs the file as __main__
    main()
