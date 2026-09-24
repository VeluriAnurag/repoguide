from pathlib import Path
from types import SimpleNamespace

import ollama
import pytest

from app.chunker import Chunk
from app.citations import validate_citations
from app.generator import (
    build_messages,
    format_context,
    generate_answer,
    is_not_found,
    replace_source_numbers,
)
from app.index import SearchResult

SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"

RESULTS = [
    SearchResult(Chunk("bookstore/db.py", 10, 15, "get_connection", "function", "def get_connection(): ..."), 0.7),
    SearchResult(Chunk("README.md", 5, 12, "Setup", "doc", "## Setup"), 0.6),
]


class FakeOllamaClient:
    """Records what it was asked and returns a canned answer."""

    def __init__(self, reply="The connection is opened in get_connection [1]."):
        self.reply = reply
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=f"  {self.reply}\n"))


class FailingClient:
    def __init__(self, error):
        self.error = error

    def chat(self, **kwargs):
        raise self.error


def test_format_context_numbers_each_source():
    assert format_context(RESULTS) == (
        "Source [1] bookstore/db.py:10-15\ndef get_connection(): ...\n\n"
        "Source [2] README.md:5-12\n## Setup"
    )


def test_build_messages_has_rules_then_context_and_question():
    system, user = build_messages("where is the db?", RESULTS)

    assert system["role"] == "system" and "ONLY" in system["content"]
    assert user["role"] == "user"
    assert "Source [1] bookstore/db.py:10-15" in user["content"]
    assert "Question: where is the db?" in user["content"]


def test_system_prompt_asks_for_source_numbers_not_paths():
    # Regression: when asked to copy file labels, the model copied example
    # paths (src/...) and placeholders (START-END) into its citations.
    system, _ = build_messages("q", RESULTS)
    assert "[1]" in system["content"]
    assert "src/" not in system["content"] and "START" not in system["content"]


def test_replace_source_numbers_with_real_citations():
    answer = "Opened in get_connection [1]. See setup [2][1] and [1, 2]."

    assert replace_source_numbers(answer, RESULTS) == (
        "Opened in get_connection [bookstore/db.py:10-15]. "
        "See setup [README.md:5-12][bookstore/db.py:10-15] and "
        "[bookstore/db.py:10-15][README.md:5-12]."
    )


def test_replace_leaves_code_and_unknown_numbers_alone():
    answer = "Uses `args[1]` and rows[0], `x = y [1]`, then items()[1] and [7]."

    assert replace_source_numbers(answer, RESULTS) == answer


def test_replace_skips_code_blocks():
    answer = "Run this [1]:\n```python\nprint(sys.argv [1])\n```"

    assert replace_source_numbers(answer, RESULTS) == (
        "Run this [bookstore/db.py:10-15]:\n```python\nprint(sys.argv [1])\n```"
    )


def test_generate_answer_calls_model_and_converts_source_numbers():
    client = FakeOllamaClient()

    answer = generate_answer("where is the db?", RESULTS, model="llama3.2", client=client)

    assert answer == "The connection is opened in get_connection [bookstore/db.py:10-15]."
    [call] = client.calls
    assert call["model"] == "llama3.2"
    assert call["options"] == {"temperature": 0}


def test_missing_model_gives_helpful_error():
    client = FailingClient(ollama.ResponseError("model not found", status_code=404))

    with pytest.raises(RuntimeError, match="ollama pull llama3.2"):
        generate_answer("q", RESULTS, client=client)


def test_ollama_not_running_gives_helpful_error():
    # Nothing listens on port 1, so this behaves like Ollama being closed.
    client = ollama.Client(host="http://127.0.0.1:1")

    with pytest.raises(RuntimeError, match="Open the Ollama app"):
        generate_answer("q", RESULTS, client=client)


# --- Real Llama 3.2 (skipped unless Ollama is running with the model) -------


def ollama_ready() -> bool:
    try:
        names = [m.model for m in ollama.list().models]
    except Exception:
        return False
    return any(name.startswith("llama3.2") for name in names)


@pytest.mark.slow
@pytest.mark.skipif(not ollama_ready(), reason="Ollama is not running or llama3.2 is not pulled")
def test_real_llm_cites_only_retrieved_chunks():
    from app.embeddings import EmbeddingModel
    from app.retriever import build_index, retrieve

    model = EmbeddingModel()
    index = build_index(str(SAMPLE_REPO), model)
    results = retrieve("where is the database connection initialized?", model, index)

    answer = generate_answer("where is the database connection initialized?", results)

    report = validate_citations(answer, [r.chunk for r in results])
    assert report.ok, answer


def test_is_not_found_ignores_case_and_final_period():
    assert is_not_found("i couldn't find that in the retrieved code")
    assert not is_not_found("The connection is opened in get_connection.")
