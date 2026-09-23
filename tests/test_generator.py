from pathlib import Path
from types import SimpleNamespace

import ollama
import pytest

from app.chunker import Chunk
from app.citations import validate_citations
from app.generator import build_messages, format_context, generate_answer
from app.index import SearchResult

SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"

RESULTS = [
    SearchResult(Chunk("bookstore/db.py", 10, 15, "get_connection", "function", "def get_connection(): ..."), 0.7),
    SearchResult(Chunk("README.md", 5, 12, "Setup", "doc", "## Setup"), 0.6),
]


class FakeOllamaClient:
    """Records what it was asked and returns a canned answer."""

    def __init__(self, reply="The connection is opened in get_connection [bookstore/db.py:10-15]."):
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


def test_format_context_labels_each_chunk_with_its_citation():
    assert format_context(RESULTS) == (
        "[bookstore/db.py:10-15]\ndef get_connection(): ...\n\n[README.md:5-12]\n## Setup"
    )


def test_build_messages_has_rules_then_context_and_question():
    system, user = build_messages("where is the db?", RESULTS)

    assert system["role"] == "system" and "ONLY" in system["content"]
    assert user["role"] == "user"
    assert "[bookstore/db.py:10-15]" in user["content"]
    assert "Question: where is the db?" in user["content"]


def test_system_prompt_example_is_not_a_realistic_path():
    # Regression: a realistic example path ([src/db.py:12-29]) got copied into
    # the model's citations. The example must stay an obvious placeholder.
    system, _ = build_messages("q", RESULTS)
    assert "src/" not in system["content"]


def test_generate_answer_calls_model_and_strips_whitespace():
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
