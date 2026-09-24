import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.chunker import Chunk
from app.retriever import build_index, embedding_text, retrieve

SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"
PROJECT_ROOT = Path(__file__).parent.parent


class FakeEmbeddingModel:
    """Stands in for BGE: one dimension per keyword.

    Anything with the same two methods as EmbeddingModel works with the
    retriever, so unit tests don't need to download or run a real model.
    Vectors are normalized, so only *which* keywords appear matters, not how
    often; each keyword should therefore appear in just one fixture chunk.
    """

    TOPICS = ["lru_cache", "hmac", "install"]

    def encode_documents(self, texts):
        return np.stack([self._vector(t) for t in texts])

    def encode_query(self, query):
        return self._vector(query)

    def _vector(self, text):
        v = np.array([text.lower().count(t) for t in self.TOPICS], dtype=np.float32)
        v += 0.01  # avoid all-zero vectors
        return v / np.linalg.norm(v)


def test_embedding_text_includes_path_and_symbol():
    chunk = Chunk("app/db.py", 3, 5, "connect", "function", "def connect(): ...")

    assert embedding_text(chunk) == (
        "File: app/db.py\nSymbol: connect (function)\n\ndef connect(): ..."
    )


def test_embedding_text_without_symbol():
    chunk = Chunk("app/db.py", 1, 2, None, "module", "import os")

    assert embedding_text(chunk) == "File: app/db.py\n\nimport os"


def test_retrieve_returns_relevant_chunk_with_metadata():
    model = FakeEmbeddingModel()
    index = build_index(str(SAMPLE_REPO), model)

    [top] = retrieve("lru_cache", model, index, k=1)

    assert top.chunk.file_path == "bookstore/db.py"
    assert top.chunk.symbol == "get_connection"
    assert (top.chunk.start_line, top.chunk.end_line) == (10, 15)


def test_build_index_on_empty_repo_raises(tmp_path):
    with pytest.raises(ValueError):
        build_index(str(tmp_path), FakeEmbeddingModel())


# --- Slow tests: use the real BGE model (downloaded on first run) ------------


SEARCH_CODE = (
    "import numpy as np, faiss\n"
    "x = np.random.rand(20000, 384).astype('float32')\n"
    "i = faiss.IndexFlatIP(384); i.add(x); i.search(x[:200], 5)\n"
)
EMBED_CODE = (
    "from app.embeddings import EmbeddingModel\n"
    "EmbeddingModel().encode_documents(['hello'] * 64)\n"
)


@pytest.mark.slow
@pytest.mark.parametrize("code", [SEARCH_CODE + EMBED_CODE, EMBED_CODE + SEARCH_CODE])
def test_embedding_and_faiss_coexist_in_either_order(code):
    # Regression: with PyTorch embeddings, two OpenMP runtimes (torch's and
    # faiss's) crashed the process in either order once search went
    # multi-threaded. Runs in a fresh process because a crash would kill pytest.
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True
    )
    assert result.returncode == 0, result.stderr.decode()[-500:]


@pytest.fixture(scope="module")
def real_index():
    from app.embeddings import EmbeddingModel

    model = EmbeddingModel()
    return model, build_index(str(SAMPLE_REPO), model)


@pytest.mark.slow
@pytest.mark.parametrize(
    "question, expected_symbol",
    [
        ("where is the database connection initialized?", "get_connection"),
        ("how are users authenticated?", "authenticate_user"),
        ("how do I install and set up the project?", "Setup"),
    ],
)
def test_real_model_ranks_expected_chunk_first(real_index, question, expected_symbol):
    model, index = real_index

    top = retrieve(question, model, index, k=1)[0]

    assert top.chunk.symbol == expected_symbol


def test_build_index_rejects_repos_over_the_size_limit(monkeypatch):
    import app.retriever

    monkeypatch.setattr(app.retriever, "MAX_CHUNKS", 3)
    with pytest.raises(ValueError, match="handles up to 3"):
        build_index(str(SAMPLE_REPO), FakeEmbeddingModel())


@pytest.mark.slow
def test_length_sorted_batches_keep_vectors_in_original_order():
    from app.embeddings import EmbeddingModel

    model = EmbeddingModel()
    texts = ["x " * 300, "short", "medium length text " * 10, "a"]

    together = model.encode_documents(texts)
    one_by_one = np.stack([model.encode_documents([t])[0] for t in texts])

    assert np.allclose(together, one_by_one, atol=1e-5)
