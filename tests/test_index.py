import numpy as np
import pytest

from app.chunker import Chunk
from app.index import RepositoryIndex


def make_chunk(name, line):
    return Chunk(f"{name}.py", line, line + 1, name, "function", f"def {name}(): ...")


def unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def index():
    chunks = [make_chunk("db", 10), make_chunk("auth", 20), make_chunk("setup", 30)]
    vectors = np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)])
    idx = RepositoryIndex()
    idx.build(chunks, vectors)
    return idx


def test_search_returns_closest_chunk_first(index):
    results = index.search(unit(0.1, 0.9, 0), k=3)

    assert [r.chunk.symbol for r in results] == ["auth", "db", "setup"]
    assert results[0].score > results[1].score > results[2].score


def test_search_preserves_metadata(index):
    [result] = index.search(unit(1, 0, 0), k=1)

    assert result.chunk == make_chunk("db", 10)
    assert result.chunk.citation == "db.py:10-11"
    assert result.score == pytest.approx(1.0)


def test_k_larger_than_index_returns_all(index):
    assert len(index.search(unit(1, 1, 1), k=50)) == 3


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        RepositoryIndex().build([make_chunk("a", 1)], np.zeros((2, 3), np.float32))


def test_search_before_build_raises():
    with pytest.raises(RuntimeError):
        RepositoryIndex().search(unit(1, 0, 0))
