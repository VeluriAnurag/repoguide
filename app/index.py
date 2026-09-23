"""FAISS vector index that keeps chunk metadata attached to every vector.

FAISS only stores vectors and returns row numbers. We keep the chunks in a
list in the same order, so row ``i`` in FAISS is always ``self.chunks[i]``.
"""

from dataclasses import dataclass

import faiss
import numpy as np

from app.chunker import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity: higher means more relevant


class RepositoryIndex:
    def __init__(self):
        self.index = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"{len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        if len(chunks) == 0:
            raise ValueError("Cannot build an index with no chunks")

        # IndexFlatIP = exact search by inner (dot) product. Our vectors are
        # normalized, so the dot product is cosine similarity. "Flat" means it
        # compares the query against every vector: simple and exact, and fast
        # enough for tens of thousands of chunks.
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self.chunks = list(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        if self.index is None:
            raise RuntimeError("Index has not been built")

        k = min(k, len(self.chunks))
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        scores, rows = self.index.search(query, k)

        return [
            SearchResult(chunk=self.chunks[row], score=float(score))
            for score, row in zip(scores[0], rows[0])
        ]
