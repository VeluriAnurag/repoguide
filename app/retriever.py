"""Retrieval: build an index for a repository and search it with a question."""

import argparse
import time

from app.chunker import Chunk, chunk_repository
from app.embeddings import EmbeddingModel
from app.github import resolve_repository
from app.index import RepositoryIndex, SearchResult

# Embedding runs at roughly 25 chunks/second on a laptop CPU, so 3,000 chunks
# is about two minutes. Bigger repos get a clear error instead of a long wait.
MAX_CHUNKS = 3000


def embedding_text(chunk: Chunk) -> str:
    """Text that gets embedded for a chunk: a short header plus the source.

    File paths and symbol names carry a lot of meaning ("db.py",
    "authenticate_user"), so we include them. This only affects the vector;
    ``chunk.text`` stays exactly the original lines for citations.
    """
    header = f"File: {chunk.file_path}"
    if chunk.symbol:
        header += f"\nSymbol: {chunk.symbol} ({chunk.chunk_type})"
    return f"{header}\n\n{chunk.text}"


def build_index(repo_path: str, embedding_model) -> RepositoryIndex:
    """Chunk a repository, embed every chunk, and load them into FAISS."""
    chunks = chunk_repository(repo_path)
    if not chunks:
        raise ValueError("No Python files or docs (.py, .md, .txt, .rst) found in this repository.")
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(
            f"This repository has {len(chunks):,} chunks. RepoGuide handles up to "
            f"{MAX_CHUNKS:,} (about two minutes of indexing on a laptop)."
        )

    embeddings = embedding_model.encode_documents([embedding_text(c) for c in chunks])
    index = RepositoryIndex()
    index.build(chunks, embeddings)
    return index


def retrieve(
    query: str, embedding_model, repository_index: RepositoryIndex, k: int = 5
) -> list[SearchResult]:
    """Embed a question and return the k most relevant chunks."""
    query_embedding = embedding_model.encode_query(query)
    return repository_index.search(query_embedding, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a repository with a question.")
    parser.add_argument("repo", help="local folder or GitHub URL")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5, help="number of results")
    parser.add_argument("--show-code", action="store_true", help="print chunk text")
    args = parser.parse_args()

    model = EmbeddingModel()
    start = time.perf_counter()
    index = build_index(str(resolve_repository(args.repo)), model)
    print(f"Indexed {len(index.chunks)} chunks in {time.perf_counter() - start:.1f}s\n")

    for rank, result in enumerate(retrieve(args.question, model, index, k=args.k), 1):
        c = result.chunk
        print(f"{rank}. {c.citation:<40} score={result.score:.3f}  {c.symbol or ''}")
        if args.show_code:
            print("   " + c.text.replace("\n", "\n   ") + "\n")


if __name__ == "__main__":
    main()
