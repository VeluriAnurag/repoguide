"""Ask a question about a repository from the command line.

Usage: python -m app.ask <folder or GitHub URL> "<question>"
"""

import argparse
import sys

from app.citations import citation_summary
from app.embeddings import EmbeddingModel
from app.generator import DEFAULT_LLM, generate_answer
from app.github import resolve_repository
from app.retriever import build_index, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about a repository.")
    parser.add_argument("repo", help="local folder or GitHub URL")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5, help="number of chunks to give the LLM")
    parser.add_argument("--model", default=DEFAULT_LLM, help="Ollama model name")
    args = parser.parse_args()

    embedding_model = EmbeddingModel()
    try:
        repo_path = resolve_repository(args.repo)
        print("Indexing repository...")
        index = build_index(str(repo_path), embedding_model)
    except (ValueError, RuntimeError) as e:
        sys.exit(f"Error: {e}")
    results = retrieve(args.question, embedding_model, index, k=args.k)

    print("Thinking...\n")
    try:
        answer = generate_answer(args.question, results, model=args.model)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")

    print(answer)
    print()
    level, message = citation_summary(answer, [r.chunk for r in results])
    print(f"WARNING: {message}" if level == "warning" else message)

    print("\nRetrieved sources:")
    for r in results:
        print(f"  {r.chunk.citation:<40} {r.chunk.symbol or ''}")


if __name__ == "__main__":
    main()
