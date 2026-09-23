"""Ask a question about a repository from the command line.

Usage: python -m app.ask <repo_path> "<question>"
"""

import argparse
import sys

from app.citations import validate_citations
from app.embeddings import EmbeddingModel
from app.generator import DEFAULT_LLM, generate_answer, is_not_found
from app.retriever import build_index, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about a repository.")
    parser.add_argument("repo_path")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5, help="number of chunks to give the LLM")
    parser.add_argument("--model", default=DEFAULT_LLM, help="Ollama model name")
    args = parser.parse_args()

    embedding_model = EmbeddingModel()
    print("Indexing repository...")
    index = build_index(args.repo_path, embedding_model)
    results = retrieve(args.question, embedding_model, index, k=args.k)

    print("Thinking...\n")
    try:
        answer = generate_answer(args.question, results, model=args.model)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")

    print(answer)
    print()
    print(citation_summary(answer, [r.chunk for r in results]))

    print("\nRetrieved sources:")
    for r in results:
        print(f"  {r.chunk.citation:<40} {r.chunk.symbol or ''}")


def citation_summary(answer: str, chunks) -> str:
    """One or two lines telling the user whether to trust the citations."""
    report = validate_citations(answer, chunks)
    if report.invalid:
        bad = ", ".join(str(c) for c in report.invalid)
        return f"WARNING: these citations don't match any retrieved code: {bad}"
    if report.valid:
        n = len(report.valid)
        return f"Citations checked: {n} of {n} match the retrieved code."
    if is_not_found(answer):
        return "No citations (the answer wasn't in the retrieved code)."
    return "WARNING: this answer has no citations, so it can't be checked."


if __name__ == "__main__":
    main()
