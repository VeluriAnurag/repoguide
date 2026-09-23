# RepoGuide – Codebase RAG Assistant

## Project Goal
Build a local-first RAG assistant that can ingest a Python repository and answer questions about its code and documentation while citing the exact source files and line ranges used for each answer.

This project is intended to become the implementation behind the resume project:

> RepoGuide – Codebase RAG Assistant  
> Python, Llama 3.2, Hugging Face, FAISS, Streamlit

## Status

| Milestone | Scope | Status |
|---|---|---|
| 1 | File discovery + AST-aware chunking | Done |
| 2 | Hugging Face BGE embeddings + FAISS retrieval (CLI demo) | Next |
| 3 | Llama 3.2 answers via Ollama, with citations | Planned |
| 4 | Citation validation | Planned |
| 5 | Streamlit UI | Planned |
| 6 | Retrieval evaluation + polish | Planned |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Inspect how a repository gets chunked:

```bash
python -m app.chunker tests/fixtures/sample_repo
```

## Tech Choices

- **Embeddings:** `BAAI/bge-small-en-v1.5` from Hugging Face, loaded with `sentence-transformers`. Runs locally on CPU.
- **Vector search:** FAISS.
- **LLM:** Llama 3.2 (3B) served locally by [Ollama](https://ollama.com). Ollama handles model download, quantization, and Apple Silicon acceleration, so RepoGuide only needs its small Python client.

## Core Requirements
1. Accept a local Python repository as input.
2. Discover relevant source files and documentation.
3. Parse Python files into meaningful chunks, preferably around functions/classes rather than arbitrary character boundaries.
4. Generate embeddings with a Hugging Face BGE embedding model.
5. Store/retrieve vectors with FAISS.
6. Run a local Llama 3.2 model for answer generation.
7. Include file-and-line citations in generated answers.
8. Verify that citations actually refer to retrieved source material.
9. Provide a simple Streamlit interface.
10. Keep the first version simple enough to run locally.

## Suggested User Flow
1. User selects a Python repository.
2. RepoGuide indexes the repository.
3. User asks a question such as:
   - "Where is authentication handled?"
   - "Explain how the database connection works."
   - "What calls this function?"
4. The retriever finds relevant chunks.
5. The LLM answers using only retrieved context.
6. The UI displays citations such as:
   `src/auth.py:42-68`

## Initial Architecture

Repository
    |
    v
File Discovery
    |
    v
Python Parser / Function-Aware Chunker
    |
    v
Text Chunks + Metadata
    |
    v
BGE Embeddings
    |
    v
FAISS Index
    |
    v
Retriever
    |
    +----> Retrieved source chunks
    |
    v
Llama 3.2
    |
    v
Citation Validator
    |
    v
Streamlit UI

## Metadata
Each indexed chunk should retain at least:
- repository-relative file path
- starting line
- ending line
- chunk type
- function/class name when available
- original source text

Example:
```json
{
  "file": "src/auth.py",
  "start_line": 42,
  "end_line": 68,
  "symbol": "authenticate_user",
  "chunk_type": "function"
}
```

## Success Criteria
A good first demo should let someone point RepoGuide at a small Python repository and ask:

"Where is the database connection initialized?"

The system should return a concise explanation plus citations that point to the actual relevant lines.

## Engineering Principles
- Prefer local inference where practical.
- Never let the LLM invent source citations.
- Keep source metadata attached to every chunk throughout retrieval.
- Make indexing deterministic and repeatable.
- Separate ingestion, retrieval, generation, and UI code.
- Add tests before adding advanced features.
