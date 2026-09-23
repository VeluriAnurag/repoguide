# RepoGuide Project Specification

## Problem
Understanding an unfamiliar codebase requires repeatedly searching files, tracing functions, reading documentation, and connecting related pieces of code. RepoGuide is a RAG assistant designed to reduce that friction.

## Primary Objective
Given a Python repository, answer natural-language questions about the codebase using retrieved repository content and provide verifiable file-and-line citations.

## MVP Scope

### In Scope
- Python repositories
- `.py` source files
- common documentation such as `.md`, `.txt`, and optionally `.rst`
- function/class-aware chunking
- BGE embeddings
- FAISS vector search
- local Llama 3.2 inference
- citation-aware answers
- Streamlit interface
- basic tests

### Out of Scope for MVP
- autonomous code modification
- executing arbitrary repository code
- production multi-user deployment
- private cloud infrastructure
- support for every programming language
- sophisticated agent/tool orchestration

## Functional Requirements

### FR1 – Repository Ingestion
The system must recursively scan a selected repository while ignoring common generated directories such as:
- `.git`
- `.venv`
- `venv`
- `__pycache__`
- `node_modules`

### FR2 – Code Chunking
Python files should be parsed with the Python AST when possible.

Chunks should preserve:
- source text
- file path
- line range
- symbol name
- chunk type

Prefer functions/classes as semantic units. If a unit is too large, split it into smaller chunks while preserving metadata.

### FR3 – Embedding
Use a Hugging Face BGE embedding model.

The embedding implementation should be isolated behind a small interface so the model can be replaced later.

### FR4 – Retrieval
FAISS should return the top-k most relevant chunks.

The retrieval result should preserve the original metadata.

### FR5 – Generation
Llama 3.2 should answer using retrieved context.

The prompt should explicitly instruct the model:
- do not invent repository facts
- use only provided context
- cite claims using the supplied file and line metadata
- say when the retrieved context is insufficient

### FR6 – Citation Validation
Before displaying an answer, validate that cited paths and line ranges correspond to retrieved chunks.

For MVP, validation can be deterministic and rule-based.

### FR7 – UI
Streamlit should provide:
- repository selection/input
- indexing status
- question input
- answer output
- citations/source panel

## Suggested Repository Structure

```text
repoguide/
├── app/
│   ├── __init__.py
│   ├── ingest.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── index.py
│   ├── retriever.py
│   ├── generator.py
│   ├── citations.py
│   └── ui.py
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_citations.py
├── data/
│   └── .gitkeep
├── requirements.txt
├── .gitignore
├── README.md
└── PROJECT_SPEC.md
```

## Suggested Dependencies

Start with:
- Python 3.11+
- faiss-cpu
- sentence-transformers
- transformers
- torch
- streamlit
- pytest

Only add additional dependencies when the implementation actually needs them.

## Example Citation Format

```text
The database connection is initialized in `src/db.py:12-29`.
The connection is then reused by the repository layer in `src/repository.py:8-21`.
```

## Testing Strategy

Start with unit tests for:
1. file filtering
2. AST chunk extraction
3. line-number preservation
4. embedding/index interfaces
5. retrieval metadata preservation
6. citation validation

Then add one end-to-end test using a tiny fixture repository.

## Development Order

1. Create ingestion and file filtering.
2. Implement AST-based chunking.
3. Add metadata tests.
4. Implement embeddings.
5. Implement FAISS indexing/retrieval.
6. Build a CLI retrieval demo.
7. Add Llama generation.
8. Add citation validation.
9. Build Streamlit UI.
10. Improve retrieval quality and documentation.

## Definition of Done for MVP

A user can select a small Python repository, index it, ask a code question, receive a grounded answer, and inspect citations that map back to real repository lines.
