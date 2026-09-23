# RepoGuide

RepoGuide answers questions about a Python codebase and shows exactly which files and lines the answer came from. Everything runs locally on my laptop, with no paid APIs.

**Stack:** Python, Llama 3.2 (via Ollama), Hugging Face BGE embeddings, FAISS, Streamlit

> Work in progress. Search works today; LLM answers, citation checks, and the UI are in progress (see [Status](#status)).

## Example

```bash
python -m app.retriever tests/fixtures/sample_repo "where is the database connection initialized?"
```

```text
1. bookstore/db.py:10-15                    score=0.712  get_connection
2. README.md:14-17                          score=0.686  Architecture
3. bookstore/db.py:1-7                      score=0.668
```

The top result is the function that opens the database connection, even though the question and the code use different words ("initialized" vs. `sqlite3.connect`).

## How It Works

```text
Python repo
  -> find files        skip folders like .git, .venv, node_modules
  -> split into chunks one chunk per function/class, with file + line numbers
  -> embed chunks      BGE turns each chunk into 384 numbers that capture its meaning
  -> FAISS index       finds the chunks closest to a question
  -> Llama 3.2         writes an answer using only those chunks        (in progress)
  -> check citations   make sure every cited line range really exists  (planned)
  -> Streamlit UI                                                       (planned)
```

Every chunk keeps its file path and line numbers the whole way through, which is what makes citations like `bookstore/db.py:10-15` possible.

## Design Decisions

**Splitting code by function, not by character count.** I use Python's built-in `ast` module to find where each function and class starts and ends, so a search result is a complete function instead of half of one. Code outside functions (imports, constants) is kept too, and large classes are split into their methods.

**Adding the file name to what gets embedded.** Before embedding a chunk, I add a short header like `File: bookstore/db.py`. File and function names say a lot about what code does, so this helps search.

**Running BGE with ONNX Runtime instead of PyTorch.** PyTorch and FAISS crashed when used in the same program on my Mac. I traced it to both libraries bringing their own copy of the same helper library (OpenMP), which conflict. I switched to ONNX Runtime, a lighter way to run the same Hugging Face model. I checked that it gives the same numbers as PyTorch before switching, and it cut the install size by about 500 MB.

**Keeping everything local.** Embeddings run on the CPU and Llama 3.2 runs through Ollama, so no code leaves the machine.

## Status

| Step | Status |
|---|---|
| File discovery + function-aware chunking | Done |
| BGE embeddings + FAISS search | Done |
| Llama 3.2 answers with citations | In progress |
| Citation checking | Planned |
| Streamlit UI | Planned |
| Measuring search quality | Planned |

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See how a repo gets split into chunks:

```bash
python -m app.chunker tests/fixtures/sample_repo
```

## Tests

```bash
pytest                  # everything (first run downloads the BGE model, ~130 MB)
pytest -m "not slow"    # quick tests only, no model download
```

The tests use a small sample project in `tests/fixtures/sample_repo`. The most important one checks that every line of code ends up in exactly one chunk with the correct line numbers, since wrong line numbers would mean wrong citations.

## Known Issues

- Test files sometimes rank above the code they test, because test names read like plain English.
- Very short chunks (like a one-line `__init__.py`) occasionally show up in results.
