# RepoGuide

RepoGuide answers questions about a Python codebase and shows exactly which files and lines the answer came from. Everything runs locally on my laptop, with no paid APIs.

**Stack:** Python, Llama 3.2 (via Ollama), Hugging Face BGE embeddings, FAISS, Streamlit

> Work in progress. Search, LLM answers, and citation checks work today; the UI is next (see [Status](#status)).

## Example

```bash
python -m app.ask tests/fixtures/sample_repo "where is the database connection initialized?"
```

```text
The database connection is initialized in the `get_connection` function, which is
decorated with `@functools.lru_cache(maxsize=1)`. [bookstore/db.py:10-15]

This function opens a SQLite connection using `sqlite3.connect(DB_PATH)` and returns
the connection object. [bookstore/db.py:10-15]

Citations checked: 2 of 2 match the retrieved code.
```

Every bracket is a citation you can check: `bookstore/db.py:10-15` is exactly where `get_connection` lives. RepoGuide also checks them automatically and warns about any citation that doesn't match the code the model was given. If the answer isn't in the code, RepoGuide says so instead of guessing (for example, asking how the sample app sends emails gives "I couldn't find that in the retrieved code").

To see just the search results without the LLM:

```bash
python -m app.retriever tests/fixtures/sample_repo "where is the database connection initialized?"
```

## How It Works

```text
Python repo
  -> find files        skip folders like .git, .venv, node_modules
  -> split into chunks one chunk per function/class, with file + line numbers
  -> embed chunks      BGE turns each chunk into 384 numbers that capture its meaning
  -> FAISS index       finds the chunks closest to a question
  -> Llama 3.2         writes an answer using only those chunks, with citations
  -> check citations   flag any citation that isn't in the retrieved chunks
  -> Streamlit UI                                                       (planned)
```

Every chunk keeps its file path and line numbers the whole way through, which is what makes citations like `bookstore/db.py:10-15` possible.

## Design Decisions

**Splitting code by function, not by character count.** I use Python's built-in `ast` module to find where each function and class starts and ends, so a search result is a complete function instead of half of one. Code outside functions (imports, constants) is kept too, and large classes are split into their methods.

**Adding the file name to what gets embedded.** Before embedding a chunk, I add a short header like `File: bookstore/db.py`. File and function names say a lot about what code does, so this helps search.

**Running BGE with ONNX Runtime instead of PyTorch.** PyTorch and FAISS crashed when used in the same program on my Mac. I traced it to both libraries bringing their own copy of the same helper library (OpenMP), which conflict. I switched to ONNX Runtime, a lighter way to run the same Hugging Face model. I checked that it gives the same numbers as PyTorch before switching, and it cut the install size by about 500 MB.

**Writing the prompt so the model cites real files.** Llama only sees the retrieved chunks, each labeled like `[bookstore/db.py:10-15]`, and is told to cite those labels. My first prompt used a realistic example citation (`[src/db.py:12-29]`), and the model copied the `src/` folder into its answers, so none of its citations pointed to real files (0 of 7). Switching to an obvious placeholder (`[folder/file.py:START-END]`) got 9 of 9 citations right on the same questions. It's a small test, which is why I also added an automatic citation check.

**Checking citations with simple rules instead of trusting the model.** After Llama answers, RepoGuide finds every `[file:start-end]` in the answer and checks that one of the retrieved chunks is from that file and covers those lines. If not, it prints a warning with the bad citation. I tested it on real output: with the old prompt it flagged every made-up `src/` citation, and on 10 harder questions about RepoGuide's own code, all 12 citations passed. One limit: it checks that a citation points to code the model was shown, not that the sentence describing that code is correct.

**Keeping everything local.** Embeddings run on the CPU and Llama 3.2 runs through Ollama, so no code leaves the machine.

## Status

| Step | Status |
|---|---|
| File discovery + function-aware chunking | Done |
| BGE embeddings + FAISS search | Done |
| Llama 3.2 answers with citations | Done |
| Citation checking | Done |
| Streamlit UI | Next |
| Measuring search quality | Planned |

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com) for the LLM step.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2    # ~2 GB, with the Ollama app running
```

See how a repo gets split into chunks:

```bash
python -m app.chunker tests/fixtures/sample_repo
```

## Tests

```bash
pytest                  # everything (first run downloads the BGE model, ~130 MB;
                        # the LLM test is skipped if Ollama isn't running)
pytest -m "not slow"    # quick tests only, no model download
```

The tests use a small sample project in `tests/fixtures/sample_repo`. The most important one checks that every line of code ends up in exactly one chunk with the correct line numbers, since wrong line numbers would mean wrong citations.

## Known Issues

- Test files sometimes rank above the code they test, because test names read like plain English.
- Very short chunks (like a one-line `__init__.py`) occasionally show up in results.
