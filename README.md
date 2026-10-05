# 🔎 RepoGuide

**Ask questions about any Python codebase and get answers that cite the exact files and lines.**

Paste a public GitHub link, ask something like *"Where is the Value class defined?"*, and RepoGuide answers using only the repo's real code, with citations like `micrograd/engine.py:2-3` that you can open and check. Everything runs on my laptop with open models: no paid AI APIs.

**Built with:** Python · Llama 3.2 (Ollama) · Hugging Face BGE embeddings · FAISS · Streamlit · pytest

![RepoGuide demo: paste a GitHub link, ask simple questions, get answers with checked citations](docs/demo.gif)

*35-second demo: indexing [karpathy/micrograd](https://github.com/karpathy/micrograd) and asking two simple questions. [Full-quality video](docs/repoguide-demo.mp4).*

| Home | Answer with checked citations |
|---|---|
| ![RepoGuide home page](docs/screenshot-home.png) | ![An answer with citations](docs/screenshot-answer.png) |

## What It Does

- **Works with any public GitHub repo** (or a local folder). Paste a link and it downloads and indexes the code in seconds.
- **Answers in plain English** using Llama 3.2, but only from code it actually found in the repo.
- **Cites every claim** with a file and line range, and shows the cited code right under the answer.
- **Checks its own citations.** If the model cites something it wasn't shown, RepoGuide flags it instead of hiding it.
- **Says when it doesn't know.** Asking how the sample app sends emails gets "I couldn't find that in the retrieved code."

## How It Works

```text
GitHub link or folder
  -> download          shallow git clone (latest version only)
  -> split into chunks one chunk per function/class using Python's ast module,
                       each with its file path and line numbers
  -> embed             BGE turns each chunk into 384 numbers that capture its meaning
  -> search            FAISS finds the 5 chunks closest to the question
  -> answer            Llama 3.2 answers from those 5 chunks and cites them as [1], [2]
  -> cite + check      code swaps [1] for the real file:lines and checks every citation
  -> show              Streamlit chat with the answer, the check, and the cited code
```

The key idea: every chunk keeps its file path and line numbers the whole way through, so citations always point to real lines.

## What I Learned Building It

**Split code where Python splits it.** I use the `ast` module to find where each function and class starts and ends, so a search result is a complete function, not half of one. A test checks that every line of code lands in exactly one chunk with the right line numbers. I also ran that check on Python's entire standard library (2,375 files, 0 errors).

**PyTorch and FAISS crashed together on my Mac.** I traced it to both libraries bundling their own copy of OpenMP (a library for using multiple CPU cores). The common fix online hides the error but can give wrong results, so instead I switched to ONNX Runtime to run the same Hugging Face model. I checked the outputs matched PyTorch (difference ~0.0000002) before switching, and it cut the install by about 500 MB.

**Small models copy examples.** My first prompt showed the model an example citation like `[src/db.py:12-29]`, and it copied `src/` into its answers: 0 of 7 citations pointed to real files. Now the model only cites source numbers (`[1]`, `[2]`) and my code fills in the real file and lines. On 8 questions about the `requests` repo, that went from 10 valid / 5 broken citations to **20 valid / 0 broken**.

**Don't trust the model's citations; check them.** A simple rule-based checker confirms each citation points to a chunk the model was actually given, and flags broken ones like `[7]` (when there were only 5 sources) or `[Retry:START-END]`.

**Measure before guessing.** Indexing Flask took 2 minutes. I suspected wasted work from padding (each batch is padded to its longest chunk), but my first measurement said no. That measurement was wrong (it counted the padding), and once fixed it showed half the work was wasted. Sorting chunks by length made indexing **2x faster** (120s → 58s) with identical results.

**Treat downloaded code as untrusted.** RepoGuide only reads downloaded repos, never runs them. GitHub links must match a strict pattern, and git is never allowed to prompt for a password. For a public version, folder paths are blocked so visitors can't read the server's own files.

## Results So Far

| | |
|---|---|
| Citation accuracy on 8 questions about `psf/requests` | 20 of 20 citations valid, 8 of 8 answers passed the check |
| Real repos tested | requests, flask, tqdm, httpx, micrograd, awesome (Markdown only) |
| Indexing speed | ~25 chunks/second on a laptop CPU (micrograd: 3s, Flask: 58s) |
| Answer time | about 2-11 seconds on an M-series MacBook |
| Tests | 108 passing |

These are small, hand-picked tests. A proper search-quality benchmark is next on the roadmap.

## Run It Yourself

Requires Python 3.11+ and [Ollama](https://ollama.com).

```bash
git clone https://github.com/VeluriAnurag/repoguide.git
cd repoguide
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2        # ~2 GB, with the Ollama app open

python -m streamlit run app/ui.py
```

Then open http://localhost:8501, paste a GitHub link, and ask a question.

It also works from the terminal:

```bash
python -m app.ask https://github.com/psf/requests "How does requests handle redirects?"
python -m app.retriever tests/fixtures/sample_repo "where is the database connection set up?"   # search only, no LLM
python -m app.chunker tests/fixtures/sample_repo                                              # see how code is split
```

## Project Structure

```text
app/
  ingest.py       find files, skip folders like .git and node_modules
  chunker.py      split code by function/class with line numbers
  embeddings.py   BGE embeddings via ONNX Runtime
  index.py        FAISS index that keeps metadata attached to each vector
  retriever.py    build the index and search it
  generator.py    prompt Llama 3.2 and turn [1] into file:lines citations
  citations.py    find and check citations
  github.py       safely download public GitHub repos
  ask.py          command line Q&A
  ui.py           Streamlit web app
tests/            108 tests, plus a small sample project to test against
scripts/          demo recording and optional Hugging Face deployment
```

## Tests

```bash
pytest                  # everything (downloads the BGE model on first run; skips the LLM test if Ollama is closed)
pytest -m "not slow"    # quick tests only
```

## Limitations

- Best at "where is X / how does Y work" questions. The model only sees the 5 most relevant chunks, so big-picture questions about a whole repo are weaker.
- The citation check confirms an answer points to real code the model was shown, not that every sentence about that code is correct. Llama 3.2 (3B) sometimes adds its own reasoning.
- Repos over 3,000 chunks (large projects like Django) are too slow to index on a laptop CPU.
- Test files sometimes rank above the code they test, because test names read like plain English.

## Roadmap

- [x] Function-aware chunking with exact line numbers
- [x] BGE embeddings + FAISS search
- [x] Llama 3.2 answers with numbered-source citations
- [x] Rule-based citation checking
- [x] Streamlit chat UI with sources panel
- [x] Public GitHub repos
- [ ] Search-quality benchmark (does the right code show up in the top 5?)
- [ ] Use the benchmark to fix test files outranking source code
- [ ] Public hosted version (ready to deploy on Hugging Face Spaces with `scripts/deploy_space.py`; Docker Spaces need a paid PRO account)

## Author

**Anurag Veluri**, Computational Modeling and Data Analytics at Virginia Tech
[LinkedIn](https://www.linkedin.com/in/anurag-veluri-bb7069308/) · [GitHub](https://github.com/VeluriAnurag)
