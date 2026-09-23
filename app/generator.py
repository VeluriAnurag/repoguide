"""Answer generation with a local Llama 3.2 model served by Ollama.

The model only sees the chunks we retrieved, each labeled with its citation
(e.g. [bookstore/db.py:10-15]). The system prompt tells it to answer from
those chunks only and to cite the labels. Milestone 4 checks the citations.
"""

import ollama

DEFAULT_LLM = "llama3.2"

# What the model is told to say when the context doesn't answer the question.
NOT_FOUND_MESSAGE = "I couldn't find that in the retrieved code."

# The citation example below is deliberately a placeholder. With a realistic
# example like [src/db.py:12-29], Llama 3.2 copied the "src/" folder into its
# citations (0 of 7 valid in a small test); with this one, 9 of 9 were valid.

SYSTEM_PROMPT = f"""You are RepoGuide, an assistant that explains a Python codebase.

Rules:
1. Answer using ONLY the code and docs in the context. Do not use outside knowledge about this repository.
2. After every claim, cite the source label exactly as written in the context, in square brackets, e.g. [folder/file.py:START-END] where you copy the real label from the context.
3. Only cite labels that appear in the context. Never make up file names or line numbers.
4. If the context does not contain the answer, say "{NOT_FOUND_MESSAGE}" and stop.
5. Be concise: a few sentences, not an essay."""


def is_not_found(answer: str) -> bool:
    """True if the model said the retrieved code doesn't answer the question."""
    return NOT_FOUND_MESSAGE.lower().rstrip(".") in answer.lower()


def format_context(results) -> str:
    """Turn search results into labeled blocks the model can cite."""
    blocks = [f"[{r.chunk.citation}]\n{r.chunk.text}" for r in results]
    return "\n\n".join(blocks)


def build_messages(question: str, results) -> list[dict]:
    user_message = (
        f"Context:\n\n{format_context(results)}\n\n"
        f"Question: {question}\n"
        "Answer with citations:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def generate_answer(question: str, results, model: str = DEFAULT_LLM, client=None) -> str:
    """Ask the local LLM to answer the question from the retrieved chunks."""
    client = client or ollama.Client()
    try:
        response = client.chat(
            model=model,
            messages=build_messages(question, results),
            # temperature 0 = always pick the most likely next word, so the
            # same question gives (nearly) the same answer every time.
            options={"temperature": 0},
        )
    except ConnectionError as e:
        raise RuntimeError("Can't reach Ollama. Open the Ollama app and try again.") from e
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise RuntimeError(f"Model '{model}' isn't downloaded. Run: ollama pull {model}") from e
        raise

    return response.message.content.strip()
