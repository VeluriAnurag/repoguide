"""Answer generation with a local Llama 3.2 model served by Ollama.

The model sees the retrieved chunks as numbered sources and cites them by
number, like [1] or [2][3]. Our code then swaps each number for the chunk's
real location (e.g. [bookstore/db.py:10-15]), so the model never has to copy
file paths or line numbers itself. citations.py checks the final answer.

Why numbers: in a test on 8 questions about the psf/requests repo, asking the
model to copy labels like [docs/user/advanced.rst:1051-1100] gave 10 valid and
5 broken citations; numbered sources gave 20 valid and 0 broken.
"""

import os
import re

import ollama

# The Ollama model to use; can be changed with the REPOGUIDE_LLM variable
# (e.g. a smaller "llama3.2:1b" on a slow server).
DEFAULT_LLM = os.environ.get("REPOGUIDE_LLM", "llama3.2")

# Upper limit on answer length, so answers stay short and fast on slow CPUs.
MAX_ANSWER_TOKENS = 512

# What the model is told to say when the context doesn't answer the question.
NOT_FOUND_MESSAGE = "I couldn't find that in the retrieved code."

SYSTEM_PROMPT = f"""You are RepoGuide, an assistant that explains a Python codebase.

Rules:
1. Answer using ONLY the numbered sources in the context. Do not use outside knowledge about this repository.
2. After every claim, cite the source number in square brackets, like [1] or [2][3].
3. Only use source numbers that appear in the context.
4. If the sources do not contain the answer, say "{NOT_FOUND_MESSAGE}" and stop.
5. Be concise: a few sentences, not an essay."""

# A source reference like [2] or [1, 3], unless it's attached to a name or
# closing bracket/parenthesis, as in code such as args[1] or f(x)[0].
SOURCE_NUMBERS = re.compile(r"(?<![\w)])\[(\d+(?:\s*,\s*\d+)*)\]")

# Code in backticks (```blocks``` or `inline`) is left untouched.
CODE_SPANS = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)


def is_not_found(answer: str) -> bool:
    """True if the model said the retrieved code doesn't answer the question."""
    return NOT_FOUND_MESSAGE.lower().rstrip(".") in answer.lower()


def format_context(results) -> str:
    """Turn search results into numbered sources the model can cite."""
    blocks = [
        f"Source [{i}] {r.chunk.citation}\n{r.chunk.text}"
        for i, r in enumerate(results, start=1)
    ]
    return "\n\n".join(blocks)


def build_messages(question: str, results) -> list[dict]:
    user_message = (
        f"Context:\n\n{format_context(results)}\n\n"
        f"Question: {question}\n"
        "Answer with source numbers:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def replace_source_numbers(answer: str, results) -> str:
    """Swap [1], [2]... for the real file and line citations.

    Numbers that don't match a source (e.g. [7] with only 5 sources) are left
    as they are so the citation check can flag them.
    """

    def to_citations(match: re.Match) -> str:
        numbers = [int(n) for n in re.findall(r"\d+", match.group(1))]
        if not all(1 <= n <= len(results) for n in numbers):
            return match.group(0)
        return "".join(f"[{results[n - 1].chunk.citation}]" for n in numbers)

    # re.split with a capturing group keeps the code spans at odd positions.
    parts = CODE_SPANS.split(answer)
    for i in range(0, len(parts), 2):
        parts[i] = SOURCE_NUMBERS.sub(to_citations, parts[i])
    return "".join(parts)


def generate_answer(question: str, results, model: str = DEFAULT_LLM, client=None) -> str:
    """Ask the local LLM to answer the question from the retrieved chunks."""
    client = client or ollama.Client()
    try:
        response = client.chat(
            model=model,
            messages=build_messages(question, results),
            # temperature 0 = always pick the most likely next word, so the
            # same question gives (nearly) the same answer every time.
            options={"temperature": 0, "num_predict": MAX_ANSWER_TOKENS},
        )
    except ConnectionError as e:
        raise RuntimeError("Can't reach Ollama. Open the Ollama app and try again.") from e
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise RuntimeError(f"Model '{model}' isn't downloaded. Run: ollama pull {model}") from e
        raise

    return replace_source_numbers(response.message.content.strip(), results)
