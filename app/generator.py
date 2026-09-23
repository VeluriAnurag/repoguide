"""LLM generation layer."""


SYSTEM_PROMPT = """You are RepoGuide, a codebase explanation assistant.

Answer questions using only the repository context supplied to you.
Do not invent files, functions, behavior, or citations.
When making a claim about the repository, cite the supplied file path and line range.
If the context is insufficient, say that the available context is insufficient.
"""


def generate_answer(question: str, retrieved_chunks, model=None) -> str:
    # TODO: build a grounded prompt and call local Llama 3.2.
    raise NotImplementedError
