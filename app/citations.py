"""Check that the citations in an answer point to chunks the LLM was given.

This is plain rule-based code, no AI: the same answer always gets the same
result. A citation like [bookstore/db.py:12-13] is valid if some retrieved
chunk is from that file and its line range covers lines 12-13.

It also flags "malformed" citations: brackets that look like the model tried
to cite something but didn't produce a real location, such as a source number
that doesn't exist ([7] with 5 sources) or [Retry:START-END].
"""

import re
from dataclasses import dataclass

from app.chunker import Chunk
from app.generator import CODE_SPANS, is_not_found

# Finds text inside square brackets, e.g. "[a.py:1-2, b.py:5]".
BRACKETS = re.compile(r"\[([^\[\]]+)\]")

# One citation inside the brackets: "path:start-end" or "path:line".
# Requiring ":<number>" means things like list[int] are ignored.
CITATION = re.compile(r"([^\s,;:]+):(\d+)(?:-(\d+))?")

# A bracket standing on its own (not code like args[1]) and not a Markdown
# link like [text](url). Used to spot malformed citation attempts.
STANDALONE_BRACKET = re.compile(r"(?<![\w)])\[([^\[\]]+)\](?!\()")
LEFTOVER_SOURCE_NUMBER = re.compile(r"^\d+(?:\s*,\s*\d+)*$")


@dataclass(frozen=True)
class Citation:
    file_path: str
    start_line: int
    end_line: int

    def __str__(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class CitationReport:
    valid: list[Citation]
    invalid: list[Citation]
    malformed: list[str]

    @property
    def ok(self) -> bool:
        """True when there is at least one citation and nothing is wrong."""
        return bool(self.valid) and not self.invalid and not self.malformed


def extract_citations(answer: str) -> list[Citation]:
    """Find every citation in an answer, in order, without duplicates."""
    found = []
    for bracket_text in BRACKETS.findall(answer):
        for path, start, end in CITATION.findall(bracket_text):
            citation = Citation(path, int(start), int(end or start))
            if citation not in found:
                found.append(citation)
    return found


def find_malformed(answer: str) -> list[str]:
    """Bracketed text that looks like a failed citation, e.g. "[7]"."""
    prose = CODE_SPANS.sub(" ", answer)  # ignore code in backticks
    malformed = []
    for text in STANDALONE_BRACKET.findall(prose):
        text = text.strip()
        looks_like_citation = LEFTOVER_SOURCE_NUMBER.match(text) or (
            ":" in text and not text.startswith("http") and not CITATION.search(text)
        )
        if looks_like_citation and f"[{text}]" not in malformed:
            malformed.append(f"[{text}]")
    return malformed


def is_supported(citation: Citation, chunks: list[Chunk]) -> bool:
    """True if a retrieved chunk from the same file covers the cited lines."""
    if citation.start_line > citation.end_line:
        return False
    return any(
        chunk.file_path == citation.file_path
        and chunk.start_line <= citation.start_line
        and citation.end_line <= chunk.end_line
        for chunk in chunks
    )


def validate_citations(answer: str, retrieved_chunks: list[Chunk]) -> CitationReport:
    """Sort the answer's citations into valid and invalid ones."""
    valid, invalid = [], []
    for citation in extract_citations(answer):
        if is_supported(citation, retrieved_chunks):
            valid.append(citation)
        else:
            invalid.append(citation)
    return CitationReport(valid=valid, invalid=invalid, malformed=find_malformed(answer))


def citation_summary(answer: str, retrieved_chunks: list[Chunk]) -> tuple[str, str]:
    """Tell the user whether to trust the citations.

    Returns (level, message) where level is "ok", "warning", or "info", so
    the terminal and the web UI can each display it their own way.
    """
    report = validate_citations(answer, retrieved_chunks)
    if report.invalid or report.malformed:
        bad = ", ".join([str(c) for c in report.invalid] + report.malformed)
        return "warning", f"These citations don't match any retrieved code: {bad}"
    if report.valid:
        n = len(report.valid)
        return "ok", f"Citations checked: {n} of {n} match the retrieved code."
    if is_not_found(answer):
        return "info", "No citations (the answer wasn't in the retrieved code)."
    return "warning", "This answer has no citations, so it can't be checked."
