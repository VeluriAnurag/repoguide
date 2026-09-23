"""Check that the citations in an answer point to chunks the LLM was given.

This is plain rule-based code, no AI: the same answer always gets the same
result. A citation like [bookstore/db.py:12-13] is valid if some retrieved
chunk is from that file and its line range covers lines 12-13.
"""

import re
from dataclasses import dataclass

from app.chunker import Chunk

# Finds text inside square brackets, e.g. "[a.py:1-2, b.py:5]".
BRACKETS = re.compile(r"\[([^\[\]]+)\]")

# One citation inside the brackets: "path:start-end" or "path:line".
# Requiring ":<number>" means things like list[int] are ignored.
CITATION = re.compile(r"([^\s,;:]+):(\d+)(?:-(\d+))?")


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

    @property
    def ok(self) -> bool:
        """True when there is at least one citation and none are invalid."""
        return bool(self.valid) and not self.invalid


def extract_citations(answer: str) -> list[Citation]:
    """Find every citation in an answer, in order, without duplicates."""
    found = []
    for bracket_text in BRACKETS.findall(answer):
        for path, start, end in CITATION.findall(bracket_text):
            citation = Citation(path, int(start), int(end or start))
            if citation not in found:
                found.append(citation)
    return found


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
    return CitationReport(valid=valid, invalid=invalid)
