"""Split repository files into chunks for embedding and retrieval.

A chunk is a contiguous range of lines from one file plus metadata describing
it. The metadata travels with the chunk through embedding, retrieval, and
generation so answers can cite ``file:start-end``.

Invariant (enforced by tests): ``chunk.text`` is exactly lines
``start_line..end_line`` (1-based, inclusive) of the original file.

Chunk types:
    function / class / method  - Python definitions found via the AST
    module                     - top-level Python code outside definitions
    code                       - Python that failed to parse (line windows)
    doc                        - Markdown sections, .txt/.rst windows
"""

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from app.ingest import discover_files, read_text_file

# Upper bound on chunk size. BGE embedding models read at most 512 tokens, so
# anything much longer would be silently truncated. ~50 lines of code is
# roughly 400-500 tokens. We will revisit this with real token counts later.
MAX_CHUNK_LINES = 50

DEFINITION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.*)")


@dataclass(frozen=True)
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    symbol: str | None
    chunk_type: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_repository(repo_path: str | Path) -> list[Chunk]:
    """Discover, read, and chunk every indexable file in a repository."""
    root = Path(repo_path).resolve()
    chunks = []
    for rel_path in discover_files(root):
        text = read_text_file(root / rel_path)
        if text is None:
            continue
        chunks.extend(chunk_file(rel_path.as_posix(), text))
    return chunks


def chunk_file(file_path: str, text: str) -> list[Chunk]:
    """Pick a chunking strategy based on the file extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".py":
        return chunk_python_file(file_path, text)
    if suffix == ".md":
        return chunk_markdown_file(file_path, text)
    return chunk_text_file(file_path, text)


def chunk_python_file(file_path: str, source: str) -> list[Chunk]:
    """Chunk Python source around top-level functions and classes.

    Each top-level function/class becomes its own chunk (decorators included).
    Code between definitions (imports, constants, ``if __name__ == ...``)
    becomes ``module`` chunks. Classes that are too large are split into
    per-method chunks. If the file cannot be parsed, fall back to line windows.
    """
    lines = _split_lines(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _window_chunks(file_path, lines, 1, len(lines), None, "code")

    return _chunk_region(file_path, lines, tree.body, 1, len(lines), parent=None)


def chunk_markdown_file(file_path: str, text: str) -> list[Chunk]:
    """Chunk Markdown into sections that each start at a heading."""
    lines = _split_lines(text)
    sections = []  # (start_line, heading_text)
    in_code_fence = False

    for line_no, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence
        match = MARKDOWN_HEADING.match(line)
        if match and not in_code_fence:
            sections.append((line_no, match.group(1).strip()))

    # Text before the first heading is its own section with no title.
    if not sections or sections[0][0] != 1:
        sections.insert(0, (1, None))

    chunks = []
    for i, (start, heading) in enumerate(sections):
        end = sections[i + 1][0] - 1 if i + 1 < len(sections) else len(lines)
        chunks.extend(_window_chunks(file_path, lines, start, end, heading, "doc"))
    return chunks


def chunk_text_file(file_path: str, text: str) -> list[Chunk]:
    """Chunk plain text (.txt, .rst) into fixed-size line windows."""
    lines = _split_lines(text)
    return _window_chunks(file_path, lines, 1, len(lines), None, "doc")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_lines(text: str) -> list[str]:
    """Split text into lines the same way the Python parser counts them.

    ``str.splitlines()`` also breaks on characters like form feed (\\x0c),
    which the parser does not treat as newlines. Using it would shift every
    line number after such a character and break citations.
    """
    return re.split(r"\r\n|\r|\n", text)


def _chunk_region(
    file_path: str,
    lines: list[str],
    body: list[ast.stmt],
    region_start: int,
    region_end: int,
    parent: str | None,
) -> list[Chunk]:
    """Chunk lines region_start..region_end, where ``body`` holds its statements.

    Definitions get their own chunks; everything between them is grouped into
    "gap" chunks. With ``parent=None`` this handles a whole module; with a
    class name it handles the inside of a large class.
    """
    gap_type = "class" if parent else "module"
    chunks = []
    cursor = region_start

    for node in body:
        if not isinstance(node, DEFINITION_TYPES):
            continue
        start, end = _definition_span(node)
        chunks.extend(_window_chunks(file_path, lines, cursor, start - 1, parent, gap_type))
        chunks.extend(_definition_chunks(file_path, lines, node, parent))
        cursor = end + 1

    chunks.extend(_window_chunks(file_path, lines, cursor, region_end, parent, gap_type))
    return chunks


def _definition_chunks(
    file_path: str, lines: list[str], node: ast.AST, parent: str | None
) -> list[Chunk]:
    """Chunk one function or class definition."""
    start, end = _definition_span(node)
    symbol = f"{parent}.{node.name}" if parent else node.name

    if isinstance(node, ast.ClassDef):
        if end - start + 1 > MAX_CHUNK_LINES:
            # Too big for one chunk: the class header/attributes become
            # "class" chunks and each method gets its own chunk.
            return _chunk_region(file_path, lines, node.body, start, end, parent=symbol)
        chunk_type = "class"
    else:
        chunk_type = "method" if parent else "function"

    return _window_chunks(file_path, lines, start, end, symbol, chunk_type)


def _definition_span(node: ast.AST) -> tuple[int, int]:
    """Return (first_line, last_line) of a definition, including decorators."""
    decorator_lines = [d.lineno for d in node.decorator_list]
    return min([node.lineno, *decorator_lines]), node.end_lineno


def _window_chunks(
    file_path: str,
    lines: list[str],
    start: int,
    end: int,
    symbol: str | None,
    chunk_type: str,
) -> list[Chunk]:
    """Make chunks for lines start..end, splitting into MAX_CHUNK_LINES windows.

    Leading/trailing blank lines are trimmed, and all-blank ranges produce no
    chunks. Split pieces keep the same symbol and type as the original unit.
    """
    while start <= end and not lines[start - 1].strip():
        start += 1
    while end >= start and not lines[end - 1].strip():
        end -= 1

    chunks = []
    for window_start in range(start, end + 1, MAX_CHUNK_LINES):
        window_end = min(window_start + MAX_CHUNK_LINES - 1, end)
        chunks.append(
            Chunk(
                file_path=file_path,
                start_line=window_start,
                end_line=window_end,
                symbol=symbol,
                chunk_type=chunk_type,
                text="\n".join(lines[window_start - 1 : window_end]),
            )
        )
    return chunks


if __name__ == "__main__":
    # Debug tool: python -m app.chunker path/to/repo
    if len(sys.argv) != 2:
        sys.exit("usage: python -m app.chunker <repo_path>")

    all_chunks = chunk_repository(sys.argv[1])
    for c in all_chunks:
        print(f"{c.citation:<45} {c.chunk_type:<9} {c.symbol or ''}")
    print(f"\n{len(all_chunks)} chunks")
