import re
from pathlib import Path

from app import chunker
from app.chunker import (
    chunk_markdown_file,
    chunk_python_file,
    chunk_repository,
    chunk_text_file,
)

SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def lines_of(text):
    return re.split(r"\r\n|\r|\n", text)


def assert_text_matches_lines(chunks, source):
    """The core invariant: a chunk's text is exactly its cited lines."""
    lines = lines_of(source)
    for c in chunks:
        assert c.text == "\n".join(lines[c.start_line - 1 : c.end_line]), c.citation


def by_symbol(chunks, symbol):
    return [c for c in chunks if c.symbol == symbol]


# --- Python chunking --------------------------------------------------------


def test_top_level_function_and_class_chunks():
    source = '''def hello(name):
    return f"Hello {name}"


class Example:
    def run(self):
        return 1
'''
    chunks = chunk_python_file("example.py", source)

    assert [(c.symbol, c.chunk_type, c.start_line, c.end_line) for c in chunks] == [
        ("hello", "function", 1, 2),
        ("Example", "class", 5, 7),
    ]
    assert_text_matches_lines(chunks, source)


def test_small_class_methods_are_not_duplicated():
    source = "class A:\n    def m(self):\n        pass\n"
    chunks = chunk_python_file("a.py", source)

    assert [c.symbol for c in chunks] == ["A"]


def test_decorators_are_included_in_definition_chunk():
    source = "import functools\n\n@functools.cache\n@other\ndef f():\n    return 1\n"
    [module, func] = chunk_python_file("f.py", source)

    assert module.chunk_type == "module"
    assert (func.symbol, func.start_line, func.end_line) == ("f", 3, 6)
    assert func.text.startswith("@functools.cache")


def test_module_level_code_is_kept():
    source = (
        "import os\n"
        'DB_URL = "sqlite://"\n'
        "\n"
        "def f():\n"
        "    pass\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    f()\n"
    )
    chunks = chunk_python_file("m.py", source)

    assert [(c.chunk_type, c.start_line, c.end_line) for c in chunks] == [
        ("module", 1, 2),
        ("function", 4, 5),
        ("module", 7, 8),
    ]
    assert_text_matches_lines(chunks, source)


def test_large_class_is_split_into_methods(monkeypatch):
    monkeypatch.setattr(chunker, "MAX_CHUNK_LINES", 5)
    source = (
        "class Big:\n"
        '    """Docstring."""\n'
        "    LIMIT = 3\n"
        "\n"
        "    def a(self):\n"
        "        return 1\n"
        "\n"
        "    @property\n"
        "    def b(self):\n"
        "        return 2\n"
    )
    chunks = chunk_python_file("big.py", source)

    assert [(c.symbol, c.chunk_type, c.start_line, c.end_line) for c in chunks] == [
        ("Big", "class", 1, 3),
        ("Big.a", "method", 5, 6),
        ("Big.b", "method", 8, 10),
    ]
    assert_text_matches_lines(chunks, source)


def test_oversized_function_is_split_with_metadata(monkeypatch):
    monkeypatch.setattr(chunker, "MAX_CHUNK_LINES", 4)
    body = "".join(f"    x{i} = {i}\n" for i in range(9))
    source = "def long():\n" + body
    chunks = chunk_python_file("long.py", source)

    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 4), (5, 8), (9, 10)]
    assert all(c.symbol == "long" and c.chunk_type == "function" for c in chunks)
    assert_text_matches_lines(chunks, source)


def test_syntax_error_falls_back_to_line_windows():
    source = "def broken(:\n    pass\n"
    [chunk] = chunk_python_file("broken.py", source)

    assert (chunk.chunk_type, chunk.start_line, chunk.end_line) == ("code", 1, 2)


def test_form_feed_does_not_shift_line_numbers():
    # str.splitlines() would treat \x0c as a line break; the parser does not.
    source = "x = 1\n\x0c\ndef f():\n    pass\n"
    [module, func] = chunk_python_file("ff.py", source)

    assert (func.start_line, func.end_line) == (3, 4)
    assert_text_matches_lines([module, func], source)


def test_empty_file_produces_no_chunks():
    assert chunk_python_file("empty.py", "") == []
    assert chunk_python_file("blank.py", "\n\n   \n") == []


# --- Documentation chunking --------------------------------------------------


def test_markdown_splits_on_headings_but_not_inside_code_fences():
    source = (SAMPLE_REPO / "README.md").read_text()
    chunks = chunk_markdown_file("README.md", source)

    assert [c.symbol for c in chunks] == ["Bookstore", "Setup", "Architecture"]
    assert all(c.chunk_type == "doc" for c in chunks)
    assert_text_matches_lines(chunks, source)


def test_markdown_text_before_first_heading():
    chunks = chunk_markdown_file("x.md", "intro line\n\n# Title\nbody\n")

    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == [
        (None, 1, 1),
        ("Title", 3, 4),
    ]


def test_text_file_uses_line_windows(monkeypatch):
    monkeypatch.setattr(chunker, "MAX_CHUNK_LINES", 3)
    source = "a\nb\nc\nd\n"
    chunks = chunk_text_file("notes.txt", source)

    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 3), (4, 4)]


# --- Whole repository --------------------------------------------------------


def test_chunk_repository_on_sample_repo():
    chunks = chunk_repository(SAMPLE_REPO)
    citations = {c.citation: c for c in chunks}

    assert "bookstore/db.py:10-15" in citations
    assert citations["bookstore/db.py:10-15"].symbol == "get_connection"
    assert any(c.symbol == "authenticate_user" for c in chunks)
    assert {c.file_path for c in chunks} == {
        "README.md",
        "requirements.txt",
        "bookstore/__init__.py",
        "bookstore/auth.py",
        "bookstore/db.py",
    }


def test_every_python_line_is_in_exactly_one_chunk():
    # No code is dropped and no code is indexed twice.
    for path in SAMPLE_REPO.rglob("*.py"):
        source = path.read_text()
        chunks = chunk_python_file(path.name, source)
        assert_text_matches_lines(chunks, source)

        covered = [n for c in chunks for n in range(c.start_line, c.end_line + 1)]
        assert len(covered) == len(set(covered)), f"overlapping chunks in {path}"

        non_blank = {i for i, line in enumerate(lines_of(source), 1) if line.strip()}
        assert non_blank <= set(covered), f"dropped lines in {path}"


def test_indexing_is_deterministic():
    assert chunk_repository(SAMPLE_REPO) == chunk_repository(SAMPLE_REPO)
