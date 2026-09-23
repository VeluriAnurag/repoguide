from pathlib import Path

import pytest

from app import ingest
from app.ingest import discover_files, read_text_file


def test_discovers_supported_files_and_skips_ignored_dirs(tmp_path):
    (tmp_path / "main.py").write_text("print('hi')")
    (tmp_path / "README.md").write_text("# Test")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "bad.py").write_text("print('ignored')")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "notes.md").write_text("ignored")

    files = discover_files(tmp_path)

    assert files == [Path("README.md"), Path("main.py")]


def test_returns_repo_relative_paths(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "auth.py").write_text("x = 1")

    assert discover_files(tmp_path) == [Path("src/pkg/auth.py")]


def test_repo_inside_a_folder_named_like_an_ignored_dir(tmp_path):
    # Regression: the starter code checked the absolute path, so a repo stored
    # under e.g. ~/venv/myrepo indexed zero files.
    repo = tmp_path / "venv" / "myrepo"
    repo.mkdir(parents=True)
    (repo / "main.py").write_text("x = 1")

    assert discover_files(repo) == [Path("main.py")]


def test_skips_files_over_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "MAX_FILE_BYTES", 10)
    (tmp_path / "small.py").write_text("x = 1")
    (tmp_path / "huge.py").write_text("x = 1\n" * 100)

    assert discover_files(tmp_path) == [Path("small.py")]


def test_missing_directory_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        discover_files(tmp_path / "does-not-exist")


def test_read_text_file_rejects_non_utf8(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("hello", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00binary")

    assert read_text_file(good) == "hello"
    assert read_text_file(bad) is None
