"""Repository file discovery.

Finds the files RepoGuide should index and reads them safely. Paths are always
returned relative to the repository root (e.g. ``src/auth.py``) because those
are the paths that appear in citations.
"""

import os
from pathlib import Path

# Directory names that never contain code worth indexing.
IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "build",
    "dist",
}

SUPPORTED_SUFFIXES = {".py", ".md", ".txt", ".rst"}

# Skip very large files (generated code, data dumps); they add noise, not insight.
MAX_FILE_BYTES = 1_000_000


def discover_files(repo_path: str | Path) -> list[Path]:
    """Return repo-relative paths of all indexable files, sorted for determinism."""
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {repo_path}")

    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Editing dirnames in place tells os.walk not to descend into ignored
        # directories at all, which matters for huge folders like node_modules.
        dirnames[:] = [
            d for d in dirnames if d not in IGNORED_DIRS and not d.endswith(".egg-info")
        ]

        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if path.is_symlink() or path.stat().st_size > MAX_FILE_BYTES:
                continue
            found.append(path.relative_to(root))

    return sorted(found)


def read_text_file(path: Path) -> str | None:
    """Read a file as UTF-8, returning None if it is not valid text."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
