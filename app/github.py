"""Download public GitHub repositories so they can be indexed like a local folder.

RepoGuide only reads the downloaded files; it never runs code from them.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

CLONE_DIR = Path(__file__).parent.parent / "data" / "repos"
CLONE_TIMEOUT_SECONDS = 120

# Keep at most this many downloaded repos; the least recently used are
# deleted first. Matters on a public server where anyone can add repos.
MAX_STORED_REPOS = 10

# Accepts https://github.com/<owner>/<repo> (with or without https:// or www.)
# and git@github.com:<owner>/<repo>, optionally ending in .git, a slash, a
# ?query, #anchor, or extra parts like /tree/main, which are all ignored. The
# strict character sets mean nothing unusual can reach the git command.
GITHUB_URL = re.compile(
    r"^(?:(?:https://)?(?:www\.)?github\.com/|git@github\.com:)"
    r"([A-Za-z0-9-]+)/([A-Za-z0-9._-]+?)(?:\.git)?(?:[/?#].*)?$"
)


def parse_github_url(url: str) -> tuple[str, str]:
    """Return (owner, repo) from a GitHub URL, or raise ValueError."""
    match = GITHUB_URL.match(url.strip())
    if not match:
        raise ValueError(f"Not a GitHub repository URL: {url}")
    return match.group(1), match.group(2)


def clone_github_repo(url: str, clone_dir: Path = CLONE_DIR) -> Path:
    """Download a public repo (latest version only) and return its folder.

    If it was downloaded before, the existing copy is reused.
    """
    owner, repo = parse_github_url(url)
    destination = clone_dir / f"{owner}__{repo}"
    if (destination / ".git").is_dir():
        destination.touch()  # mark as recently used
        return destination

    clone_dir.mkdir(parents=True, exist_ok=True)
    remove_old_repos(clone_dir, keep=MAX_STORED_REPOS - 1)
    command = [
        "git", "clone",
        "--depth", "1",       # latest version only, no history
        "--single-branch",
        "--",                 # everything after this is a URL/path, never an option
        f"https://github.com/{owner}/{repo}.git",
        str(destination),
    ]
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",   # fail instead of asking for a password
        "GIT_LFS_SKIP_SMUDGE": "1",   # skip large binary files stored with Git LFS
    }

    try:
        subprocess.run(
            command, env=env, capture_output=True, text=True,
            timeout=CLONE_TIMEOUT_SECONDS, check=True,
        )
    except subprocess.TimeoutExpired as e:
        shutil.rmtree(destination, ignore_errors=True)
        raise RuntimeError(f"Downloading {owner}/{repo} took too long.") from e
    except subprocess.CalledProcessError as e:
        shutil.rmtree(destination, ignore_errors=True)
        raise RuntimeError(
            f"Couldn't download {owner}/{repo}. Check the URL and that the repository is public."
        ) from e

    return destination


def remove_old_repos(clone_dir: Path, keep: int) -> None:
    """Delete the least recently used downloads so only `keep` remain."""
    repos = sorted(
        (d for d in clone_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for old in repos[keep:]:
        shutil.rmtree(old, ignore_errors=True)


def canonical_source(source: str) -> str:
    """One standard form per repo, so every way of writing the same GitHub
    link maps to https://github.com/<owner>/<repo>. Local paths are returned
    unchanged. The result can still be passed to resolve_repository."""
    source = source.strip()
    if "github.com" in source:
        owner, repo = parse_github_url(source)
        return f"https://github.com/{owner}/{repo}"
    return source


def resolve_repository(source: str, allow_local: bool = True) -> Path:
    """Turn user input (a local folder or a GitHub URL) into a local folder.

    On the public website allow_local is False: letting strangers type a
    folder path would let them read the server's own files.
    """
    source = source.strip()
    if "github.com" in source:
        return clone_github_repo(source)
    if "://" in source or source.startswith("git@"):
        raise ValueError(f"Not a GitHub repository URL: {source}. Only public GitHub repos are supported.")
    if not allow_local:
        raise ValueError("Paste a public GitHub repository link (like https://github.com/psf/requests).")

    path = Path(source).expanduser()
    if not path.is_dir():
        raise ValueError(f"That folder doesn't exist: {source}")
    return path
