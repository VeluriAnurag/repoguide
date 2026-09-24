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

# https://github.com/<owner>/<repo>, optionally with .git, a trailing slash,
# or extra parts like /tree/main (which are ignored). The strict character
# sets mean nothing unusual can reach the git command.
GITHUB_URL = re.compile(
    r"^(?:https://)?(?:www\.)?github\.com/([A-Za-z0-9-]+)/([A-Za-z0-9._-]+?)(?:\.git)?(?:/.*)?$"
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
        return destination

    clone_dir.mkdir(parents=True, exist_ok=True)
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


def resolve_repository(source: str) -> Path:
    """Turn user input (a local folder or a GitHub URL) into a local folder."""
    source = source.strip()
    if "github.com" in source:
        return clone_github_repo(source)

    path = Path(source).expanduser()
    if not path.is_dir():
        raise ValueError(f"That folder doesn't exist: {source}")
    return path
