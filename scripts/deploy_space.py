"""Upload RepoGuide to a Hugging Face Space (the public website).

Usage:
    hf auth login                                   # once, with your own token
    python scripts/deploy_space.py <your-hf-username>/repoguide

Hugging Face builds the Dockerfile and serves the app at
https://huggingface.co/spaces/<your-hf-username>/repoguide
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).parent.parent

# Settings Hugging Face reads from the top of the Space's README. They're only
# added to the uploaded copy, so the GitHub README stays clean.
SPACE_SETTINGS = """---
title: RepoGuide
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
short_description: Ask questions about any Python repo, with cited answers
---

"""

SKIP = [
    ".venv/*", ".git/*", ".claude/*", "data/repos/*",
    "**/__pycache__/*", ".pytest_cache/*", "**/.DS_Store", "README.md",
]


def main() -> None:
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python scripts/deploy_space.py <your-hf-username>/<space-name>")
    space_id = sys.argv[1]

    api = HfApi()
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        folder_path=ROOT, repo_id=space_id, repo_type="space",
        ignore_patterns=SKIP, commit_message="Deploy RepoGuide",
    )
    readme = SPACE_SETTINGS + (ROOT / "README.md").read_text()
    api.upload_file(
        path_or_fileobj=readme.encode(), path_in_repo="README.md",
        repo_id=space_id, repo_type="space", commit_message="Update README",
    )
    print(f"Uploaded. Hugging Face is building it now (about 10-15 minutes):")
    print(f"https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
