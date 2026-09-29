import subprocess

import pytest

from app import github
from app.github import clone_github_repo, parse_github_url, resolve_repository


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/psf/requests",
        "https://github.com/psf/requests.git",
        "https://github.com/psf/requests/",
        "https://github.com/psf/requests/tree/main/src",
        "https://www.github.com/psf/requests",
        "github.com/psf/requests",
        "  https://github.com/psf/requests  ",
        "https://github.com/psf/requests?tab=readme-ov-file",
        "https://github.com/psf/requests#readme",
        "https://github.com/psf/requests/blob/main/README.md",
        "git@github.com:psf/requests.git",
    ],
)
def test_parses_common_github_url_forms(url):
    assert parse_github_url(url) == ("psf", "requests")


def test_keeps_dots_and_dashes_in_repo_name():
    assert parse_github_url("https://github.com/my-org/my.repo-2") == ("my-org", "my.repo-2")


@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.com/psf/requests",
        "https://github.com/psf",
        "https://evil.com/github.com/psf/requests",
        "https://github.com/--upload-pack=touch/x",
        "https://github.com/psf/requests;rm -rf ~",
        "http://github.com/psf/requests",
    ],
)
def test_rejects_anything_that_is_not_a_plain_github_repo_url(url):
    with pytest.raises(ValueError):
        parse_github_url(url)


def test_reuses_existing_download_without_running_git(tmp_path, monkeypatch):
    (tmp_path / "psf__requests" / ".git").mkdir(parents=True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("git should not run"))

    assert clone_github_repo("https://github.com/psf/requests", tmp_path) == tmp_path / "psf__requests"


def test_failed_download_cleans_up_and_explains(tmp_path, monkeypatch):
    def fake_git(command, **kwargs):
        (tmp_path / "psf__nope").mkdir()  # git leaves a partial folder behind
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(subprocess, "run", fake_git)

    with pytest.raises(RuntimeError, match="public"):
        clone_github_repo("https://github.com/psf/nope", tmp_path)
    assert not (tmp_path / "psf__nope").exists()


def test_git_command_never_prompts_and_ends_options_before_url(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))

    clone_github_repo("https://github.com/psf/requests", tmp_path)

    [(command, kwargs)] = calls
    assert command[command.index("--") + 1] == "https://github.com/psf/requests.git"
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_resolve_repository_accepts_local_folder(tmp_path):
    assert resolve_repository(str(tmp_path)) == tmp_path


def test_resolve_repository_rejects_missing_folder(tmp_path):
    with pytest.raises(ValueError, match="doesn't exist"):
        resolve_repository(str(tmp_path / "missing"))


@pytest.mark.slow
def test_real_download_of_a_tiny_public_repo(tmp_path):
    try:
        path = clone_github_repo("https://github.com/octocat/Hello-World", tmp_path)
    except RuntimeError:
        pytest.skip("no internet connection")
    assert (path / "README").is_file()


def test_resolve_repository_explains_non_github_links():
    with pytest.raises(ValueError, match="Only public GitHub repos"):
        resolve_repository("https://gitlab.com/psf/requests")


def test_public_mode_refuses_local_paths(tmp_path):
    # On the public website, a folder path would let visitors read server files.
    with pytest.raises(ValueError, match="public GitHub repository link"):
        resolve_repository(str(tmp_path), allow_local=False)


def test_canonical_source_is_the_same_for_every_url_form():
    forms = [
        "https://github.com/psf/requests",
        "https://github.com/psf/requests.git",
        "https://github.com/psf/requests?tab=readme-ov-file",
        "git@github.com:psf/requests.git",
    ]
    assert {github.canonical_source(f) for f in forms} == {"https://github.com/psf/requests"}
    assert github.canonical_source("/Users/me/project") == "/Users/me/project"


def test_remove_old_repos_keeps_most_recently_used(tmp_path):
    import os

    for i, name in enumerate(["old", "middle", "new"]):
        (tmp_path / name).mkdir()
        os.utime(tmp_path / name, (1000 + i, 1000 + i))

    github.remove_old_repos(tmp_path, keep=2)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["middle", "new"]
