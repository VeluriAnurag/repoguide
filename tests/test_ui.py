from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.ui import highlight_citations, repo_display_name

UI_SCRIPT = str(Path(__file__).parent.parent / "app" / "ui.py")
SAMPLE_REPO = str(Path(__file__).parent / "fixtures" / "sample_repo")


def load_app() -> AppTest:
    at = AppTest.from_file(UI_SCRIPT, default_timeout=60)
    return at.run()


def test_page_loads_with_welcome_steps():
    at = load_app()

    assert not at.exception
    assert at.title[0].value == "🔎 RepoGuide"
    assert any("Paste a repo" in m.value for m in at.markdown)
    assert len(at.text_input) == 1  # only the repo box
    assert len(at.chat_input) == 0  # no chat until a repo is indexed


def test_empty_box_asks_for_a_link():
    at = load_app()

    at.sidebar.button[0].click().run()

    assert not at.exception
    assert at.sidebar.error[0].value == "Paste a GitHub link or a folder path first."


def test_invalid_link_shows_error_instead_of_crashing():
    at = load_app()

    at.sidebar.text_input[0].set_value("https://gitlab.com/psf/requests")
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert "Not a GitHub repository URL" in at.sidebar.error[0].value


def test_missing_folder_shows_error_instead_of_crashing():
    at = load_app()

    at.sidebar.text_input[0].set_value("/definitely/not/a/real/folder")
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert at.sidebar.error[0].value.startswith("That folder doesn't exist")


@pytest.mark.slow
def test_indexing_sample_repo_shows_stats_and_chat():
    at = load_app()

    at.sidebar.text_input[0].set_value(SAMPLE_REPO)
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert [m.value for m in at.sidebar.metric] == ["5", "12", at.sidebar.metric[2].value]
    assert len(at.chat_input) == 1


def test_highlight_citations_turns_citations_into_code():
    answer = "Opened in [bookstore/db.py:10-15]. See [the docs] and `x[1]`."

    assert highlight_citations(answer) == (
        "Opened in `bookstore/db.py:10-15`. See [the docs] and `x[1]`."
    )


def test_repo_display_name():
    assert repo_display_name("https://github.com/psf/requests?tab=readme") == "psf/requests"
    assert repo_display_name("/Users/me/code/myproject") == "myproject"


def test_author_credit_and_about_dialog():
    at = load_app()

    assert any("Anurag Veluri" in m.value for m in at.markdown)

    [about] = [b for b in at.button if b.label == "About"]
    about.click().run()

    assert not at.exception
    assert any("How it works" in m.value for m in at.markdown)
