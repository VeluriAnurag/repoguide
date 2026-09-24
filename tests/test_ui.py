from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

UI_SCRIPT = str(Path(__file__).parent.parent / "app" / "ui.py")
SAMPLE_REPO = str(Path(__file__).parent / "fixtures" / "sample_repo")


def load_app() -> AppTest:
    at = AppTest.from_file(UI_SCRIPT, default_timeout=60)
    return at.run()


def test_page_loads_and_asks_for_a_repository_first():
    at = load_app()

    assert not at.exception
    assert at.title[0].value == "RepoGuide"
    assert "Index a repository" in at.sidebar.info[0].value
    assert len(at.text_input) == 1  # only the repo path; no question box yet


def test_missing_folder_shows_error_instead_of_crashing():
    at = load_app()

    at.sidebar.text_input[0].set_value("/definitely/not/a/real/folder")
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert at.sidebar.error[0].value.startswith("That folder doesn't exist")


@pytest.mark.slow
def test_indexing_sample_repo_shows_stats_and_question_box():
    at = load_app()

    at.sidebar.text_input[0].set_value(SAMPLE_REPO)
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert at.sidebar.success[0].value.startswith("12 chunks from 5 files")
    assert len(at.text_input) == 2  # repo path + question box
