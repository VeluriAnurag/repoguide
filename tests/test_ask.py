from app.ask import citation_summary
from app.chunker import Chunk
from app.generator import NOT_FOUND_MESSAGE, is_not_found

CHUNKS = [Chunk("bookstore/db.py", 10, 15, "get_connection", "function", "...")]


def test_summary_when_all_citations_are_valid():
    summary = citation_summary("Opened here [bookstore/db.py:10-15].", CHUNKS)

    assert summary == "Citations checked: 1 of 1 match the retrieved code."


def test_summary_names_invalid_citations():
    summary = citation_summary("Opened here [src/db.py:10-15].", CHUNKS)

    assert summary.startswith("WARNING")
    assert "src/db.py:10-15" in summary


def test_summary_when_answer_was_not_found():
    summary = citation_summary(NOT_FOUND_MESSAGE, CHUNKS)

    assert summary == "No citations (the answer wasn't in the retrieved code)."


def test_summary_warns_when_real_answer_has_no_citations():
    summary = citation_summary("The connection is opened in get_connection.", CHUNKS)

    assert summary == "WARNING: this answer has no citations, so it can't be checked."


def test_is_not_found_ignores_case_and_final_period():
    assert is_not_found("i couldn't find that in the retrieved code")
    assert not is_not_found("The connection is opened in get_connection.")
