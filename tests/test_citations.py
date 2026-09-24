from app.chunker import Chunk
from app.citations import Citation, citation_summary, extract_citations, validate_citations
from app.generator import NOT_FOUND_MESSAGE

CHUNKS = [
    Chunk("bookstore/db.py", 10, 15, "get_connection", "function", "..."),
    Chunk("bookstore/auth.py", 13, 18, "authenticate_user", "function", "..."),
    Chunk("README.md", 5, 12, "Setup", "doc", "..."),
]


# --- Finding citations -------------------------------------------------------


def test_extracts_citation_with_line_range():
    assert extract_citations("Opened here [bookstore/db.py:10-15].") == [
        Citation("bookstore/db.py", 10, 15)
    ]


def test_extracts_single_line_citation():
    assert extract_citations("See [bookstore/db.py:12]") == [Citation("bookstore/db.py", 12, 12)]


def test_extracts_several_citations_in_one_bracket():
    answer = "Both [bookstore/db.py:10-15, README.md:5-12] and [bookstore/auth.py:13-18; README.md:6]"

    assert extract_citations(answer) == [
        Citation("bookstore/db.py", 10, 15),
        Citation("README.md", 5, 12),
        Citation("bookstore/auth.py", 13, 18),
        Citation("README.md", 6, 6),
    ]


def test_path_with_spaces_is_read_whole():
    chunk = Chunk("docs/User Guide.md", 1, 5, None, "doc", "...")

    report = validate_citations("See [docs/User Guide.md:1-5, README.md:5-12].", [chunk, *CHUNKS])

    assert [str(c) for c in report.valid] == ["docs/User Guide.md:1-5", "README.md:5-12"]
    assert report.ok


def test_ignores_brackets_that_are_not_citations():
    answer = "It returns `list[int]`, see [the docs], and uses users[username]."

    assert extract_citations(answer) == []


def test_repeated_citation_is_listed_once():
    answer = "A [bookstore/db.py:10-15]. B [bookstore/db.py:10-15]."

    assert extract_citations(answer) == [Citation("bookstore/db.py", 10, 15)]


# --- Checking citations ------------------------------------------------------


def test_exact_match_is_valid():
    report = validate_citations("x [bookstore/db.py:10-15]", CHUNKS)

    assert report.valid == [Citation("bookstore/db.py", 10, 15)]
    assert report.invalid == []
    assert report.ok


def test_narrower_range_inside_chunk_is_valid():
    report = validate_citations("x [bookstore/db.py:12-13] y [bookstore/db.py:15]", CHUNKS)

    assert len(report.valid) == 2 and report.ok


def test_wrong_folder_is_invalid():
    # The real bug from Milestone 3: the model wrote src/ instead of bookstore/.
    report = validate_citations("x [src/db.py:10-15]", CHUNKS)

    assert report.invalid == [Citation("src/db.py", 10, 15)]
    assert not report.ok


def test_range_past_the_retrieved_chunk_is_invalid():
    report = validate_citations("x [bookstore/db.py:10-40]", CHUNKS)

    assert report.invalid == [Citation("bookstore/db.py", 10, 40)]


def test_range_spanning_two_chunks_is_invalid():
    # Lines 10-18 are never covered by a single retrieved chunk.
    chunks = CHUNKS + [Chunk("bookstore/db.py", 16, 18, None, "module", "...")]

    assert not validate_citations("x [bookstore/db.py:10-18]", chunks).ok


def test_backwards_range_is_invalid():
    report = validate_citations("x [bookstore/db.py:15-10]", CHUNKS)

    assert report.invalid == [Citation("bookstore/db.py", 15, 10)]


def test_mix_of_valid_and_invalid():
    report = validate_citations("a [README.md:5-12] b [README.md:40-50]", CHUNKS)

    assert report.valid == [Citation("README.md", 5, 12)]
    assert report.invalid == [Citation("README.md", 40, 50)]
    assert not report.ok


def test_answer_without_citations_is_not_ok():
    report = validate_citations("I couldn't find that in the retrieved code.", CHUNKS)

    assert report.valid == [] and report.invalid == []
    assert not report.ok


# --- Malformed citations -------------------------------------------------------


def test_placeholder_copied_by_the_model_is_malformed():
    # Real Llama output on the psf/requests repo.
    report = validate_citations("It uses Retry [urllib3.util.Retry:START-END].", CHUNKS)

    assert report.malformed == ["[urllib3.util.Retry:START-END]"]
    assert not report.ok


def test_source_number_that_does_not_exist_is_malformed():
    report = validate_citations("Opened here [bookstore/db.py:10-15] and [7].", CHUNKS)

    assert report.malformed == ["[7]"]
    assert not report.ok


def test_normal_brackets_code_and_links_are_not_malformed():
    answer = (
        "See [the docs](https://example.com) and [https://x.io]. "
        "`args[1]` and rows[0] and `{'a': 1}[x]` are code. Valid [bookstore/db.py:10-15]."
    )

    report = validate_citations(answer, CHUNKS)

    assert report.malformed == []
    assert report.ok


# --- Summary shown to the user ------------------------------------------------


def test_summary_when_all_citations_are_valid():
    assert citation_summary("Opened here [bookstore/db.py:10-15].", CHUNKS) == (
        "ok",
        "Citations checked: 1 of 1 match the retrieved code.",
    )


def test_summary_names_invalid_citations():
    level, message = citation_summary("Opened here [src/db.py:10-15].", CHUNKS)

    assert level == "warning"
    assert "src/db.py:10-15" in message


def test_summary_when_answer_was_not_found():
    assert citation_summary(NOT_FOUND_MESSAGE, CHUNKS) == (
        "info",
        "No citations (the answer wasn't in the retrieved code).",
    )


def test_summary_names_malformed_citations():
    level, message = citation_summary("Uses Retry [Retry:START-END].", CHUNKS)

    assert level == "warning"
    assert "[Retry:START-END]" in message


def test_summary_warns_when_real_answer_has_no_citations():
    assert citation_summary("The connection is opened in get_connection.", CHUNKS) == (
        "warning",
        "This answer has no citations, so it can't be checked.",
    )
