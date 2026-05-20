from unittest.mock import MagicMock, patch

from src.wiki_query import (
    _parse_json_array,
    format_answer_for_display,
    select_pages,
)


def test_format_answer_strips_wiki_citations():
    raw = (
        "He works at GitGuardian (wiki/sources/raw-cv-barranchina-pdf.md; "
        "wiki/sources/website-timeline-json.md)."
    )
    assert "wiki/" not in format_answer_for_display(raw)
    assert "GitGuardian" in format_answer_for_display(raw)


def test_format_answer_strips_wiki_meta_language():
    raw = (
        "## Short answer\n\nAccording to the wiki, Agustin works at GitGuardian.\n\n"
        "## Details from the wiki\n\nListed in his CV."
    )
    out = format_answer_for_display(raw)
    assert "wiki" not in out.lower()
    assert "GitGuardian" in out


def test_parse_json_array():
    assert _parse_json_array('["overview.md", "projects/cvnn.md"]') == [
        "overview.md",
        "projects/cvnn.md",
    ]


@patch("src.wiki_query.read_index", return_value="# index\n")
@patch(
    "src.wiki_query.list_wiki_pages",
    return_value=["overview.md", "projects/cvnn.md"],
)
@patch("src.wiki_query.grep_wiki", return_value=[])
@patch("src.wiki_query.follow_links", return_value=[])
def test_select_pages_llm(mock_links, mock_grep, mock_list, mock_index):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content='["projects/cvnn.md"]')
    pages = select_pages("What is CVNN?", llm)
    assert "overview.md" in pages
    assert "projects/cvnn.md" in pages


@patch("src.wiki_query.read_index", return_value="# index\n")
@patch("src.wiki_query.list_wiki_pages", return_value=["overview.md"])
@patch("src.wiki_query.follow_links", return_value=[])
def test_select_pages_grep_fallback(mock_links, mock_list, mock_index):
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("api down")
    with patch(
        "src.wiki_query.grep_wiki", return_value=[("projects/foo.md", "cvnn")]
    ):
        pages = select_pages("cvnn project", llm)
    assert "overview.md" in pages
    assert "projects/foo.md" in pages
