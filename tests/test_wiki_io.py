import pytest

from src.wiki_io import (
    WIKI_ROOT,
    append_index_entries,
    follow_links,
    list_wiki_pages,
    read_index,
    resolve_wiki_path,
    write_page,
)


def test_resolve_wiki_path_rejects_traversal():
    with pytest.raises(ValueError):
        resolve_wiki_path("../../etc/passwd")


def test_read_index():
    assert "Wiki index" in read_index()


def test_list_wiki_pages_includes_overview():
    pages = list_wiki_pages()
    assert "overview.md" in pages


def test_write_and_follow_links(tmp_path, monkeypatch):
    monkeypatch.setattr("src.wiki_io.WIKI_ROOT", tmp_path / "wiki")
    root = tmp_path / "wiki"
    root.mkdir()
    write_page("concepts/test.md", "See [[projects/demo]] for more.\n")
    write_page("projects/demo.md", "# Demo\n")
    links = follow_links("concepts/test.md")
    assert "projects/demo.md" in links


def test_append_index_entries(tmp_path, monkeypatch):
    monkeypatch.setattr("src.wiki_io.WIKI_ROOT", tmp_path / "wiki")
    monkeypatch.setattr("src.wiki_io.INDEX_PATH", tmp_path / "wiki" / "index.md")
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "index.md").write_text(
        "# Wiki index\n\n## Projects\n\n| Page | Summary |\n|------|----------|\n",
        encoding="utf-8",
    )
    append_index_entries(
        [{"section": "Projects", "link": "projects/foo", "summary": "A project"}]
    )
    text = (tmp_path / "wiki" / "index.md").read_text(encoding="utf-8")
    assert "projects/foo.md" in text
    assert "A project" in text
