from pathlib import Path

from src.loaders import load_raw_text


def test_load_raw_text_utf8_markdown_with_emoji():
    path = Path("raw/github/NEGU93.md")
    if not path.is_file():
        return
    text = load_raw_text(path)
    assert "GitGuardian" in text
    assert "Agustin" in text
