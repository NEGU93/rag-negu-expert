"""Read/write helpers for the markdown wiki."""

import re
from pathlib import Path

from src.logger_init import logger

WIKI_ROOT = Path("wiki")
INDEX_PATH = WIKI_ROOT / "index.md"
LOG_PATH = WIKI_ROOT / "log.md"

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:\|[^\]]+)?\]\]")
MD_LINK_RE = re.compile(r"\]\(([^)#]+\.md)\)")

INDEX_SECTIONS = {
    "overview": "Overview",
    "entities": "Entities",
    "concepts": "Concepts",
    "projects": "Projects",
    "education": "Education",
    "publications": "Publications",
    "events": "Events",
    "sources": "Sources",
}


def resolve_wiki_path(rel_path: str) -> Path:
    """Resolve a wiki-relative path; reject traversal outside wiki/."""
    rel = rel_path.replace("\\", "/").lstrip("/")
    if rel.startswith("wiki/"):
        rel = rel[5:]
    full = (WIKI_ROOT / rel).resolve()
    wiki_resolved = WIKI_ROOT.resolve()
    if not str(full).startswith(str(wiki_resolved)):
        raise ValueError(f"Path escapes wiki root: {rel_path}")
    return full


def read_index() -> str:
    return INDEX_PATH.read_text(encoding="utf-8")


def read_page(rel_path: str) -> str:
    path = resolve_wiki_path(rel_path)
    if not path.is_file():
        raise FileNotFoundError(rel_path)
    return path.read_text(encoding="utf-8")


def write_page(rel_path: str, content: str) -> Path:
    path = resolve_wiki_path(rel_path)
    if not rel_path.endswith(".md"):
        raise ValueError("Wiki pages must be .md files")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info(f"Wrote wiki page: {rel_path}")
    return path


def list_wiki_pages(category: str | None = None) -> list[str]:
    """Return wiki-relative paths for all .md files (excluding index/log at root optional)."""
    pages = []
    for path in sorted(WIKI_ROOT.rglob("*.md")):
        rel = path.relative_to(WIKI_ROOT).as_posix()
        if category:
            prefix = category.rstrip("/") + "/"
            if not rel.startswith(prefix) and rel != f"{category}.md":
                continue
        pages.append(rel)
    return pages


def append_log(entry: str) -> None:
    """Append a log entry (include ## header in entry)."""
    text = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else "# Wiki log\n\n"
    if not text.endswith("\n"):
        text += "\n"
    text += "\n" + entry.strip() + "\n"
    LOG_PATH.write_text(text, encoding="utf-8")


def _normalize_link_target(target: str) -> str:
    target = target.strip()
    if target.endswith(".md"):
        return target
    return f"{target}.md"


def follow_links(rel_path: str) -> list[str]:
    """Parse wikilinks and markdown .md links from a page; return existing wiki paths."""
    try:
        content = read_page(rel_path)
    except FileNotFoundError:
        return []

    found: list[str] = []
    for match in WIKILINK_RE.finditer(content):
        found.append(_normalize_link_target(match.group(1)))
    for match in MD_LINK_RE.finditer(content):
        found.append(_normalize_link_target(match.group(1)))

    existing = []
    seen = set()
    for target in found:
        if target in seen:
            continue
        seen.add(target)
        try:
            resolve_wiki_path(target)
            if resolve_wiki_path(target).is_file():
                existing.append(target)
        except (ValueError, FileNotFoundError):
            pass
    return existing


def grep_wiki(query: str, max_results: int = 20) -> list[tuple[str, str]]:
    """Simple case-insensitive search in page titles and bodies."""
    q = query.lower()
    hits: list[tuple[str, str]] = []
    for rel in list_wiki_pages():
        if rel in ("index.md", "log.md"):
            continue
        try:
            body = read_page(rel)
        except FileNotFoundError:
            continue
        if q in rel.lower() or q in body.lower():
            line = next(
                (ln.strip() for ln in body.splitlines() if q in ln.lower()),
                "",
            )
            hits.append((rel, line[:120]))
            if len(hits) >= max_results:
                break
    return hits


def append_index_entries(entries: list[dict]) -> None:
    """
    Append rows to index.md sections.
    Each entry: {"section": "Projects", "link": "projects/cvnn", "summary": "..."}
    """
    if not entries:
        return

    index = read_index()
    for entry in entries:
        section = entry.get("section", "Sources")
        link = entry["link"].removesuffix(".md")
        summary = entry.get("summary", "")
        row = f"| [{Path(link).name}]({link}.md) | {summary} |"

        header = f"## {section}"
        if header not in index:
            index += f"\n\n{header}\n\n| Page | Summary |\n|------|----------|\n{row}\n"
            continue

        parts = index.split(header, 1)
        after = parts[1]
        if row in after:
            continue

        table_end = re.search(r"\n\n## |\n\n_[^\n]+_$", after)
        if "| Page | Summary |" in after[:500]:
            insert_at = after.find("\n\n", after.find("|------|"))
            if insert_at == -1:
                insert_at = len(after)
            else:
                insert_at += 2
            after = after[:insert_at] + row + "\n" + after[insert_at:]
        else:
            after = (
                f"\n\n| Page | Summary |\n|------|----------|\n{row}\n" + after
            )

        index = parts[0] + header + after

    INDEX_PATH.write_text(index, encoding="utf-8")


def section_for_wiki_path(rel_path: str) -> str:
    """Map wiki path prefix to index section title."""
    rel = rel_path.replace("\\", "/")
    if rel == "overview.md":
        return "Overview"
    prefix = rel.split("/")[0] if "/" in rel else ""
    return INDEX_SECTIONS.get(prefix, "Sources")
