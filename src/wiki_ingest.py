"""Ingest one raw source into the wiki via structured LLM output."""

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from src.loaders import LOADERS, RAW_DIR, load_raw_text
from src.logger_init import logger
from src.wiki_io import (
    WIKI_ROOT,
    append_index_entries,
    append_log,
    read_index,
    read_page,
    section_for_wiki_path,
    write_page,
)

MODEL = "gpt-5-mini"
DEFAULT_MAX_RAW_CHARS = 80_000

# GitHub profile README (repo named like the user, e.g. NEGU93/NEGU93)
PROFILE_README_FILENAMES = frozenset({"NEGU93.md", "profile-readme.md"})

INGEST_SYSTEM = """You are a wiki maintainer for Jose Agustin BARRACHINA (NEGU).
Integrate ONE raw source into an existing markdown wiki.

Rules:
- Output ONLY valid JSON (no markdown fences).
- Wiki paths are relative to wiki/ (e.g. "projects/cvnn.md").
- Use kebab-case filenames, YAML frontmatter with tags/sources/updated.
- Use wikilinks [[path/to/page]] cross-references.
- Note contradictions with existing wiki explicitly.
- Always include a sources/ page for this raw file.
- index_entries.section must be one of: Overview, Entities, Concepts, Projects, Education, Publications, Events, Sources, Certificates & courses
"""

INGEST_USER_TEMPLATE = """## Raw source
Path: {raw_path}
Category: {category}

```
{raw_text}
```

## Current wiki index
{index_excerpt}

## Existing related pages (excerpts)
{related_excerpt}

## Task
Return JSON:
{{
  "pages": [
    {{"path": "sources/raw-....md", "action": "create|update", "content": "full markdown"}}
  ],
  "index_entries": [
    {{"section": "Projects", "link": "projects/foo", "summary": "one line"}}
  ],
  "log_line": "short description of what was updated"
}}
"""

PROFILE_README_TASK = """
## Required for this source (GitHub profile README)
This raw file is the public GitHub profile README — the canonical "who is he / what does he do now" page.

You MUST include in "pages":
- `overview.md` — career hub: current role first (employer, title, location if known), then PhD/research highlights, then notable projects. Replace any stub content.
- `entities/jose-barrachina.md` — person entity with the same current-role facts.
- `sources/raw-github-negu93.md` (or matching slug) — source summary for this raw file.

Also add index_entries for Overview (overview) and Entities (jose-barrachina) if missing.
Do NOT file the profile only under projects/; overview.md is the main hub.
"""


def is_profile_readme(raw_path: Path) -> bool:
    return raw_path.name in PROFILE_README_FILENAMES


@dataclass
class IngestResult:
    raw_path: str
    pages_written: list[str]
    index_entries: int
    log_line: str


def raw_path_to_slug(raw_path: Path) -> str:
    """raw/publications/Foo Bar.pdf -> raw-publications-foo-bar"""
    parts = raw_path.as_posix().replace("\\", "/").split("/")
    name = Path(parts[-1]).stem if parts else "unknown"
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    prefix = "-".join(
        re.sub(r"[^a-z0-9]+", "-", p.lower()).strip("-")
        for p in parts[:-1]
        if p and p != RAW_DIR
    )
    return f"{prefix}-{slug}" if prefix else slug


def _load_related_excerpt(raw_path: Path, max_chars: int = 12_000) -> str:
    """Pull index + any existing wiki pages matching raw folder or filename."""
    parts = []
    try:
        parts.append(read_index()[:4000])
    except FileNotFoundError:
        parts.append("(no index yet)")

    folder = raw_path.parent.name
    keywords = [
        raw_path.stem.lower(),
        folder.lower(),
        raw_path.stem.lower().replace("_", "-"),
    ]

    if WIKI_ROOT.exists():
        for md in sorted(WIKI_ROOT.rglob("*.md")):
            rel = md.relative_to(WIKI_ROOT).as_posix()
            if rel in ("index.md", "log.md"):
                continue
            name_lower = md.stem.lower()
            if not any(k in name_lower or k in rel.lower() for k in keywords if k):
                continue
            try:
                body = read_page(rel)[:2000]
                parts.append(f"\n### {rel}\n{body}")
            except FileNotFoundError:
                pass

    text = "\n".join(parts)
    return text[:max_chars] if len(text) > max_chars else text


def _parse_ingest_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def apply_ingest_payload(payload: dict) -> tuple[list[str], int]:
    written = []
    for page in payload.get("pages", []):
        rel = page["path"].replace("\\", "/")
        if not rel.endswith(".md"):
            rel += ".md"
        write_page(rel, page["content"])
        written.append(rel)

    entries = payload.get("index_entries", [])
    for entry in entries:
        if "section" not in entry and "link" in entry:
            entry["section"] = section_for_wiki_path(entry["link"])
    append_index_entries(entries)
    return written, len(entries)


def ingest_source(
    raw_path: Path,
    llm: ChatOpenAI | None = None,
    max_raw_chars: int = DEFAULT_MAX_RAW_CHARS,
    dry_run: bool = False,
) -> IngestResult:
    """
    Read one raw file (including PDFs via PyMuPDF) and update the wiki.
    """
    raw_path = Path(raw_path)
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    ext = raw_path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(
            f"Unsupported file type {ext}. Supported: {', '.join(LOADERS)}"
        )

    llm = llm or ChatOpenAI(temperature=0.3, model_name=MODEL)

    logger.info(f"Ingesting raw source: {raw_path}")
    raw_text = load_raw_text(raw_path, max_chars=max_raw_chars)
    if not raw_text.strip():
        raise ValueError(f"No text extracted from {raw_path} (PDF may be image-only)")

    category = raw_path.parent.name
    related = _load_related_excerpt(raw_path)
    index_excerpt = read_index()[:4000] if (WIKI_ROOT / "index.md").exists() else ""

    user_msg = INGEST_USER_TEMPLATE.format(
        raw_path=raw_path.as_posix(),
        category=category,
        raw_text=raw_text,
        index_excerpt=index_excerpt,
        related_excerpt=related,
    )
    if is_profile_readme(raw_path):
        user_msg += PROFILE_README_TASK

    response = llm.invoke(
        [
            SystemMessage(content=INGEST_SYSTEM),
            HumanMessage(content=user_msg),
        ]
    )
    payload = _parse_ingest_json(response.content)

    if dry_run:
        return IngestResult(
            raw_path=str(raw_path),
            pages_written=[p.get("path", "") for p in payload.get("pages", [])],
            index_entries=len(payload.get("index_entries", [])),
            log_line=payload.get("log_line", ""),
        )

    written, n_index = apply_ingest_payload(payload)

    today = date.today().isoformat()
    log_line = payload.get("log_line", f"Ingested {raw_path.name}")
    log_entry = (
        f"## [{today}] ingest | {raw_path.name}\n"
        f"- Raw: `{raw_path.as_posix()}`\n"
        f"- Pages: {', '.join(f'`{p}`' for p in written)}\n"
        f"- Note: {log_line}"
    )
    append_log(log_entry)

    return IngestResult(
        raw_path=str(raw_path),
        pages_written=written,
        index_entries=n_index,
        log_line=log_line,
    )
