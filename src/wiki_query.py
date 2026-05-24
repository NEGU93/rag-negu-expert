"""Answer questions from the markdown wiki (index-first navigation)."""

import json
import re

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.logger_init import logger
from src.wiki_io import (
    follow_links,
    grep_wiki,
    list_wiki_pages,
    read_index,
    read_page,
)

MODEL = "gpt-5-mini"
MAX_PAGES = 8
MAX_CHARS_PER_PAGE = 12_000

INITIAL_MESSAGE = """Hello! I'm an AI assistant specialized in providing information about Jose Agustin BARRACHINA. I have access to a curated wiki about his background, projects, skills, and experience.

How can I help you learn more about him today?"""

SYSTEM_PROMPT = """You are an AI assistant specialized in providing information about Jose Agustin BARRACHINA (also known as Agustin, NEGU, or Jose). All pronouns ("he", "him") refer to him.

Voice and style (critical):
- Speak as a knowledgeable colleague who simply knows his career — never meta-comment on where information comes from
- Never use the word "wiki" or phrases like "according to the wiki", "from the wiki", "recorded in the wiki", "details from the wiki", "the documents indicate", or "available information" as a hedge
- Do not cite sources, file paths, or reference materials unless the user explicitly asks where you got something
- No parenthetical citations like (wiki/...) or (CV path)
- State facts directly: "He works at GitGuardian as..." not "The wiki says he works at..."

Formatting:
- Markdown with clear headings (e.g. ## Current employment), lists, and bold when helpful
- Professional, conversational tone; include dates, projects, and specifics

Sources (only if the user explicitly asks for sources, citations, or provenance):
- Add a brief ## Sources section with human-readable labels only (e.g. "CV", "Career timeline") — still never say "wiki"

Response rules:
- Relevant context: answer comprehensively in natural prose
- No relevant context: say you don't have enough information about Jose Agustin BARRACHINA
- Questions about someone else: say you can only provide information about Jose Agustin BARRACHINA

Special cases:
- Yes/no: answer plus supporting details
- Timelines: chronological order
- Uncertainty: "His CV lists..." or "He has described..." — never mention wiki or generic "documents"
"""

_WIKI_CITE_RE = re.compile(r"\s*\(wiki/[^)]+\)")
# Meta phrases the model sometimes uses despite instructions
_WIKI_META_REPLACEMENTS = [
    (re.compile(r"\bAccording to the wiki,?\s*", re.I), ""),
    (re.compile(r"\bfrom the wiki\b", re.I), ""),
    (re.compile(r"\bin the wiki\b", re.I), ""),
    (re.compile(r"\brecorded in the wiki\b", re.I), ""),
    (re.compile(r"\bare recorded in the wiki\.?", re.I), "."),
    (re.compile(r"\bthe wiki\b", re.I), ""),
    (
        re.compile(r"^#{1,3}\s*Details from the wiki\s*$", re.I | re.M),
        "## Details",
    ),
    (
        re.compile(
            r"^#{1,3}\s*Short answer:?\s*According to the wiki,?\s*",
            re.I | re.M,
        ),
        "## Short answer\n\n",
    ),
]


def format_answer_for_display(text: str) -> str:
    """Strip citations and meta 'wiki' language before showing in Gradio."""
    cleaned = _WIKI_CITE_RE.sub("", text)
    for pattern, repl in _WIKI_META_REPLACEMENTS:
        cleaned = pattern.sub(repl, cleaned)
    cleaned = re.sub(r"  +", " ", cleaned)
    cleaned = re.sub(r" +\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


SELECT_PROMPT = """Given the wiki index and page list, pick up to {max_pages} markdown page paths most relevant to the question.
Return ONLY a JSON array of strings, e.g. ["overview.md", "projects/cvnn.md"].
Use paths exactly as listed (include .md). Prefer specific pages over index.md.

Question: {question}

Wiki index:
{index}

All pages ({n_pages}):
{page_list}
"""


def _parse_json_array(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Expected JSON array")
    return [str(p) for p in data]


def select_pages(question: str, llm: ChatOpenAI) -> list[str]:
    """Choose wiki pages via LLM; fall back to keyword grep."""
    pages = list_wiki_pages()
    pages = [p for p in pages if p not in ("index.md", "log.md")]

    selected: list[str] = []
    if read_index_path := _safe_read_index():
        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content="You select wiki pages for Q&A. Output JSON only."
                    ),
                    HumanMessage(
                        content=SELECT_PROMPT.format(
                            max_pages=MAX_PAGES,
                            question=question,
                            index=read_index_path[:8000],
                            n_pages=len(pages),
                            page_list="\n".join(pages[:200]),
                        )
                    ),
                ]
            )
            selected = _parse_json_array(response.content)
        except Exception as e:
            logger.warning(f"LLM page selection failed: {e}")

    selected = [p for p in selected if p in pages]

    if "overview.md" in pages and "overview.md" not in selected:
        selected.insert(0, "overview.md")

    if len(selected) < 3:
        for path, _ in grep_wiki(question, max_results=MAX_PAGES):
            if path not in selected:
                selected.append(path)

    expanded: list[str] = []
    seen = set()
    for rel in selected:
        if rel not in seen:
            seen.add(rel)
            expanded.append(rel)
        for linked in follow_links(rel):
            if linked not in seen and len(expanded) < MAX_PAGES + 4:
                seen.add(linked)
                expanded.append(linked)

    return expanded[:MAX_PAGES]


def _safe_read_index() -> str:
    try:
        return read_index()
    except FileNotFoundError:
        return ""


def build_wiki_context(
    question: str, llm: ChatOpenAI
) -> tuple[str, list[str]]:
    paths = select_pages(question, llm)
    if not paths:
        return "", []

    blocks = []
    for rel in paths:
        try:
            body = read_page(rel)
            if len(body) > MAX_CHARS_PER_PAGE:
                body = body[:MAX_CHARS_PER_PAGE] + "\n\n[... truncated ...]"
            blocks.append(f"### {rel}\n{body}")
        except FileNotFoundError:
            logger.warning(f"Missing wiki page: {rel}")

    return "\n\n---\n\n".join(blocks), paths


def wiki_answer(question: str, llm: ChatOpenAI, chat_history: str = "") -> str:
    context, paths = build_wiki_context(question, llm)
    if not context.strip():
        return (
            "I don't have enough information to answer that question about "
            "Jose Agustin BARRACHINA."
        )

    user_content = f"""Reference material:

{context}

"""
    if chat_history.strip():
        user_content += f"Recent conversation:\n{chat_history}\n\n"
    user_content += f"Question: {question}\n\nAnswer:"

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
    )
    return response.content


class WikiChat:
    """Gradio-compatible chat with simple in-process conversation history."""

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
        self._history: list[tuple[str, str]] = []

    def _history_text(self, last_n: int = 6) -> str:
        lines = []
        for role, content in self._history[-last_n:]:
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    def invoke(self, inputs: dict, *, for_display: bool = True) -> dict:
        question = inputs["question"]
        answer = wiki_answer(
            question, self.llm, chat_history=self._history_text()
        )
        if for_display:
            answer = format_answer_for_display(answer)
        self._history.append(("user", question))
        self._history.append(("assistant", answer))
        return {"answer": answer}


def create_wiki_chain() -> WikiChat:
    return WikiChat()
