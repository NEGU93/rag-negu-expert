---
title: rag-negu-expert
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# NEGU Expert (LLM Wiki)

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/NEGU93/rag-negu-expert)
[![Gradio](https://img.shields.io/badge/Gradio-5.42-orange)](https://gradio.app/)

A personal **portfolio** project: an expert chatbot about **Jose Agustin BARRACHINA**, built to showcase applied LLM work on a real corpus (CV, papers, GitHub, timeline).

## From RAG to LLM Wiki

This repo **started as classic RAG** — LangChain, OpenAI embeddings, ChromaDB, chunk retrieval at query time, Gradio on Hugging Face Spaces. That version is a solid baseline and worth knowing if you hire for ML / LLM engineering.

I **migrated to an LLM Wiki** pattern (persistent markdown knowledge base maintained by the model) because, for this use case, it fit the goal better:

- **Performance:** no embedding rebuild or vector DB on every deploy; the Space loads a pre-built `wiki/` and answers in seconds.
- **Objective:** a stable personal expert over *my* documents — synthesis and cross-links matter more than re-finding chunks on every question.
- **Experiment:** I wanted to try something new on a portfolio piece, not only ship the default RAG stack.

So today the live app is **wiki-first** (`raw/` → compiled `wiki/` → index navigation → chat). The name *rag-negu-expert* is historical; the headline stack is the wiki, with RAG as the earlier chapter of the same project.

## Live demo

[Try the interactive demo](https://huggingface.co/spaces/NEGU93/rag-negu-expert)

## Architecture (current)

| Layer | Path | Role |
|-------|------|------|
| Raw sources | `raw/` | Immutable PDFs, READMEs, CV, etc. |
| Wiki | `wiki/` | LLM-maintained markdown + cross-links |
| Schema | `AGENTS.md` | Ingest / query / lint conventions |

**Query:** read `wiki/index.md` → open relevant pages → synthesize a natural answer.

## Local setup

```bash
uv sync
# add OPENAI_API_KEY to .env
uv run python src/app.py
```

## Bootstrap wiki from raw sources

When `raw/` changes, re-ingest (wiki does not update automatically):

```bash
uv run python scripts/bootstrap_wiki.py --limit 1   # smoke test
uv run python scripts/bootstrap_wiki.py             # full corpus
uv run python scripts/bootstrap_wiki.py --file raw/CV/cv.tex
uv run python scripts/bootstrap_wiki.py --file raw/github/NEGU93.md --force  # re-run profile → overview
```

Uses **PyMuPDF** for PDFs. Resume via `wiki/.bootstrap_checkpoint.json`.

## Author

**J Agustin Barrachina (NEGU93)**

- GitHub: [@NEGU93](https://github.com/NEGU93)
- Hugging Face: [@NEGU93](https://huggingface.co/NEGU93)
