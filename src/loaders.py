"""Load text from immutable raw source files (used by wiki ingest)."""

from pathlib import Path

from langchain_community.document_loaders import (
    JSONLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredImageLoader,
    UnstructuredXMLLoader,
)

from src.logger_init import logger

RAW_DIR = "raw"


def _utf8_text_loader(path: str) -> TextLoader:
    """TextLoader with explicit UTF-8 (Windows default cp1252 breaks emoji in markdown)."""
    return TextLoader(path, encoding="utf-8")


LOADERS = {
    ".pdf": PyMuPDFLoader,
    ".xml": UnstructuredXMLLoader,
    ".md": _utf8_text_loader,
    ".tex": _utf8_text_loader,
    ".txt": _utf8_text_loader,
    ".jpg": UnstructuredImageLoader,
    ".json": lambda path: JSONLoader(
        file_path=path, jq_schema=".", text_content=False
    ),
}


def load_single_file(file_path: Path, loader_class):
    """Load a single file with proper error handling."""
    try:
        loader = loader_class(str(file_path))
        return loader.load()
    except Exception as e:
        logger.error(f"Failed to load {file_path.name}: {e}")
        return []


def add_raw_metadata(doc, doc_type: str, file_path: Path):
    doc.metadata["doc_type"] = doc_type
    if "source" in doc.metadata:
        source_path = Path(doc.metadata["source"])
        doc.metadata["filename"] = source_path.name
        doc.metadata["file_path"] = str(source_path)
    else:
        doc.metadata["filename"] = file_path.name
        doc.metadata["file_path"] = str(file_path)
    return doc


def load_raw_documents(folder_path: str = RAW_DIR):
    """Walk raw/ and return LangChain Documents with metadata (no chunking)."""
    data_path = Path(folder_path)
    if not data_path.is_dir():
        logger.warning(f"Raw folder not found: {data_path}")
        return []

    documents = []
    file_stats = {"loaded": 0, "skipped": 0, "errors": 0}

    for folder in sorted(f for f in data_path.iterdir() if f.is_dir()):
        doc_type = folder.name
        for file_path in folder.rglob("*"):
            if not file_path.is_file():
                continue
            ext = file_path.suffix.lower()
            if ext not in LOADERS:
                file_stats["skipped"] += 1
                continue
            file_docs = load_single_file(file_path, LOADERS[ext])
            if file_docs:
                documents.extend(
                    add_raw_metadata(doc, doc_type, file_path) for doc in file_docs
                )
                file_stats["loaded"] += 1
            else:
                file_stats["errors"] += 1

    logger.info(
        f"Raw load: {file_stats['loaded']} loaded, "
        f"{file_stats['skipped']} skipped, {file_stats['errors']} errors"
    )
    return documents


def load_raw_text(file_path: Path, max_chars: int | None = None) -> str:
    """Extract plain text from one raw file (for wiki ingest)."""
    ext = file_path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported extension: {ext}")

    docs = load_single_file(file_path, LOADERS[ext])
    text = "\n\n".join(d.page_content for d in docs if d.page_content)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... truncated ...]"
    return text
