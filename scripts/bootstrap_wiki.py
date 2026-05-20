#!/usr/bin/env python3
"""
Batch-ingest all files under raw/ into the wiki.

PDFs, images, and other formats are read via src/loaders.py (PyMuPDF for PDFs).

Usage:
  python scripts/bootstrap_wiki.py
  python scripts/bootstrap_wiki.py --dry-run
  python scripts/bootstrap_wiki.py --limit 3
  python scripts/bootstrap_wiki.py --file raw/CV/cv.tex
  python scripts/bootstrap_wiki.py --file raw/github/NEGU93.md --force
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.loaders import LOADERS, RAW_DIR
from src.logger_init import logger
from src.wiki_ingest import ingest_source

CHECKPOINT_PATH = ROOT / "wiki" / ".bootstrap_checkpoint.json"

# Lower number = earlier in bootstrap order
FOLDER_PRIORITY = {
    "CV": 0,
    "website": 1,
    "education": 2,
    "publications": 3,
    "github": 4,
    "events_conferences": 5,
    "comptetitions": 6,
    "certificates": 7,
    "courses": 8,
    "languages": 9,
}


def checkpoint_key(path: Path) -> str:
    """Stable key for resume (resolved absolute path)."""
    return path.resolve().as_posix()


def load_checkpoint() -> set[str]:
    if not CHECKPOINT_PATH.exists():
        return set()
    data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {checkpoint_key(Path(p)) for p in data.get("completed", [])}


def is_completed(path: Path, completed: set[str]) -> bool:
    return checkpoint_key(path) in completed


def save_checkpoint(completed: set[str]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(
        json.dumps({"completed": sorted(completed)}, indent=2),
        encoding="utf-8",
    )


def iter_raw_files(raw_root: Path) -> list[Path]:
    files = []
    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in LOADERS:
            files.append(path)
    return sorted(files, key=_sort_key)


def _sort_key(path: Path) -> tuple:
    try:
        rel = path.relative_to(RAW_DIR)
    except ValueError:
        rel = path
    folder = rel.parts[0] if len(rel.parts) > 1 else ""
    priority = FOLDER_PRIORITY.get(folder, 99)
    return (priority, str(rel).lower())


def main() -> int:
    load_dotenv(ROOT / ".env", override=True)

    parser = argparse.ArgumentParser(description="Bootstrap wiki from raw/")
    parser.add_argument("--dry-run", action="store_true", help="LLM only, no writes")
    parser.add_argument("--limit", type=int, default=0, help="Max files to process (0=all)")
    parser.add_argument("--file", type=str, help="Ingest a single raw file")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if already in checkpoint (use with --file)",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    args = parser.parse_args()

    raw_root = ROOT / RAW_DIR
    if not raw_root.is_dir():
        logger.error(f"Missing {raw_root}")
        return 1

    if args.reset and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint cleared")

    completed = load_checkpoint()

    if args.file:
        targets = [ROOT / args.file]
    else:
        targets = iter_raw_files(raw_root)

    if args.force:
        pending = list(targets)
    else:
        pending = [p for p in targets if not is_completed(p, completed)]
    if args.limit > 0:
        pending = pending[: args.limit]

    skipped = len(targets) - len(pending) if not args.force else 0
    logger.info(
        f"Bootstrap: {len(pending)} file(s) to process "
        f"({skipped} skipped as already done, {len(completed)} in checkpoint)"
    )

    ok, fail = 0, 0
    for i, path in enumerate(pending, 1):
        logger.info(f"[{i}/{len(pending)}] {path.as_posix()}")
        try:
            result = ingest_source(path, dry_run=args.dry_run)
            logger.info(
                f"  -> {len(result.pages_written)} pages, "
                f"{result.index_entries} index rows"
            )
            if not args.dry_run:
                completed.add(checkpoint_key(path))
                save_checkpoint(completed)
            ok += 1
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            fail += 1

        if args.delay and i < len(pending):
            time.sleep(args.delay)

    logger.info(f"Done: {ok} ok, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
