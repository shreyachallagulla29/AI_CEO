# =============================================================================
# storage.py — JSON document creation, cleaning, deduplication, and persistence
#
# Responsibilities:
#   - Clean raw scraped text
#   - Build the standard Document schema (matching project requirement)
#   - Deduplicate by URL
#   - Load / save the JSON dataset file
#   - Merge new documents into the existing dataset
# =============================================================================

import re
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path

import config
from text_utils import clean_text, clean_title  # consolidated cleaner

logger = logging.getLogger("storage")


# clean_text() and clean_title() now live in text_utils.py
# They are imported at the top of this file and used in create_document() below.


# ===========================================================================
# Document Schema Builder
# ===========================================================================

def _generate_id(url: str, index: int) -> str:
    """
    Generate a stable, readable document ID.
    Format: doc_<6-char-url-hash>  (deterministic from URL, so same URL → same ID)
    Falls back to a sequential id if URL is empty.
    """
    if url:
        short_hash = hashlib.md5(url.encode()).hexdigest()[:6]
        return f"doc_{short_hash}"
    return f"doc_{index:04d}"


def create_document(raw: dict, source_type: str, index: int) -> dict | None:
    """
    Convert a raw scraper dict into the standard Document schema.

    Args:
        raw:         Raw dict from any scraper (keys: source_name, url, title, content, date, publisher)
        source_type: Scraper category key (e.g. "newsapi", "rss", "reddit", "ir_page")
        index:       Positional index (used as fallback ID)

    Returns:
        Standardised document dict, or None if content is too short to be useful.

    Document schema:
    {
        "id":        "doc_abc123",
        "source":    "reddit",
        "url":       "https://...",
        "title":     "...",
        "content":   "Full cleaned text here...",
        "date":      "2026-06-10",
        "company":   "Lufthansa",
        "type":      "community_post",
        "sentiment": null
    }
    """
    url     = raw.get("url", "").strip()
    title   = clean_title(raw.get("title", ""))
    content = clean_text(raw.get("content", ""))

    # Discard documents that are too short to be informative
    if len(content) < config.MIN_CONTENT_LENGTH:
        logger.debug(f"Skipping short document (len={len(content)}): {url}")
        return None

    # Normalise date to YYYY-MM-DD; default to today if missing/malformed
    raw_date = raw.get("date", "")
    try:
        # Accept both full ISO timestamps and plain dates
        date_part = raw_date[:10] if raw_date else ""
        datetime.strptime(date_part, "%Y-%m-%d")
        date_str = date_part
    except (ValueError, TypeError):
        date_str = datetime.now().strftime("%Y-%m-%d")

    doc_type = config.DOC_TYPES.get(source_type, "news_article")

    return {
        "id":        _generate_id(url, index),
        "source":    raw.get("source_name", source_type),
        "url":       url,
        "title":     title,
        "content":   content,
        "date":      date_str,
        "company":   config.DEFAULT_COMPANY,
        "type":      doc_type,
        "sentiment": None,          # filled later by the sentiment analysis module
        # --- Extra metadata (useful for analysis, not in minimal schema) ---
        "publisher": raw.get("publisher", ""),
        "schema_version": config.DOCUMENT_SCHEMA_VERSION,
    }


def build_documents(raw_data: dict[str, list[dict]]) -> list[dict]:
    """
    Convert all raw scraper output into clean Document objects.

    Args:
        raw_data: Output of scraper.run_all_scrapers()
                  { "newsapi": [...], "rss": [...], "reddit": [...], "ir_page": [...] }

    Returns:
        List of valid Document dicts.
    """
    documents = []
    global_index = 0

    for source_type, raw_list in raw_data.items():
        logger.info(f"Building documents from [{source_type}]: {len(raw_list)} raw items")
        source_count = 0

        for raw in raw_list:
            doc = create_document(raw, source_type, global_index)
            if doc:
                documents.append(doc)
                source_count += 1
            global_index += 1

        logger.info(f"  → {source_count} valid documents from [{source_type}]")

    logger.info(f"Total documents before deduplication: {len(documents)}")
    return documents


# ===========================================================================
# Deduplication
# ===========================================================================

def deduplicate(documents: list[dict]) -> list[dict]:
    """
    Remove duplicate documents by URL (config.DEDUP_KEY).
    When duplicates exist, keeps the first occurrence (usually most recent).

    Also reassigns IDs sequentially after dedup so IDs remain unique.
    """
    seen = set()
    unique = []

    for doc in documents:
        key = doc.get(config.DEDUP_KEY, "")
        if not key or key not in seen:
            seen.add(key)
            unique.append(doc)

    removed = len(documents) - len(unique)
    logger.info(f"Deduplication: removed {removed} duplicates, {len(unique)} unique documents remain")
    return unique


# ===========================================================================
# JSON Persistence
# ===========================================================================

def load_dataset(filepath: Path | None = None) -> list[dict]:
    """
    Load the existing JSON dataset from disk.
    Returns an empty list if the file doesn't exist yet.
    """
    filepath = filepath or config.RAW_JSON_FILE
    if not filepath.exists():
        logger.info(f"No existing dataset at {filepath}, starting fresh")
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} existing documents from {filepath}")
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load dataset: {e}")
        return []


def save_dataset(documents: list[dict], filepath: Path | None = None) -> None:
    """
    Save the document list to JSON with pretty-printing.
    Overwrites the file completely (source of truth = in-memory list).
    """
    filepath = filepath or config.RAW_JSON_FILE
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(documents)} documents to {filepath}")
    except OSError as e:
        logger.error(f"Failed to save dataset: {e}")
        raise


def merge_and_save(
    new_documents: list[dict],
    filepath: Path | None = None,
) -> list[dict]:
    """
    Merge new documents into the existing dataset, deduplicate, and save.

    Workflow:
      1. Load existing dataset
      2. Append new documents
      3. Deduplicate by URL
      4. Save back to disk
      5. Return final merged list

    Args:
        new_documents: Freshly built Document dicts (output of build_documents())
        filepath:      Optional override for JSON file path

    Returns:
        Final deduplicated document list.
    """
    filepath = filepath or config.RAW_JSON_FILE

    existing = load_dataset(filepath)
    combined = existing + new_documents

    logger.info(f"Merging: {len(existing)} existing + {len(new_documents)} new = {len(combined)} total")

    final = deduplicate(combined)
    save_dataset(final, filepath)

    logger.info(f"Dataset ready: {len(final)} unique documents at {filepath}")
    return final


# ===========================================================================
# Convenience: stats summary
# ===========================================================================

def dataset_stats(documents: list[dict]) -> dict:
    """Return a quick summary of the dataset."""
    if not documents:
        return {"total": 0}

    by_source = {}
    by_type   = {}
    by_date   = {}

    for doc in documents:
        src  = doc.get("source", "unknown")
        dtype = doc.get("type", "unknown")
        date = doc.get("date", "unknown")[:7]   # YYYY-MM

        by_source[src]  = by_source.get(src, 0) + 1
        by_type[dtype]  = by_type.get(dtype, 0) + 1
        by_date[date]   = by_date.get(date, 0) + 1

    return {
        "total":      len(documents),
        "by_source":  by_source,
        "by_type":    by_type,
        "by_month":   dict(sorted(by_date.items())),
        "company":    config.COMPANY,
        "saved_to":   str(config.RAW_JSON_FILE),
    }


# ===========================================================================
# Entry point (run standalone to inspect an existing dataset)
# ===========================================================================
if __name__ == "__main__":
    import json as _json

    docs = load_dataset()
    stats = dataset_stats(docs)
    print(_json.dumps(stats, indent=2))
