# =============================================================================
# run_pipeline.py — End-to-end pipeline runner
#
# Stages:
#   1. Scrape   — reads links.json, scrapes each URL via scraper_1.py
#                 company name is read from links.json "Company_Name" field
#                 and injected into every raw document
#   2. Build    — clean raw data → Document schema → save combined JSON dataset
#   3. Process  — clean text → deduplicate → chunk
#   4. Embed    — generate vectors (BGE or MiniLM, set in config.py)
#   5. Store    — group chunks by company → upsert each group to its own
#                 ChromaDB collection (see COMPANY_COLLECTION_MAP in config.py)
#
# Collections:
#   Lufthansa        → Lufthansa_Knowledge
#   Air India        → AirIndia_Knowledge
#   United Airlines  → UnitedAirlines_Knowledge
#   Delta Air Lines  → DeltaAirlines_Knowledge
#   American Airlines→ AmericanAirlines_Knowledge
#
# Run:
#   python run_pipeline.py                          # full run (scrape + store)
#   python run_pipeline.py --skip-scrape            # use existing JSON on disk
#   python run_pipeline.py --links links.json       # override links file path
# =============================================================================

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-15s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_links(links_file: Path) -> dict[str, list[dict]]:
    """
    Load links.json and group entries by Company_Name.

    Returns:
        {"Lufthansa": [{"link": "...", "scrap_depth": 1}, ...], "Air India": [...], ...}
    """
    if not links_file.exists():
        logger.error(f"links.json not found at {links_file}")
        sys.exit(1)

    with open(links_file, encoding="utf-8") as f:
        entries = json.load(f)

    by_company: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        company = entry.get("Company_Name", "").strip()
        link    = entry.get("link", "").strip()
        depth   = int(entry.get("scrap_depth", 1))
        if company and link:
            by_company[company].append({"link": link, "scrap_depth": depth})
        else:
            logger.warning(f"Skipping malformed entry in links.json: {entry}")

    logger.info(
        f"Loaded links.json — {sum(len(v) for v in by_company.values())} URLs "
        f"across {len(by_company)} companies: {list(by_company.keys())}"
    )
    return dict(by_company)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(skip_scrape: bool = False, links_file: Path = None):

    links_file = links_file or config.LINKS_JSON_FILE

    # ── Stage 1: Scrape ─────────────────────────────────────────────────────
    if skip_scrape:
        logger.info("─── Stage 1: SKIPPED (--skip-scrape) ───────────────────")
        from clean_storage import load_dataset
        documents = load_dataset()
        if not documents:
            logger.error("No existing dataset found. Run without --skip-scrape first.")
            sys.exit(1)
        logger.info(f"Loaded {len(documents)} existing documents from disk")

    else:
        logger.info("─── Stage 1: Scraping from links.json ───────────────────")
        from scraper_1 import run_hyperbrowser_scrapers
        from clean_storage import build_documents, merge_and_save, dataset_stats

        by_company = _load_links(links_file)

        # Scrape each company's URLs separately so "company" field is correctly
        # tagged in every raw document (scraper_1 passes it through to each dict)
        all_raw_items: list[dict] = []
        for company_name, targets in by_company.items():
            logger.info(
                f"  Scraping {company_name}: {len(targets)} URL(s)"
            )
            raw = run_hyperbrowser_scrapers(targets=targets, company=company_name)
            # raw = {"hyperbrowser": [...]}; flatten all source lists
            for items in raw.values():
                all_raw_items.extend(items)
            logger.info(
                f"  → {sum(len(v) for v in raw.values())} pages scraped for {company_name}"
            )

        logger.info(f"Total raw items from all companies: {len(all_raw_items)}")

        # Wrap in the standard {"hyperbrowser": [...]} structure for build_documents()
        raw_data  = {"hyperbrowser": all_raw_items}
        documents = build_documents(raw_data)
        documents = merge_and_save(documents)   # deduplicates + saves combined JSON
        # logger.info(f"Dataset:\n{json.dumps(dataset_stats(documents), indent=2)}")

    if not documents:
        logger.error("No documents to process. Exiting.")
        sys.exit(1)

    # ── Stage 2: Process ────────────────────────────────────────────────────
    logger.info("─── Stage 2: Processing (clean → dedupe → chunk) ───────")
    from processor import run_processing_pipeline, chunk_stats

    chunks = run_processing_pipeline(documents)
    logger.info(f"Chunks:\n{json.dumps(chunk_stats(chunks), indent=2)}")

    if not chunks:
        logger.error("No chunks produced. Check MIN_CONTENT_LENGTH in config.py.")
        sys.exit(1)

    # ── Stage 3: Embed ──────────────────────────────────────────────────────
    logger.info(f"─── Stage 3: Embedding ({config.EMBEDDING_MODEL}) ──────────────────")
    from embedder import generate_embeddings

    embedded = generate_embeddings(chunks)
    logger.info(f"Embedded {len(embedded)} chunks  (dim={len(embedded[0]['embedding'])})")

    # ── Stage 4: Store — per-company ChromaDB collections ───────────────────
    logger.info("─── Stage 4: Storing to ChromaDB Cloud (per-company) ────")
    from vector_store import ChromaVectorStore

    # Group embedded chunks by company
    by_company_chunks: dict[str, list[dict]] = defaultdict(list)
    for chunk in embedded:
        company = chunk.get("company", "").strip() or config.DEFAULT_COMPANY
        by_company_chunks[company].append(chunk)

    total_stored = 0
    summary      = {}

    for company_name, company_chunks in by_company_chunks.items():
        collection_name = config.COMPANY_COLLECTION_MAP.get(
            company_name, config.CHROMA_COLLECTION_NAME
        )
        if company_name not in config.COMPANY_COLLECTION_MAP:
            logger.warning(
                f"'{company_name}' not in COMPANY_COLLECTION_MAP — "
                f"falling back to '{collection_name}'"
            )

        logger.info(
            f"  {company_name}: {len(company_chunks)} chunks → "
            f"collection '{collection_name}'"
        )
        store  = ChromaVectorStore(collection_name=collection_name)
        saved  = store.upsert_embeddings(company_chunks)
        total  = store.count()
        total_stored += saved
        summary[company_name] = {
            "collection":    collection_name,
            "chunks_stored": saved,
            "total_in_db":   total,
        }
        logger.info(f"    → {saved} stored  |  {total} total in '{collection_name}'")

    logger.info("─── Pipeline complete ✓ ─────────────────────────────────")
    logger.info(f"Summary:\n{json.dumps(summary, indent=2)}")

    return {
        "chunks_embedded": len(embedded),
        "chunks_stored":   total_stored,
        "per_company":     summary,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-company RAG pipeline")
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping; use the existing JSON dataset on disk",
    )
    parser.add_argument(
        "--links",
        type=str,
        default=None,
        help=f"Path to links.json (default: {config.LINKS_JSON_FILE})",
    )
    args = parser.parse_args()

    links_path = Path(args.links) if args.links else None
    result     = run(skip_scrape=args.skip_scrape, links_file=links_path)
    print(f"\nDone:\n{json.dumps(result, indent=2)}")
