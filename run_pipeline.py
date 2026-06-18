# =============================================================================
# run_pipeline.py — End-to-end pipeline runner
#
# Stages:
#   1. Scrape      — NewsAPI, RSS, Reddit, Lufthansa IR
#   2. Build & Save — clean raw data → JSON documents → save to disk
#   3. Process     — clean text → deduplicate → chunk
#   4. Embed       — generate vectors (BGE or MiniLM, set in config.py)
#   5. Store       — upsert embeddings into ChromaDB Cloud
#
# Run:
#   python run_pipeline.py
#   python run_pipeline.py --skip-scrape     (use existing JSON on disk)
# =============================================================================

import argparse
import json
import logging
import sys
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
# Pipeline
# ---------------------------------------------------------------------------

def run(skip_scrape: bool = False):

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
        logger.info("─── Stage 1: Scraping ───────────────────────────────────")
        from scraper import run_all_scrapers
        from clean_storage import build_documents, merge_and_save, dataset_stats

        raw_data  = run_all_scrapers()
        counts    = {src: len(items) for src, items in raw_data.items()}
        logger.info(f"Scraped: {counts}")

        documents = build_documents(raw_data)
        documents = merge_and_save(documents)          # deduplicates + saves JSON
        logger.info(f"Dataset: {json.dumps(dataset_stats(documents), indent=2)}")

    if not documents:
        logger.error("No documents to process. Exiting.")
        sys.exit(1)

    # ── Stage 2: Process ────────────────────────────────────────────────────
    logger.info("─── Stage 2: Processing (clean → dedupe → chunk) ───────")
    from processor import run_processing_pipeline, chunk_stats

    chunks = run_processing_pipeline(documents)
    logger.info(f"Chunks: {json.dumps(chunk_stats(chunks), indent=2)}")

    if not chunks:
        logger.error("No chunks produced. Check MIN_CONTENT_LENGTH in config.py.")
        sys.exit(1)

    # ── Stage 3: Embed ──────────────────────────────────────────────────────
    logger.info(f"─── Stage 3: Embedding ({config.EMBEDDING_MODEL}) ──────────────────")
    from embedder import generate_embeddings

    embedded = generate_embeddings(chunks)
    logger.info(f"Embedded {len(embedded)} chunks  (dim={len(embedded[0]['embedding'])})")

    # ── Stage 4: Store ──────────────────────────────────────────────────────
    logger.info("─── Stage 4: Storing to ChromaDB Cloud ──────────────────")
    from vector_store import ChromaVectorStore

    chroma = ChromaVectorStore()
    saved  = chroma.upsert_embeddings(embedded)
    total  = chroma.count()

    logger.info(f"Upserted {saved} chunks  |  Total vectors in ChromaDB: {total}")
    logger.info("─── Pipeline complete ✓ ─────────────────────────────────")

    return {"chunks_embedded": len(embedded), "chunks_stored": saved, "total_in_db": total}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lufthansa RAG pipeline")
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping; use the existing JSON dataset on disk",
    )
    args = parser.parse_args()

    result = run(skip_scrape=args.skip_scrape)
    print(f"\nDone: {result}")
