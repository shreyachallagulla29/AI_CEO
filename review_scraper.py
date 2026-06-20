"""review_scraper.py — Skytrax airline reviews via Apify → clean → save → ChromaDB."""

import json, logging
from datetime import datetime
import config
from text_utils import get_cleaner

logger = logging.getLogger("review_scraper")


def scrape_and_save(max_reviews: int = None, cutoff_date: str = "") -> list[dict]:
    """
    Scrape Skytrax reviews for every airline in COMPANY_SKYTRAX_URLS,
    clean the 'comment' field with TextCleaner, and save to REVIEWS_JSON_FILE.

    Returns the list of raw dicts in standard pipeline format.
    """
    from apify_client import ApifyClient

    cleaner = get_cleaner()
    client  = ApifyClient(config.APIFY_API_TOKEN)
    max_r   = max_reviews or config.REVIEWS_MAX_PER_COMPANY
    reviews = []

    for company, url in config.COMPANY_SKYTRAX_URLS.items():
        logger.info(f"Scraping reviews: {company}")
        run = client.actor("knagymate/airlinequality-skytrax-reviews-scraper").call(
            run_input={"startUrl": url, "maxReviews": max_r, "cutoffDate": cutoff_date}
        )
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            reviews.append({
                "source_name": "skytrax_review",
                "url":         item.get("url", ""),
                "title":       item.get("title", ""),
                "content":     cleaner.clean(item.get("comment", "")),
                "date":        (item.get("date") or datetime.now().strftime("%Y-%m-%d"))[:10],
                "publisher":   "airlinequality.com",
                "company":     company,
            })
        logger.info(f"  → {len(reviews)} total so far")

    config.REVIEWS_JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.REVIEWS_JSON_FILE.write_text(
        json.dumps(reviews, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Saved {len(reviews)} reviews → {config.REVIEWS_JSON_FILE}")
    return reviews


def run_review_pipeline(max_reviews: int = None) -> dict:
    """Scrape → build docs → chunk → embed → push to Reviews_Knowledge."""
    from clean_storage import build_documents
    from processor    import run_processing_pipeline
    from embedder     import generate_embeddings
    from vector_store import ChromaVectorStore

    reviews  = scrape_and_save(max_reviews=max_reviews)
    docs     = build_documents({"reviews": reviews})
    chunks   = run_processing_pipeline(docs)
    embedded = generate_embeddings(chunks)

    store = ChromaVectorStore(collection_name=config.COLLECTION_MAP["reviews"])
    saved = store.upsert_embeddings(embedded)
    logger.info(f"Reviews: {saved} chunks → '{config.COLLECTION_MAP['reviews']}'")
    return {"reviews_scraped": len(reviews), "chunks_stored": saved, "total_in_db": store.count()}


if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(asctime)s  %(name)-15s  %(levelname)s  %(message)s")
    print(run_review_pipeline(max_reviews=10))
