# =============================================================================
# config.py — Central configuration for the Lufthansa Strategic Intelligence Agent
# =============================================================================

import os
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Storage / File Paths
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
RAW_JSON_FILE = DATA_DIR / "raw_dataset.json"
REVIEWS_JSON_FILE = DATA_DIR / "reviews_raw_dataset.json"
LOG_FILE      = BASE_DIR / "logs" / "scraper.log"

# Create dirs if missing
DATA_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Document Schema Defaults
# ---------------------------------------------------------------------------
DOCUMENT_SCHEMA_VERSION = "1.0"

# Allowed document types (maps to source categories in requirements)
DOC_TYPES = {
    "newsapi":       "news_article",
    "rss":           "news_article",
    "reddit":        "community_post",
    "ir_page":       "press_release",
    "newsroom":      "press_release",
    "hyperbrowser":  "web_page",       # scraper_1.py JS-rendered pages
}

# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
DEDUP_KEY = "url"               # field used to detect duplicates

# ---------------------------------------------------------------------------
# Scraping Run Control
# ---------------------------------------------------------------------------
MIN_CONTENT_LENGTH = 80         # discard documents shorter than this (chars)
MAX_CONTENT_LENGTH = 50_000     # truncate documents longer than this (chars)

# ---------------------------------------------------------------------------
# Chunking  (processor.py)
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 1000            # characters per chunk (≈ 250–300 tokens for most models)
CHUNK_OVERLAP = 150             # character overlap between adjacent chunks
CHUNK_MIN_SIZE = 100            # discard chunks shorter than this (chars)

# ---------------------------------------------------------------------------
# Embedding Model  (embedder.py)
# ---------------------------------------------------------------------------
# Switch between models by changing EMBEDDING_MODEL to "bge" or "minilm"
EMBEDDING_MODEL = "bge"         # "bge" → BAAI/bge-base-en-v1.5  |  "minilm" → all-MiniLM-L6-v2

EMBEDDING_MODELS = {
    "bge":    "BAAI/bge-base-en-v1.5",
    "minilm": "all-MiniLM-L6-v2",
}

EMBEDDING_BATCH_SIZE = 32       # chunks processed per inference batch
EMBEDDING_DEVICE     = "cpu"    # "cpu" | "cuda" | "mps"

# BGE requires a query prefix at retrieval time (not during indexing)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------------------
# ChromaDB Cloud  — hosted vector store  (vector_store.py)
# Get these from: https://trychroma.com → your database dashboard
# ---------------------------------------------------------------------------
CHROMA_API_KEY         = os.getenv("CHROMA_API_KEY",  "ck-42Hn2rzqUrXu5k3GtwmDaJtDmfNh6xa8MUs39d9EanM4")
CHROMA_TENANT          = os.getenv("CHROMA_TENANT",   "3cd57535-7db4-499f-a4c4-91022866ca80")
CHROMA_DATABASE        = os.getenv("CHROMA_DATABASE", "ragsystem")
CHROMA_COLLECTION_NAME = "Lufthansa_Knowledge"
CHROMA_DISTANCE_METRIC = "cosine"   # "cosine" | "l2" | "ip"
CHROMA_TOP_K           = 5          # default results per query
CHROMA_PERSIST_DIR = BASE_DIR / "chromadb"

# Maps each company name (as it appears in links.json "Company_Name") to its
# ChromaDB collection. Unknown companies fall back to CHROMA_COLLECTION_NAME.
COLLECTION_MAP: dict[str, str] = {
    "Lufthansa":         "Lufthansa_Knowledge",
    "Air India":         "AirIndia_Knowledge",
    "United Airlines":   "UnitedAirlines_Knowledge",
    "Delta Air Lines":   "DeltaAirlines_Knowledge",
    "American Airlines": "AmericanAirlines_Knowledge",
    "reviews" : "Reviews_Knowledge"
}

# Path to the links.json file that drives scraper_1.py
LINKS_JSON_FILE = BASE_DIR / "links.json"

# ---------------------------------------------------------------------------
# Hyperbrowser — JS-rendered web scraping  (scraper_1.py)
# Get an API key at: https://app.hyperbrowser.ai
# ---------------------------------------------------------------------------
HYPERBROWSER_API_KEY = os.getenv("HYPERBROWSER_API_KEY", "hb_657e76ab1a12fd41125b5d278313")

# Each entry: {"link": "<URL>", "scrap_depth": <int>}
#   scrap_depth=0 → only the given page
#   scrap_depth=1 → given page + all direct links found on it
#   scrap_depth=N → N levels of link traversal (BFS)
with open("links.json","r") as f:
    HYPERBROWSER_TARGETS: list[dict] = json.load(f)


HYPERBROWSER_MAX_LINKS_PER_PAGE = 20    # cap links followed per page (prevents exponential growth)
HYPERBROWSER_REQUEST_DELAY      = 1.5   # seconds between requests (be polite)

# ---------------------------------------------------------------------------
# Apify — Skytrax airline reviews  (reviews_scraper.py)
# Get an API token at: https://console.apify.com/account/integrations
# ---------------------------------------------------------------------------
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "apify_api_ZUUmhKa9cXRRFcDhOBIga3orSaDQfH4rIXsK")

# Reviews file (flat list — one item per review, all companies combined)
REVIEWS_JSON_FILE = DATA_DIR / "reviews_raw_dataset.json"

# Max reviews to fetch per company per run (set to None for no cap)
REVIEWS_MAX_PER_COMPANY = 500

# Maps company name → AirlineQuality.com review page URL
# Slug format: https://www.airlinequality.com/airline-reviews/<slug>
COMPANY_SKYTRAX_URLS: dict[str, str] = {
    "Lufthansa":         "https://www.airlinequality.com/airline-reviews/lufthansa",
    "Air India":         "https://www.airlinequality.com/airline-reviews/air-india",
    "United Airlines":   "https://www.airlinequality.com/airline-reviews/united-airlines",
    "Delta Air Lines":   "https://www.airlinequality.com/airline-reviews/delta-air-lines",
    "American Airlines": "https://www.airlinequality.com/airline-reviews/american-airlines",
}