# =============================================================================
# config.py — Central configuration for the Lufthansa Strategic Intelligence Agent
# =============================================================================

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Company
# ---------------------------------------------------------------------------
COMPANY = "Lufthansa"
COMPANY_ALIASES = ["Lufthansa", "Deutsche Lufthansa", "LH", "DLAKF"]
COMPANY_TICKER = "LHA.DE"
INDUSTRY = "Aviation / Airline"

# Keywords used across all scrapers to filter relevant content
SEARCH_KEYWORDS = [
    "Lufthansa",
    "Deutsche Lufthansa",
    "Lufthansa Group",
    "Lufthansa cargo",
    "Lufthansa technik",
    "LH airline",
    "Miles & More",
]

# ---------------------------------------------------------------------------
# API Keys  (set as environment variables — never hard-code secrets)
# ---------------------------------------------------------------------------
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "dbacc7a9531c4e958287d41f8233b783")

REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID",     "YOUR_REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "YOUR_REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT",    "lufthansa_intel_agent/1.0")

# ---------------------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------------------
NEWSAPI_BASE_URL  = "https://newsapi.org/v2/everything"
NEWSAPI_PAGE_SIZE = 100          # max per request (free tier: 100)
NEWSAPI_MAX_PAGES = 1            # free tier only supports page 1 (426 on page 2+)
NEWSAPI_LANGUAGE  = "en"
NEWSAPI_SORT_BY   = "publishedAt"
NEWSAPI_FROM_DAYS = 30           # look back this many days

# ---------------------------------------------------------------------------
# RSS Feeds  (aviation, finance, European business news)
# ---------------------------------------------------------------------------
RSS_FEEDS = {
    "Reuters Business":      "https://feeds.reuters.com/reuters/businessNews",
    "Reuters Transport":     "https://feeds.reuters.com/reuters/industrialsNews",
    "BBC Business":          "https://feeds.bbci.co.uk/news/business/rss.xml",
    "Simple Flying":         "https://simpleflying.com/feed/",
    "Aviation Week":         "https://aviationweek.com/rss.xml",
    "The Points Guy":        "https://thepointsguy.com/feed/",
    "Air Transport World":   "https://atwonline.com/rss.xml",
    "FlightGlobal":          "https://www.flightglobal.com/rss/",
    "Handelsblatt English":  "https://www.handelsblatt.com/contentexport/feed/english",
    "Yahoo Finance LH":      "https://finance.yahoo.com/rss/headline?s=LHA.DE",
}

RSS_MAX_ENTRIES_PER_FEED = 50    # cap per feed per run

# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------
REDDIT_SUBREDDITS = [
    "aviation",
    "flights",
    "frequentflyers",
    "awardtravel",
    "churning",          # miles & credit card strategy
    "lufthansa",         # direct brand subreddit
    "europe",
    "stocks",
    "investing",
]

REDDIT_POST_LIMIT     = 50       # posts per subreddit
REDDIT_COMMENT_DEPTH  = 2        # top-level + one reply level
REDDIT_MIN_SCORE      = 5        # ignore posts below this upvote threshold
REDDIT_TIME_FILTER    = "month"  # "day" | "week" | "month" | "year"
REDDIT_SEARCH_LIMIT   = 25       # extra keyword-search posts per subreddit

# ---------------------------------------------------------------------------
# Lufthansa Official & IR Scraping
# ---------------------------------------------------------------------------
LUFTHANSA_IR_URL        = "https://investor-relations.lufthansagroup.com/en/news/financial-news.html"
LUFTHANSA_NEWSROOM_URL  = "https://newsroom.lufthansagroup.com/en/"
# LUFTHANSA_PRESS_URL     = "https://www.lufthansagroup.com/en/press.html"

IR_MAX_PAGES     = 3             # pagination pages to scrape
IR_MAX_ARTICLES  = 50            # max press release articles to fetch per run (set to None for no limit)
IR_REQUEST_DELAY = 2.0           # seconds between requests (be polite)
IR_TIMEOUT       = 15            # request timeout in seconds
IR_USER_AGENT    = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Storage / File Paths
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
RAW_JSON_FILE = DATA_DIR / "lufthansa_raw_dataset.json"
LOG_FILE      = BASE_DIR / "logs" / "scraper.log"

# Create dirs if missing
DATA_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Document Schema Defaults
# ---------------------------------------------------------------------------
DOCUMENT_SCHEMA_VERSION = "1.0"
DEFAULT_COMPANY         = COMPANY

# Allowed document types (maps to source categories in requirements)
DOC_TYPES = {
    "newsapi":    "news_article",
    "rss":        "news_article",
    "reddit":     "community_post",
    "ir_page":    "press_release",
    "newsroom":   "press_release",
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
CHROMA_API_KEY         = os.getenv("CHROMA_API_KEY",  "ck-7sDAfNnynzaFSF29onskcSJUTqbJF6mRJEe4ZaiKm72c")
CHROMA_TENANT          = os.getenv("CHROMA_TENANT",   "3cd57535-7db4-499f-a4c4-91022866ca80")
CHROMA_DATABASE        = os.getenv("CHROMA_DATABASE", "ragsystem")
CHROMA_COLLECTION_NAME = "Knowledge_Database"
CHROMA_DISTANCE_METRIC = "cosine"   # "cosine" | "l2" | "ip"
CHROMA_TOP_K           = 5          # default results per query
CHROMA_PERSIST_DIR = BASE_DIR / "chromadb"
