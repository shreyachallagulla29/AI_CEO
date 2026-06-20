# =============================================================================
# competitor_scraper.py — Scraping layer for Lufthansa's key competitors
#
# Competitors covered:
#   - Air India       (Tata Group, India)
#   - Ryanair         (low-cost, Europe)
#   - United Airlines (US major carrier)
#
# Sources per company (same four as scraper.py):
#   1. NewsAPI       — financial & industry news articles
#   2. RSS Feeds     — aviation, finance, business press
#   3. Reddit (PRAW) — community sentiment and discussions
#   4. IR / Newsroom — official press releases
#
# Output format matches scraper.py so the same pipeline (clean_storage →
# processor → embedder → vector_store) handles competitor data unchanged.
#
# Quick usage:
#   from competitor_scraper import run_competitor_scrapers
#   data = run_competitor_scrapers()
#   # → {"Air India": {"newsapi": [...], "rss": [...], ...}, "Ryanair": {...}, ...}
#
#   # Or scrape a single company:
#   from competitor_scraper import scrape_company, COMPETITORS
#   data = scrape_company(COMPETITORS["ryanair"])
# =============================================================================

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import requests
import feedparser
from bs4 import BeautifulSoup

import config

logger = logging.getLogger("competitor_scraper")


# =============================================================================
# Company Configuration
# =============================================================================

@dataclass
class CompanyConfig:
    """All scraping parameters for one competitor airline."""

    name:        str                       # display name, e.g. "Air India"
    key:         str                       # snake_case key, e.g. "air_india"
    keywords:    list[str]                 # search terms for NewsAPI / RSS / Reddit filtering
    ticker:      str                       # stock ticker (used as fallback keyword)
    rss_feeds:   dict[str, str]            # {feed_label: feed_url}
    reddit_subs: list[str]                 # subreddits to scrape
    ir_url:      str                       # investor relations / newsroom listing URL
    ir_publisher: str                      # label for press release publisher field
    ir_skip_paths: tuple[str, ...] = field(default_factory=tuple)   # paths to skip in IR scraper
    ir_blocked_domains: tuple[str, ...] = field(default_factory=tuple)


# =============================================================================
# Per-company configurations
# =============================================================================

# Shared aviation/finance RSS feeds (relevant to all airlines)
_SHARED_RSS = {
    "Reuters Transport":   "https://feeds.reuters.com/reuters/industrialsNews",
    "BBC Business":        "https://feeds.bbci.co.uk/news/business/rss.xml",
    "Simple Flying":       "https://simpleflying.com/feed/",
    "Aviation Week":       "https://aviationweek.com/rss.xml",
    "Air Transport World": "https://atwonline.com/rss.xml",
    "FlightGlobal":        "https://www.flightglobal.com/rss/",
    "Yahoo Finance":       "https://finance.yahoo.com/news/rssindex",
}

COMPETITORS: dict[str, CompanyConfig] = {

    # ── Air India ─────────────────────────────────────────────────────────────
    "air_india": CompanyConfig(
        name    = "Air India",
        key     = "air_india",
        ticker  = "AIRINDIA",
        keywords = [
            "Air India",
            "Air India Limited",
            "Air India Express",
            "Tata Airlines",
            "IndiGo",          # key domestic rival context
            "AIRL.NS",
        ],
        rss_feeds = {
            **_SHARED_RSS,
            "The Hindu Business": "https://www.thehindu.com/business/Industry/?service=rss",
            "Mint Aviation":      "https://www.livemint.com/rss/aviation.xml",
            "Business Standard":  "https://www.business-standard.com/rss/latest.rss",
            "Economic Times":     "https://economictimes.indiatimes.com/industry/transportation/airlines-/-aviation/rssfeeds/28575842.cms",
        },
        reddit_subs = [
            "aviation",
            "flights",
            "india",
            "IndiaSpeaks",
            "frequentflyers",
            "awardtravel",
        ],
        ir_url       = "https://www.airindia.com/in/en/press-releases.html",
        ir_publisher = "Air India Press Office",
        ir_skip_paths = (
            "/baggage", "/services", "/faq", "/contact",
            "/check-in", "/manage-booking", "/offers",
        ),
        ir_blocked_domains = (),
    ),

    # ── Ryanair ───────────────────────────────────────────────────────────────
    "ryanair": CompanyConfig(
        name    = "Ryanair",
        key     = "ryanair",
        ticker  = "RYA.IR",
        keywords = [
            "Ryanair",
            "Ryanair Holdings",
            "Michael O'Leary",
            "RYA.IR",
            "Ryanair DAC",
        ],
        rss_feeds = {
            **_SHARED_RSS,
            "Ryanair Investor News": "https://investor.ryanair.com/feed/",
            "Irish Times Business":  "https://www.irishtimes.com/business/?page=1",
            "The Points Guy":        "https://thepointsguy.com/feed/",
            "Handelsblatt English":  "https://www.handelsblatt.com/contentexport/feed/english",
        },
        reddit_subs = [
            "Ryanair",
            "aviation",
            "flights",
            "europe",
            "frequentflyers",
            "travel",
        ],
        ir_url       = "https://investor.ryanair.com/news/",
        ir_publisher = "Ryanair Investor Relations",
        ir_skip_paths = (
            "/agm/", "/governance/", "/financial-reports/", "/contact",
            "/shareholder", "/dividends", "/share-price",
        ),
        ir_blocked_domains = (),
    ),

    # ── United Airlines ───────────────────────────────────────────────────────
    "united_airlines": CompanyConfig(
        name    = "United Airlines",
        key     = "united_airlines",
        ticker  = "UAL",
        keywords = [
            "United Airlines",
            "United Continental",
            "United Airlines Holdings",
            "UAL",
            "United MileagePlus",
        ],
        rss_feeds = {
            **_SHARED_RSS,
            "United Airlines Hub":   "https://hub.united.com/feed/",
            "The Points Guy":        "https://thepointsguy.com/feed/",
            "View from the Wing":    "https://viewfromthewing.com/feed/",
            "One Mile at a Time":    "https://onemileatatime.com/feed/",
            "WSJ Airlines":          "https://feeds.a.dj.com/rss/RSSWSJD.xml",
        },
        reddit_subs = [
            "unitedairlines",
            "aviation",
            "flights",
            "frequentflyers",
            "awardtravel",
            "stocks",
        ],
        ir_url       = "https://ir.united.com/news-releases/news-releases",
        ir_publisher = "United Airlines Investor Relations",
        ir_skip_paths = (
            "/governance", "/financial-information", "/stock-information",
            "/sec-filings", "/committee", "/contact",
        ),
        ir_blocked_domains = (),
    ),
}


# =============================================================================
# Scraper 1 — NewsAPI
# =============================================================================

class CompetitorNewsAPIScraper:
    """
    NewsAPI scraper parameterized for a specific competitor.
    Builds a keyword query from the company's keyword list.
    """

    def __init__(self, company: CompanyConfig):
        self.company   = company
        self.api_key   = config.NEWS_API_KEY
        self.base_url  = config.NEWSAPI_BASE_URL
        self.from_date = (
            datetime.now(timezone.utc) - timedelta(days=config.NEWSAPI_FROM_DAYS)
        ).strftime("%Y-%m-%d")

    def _build_query(self) -> str:
        return " OR ".join(f'"{kw}"' for kw in self.company.keywords)

    def scrape(self) -> list[dict]:
        results = []
        logger.info(f"NewsAPI [{self.company.name}] — starting scrape")

        for page in range(1, config.NEWSAPI_MAX_PAGES + 1):
            try:
                params = {
                    "apiKey":   self.api_key,
                    "q":        self._build_query(),
                    "language": config.NEWSAPI_LANGUAGE,
                    "sortBy":   config.NEWSAPI_SORT_BY,
                    "pageSize": config.NEWSAPI_PAGE_SIZE,
                    "from":     self.from_date,
                    "page":     page,
                }
                resp = requests.get(self.base_url, params=params, timeout=15)
                resp.raise_for_status()
                data     = resp.json()
                articles = data.get("articles", [])

                if not articles:
                    logger.info(f"NewsAPI [{self.company.name}] — no more articles at page {page}")
                    break

                for art in articles:
                    results.append({
                        "source_name": "newsapi",
                        "url":         art.get("url", ""),
                        "title":       art.get("title", ""),
                        "content":     (art.get("content") or art.get("description") or ""),
                        "date":        (art.get("publishedAt") or "")[:10],
                        "publisher":   art.get("source", {}).get("name", ""),
                        "company":     self.company.name,   # ← passed to create_document()
                    })

                logger.info(f"NewsAPI [{self.company.name}] page {page}: {len(articles)} articles")
                time.sleep(1)

            except requests.HTTPError as e:
                logger.error(f"NewsAPI [{self.company.name}] HTTP error: {e}")
                break
            except Exception as e:
                logger.error(f"NewsAPI [{self.company.name}] error: {e}")
                break

        logger.info(f"NewsAPI [{self.company.name}] — total: {len(results)}")
        return results


# =============================================================================
# Scraper 2 — RSS Feeds
# =============================================================================

class CompetitorRSSFeedScraper:
    """
    RSS scraper parameterized for a specific competitor.
    Filters entries that mention any of the company's keywords.
    """

    def __init__(self, company: CompanyConfig):
        self.company = company

    def _is_relevant(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.company.keywords)

    def _parse_date(self, entry) -> str:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                return datetime(*entry.published_parsed[:3]).strftime("%Y-%m-%d")
            except Exception:
                pass
        if hasattr(entry, "updated_parsed") and entry.updated_parsed:
            try:
                return datetime(*entry.updated_parsed[:3]).strftime("%Y-%m-%d")
            except Exception:
                pass
        return datetime.now().strftime("%Y-%m-%d")

    def _get_content(self, entry) -> str:
        if hasattr(entry, "content") and entry.content:
            return entry.content[0].get("value", "")
        return getattr(entry, "summary", "")

    def scrape(self) -> list[dict]:
        results = []
        logger.info(f"RSS [{self.company.name}] — scraping {len(self.company.rss_feeds)} feeds")

        for feed_name, feed_url in self.company.rss_feeds.items():
            try:
                feed  = feedparser.parse(feed_url)
                count = 0

                for entry in feed.entries[: config.RSS_MAX_ENTRIES_PER_FEED]:
                    title   = getattr(entry, "title", "")
                    content = self._get_content(entry)

                    if not self._is_relevant(f"{title} {content}"):
                        continue

                    results.append({
                        "source_name": "rss",
                        "url":         getattr(entry, "link", ""),
                        "title":       title,
                        "content":     content,
                        "date":        self._parse_date(entry),
                        "publisher":   feed_name,
                        "company":     self.company.name,
                    })
                    count += 1

                logger.info(f"RSS [{self.company.name}] [{feed_name}] — {count} relevant entries")
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"RSS [{self.company.name}] [{feed_name}] error: {e}")

        logger.info(f"RSS [{self.company.name}] — total: {len(results)}")
        return results


# =============================================================================
# Scraper 3 — Reddit (PRAW)
# =============================================================================

class CompetitorRedditScraper:
    """
    Reddit scraper parameterized for a specific competitor.
    Searches within relevant subreddits using the company's keywords.
    """

    def __init__(self, company: CompanyConfig):
        self.company = company
        import praw as _praw
        self.reddit = _praw.Reddit(
            client_id     = config.REDDIT_CLIENT_ID,
            client_secret = config.REDDIT_CLIENT_SECRET,
            user_agent    = config.REDDIT_USER_AGENT,
            read_only     = True,
        )

    def _post_to_dict(self, post, subreddit_name: str) -> dict | None:
        if post.score < config.REDDIT_MIN_SCORE:
            return None

        body     = post.selftext.strip() if post.selftext else ""
        combined = f"{post.title}. {body}".strip()

        # Must mention at least one company keyword (avoid off-topic hot posts)
        if not any(kw.lower() in combined.lower() for kw in self.company.keywords):
            return None

        # Collect top comments
        try:
            post.comments.replace_more(limit=0)
            comments = [
                c.body for c in list(post.comments)[:10]
                if hasattr(c, "body") and c.body not in ("[deleted]", "[removed]")
            ]
            if comments:
                combined += "\n\nComments:\n" + "\n".join(comments)
        except Exception:
            pass

        return {
            "source_name": "reddit",
            "url":         f"https://reddit.com{post.permalink}",
            "title":       post.title,
            "content":     combined,
            "date":        datetime.utcfromtimestamp(post.created_utc).strftime("%Y-%m-%d"),
            "publisher":   f"r/{subreddit_name}",
            "company":     self.company.name,
        }

    def _scrape_subreddit(self, subreddit_name: str) -> list[dict]:
        results = []
        try:
            sub = self.reddit.subreddit(subreddit_name)

            # Hot posts (may mention the company if they're discussing it)
            for post in sub.hot(limit=config.REDDIT_POST_LIMIT):
                row = self._post_to_dict(post, subreddit_name)
                if row:
                    results.append(row)

            # Keyword search — more targeted
            for keyword in self.company.keywords[:3]:
                for post in sub.search(
                    keyword,
                    limit=config.REDDIT_SEARCH_LIMIT,
                    time_filter=config.REDDIT_TIME_FILTER,
                ):
                    row = self._post_to_dict(post, subreddit_name)
                    if row:
                        results.append(row)

            time.sleep(1)

        except Exception as e:
            logger.error(f"Reddit [{self.company.name}] [r/{subreddit_name}] error: {e}")

        return results

    def scrape(self) -> list[dict]:
        results = []
        logger.info(f"Reddit [{self.company.name}] — scraping {len(self.company.reddit_subs)} subreddits")

        for sub_name in self.company.reddit_subs:
            sub_results = self._scrape_subreddit(sub_name)
            logger.info(f"Reddit [{self.company.name}] [r/{sub_name}] — {len(sub_results)} posts")
            results.extend(sub_results)

        logger.info(f"Reddit [{self.company.name}] — total: {len(results)}")
        return results


# =============================================================================
# Scraper 4 — Investor Relations / Newsroom
# =============================================================================

class CompetitorIRScraper:
    """
    Generic IR / newsroom scraper for competitor airlines.
    Follows links from a listing page to extract press release text.

    Designed to work with Ryanair, Air India, and United Airlines IR pages.
    URL filtering is default-allow (same safe strategy as LufthansaIRScraper).
    """

    HEADERS = {
        "User-Agent":      config.IR_USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept":          "text/html,application/xhtml+xml",
    }

    def __init__(self, company: CompanyConfig):
        self.company = company

    def _fetch_html(self, url: str) -> BeautifulSoup | None:
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=config.IR_TIMEOUT)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            logger.error(f"IR [{self.company.name}] fetch error [{url}]: {e}")
            return None

    def _extract_article_text(self, url: str) -> str:
        """Follow a press release link and extract its main body text."""
        soup = self._fetch_html(url)
        if not soup:
            return ""
        for selector in [
            "article", "main", ".press-release-content",
            ".content-body", ".article-body", ".news-detail",
            ".release-body", "#content", "#main-content",
        ]:
            tag = soup.select_one(selector)
            if tag:
                return tag.get_text(separator=" ", strip=True)
        return " ".join(p.get_text(strip=True) for p in soup.find_all("p"))

    def _is_press_release_url(self, href: str) -> bool:
        """
        Default-allow: return True unless the URL matches a known bad pattern.
        Same strategy as LufthansaIRScraper._is_press_release_url().
        """
        path = href.lower()

        if any(domain in path for domain in self.company.ir_blocked_domains):
            return False

        if any(seg in path for seg in self.company.ir_skip_paths):
            return False

        if any(path.endswith(ext) for ext in (".pdf", ".zip", ".xlsx", ".docx", ".pptx")):
            return False

        return True

    def _parse_listing_page(self, soup: BeautifulSoup) -> list[dict]:
        """Extract press release links and fetch each article's text."""
        results  = []
        parsed   = urlparse(self.company.ir_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Try progressively broader selectors until we find something
        candidates = (
            soup.select("article a[href]")
            or soup.select(".press-release-item a[href]")
            or soup.select(".news-item a[href]")
            or soup.select(".news-list a[href]")
            or soup.select(".release-list a[href]")
            or soup.select("li a[href]")
        )

        seen_urls = set()
        for a_tag in candidates:
            href  = a_tag.get("href", "").strip()
            title = a_tag.get_text(strip=True)

            # Build absolute URL
            if href.startswith("/"):
                href = base_url + href
            elif not href.startswith("http"):
                continue

            if href in seen_urls or len(title) < 10:
                continue
            if not self._is_press_release_url(href):
                logger.debug(f"IR [{self.company.name}] skipping: {href}")
                continue

            seen_urls.add(href)

            # Try to find a date near this link
            parent   = a_tag.find_parent(["article", "li", "div", "tr"])
            date_str = datetime.now().strftime("%Y-%m-%d")
            if parent:
                date_tag = parent.find(
                    ["time", "span"],
                    class_=lambda c: c and "date" in c.lower()
                )
                if not date_tag:
                    date_tag = parent.find("time")
                if date_tag:
                    raw = date_tag.get("datetime") or date_tag.get_text(strip=True)
                    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%B %d, %Y", "%d %B %Y", "%Y-%m-%dT%H:%M:%S"):
                        try:
                            date_str = datetime.strptime(raw[:19], fmt).strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            continue

            time.sleep(config.IR_REQUEST_DELAY)
            body = self._extract_article_text(href)

            results.append({
                "source_name": "ir_page",
                "url":         href,
                "title":       title,
                "content":     body or title,
                "date":        date_str,
                "publisher":   self.company.ir_publisher,
                "company":     self.company.name,
            })

        return results

    def scrape(self) -> list[dict]:
        results     = []
        max_articles = config.IR_MAX_ARTICLES
        logger.info(f"IR [{self.company.name}] — starting (limit: {max_articles or 'unlimited'})")

        for page_num in range(1, config.IR_MAX_PAGES + 1):
            if max_articles and len(results) >= max_articles:
                break

            url = self.company.ir_url
            if page_num > 1:
                url = f"{url}?page={page_num}"

            soup = self._fetch_html(url)
            if not soup:
                break

            page_results = self._parse_listing_page(soup)
            if not page_results:
                logger.info(f"IR [{self.company.name}] page {page_num}: no items, stopping")
                break

            if max_articles:
                remaining    = max_articles - len(results)
                page_results = page_results[:remaining]

            logger.info(f"IR [{self.company.name}] page {page_num}: {len(page_results)} releases")
            results.extend(page_results)
            time.sleep(config.IR_REQUEST_DELAY)

        logger.info(f"IR [{self.company.name}] — total: {len(results)}")
        return results


# =============================================================================
# Per-company orchestrator
# =============================================================================

def scrape_company(company: CompanyConfig, skip_reddit: bool = True) -> dict[str, list[dict]]:
    """
    Run all scrapers for one competitor company.

    Args:
        company:      CompanyConfig instance (from COMPETITORS dict)
        skip_reddit:  Set True (default) when Reddit credentials are not configured

    Returns:
        {"newsapi": [...], "rss": [...], "reddit": [...], "ir_page": [...]}
        (same structure as scraper.run_all_scrapers())
    """
    logger.info(f"{'='*60}")
    logger.info(f"Scraping competitor: {company.name}")
    logger.info(f"{'='*60}")

    scrapers = [
        ("newsapi", CompetitorNewsAPIScraper(company)),
        ("rss",     CompetitorRSSFeedScraper(company)),
        ("ir_page", CompetitorIRScraper(company)),
    ]

    if not skip_reddit:
        scrapers.append(("reddit", CompetitorRedditScraper(company)))

    results: dict[str, list[dict]] = {}
    for name, scraper in scrapers:
        try:
            logger.info(f"  --- {name} ---")
            results[name] = scraper.scrape()
        except Exception as e:
            logger.error(f"[{company.name}] [{name}] scraper failed: {e}")
            results[name] = []

    total = sum(len(v) for v in results.values())
    logger.info(f"{company.name} scrape complete — {total} raw documents")
    for src, docs in results.items():
        logger.info(f"  {src}: {len(docs)}")

    return results


# =============================================================================
# Full competitor run
# =============================================================================

def run_competitor_scrapers(
    company_keys: list[str] | None = None,
    skip_reddit: bool = True,
) -> dict[str, dict[str, list[dict]]]:
    """
    Scrape all (or selected) competitors.

    Args:
        company_keys: List of keys from COMPETITORS to run, e.g. ["ryanair"].
                      Pass None to run all three.
        skip_reddit:  Skip Reddit scraping (default True — needs valid PRAW credentials)

    Returns:
        {
            "Air India":       {"newsapi": [...], "rss": [...], "ir_page": [...]},
            "Ryanair":         {"newsapi": [...], "rss": [...], "ir_page": [...]},
            "United Airlines": {"newsapi": [...], "rss": [...], "ir_page": [...]},
        }

    Pipeline integration example:
        from competitor_scraper import run_competitor_scrapers
        from clean_storage import build_documents, merge_and_save

        all_competitor_data = run_competitor_scrapers()
        for company_name, raw_data in all_competitor_data.items():
            docs = build_documents(raw_data)
            merge_and_save(docs)
    """
    keys_to_run = company_keys or list(COMPETITORS.keys())

    all_results: dict[str, dict[str, list[dict]]] = {}
    for key in keys_to_run:
        company = COMPETITORS.get(key)
        if not company:
            logger.warning(f"Unknown competitor key: '{key}'. Available: {list(COMPETITORS.keys())}")
            continue
        all_results[company.name] = scrape_company(company, skip_reddit=skip_reddit)

    grand_total = sum(
        sum(len(v) for v in sources.values())
        for sources in all_results.values()
    )
    logger.info(f"\nAll competitor scraping complete — {grand_total} total raw documents")
    return all_results


# =============================================================================
# Entry point (standalone test)
# =============================================================================

if __name__ == "__main__":
    import json
    from clean_storage import build_documents, dataset_stats

    # Scrape all competitors (Reddit off by default — needs credentials)
    all_data = run_competitor_scrapers()

    for company_name, raw_data in all_data.items():
        print(f"\n{'='*50}")
        print(f"  {company_name}")
        print(f"{'='*50}")
        for src, docs in raw_data.items():
            print(f"  {src}: {len(docs)} raw items")

        documents = build_documents(raw_data)
        stats     = dataset_stats(documents)
        print(f"  → {stats['total']} valid documents after cleaning")
