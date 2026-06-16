# =============================================================================
# scraper.py — All scraping logic for the Lufthansa Strategic Intelligence Agent
#
# Sources:
#   1. NewsAPI       — financial & industry news articles
#   2. RSS Feeds     — aviation, finance, European business news
#   3. Reddit (PRAW) — community sentiment, discussions
#   4. Lufthansa IR  — official press releases & announcements
#
# Each scraper returns a list of raw dicts.
# Call run_all_scrapers() to collect from all sources in one shot.
# =============================================================================

import time
import logging
from datetime import datetime, timedelta, timezone

import requests
import feedparser
import praw
from bs4 import BeautifulSoup

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("scraper")


# ===========================================================================
# 1. NewsAPI Scraper
# ===========================================================================
class NewsAPIScraper:
    """
    Fetches news articles via NewsAPI /v2/everything endpoint.
    Returns articles mentioning any SEARCH_KEYWORDS for Lufthansa.
    """

    def __init__(self):
        self.api_key  = config.NEWS_API_KEY
        self.base_url = config.NEWSAPI_BASE_URL
        self.from_date = (
            datetime.now(timezone.utc) - timedelta(days=config.NEWSAPI_FROM_DAYS)
        ).strftime("%Y-%m-%d")

    def _build_query(self) -> str:
        """Build OR-joined keyword query string."""
        return " OR ".join(f'"{kw}"' for kw in config.SEARCH_KEYWORDS)

    def _fetch_page(self, page: int) -> dict:
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
        return resp.json()

    def scrape(self) -> list[dict]:
        """Return raw article dicts from NewsAPI."""
        results = []
        logger.info("NewsAPI — starting scrape")

        for page in range(1, config.NEWSAPI_MAX_PAGES + 1):
            try:
                data = self._fetch_page(page)
                articles = data.get("articles", [])
                if not articles:
                    logger.info(f"NewsAPI — no more articles at page {page}")
                    break

                for art in articles:
                    results.append({
                        "source_name": "newsapi",
                        "url":         art.get("url", ""),
                        "title":       art.get("title", ""),
                        "content":     (art.get("content") or art.get("description") or ""),
                        "date":        (art.get("publishedAt") or "")[:10],
                        "publisher":   art.get("source", {}).get("name", ""),
                    })

                logger.info(f"NewsAPI — page {page}: {len(articles)} articles")
                time.sleep(1)

            except requests.HTTPError as e:
                logger.error(f"NewsAPI HTTP error page {page}: {e}")
                break
            except Exception as e:
                logger.error(f"NewsAPI unexpected error: {e}")
                break

        logger.info(f"NewsAPI — total collected: {len(results)}")
        return results


# ===========================================================================
# 2. RSS Feed Scraper
# ===========================================================================
class RSSFeedScraper:
    """
    Parses RSS/Atom feeds defined in config.RSS_FEEDS.
    Filters entries whose title or summary mentions a Lufthansa keyword.
    """

    def _is_relevant(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in config.SEARCH_KEYWORDS)

    def _parse_date(self, entry) -> str:
        """Extract date string (YYYY-MM-DD) from feedparser entry."""
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
        """Extract the best available text content from an entry."""
        if hasattr(entry, "content") and entry.content:
            return entry.content[0].get("value", "")
        if hasattr(entry, "summary"):
            return entry.summary
        return ""

    def scrape(self) -> list[dict]:
        results = []
        logger.info("RSS — starting scrape across all feeds")

        for feed_name, feed_url in config.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                count = 0

                for entry in feed.entries[: config.RSS_MAX_ENTRIES_PER_FEED]:
                    title   = getattr(entry, "title", "")
                    content = self._get_content(entry)
                    combined = f"{title} {content}"

                    if not self._is_relevant(combined):
                        continue

                    results.append({
                        "source_name": "rss",
                        "url":         getattr(entry, "link", ""),
                        "title":       title,
                        "content":     content,
                        "date":        self._parse_date(entry),
                        "publisher":   feed_name,
                    })
                    count += 1

                logger.info(f"RSS [{feed_name}] — {count} relevant entries")
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"RSS feed error [{feed_name}]: {e}")

        logger.info(f"RSS — total collected: {len(results)}")
        return results


# ===========================================================================
# 3. Reddit Scraper (PRAW)
# ===========================================================================
class RedditScraper:
    """
    Scrapes Reddit posts and top-level comments from aviation/finance subreddits.
    Uses PRAW (Python Reddit API Wrapper).

    Two strategies per subreddit:
      a) Hot/Top posts (captures trending discussions)
      b) Keyword search (targets direct Lufthansa mentions)
    """

    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT,
            read_only=True,
        )

    def _post_to_dict(self, post, subreddit_name: str) -> dict | None:
        """Convert a PRAW submission to a raw dict."""
        if post.score < config.REDDIT_MIN_SCORE:
            return None

        # Combine title + selftext for content
        body = post.selftext.strip() if post.selftext else ""
        combined = f"{post.title}. {body}".strip()

        # Collect top comments
        comments_text = []
        try:
            post.comments.replace_more(limit=0)
            for comment in list(post.comments)[:10]:
                if hasattr(comment, "body") and comment.body not in ("[deleted]", "[removed]"):
                    comments_text.append(comment.body)
        except Exception:
            pass

        if comments_text:
            combined += "\n\nComments:\n" + "\n".join(comments_text)

        return {
            "source_name": "reddit",
            "url":         f"https://reddit.com{post.permalink}",
            "title":       post.title,
            "content":     combined,
            "date":        datetime.utcfromtimestamp(post.created_utc).strftime("%Y-%m-%d"),
            "publisher":   f"r/{subreddit_name}",
        }

    def _scrape_subreddit(self, subreddit_name: str) -> list[dict]:
        results = []
        try:
            sub = self.reddit.subreddit(subreddit_name)

            # Strategy A: Hot posts
            for post in sub.hot(limit=config.REDDIT_POST_LIMIT):
                row = self._post_to_dict(post, subreddit_name)
                if row:
                    results.append(row)

            # Strategy B: Keyword search within subreddit
            for keyword in config.SEARCH_KEYWORDS[:3]:    # top 3 keywords only
                for post in sub.search(keyword, limit=config.REDDIT_SEARCH_LIMIT,
                                        time_filter=config.REDDIT_TIME_FILTER):
                    row = self._post_to_dict(post, subreddit_name)
                    if row:
                        results.append(row)

            time.sleep(1)   # respect Reddit rate limits

        except Exception as e:
            logger.error(f"Reddit error [r/{subreddit_name}]: {e}")

        return results

    def scrape(self) -> list[dict]:
        results = []
        logger.info("Reddit — starting scrape across all subreddits")

        for sub_name in config.REDDIT_SUBREDDITS:
            sub_results = self._scrape_subreddit(sub_name)
            logger.info(f"Reddit [r/{sub_name}] — {len(sub_results)} posts")
            results.extend(sub_results)

        logger.info(f"Reddit — total collected: {len(results)}")
        return results


# ===========================================================================
# 4. Lufthansa Investor Relations Scraper
# ===========================================================================
class LufthansaIRScraper:
    """
    Scrapes official Lufthansa press releases from the Investor Relations page.
    Source: https://investor-relations.lufthansa.com/en/press-releases

    Falls back to the public Newsroom if IR page structure changes.
    """

    HEADERS = {
        "User-Agent":      config.IR_USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept":          "text/html,application/xhtml+xml",
    }

    def _fetch_html(self, url: str) -> BeautifulSoup | None:
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=config.IR_TIMEOUT)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            logger.error(f"IR scraper fetch error [{url}]: {e}")
            return None

    def _extract_article_text(self, url: str) -> str:
        """Follow a press release link and extract its main body text."""
        soup = self._fetch_html(url)
        if not soup:
            return ""

        # Common selectors for press release body text
        for selector in ["article", "main", ".press-release-content",
                          ".content-body", ".article-body", "#content"]:
            tag = soup.select_one(selector)
            if tag:
                return tag.get_text(separator=" ", strip=True)

        # Fallback: all paragraph text
        return " ".join(p.get_text(strip=True) for p in soup.find_all("p"))

    # URL path segments that are definitively NOT press releases — skip these
    _SKIP_PATH_SEGMENTS = (
        "/events/", "/service/", "/fileadmin/", "/disclaimer",
        "/accessibility", "/contact", "/mailing", "/shareholders",
        "/annual-general", "/financial-calendar", "/suppliers",
        "/responsibility", "/en/home", "/home.html",
        "/investor-relations.html",   # IR root landing page
    )

    # Domains that return 403 for automated requests — skip entirely
    _BLOCKED_DOMAINS = (
        "www.lufthansagroup.com",
    )

    def _is_press_release_url(self, href: str) -> bool:
        """
        Return False only for URLs we know are navigation/service/binary pages.
        Default-ALLOW: if we can't classify a URL as bad, we follow it.
        This is safer than a whitelist when URL patterns are unknown.
        """
        path = href.lower()

        # Reject known non-article domains (return 403)
        if any(domain in path for domain in self._BLOCKED_DOMAINS):
            return False

        # Reject known navigation / service paths
        if any(skip in path for skip in self._SKIP_PATH_SEGMENTS):
            return False

        # Reject binary / download file extensions
        if any(path.endswith(ext) for ext in (".pdf", ".zip", ".xlsx", ".docx", ".pptx")):
            return False

        # Default: allow — let the fetcher decide if it's useful content
        return True

    def _parse_ir_page(self, soup: BeautifulSoup) -> list[dict]:
        """Extract press release links and metadata from IR listing page."""
        results = []

        # Derive base domain from config so it stays in sync
        from urllib.parse import urlparse
        base_domain = urlparse(config.LUFTHANSA_IR_URL).scheme + "://" + urlparse(config.LUFTHANSA_IR_URL).netloc

        # Lufthansa IR typically lists releases as <article> or <li> items with <a> links
        candidates = (
            soup.select("article a[href]")
            or soup.select(".press-release-item a[href]")
            or soup.select(".news-list a[href]")
            or soup.select("li a[href]")    # broad fallback
        )

        seen_urls = set()
        for a_tag in candidates:
            href  = a_tag.get("href", "")
            title = a_tag.get_text(strip=True)

            # Build absolute URL using the correct base domain from config
            if href.startswith("/"):
                href = base_domain + href
            if not href.startswith("http") or href in seen_urls:
                continue
            if len(title) < 10:     # skip nav links
                continue

            # Only follow URLs that look like actual press releases
            if not self._is_press_release_url(href):
                logger.debug(f"IR scraper skipping non-article URL: {href}")
                continue

            seen_urls.add(href)

            # Try to find a date near this link
            parent = a_tag.find_parent(["article", "li", "div"])
            date_str = datetime.now().strftime("%Y-%m-%d")
            if parent:
                date_tag = parent.find(["time", "span"], class_=lambda c: c and "date" in c.lower())
                if date_tag:
                    raw = date_tag.get("datetime") or date_tag.get_text(strip=True)
                    try:
                        # Handle common formats
                        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%B %d, %Y", "%d %B %Y"):
                            try:
                                date_str = datetime.strptime(raw[:10], fmt).strftime("%Y-%m-%d")
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass

            # Fetch full article text (with polite delay)
            time.sleep(config.IR_REQUEST_DELAY)
            body = self._extract_article_text(href)

            results.append({
                "source_name": "ir_page",
                "url":         href,
                "title":       title,
                "content":     body or title,
                "date":        date_str,
                "publisher":   "Lufthansa Investor Relations",
            })

        return results

    def scrape(self) -> list[dict]:
        results = []
        max_articles = config.IR_MAX_ARTICLES   # None = no limit
        logger.info(f"Lufthansa IR — starting scrape (limit: {max_articles or 'unlimited'})")

        for page_num in range(1, config.IR_MAX_PAGES + 1):
            # Stop early if we've already hit the article cap
            if max_articles and len(results) >= max_articles:
                break

            # Many IR pages use ?page=N or ?start=N pagination
            url = config.LUFTHANSA_IR_URL
            if page_num > 1:
                url = f"{url}?page={page_num}"

            soup = self._fetch_html(url)
            if not soup:
                break

            page_results = self._parse_ir_page(soup)
            if not page_results:
                logger.info(f"IR page {page_num}: no items found, stopping pagination")
                break

            # Trim to stay within the cap
            if max_articles:
                remaining = max_articles - len(results)
                page_results = page_results[:remaining]

            logger.info(f"IR page {page_num}: {len(page_results)} press releases")
            results.extend(page_results)
            time.sleep(config.IR_REQUEST_DELAY)

        logger.info(f"Lufthansa IR — total collected: {len(results)}")
        return results


# ===========================================================================
# Orchestrator — run all scrapers
# ===========================================================================
def run_all_scrapers() -> dict[str, list[dict]]:
    """
    Run every scraper and return results grouped by source.

    Returns:
        {
            "newsapi": [...],
            "rss":     [...],
            "reddit":  [...],
            "ir_page": [...],
        }
    """
    logger.info("=" * 60)
    logger.info("Starting full scraping run for: Lufthansa")
    logger.info("=" * 60)

    results = {}

    scrapers = [
        ("newsapi", NewsAPIScraper()),
        ("rss",     RSSFeedScraper()),
        # ("reddit",  RedditScraper()),
        ("ir_page", LufthansaIRScraper()),
    ]

    for name, scraper in scrapers:
        try:
            logger.info(f"\n--- Running {name} scraper ---")
            results[name] = scraper.scrape()
        except Exception as e:
            logger.error(f"Scraper [{name}] failed entirely: {e}")
            results[name] = []

    total = sum(len(v) for v in results.values())
    logger.info(f"\nScraping complete. Total raw documents: {total}")
    for name, docs in results.items():
        logger.info(f"  {name}: {len(docs)} documents")

    return results


# ===========================================================================
# Entry point (run standalone to test)
# ===========================================================================
if __name__ == "__main__":
    raw_data = run_all_scrapers()
    print(f"\nTotal raw documents collected: {sum(len(v) for v in raw_data.values())}")
