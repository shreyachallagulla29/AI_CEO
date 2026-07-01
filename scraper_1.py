# =============================================================================
# scraper_1.py — Deep web scraper using Hyperbrowser (JS-rendered pages)
#
# Uses HyperbrowserCrawlTool from langchain_hyperbrowser to scrape pages that
# require JavaScript rendering (SPAs, dynamic content).
# =============================================================================

import logging
import time
from collections import deque
from datetime import datetime
from urllib.parse import urljoin, urlparse

import config

logger = logging.getLogger("scraper_1")


# ===========================================================================
# URL helpers
# ===========================================================================

def _same_domain(base_url: str, candidate: str) -> bool:
    """
    Return True if `candidate` shares the same registered domain as `base_url`.
    Strips leading 'www.' so that www.example.com and example.com match.
    """
    base_netloc = urlparse(base_url).netloc.lower().lstrip("www.")
    cand_netloc = urlparse(candidate).netloc.lower().lstrip("www.")
    return cand_netloc == base_netloc or cand_netloc.endswith("." + base_netloc)


# File extensions and URL prefixes that are never content pages
_SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp",
    ".css", ".js", ".json", ".xml",
    ".pdf", ".zip", ".gz", ".tar",
    ".mp4", ".mp3", ".wav", ".ogg",
    ".woff", ".woff2", ".ttf", ".eot",
)
_SKIP_PREFIXES = ("mailto:", "javascript:", "tel:", "ftp:", "#")


def _is_content_url(url: str) -> bool:
    """Return True if the URL looks like an HTML content page worth scraping."""
    lower = url.lower()
    if any(lower.endswith(ext) for ext in _SKIP_EXTENSIONS):
        return False
    if any(lower.startswith(pfx) for pfx in _SKIP_PREFIXES):
        return False
    if "#" in url and not url.endswith("#"):
        # Fragment-only anchor — same page, different section; skip
        url_no_frag = url.split("#")[0]
        return bool(url_no_frag)
    return True


def _resolve_url(base: str, href: str) -> str | None:
    """
    Resolve a potentially relative href against base.
    Returns None if the result is not a valid http/https URL.
    """
    try:
        resolved = urljoin(base, href)
        scheme = urlparse(resolved).scheme
        if scheme in ("http", "https"):
            return resolved
    except Exception:
        pass
    return None


# ===========================================================================
# Metadata extraction helpers
# ===========================================================================

def _flatten_metadata(meta: dict) -> dict:
    """
    Hyperbrowser metadata values can be strings OR lists (e.g. {"title": ["Page"]}).
    Flatten lists to their first element so we can do simple string lookups.
    """
    flat = {}
    for k, v in (meta or {}).items():
        flat[k] = v[0] if isinstance(v, list) and v else (v or "")
    return flat


def _extract_title(flat_meta: dict, url: str) -> str:
    """Pick the best title from metadata, falling back to the URL."""
    for key in ("title", "og:title", "twitter:title", "DC.title"):
        val = flat_meta.get(key, "")
        if val and isinstance(val, str):
            return val.strip()
    # Last resort: the URL path tail
    path = urlparse(url).path.rstrip("/").split("/")[-1]
    return path or url


def _extract_date(flat_meta: dict) -> str:
    """
    Try common metadata keys for publication date.
    Falls back to today's date if nothing is found.
    """
    date_keys = [
        "article:published_time",
        "og:article:published_time",
        "datePublished",
        "date",
        "published_date",
        "publishedTime",
        "DC.date",
        "article.published_time",
        "pubdate",
    ]
    for key in date_keys:
        val = flat_meta.get(key, "")
        if val and isinstance(val, str) and len(val) >= 10:
            try:
                # Validate the first 10 chars are a real date
                datetime.strptime(val[:10], "%Y-%m-%d")
                return val[:10]
            except ValueError:
                continue
    return datetime.now().strftime("%Y-%m-%d")


# ===========================================================================
# Core scraper class
# ===========================================================================

class HyperbrowserDeepScraper:
    """
    Depth-controlled web scraper built on HyperbrowserCrawlTool.

    Algorithm:
      1. Enqueue the root URL at depth 0.
      2. Dequeue a URL, call HyperbrowserCrawlTool with max_pages=1.
      3. Save the page content as a raw dict.
      4. If current depth < target depth, extract all same-domain links
         and enqueue them at depth+1.
      5. Repeat until the queue is empty.

    The 'links' scrape format is explicitly requested so that CrawledPage.links
    is populated even when the default format ('markdown') would omit them.
    """

    def __init__(
        self,
        api_key: str = None,
        request_delay: float = None,
        max_links_per_page: int = None,
    ):
        """
        Args:
            api_key:            Hyperbrowser API key (falls back to config).
            request_delay:      Seconds between requests (falls back to config).
            max_links_per_page: Max links to follow per page to prevent explosion.
        """
        from langchain_hyperbrowser import HyperbrowserCrawlTool

        self.api_key            = api_key or config.HYPERBROWSER_API_KEY
        self.request_delay      = request_delay if request_delay is not None else config.HYPERBROWSER_REQUEST_DELAY
        self.max_links_per_page = max_links_per_page or config.HYPERBROWSER_MAX_LINKS_PER_PAGE

        # Lazy-import to avoid import errors if langchain_hyperbrowser not installed
        self._tool = HyperbrowserCrawlTool(api_key=self.api_key)

    # -----------------------------------------------------------------------
    # Low-level: single page fetch
    # -----------------------------------------------------------------------

    def _fetch_page(self, url: str):
        """
        Scrape a single URL using HyperbrowserCrawlTool (max_pages=1).

        Returns the first CrawledPage object, or None on failure.
        The 'links' format is included so CrawledPage.links is populated.
        """
        try:
            result = self._tool.invoke({
                "url":            url,
                "max_pages":      1,
                "scrape_options": {"formats": ["markdown", "links"]},
            })

            error = result.get("error")
            if error:
                logger.warning(f"Hyperbrowser error for {url}: {error}")
                return None

            pages = result.get("data") or []
            if not pages:
                logger.warning(f"No data returned for {url}")
                return None

            page = pages[0]

            if getattr(page, "status", None) == "failed" or getattr(page, "error", None):
                logger.warning(f"Page scrape failed: {url} — {getattr(page, 'error', 'unknown')}")
                return None

            return page

        except Exception as exc:
            logger.error(f"Exception scraping {url}: {exc}")
            return None

    # -----------------------------------------------------------------------
    # Conversion: CrawledPage → project raw dict
    # -----------------------------------------------------------------------

    def _to_raw(self, page, company: str) -> dict:
        """
        Convert a CrawledPage to the standard raw scraper dict used by
        clean_storage.build_documents().

        Standard keys: source_name, url, title, content, date, publisher, company
        """
        flat_meta = _flatten_metadata(getattr(page, "metadata", None))
        title     = _extract_title(flat_meta, page.url)
        date      = _extract_date(flat_meta)
        content   = getattr(page, "markdown", "") or ""
        publisher = urlparse(page.url).netloc

        return {
            "source_name": "hyperbrowser",
            "url":         page.url,
            "title":       title,
            "content":     content,
            "date":        date,
            "publisher":   publisher,
            "company":     company,
        }

    # -----------------------------------------------------------------------
    # BFS traversal
    # -----------------------------------------------------------------------

    def scrape_target(
        self,
        start_url: str,
        depth: int,
        company: str = None,
    ) -> list[dict]:
        """
        BFS-scrape starting at `start_url` to `depth` levels deep.

        Args:
            start_url: The root URL to begin crawling.
            depth:     0 = root only; N = N link-hops from root.
            company:   Company name to tag in each raw dict.

        Returns:
            List of raw dicts in standard format.
        """
        visited  = set()
        results  = []
        # queue entries: (url, current_depth)
        queue    = deque([(start_url, 0)])

        logger.info(f"[hyperbrowser] Starting: {start_url}  depth={depth}")

        while queue:
            url, current_depth = queue.popleft()

            # Skip if already visited or depth exceeded
            if url in visited or current_depth > depth:
                continue

            # Basic scheme guard
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                continue

            visited.add(url)
            logger.info(f"  [d={current_depth}] Fetching: {url}")

            page = self._fetch_page(url)
            if page is None:
                continue

            results.append(self._to_raw(page, company))

            # Enqueue child links (only if we haven't reached max depth)
            if current_depth < depth:
                raw_links = getattr(page, "links", None) or []
                child_links = raw_links[: self.max_links_per_page]
                added = 0

                for href in child_links:
                    resolved = _resolve_url(url, href)
                    if not resolved:
                        continue
                    if resolved in visited:
                        continue
                    if not _same_domain(start_url, resolved):
                        continue
                    if not _is_content_url(resolved):
                        continue
                    queue.append((resolved, current_depth + 1))
                    added += 1

                logger.debug(f"    → enqueued {added} child links for depth {current_depth + 1}")

            # Polite delay between requests
            if queue:
                time.sleep(self.request_delay)

        logger.info(f"  Done: {len(results)} pages scraped from {start_url}")
        return results


# ===========================================================================
# Public entry point
# ===========================================================================

def run_hyperbrowser_scrapers(
    targets: list[dict] = None,
    company: str = None,
) -> dict:
    """
    Scrape all configured targets and return results in the standard pipeline format.

    Args:
        targets: List of {"link": "<url>", "scrap_depth": int}.
                 Defaults to config.HYPERBROWSER_TARGETS.
        company: Company name to embed in every document.
                 Defaults to config.DEFAULT_COMPANY.

    Returns:
        {"hyperbrowser": [list of raw dicts]}

    The returned dict is compatible with clean_storage.build_documents():

        from scraper_1 import run_hyperbrowser_scrapers
        from clean_storage import build_documents, merge_and_save

        raw   = run_hyperbrowser_scrapers(company="Lufthansa")
        docs  = build_documents(raw)
        final = merge_and_save(docs)
    """
    targets = targets or config.HYPERBROWSER_TARGETS

    if not targets:
        logger.warning("No HYPERBROWSER_TARGETS configured — nothing to scrape")
        return {"hyperbrowser": []}

    scraper     = HyperbrowserDeepScraper()
    all_results = []

    for i, target in enumerate(targets, 1):
        url   = target.get("link", "").strip()
        depth = int(target.get("scrap_depth", 1))

        if not url:
            logger.warning(f"Target #{i}: missing 'link', skipping")
            continue

        logger.info(f"Target #{i}/{len(targets)}: {url}  depth={depth}  company={company}")
        results = scraper.scrape_target(url, depth=depth, company=company)
        all_results.extend(results)
        logger.info(f"  → {len(results)} pages  (running total: {len(all_results)})")

    logger.info(f"Hyperbrowser scrape complete: {len(all_results)} raw items total")
    return {"hyperbrowser": all_results}


# ===========================================================================
# Standalone entry point
# ===========================================================================

if __name__ == "__main__":
    import time

    while(True):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(name)-15s  %(levelname)s  %(message)s",
        )

        raw_data = run_hyperbrowser_scrapers()
        total    = len(raw_data.get("hyperbrowser", []))
        print(f"\nTotal pages scraped: {total}")

        for item in raw_data.get("hyperbrowser", [])[:3]:
            print(f"\n{'─' * 60}")
            print(f"URL:       {item['url']}")
            print(f"Title:     {item['title']}")
            print(f"Date:      {item['date']}")
            print(f"Publisher: {item['publisher']}")
            preview = item["content"][:300].replace("\n", " ")
            print(f"Content:   {preview}...")

        from clean_storage import build_documents, merge_and_save
        docs  = build_documents(raw_data)
        final = merge_and_save(docs)

        # Optionally pipe directly into the pipeline
        if total > 0:
            print("\nTo push into the pipeline:")
            print("  from clean_storage import build_documents, merge_and_save")
            print("  docs  = build_documents(raw_data)")
            print("  final = merge_and_save(docs)")

        time.sleep(7*24*60*60)