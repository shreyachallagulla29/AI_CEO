# =============================================================================
# text_utils.py — Shared text cleaning utilities
#
# Single source of truth for all text/HTML cleaning used across the pipeline.
# Replaces the duplicate cleaning code that existed in both:
#   - clean_storage.py  (clean_text, clean_title)
#   - processor.py      (TextCleaner class)
#
# What's combined here:
#   ✓ BeautifulSoup HTML parsing          (from clean_storage.py — more thorough)
#   ✓ NFKC Unicode normalization          (from clean_storage.py — was missing in processor)
#   ✓ Email removal                       (from clean_storage.py — was missing in processor)
#   ✓ 13 merged boilerplate patterns      (processor had 10, clean_storage had 5, merged unique)
#   ✓ Non-printable character removal     (from processor.py — was missing in clean_storage)
#   ✓ Truncation uses config.MAX_CONTENT_LENGTH (fixes hardcoded 4000-char bug in clean_storage)
#
# Usage:
#   from text_utils import TextCleaner
#   cleaner = TextCleaner()
#   clean   = cleaner.clean("raw html or plain text here")
#   title   = cleaner.clean_title("<b>My Title</b>")
# =============================================================================

import re
import logging
import unicodedata

import config

logger = logging.getLogger("text_utils")


class TextCleaner:
    """
    Cleans raw HTML or plain text for downstream embedding and analysis.

    Stages (in order):
      1. HTML parsing    — BeautifulSoup removes tags/noise; falls back to regex for plain text
      2. Unicode         — NFKC normalization (fixes ligatures, curly quotes, \xa0 spaces)
      3. URL/email       — strip bare URLs and email addresses
      4. Boilerplate     — remove cookie notices, social prompts, legal boilerplate
      5. Non-printable   — strip control characters (keeps \n, \t)
      6. Whitespace      — collapse excessive spaces and newlines
      7. Truncation      — cap at config.MAX_CONTENT_LENGTH
    """

    # ── Boilerplate patterns (merged from both original files) ────────────────
    _BOILERPLATE = [
        r"all rights reserved\.?",
        r"click here to read more",
        r"subscribe to our newsletter",
        r"sign up for.*newsletter",
        r"read more:.*",
        r"advertisement",
        r"cookie(s)? (policy|settings|preferences)",
        r"this site uses cookies",
        r"by continuing.*agree",
        r"privacy policy",
        r"terms (and conditions|of use|of service)",
        r"follow us on (twitter|x|facebook|instagram|linkedin)",
        r"share this article",
        r"javascript (must be|is) enabled",
        r"loading\.\.\.",
    ]
    _BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE), re.IGNORECASE)

    # ── HTML noise elements to strip before extracting text ───────────────────
    _NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "button"]
    _NOISE_SELECTORS = [".cookie-banner", ".sidebar", ".social-share", "#comments"]

    def clean(self, text: str, max_length: int = None) -> str:
        """
        Clean raw HTML or plain text.

        Args:
            text:       Raw HTML string or plain text.
            max_length: Hard character cap (defaults to config.MAX_CONTENT_LENGTH).

        Returns:
            Cleaned plain text string.
        """
        if not text or not text.strip():
            return ""

        max_length = max_length or config.MAX_CONTENT_LENGTH

        # ── 1. HTML parsing ───────────────────────────────────────────────────
        if "<" in text and ">" in text:
            try:
                from bs4 import BeautifulSoup, Comment

                soup = BeautifulSoup(text, "html.parser")

                # Remove noise tags
                for tag in self._NOISE_TAGS:
                    for el in soup(tag):
                        el.decompose()

                # Remove noise CSS selectors
                for selector in self._NOISE_SELECTORS:
                    for el in soup.select(selector):
                        el.decompose()

                # Remove HTML comments
                for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
                    comment.extract()

                # Add spacing around block elements so words don't run together
                for tag in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"]):
                    tag.insert_before("\n")
                    tag.insert_after("\n")

                text = soup.get_text(separator=" ")

            except ImportError:
                # BeautifulSoup not available — fall back to regex strip
                logger.warning("BeautifulSoup not installed; falling back to regex HTML strip")
                text = re.sub(r"<[^>]+>", " ", text)
        # else: plain text — no HTML parsing needed

        # ── 2. Unicode normalization ──────────────────────────────────────────
        text = unicodedata.normalize("NFKC", text)

        # ── 3. URL and email removal ──────────────────────────────────────────
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)

        # ── 4. Boilerplate removal ────────────────────────────────────────────
        text = self._BOILERPLATE_RE.sub(" ", text)

        # ── 5. Non-printable character removal (keep \n and \t) ───────────────
        text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)

        # ── 6. Whitespace normalization ───────────────────────────────────────
        text = re.sub(r"[ \t]+", " ", text)           # collapse spaces/tabs
        text = re.sub(r"\n{3,}", "\n\n", text)         # max two consecutive newlines
        text = "\n".join(line.strip() for line in text.splitlines())
        text = text.strip()

        # ── 7. Truncation ─────────────────────────────────────────────────────
        if len(text) > max_length:
            text = text[:max_length] + "…"

        return text

    def clean_title(self, title: str) -> str:
        """Strip HTML tags and collapse whitespace in a title string."""
        if not title:
            return ""
        title = re.sub(r"<[^>]+>", " ", title)
        title = unicodedata.normalize("NFKC", title)
        title = re.sub(r"\s+", " ", title).strip()
        return title

    def clean_document(self, doc: dict) -> dict:
        """Return a copy of doc with cleaned content and title fields."""
        cleaned = dict(doc)
        cleaned["content"] = self.clean(doc.get("content", ""))
        cleaned["title"]   = self.clean_title(doc.get("title", ""))
        return cleaned

    def clean_all(self, documents: list[dict]) -> list[dict]:
        """Clean every document; drop those whose content falls below MIN_CONTENT_LENGTH."""
        cleaned = [self.clean_document(d) for d in documents]
        before  = len(cleaned)
        cleaned = [d for d in cleaned if len(d.get("content", "")) >= config.MIN_CONTENT_LENGTH]
        dropped = before - len(cleaned)
        if dropped:
            logger.info(
                f"Cleaning: dropped {dropped} docs below min length "
                f"({config.MIN_CONTENT_LENGTH} chars)"
            )
        logger.info(f"Cleaning complete → {len(cleaned)} documents remain")
        return cleaned


# ---------------------------------------------------------------------------
# Module-level singleton (avoids re-instantiating in tight loops)
# ---------------------------------------------------------------------------
_cleaner: TextCleaner | None = None


def get_cleaner() -> TextCleaner:
    """Return the shared TextCleaner instance (lazy init)."""
    global _cleaner
    if _cleaner is None:
        _cleaner = TextCleaner()
    return _cleaner


# ---------------------------------------------------------------------------
# Convenience wrappers (for backward-compat with clean_storage.py call sites)
# ---------------------------------------------------------------------------

def clean_text(text: str, max_length: int = None) -> str:
    """Clean raw HTML or plain text. Thin wrapper around TextCleaner.clean()."""
    return get_cleaner().clean(text, max_length=max_length)


def clean_title(title: str) -> str:
    """Clean a title string. Thin wrapper around TextCleaner.clean_title()."""
    return get_cleaner().clean_title(title)
