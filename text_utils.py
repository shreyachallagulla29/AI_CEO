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
      3. Markdown        — strip link/image/bold/heading syntax before URLs are removed
      4. URL/email       — strip bare URLs and email addresses
      5. Boilerplate     — remove cookie notices, social prompts, legal boilerplate
      6. Non-printable   — strip control characters (keeps \n, \t)
      7. Whitespace      — collapse excessive spaces and newlines
      8. Truncation      — cap at config.MAX_CONTENT_LENGTH
    """

    # ── Boilerplate patterns ───────────────────────────────────────────────────
    # NOTE: every entry here is a proper regex — square brackets must be escaped
    # when matching literal characters (e.g. \[Learn more\], not [Learn more]).
    _BOILERPLATE = [
        r"all rights reserved\.?",
        r"click here to read more",
        r"subscribe to our newsletter",
        r"sign up for.*?newsletter",
        r"read more:.*",
        # Markdown link remnants ([text]() and bare [text])
        r"\[learn more\]\s*\(",           # markdown [Learn more]( after URL strip
        r"\[close\]\s*\(",                # markdown [Close](
        r"\[accept cookies?\]\s*\(",
        r"\[decline cookies?\]\s*\(",
        r"\blearn more\b",               # plain text after markdown is stripped
        r"advertisement",
        r"cookie(s)? (policy|settings|preferences)",
        r"this site uses cookies",
        r"some modules? (?:are|is) disabled because cookies are declined[^.]*\.",
        r"accept cookies? to experience the full functionality[^.]*\.",
        r"we use cookies? to optimize[^.]*\.",
        r"no personal information is stored[^.]*\.",
        r"\baccept cookies?\b",
        r"\bdecline cookies?\b",
        r"by continuing.*?agree",
        r"privacy policy",
        r"terms (and conditions|of use|of service)",
        r"follow us on (twitter|x|facebook|instagram|linkedin)",
        r"share this article",
        r"javascript (must be|is) enabled",
        r"loading\.\.\.",
        r"\bprevious\s*next\b",          # pagination arrow labels
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

        # ── 3. Markdown syntax stripping ─────────────────────────────────────
        # Must run BEFORE URL removal so [text](url) doesn't become [text](
        # Images: ![alt](url) → remove entirely (no useful text)
        text = re.sub(r"!\[([^\]]{0,200})\]\([^)]{0,1000}\)", "", text)
        # Known UI-only button labels: remove the whole link including display text
        # e.g. [Close]( [Accept Cookies]( [Decline Cookies]( [Learn more](
        _UI_LABELS = r"(?:Close|Accept Cookies?|Decline Cookies?|Learn [Mm]ore|Previous\s*Next|Previous|Next)"
        text = re.sub(rf"\[{_UI_LABELS}\]\s*\(", " ", text, flags=re.IGNORECASE)
        text = re.sub(rf"\[{_UI_LABELS}\]", " ", text, flags=re.IGNORECASE)
        # Links: [display text](url) → keep the display text
        text = re.sub(r"\[([^\]]{0,300})\]\([^)]{0,1000}\)", r"\1", text)
        # Leftover dangling [text]( (URL was whitespace-broken, paren still open)
        text = re.sub(r"\[([^\]]{0,300})\]\(", r"\1 ", text)
        # Bare brackets with no link: [text] that remained after above passes
        text = re.sub(r"\[([^\]\[]{0,300})\]", r"\1", text)
        # Markdown bold/italic: **text** → text, *text* → text, __text__ → text
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"\*([^*\n]+)\*", r"\1", text)
        text = re.sub(r"_([^_\n]+)_", r"\1", text)
        # Markdown headings: ## Heading → Heading
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Markdown horizontal rules: --- or *** or ___
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

        # ── 4. URL and email removal ──────────────────────────────────────────
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)

        # ── 5. Boilerplate removal ────────────────────────────────────────────
        text = self._BOILERPLATE_RE.sub(" ", text)

        # ── 6. Non-printable character removal (keep \n and \t) ───────────────
        text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)

        # ── 7. Whitespace normalization ───────────────────────────────────────
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
