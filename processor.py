# =============================================================================
# processor.py — Text Processing Pipeline
#
# Stages (in order):
#   1. Clean  — strip HTML, normalise whitespace, remove boilerplate
#   2. Dedupe — remove documents with duplicate URLs or identical content hash
#   3. Chunk  — split long documents into overlapping sentence-aware windows
#
# Input:  list of Document dicts (output of storage.build_documents())
# Output: list of Chunk dicts ready for embedding
#
# Chunk schema:
# {
#   "chunk_id":     "doc_abc123_c0",
#   "doc_id":       "doc_abc123",
#   "chunk_index":  0,
#   "total_chunks": 3,
#   "chunk_text":   "cleaned chunk text...",
#   "source":       "reddit",
#   "url":          "https://...",
#   "title":        "Original document title",
#   "date":         "2026-06-10",
#   "company":      "Lufthansa",
#   "type":         "community_post",
#   "publisher":    "r/aviation",
# }
#
# Tuning knobs in config.py:
#   CHUNK_SIZE     — target characters per chunk  (default: 1000)
#   CHUNK_OVERLAP  — overlap between chunks       (default: 150)
#   CHUNK_MIN_SIZE — discard chunks shorter than  (default: 100)
# =============================================================================

import re
import hashlib
import logging

import config

logger = logging.getLogger("processor")


# ===========================================================================
# Stage 1 — Text Cleaning
# ===========================================================================

class TextCleaner:
    """
    Cleans raw document text for downstream embedding and analysis.
    Removes HTML, boilerplate phrases, non-printable characters,
    and collapses whitespace.
    """

    _BOILERPLATE = [
        r"all rights reserved\.?",
        r"click here to read more",
        r"subscribe to our newsletter",
        r"read more:.*",
        r"advertisement",
        r"cookie(s)? (policy|settings|preferences)",
        r"privacy policy",
        r"terms (and conditions|of use|of service)",
        r"follow us on (twitter|x|facebook|instagram|linkedin)",
        r"share this article",
        r"sign up for.*newsletter",
        r"javascript (must be|is) enabled",
        r"loading\.\.\.",
        r"this site uses cookies",
        r"by continuing.*agree",
    ]
    _BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE), re.IGNORECASE)

    _HTML_ENTITIES = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
        "&#39;": "'", "&nbsp;": " ", "&mdash;": "—", "&ndash;": "–",
        "&hellip;": "…", "&copy;": "©", "&reg;": "®",
    }

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # 1. Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # 2. Decode HTML entities
        for entity, char in self._HTML_ENTITIES.items():
            text = text.replace(entity, char)

        # 3. Remove bare URLs (not useful for semantic content)
        text = re.sub(r"https?://\S+", "", text)

        # 4. Remove boilerplate phrases
        text = self._BOILERPLATE_RE.sub(" ", text)

        # 5. Remove non-printable / control characters (keep newlines + tabs)
        text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)

        # 6. Collapse excessive whitespace
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        # 7. Truncate to hard limit
        if len(text) > config.MAX_CONTENT_LENGTH:
            text = text[: config.MAX_CONTENT_LENGTH] + "…"

        return text

    def clean_document(self, doc: dict) -> dict:
        """Return a copy of doc with cleaned content and title fields."""
        cleaned = dict(doc)
        cleaned["content"] = self.clean(doc.get("content", ""))
        cleaned["title"]   = re.sub(r"\s+", " ", doc.get("title", "")).strip()
        return cleaned

    def clean_all(self, documents: list[dict]) -> list[dict]:
        """Clean every document; drop those that become too short."""
        cleaned = [self.clean_document(d) for d in documents]
        before  = len(cleaned)
        cleaned = [d for d in cleaned
                   if len(d.get("content", "")) >= config.MIN_CONTENT_LENGTH]
        dropped = before - len(cleaned)
        if dropped:
            logger.info(f"Cleaning: dropped {dropped} docs below min length ({config.MIN_CONTENT_LENGTH} chars)")
        logger.info(f"Cleaning complete → {len(cleaned)} documents remain")
        return cleaned


# ===========================================================================
# Stage 2 — Deduplication
# ===========================================================================

class Deduplicator:
    """
    Two-pass deduplication:
      Pass 1 — Exact URL match  (catches reposts of the same article)
      Pass 2 — Content hash     (MD5 of first 500 chars; catches near-dupes
                                 with different URLs but same text)
    Both passes keep the first occurrence.
    """

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.md5(text[:500].strip().lower().encode()).hexdigest()

    def deduplicate(self, documents: list[dict]) -> list[dict]:
        seen_urls   = set()
        seen_hashes = set()
        unique      = []

        for doc in documents:
            url  = doc.get("url", "").strip()
            h    = self._content_hash(doc.get("content", ""))

            if url and url in seen_urls:
                continue
            if h in seen_hashes:
                continue

            seen_urls.add(url)
            seen_hashes.add(h)
            unique.append(doc)

        removed = len(documents) - len(unique)
        logger.info(
            f"Deduplication: removed {removed} duplicates → {len(unique)} unique documents"
        )
        return unique


# ===========================================================================
# Stage 3 — Chunking
# ===========================================================================

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentence-like segments using punctuation + paragraph breaks.
    Used to make chunk boundaries fall on natural language boundaries.
    """
    # Split on sentence-ending punctuation followed by whitespace + capital
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\"\'\(])', text)
    # Also split on paragraph breaks
    result = []
    for part in parts:
        paragraphs = [p.strip() for p in part.split("\n\n") if p.strip()]
        result.extend(paragraphs if paragraphs else [part])
    return result if result else [text]


class TextChunker:
    """
    Splits document content into fixed-size overlapping windows,
    respecting sentence boundaries where possible.

    Why character-based?
    → Tokenizer-free at this stage. chunk_size=1000 chars ≈ 250-300 tokens,
      which is safe for both BGE (512-token limit) and MiniLM (256-token limit).
    """

    def __init__(
        self,
        chunk_size:  int = None,
        overlap:     int = None,
        min_size:    int = None,
    ):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.overlap    = overlap    or config.CHUNK_OVERLAP
        self.min_size   = min_size   or config.CHUNK_MIN_SIZE

    def _chunk_text(self, text: str) -> list[str]:
        """Split one text string into a list of chunk strings."""
        if not text or len(text.strip()) == 0:
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()]

        sentences = _split_sentences(text)
        chunks    = []
        current   = ""

        for sentence in sentences:
            candidate = (current + " " + sentence).strip() if current else sentence
            if current and len(candidate) > self.chunk_size:
                # Finalise current chunk
                chunks.append(current.strip())
                # Start next chunk with overlap from the tail of the previous
                overlap_text = current[-self.overlap:] if len(current) > self.overlap else current
                current = (overlap_text + " " + sentence).strip()
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def chunk_document(self, doc: dict) -> list[dict]:
        """Return a list of Chunk dicts for a single Document dict."""
        raw_chunks = self._chunk_text(doc.get("content", ""))
        raw_chunks = [c for c in raw_chunks if len(c) >= self.min_size]

        if not raw_chunks:
            logger.debug(f"No usable chunks from [{doc.get('id')}] — too short after cleaning")
            return []

        doc_id = doc.get("id", "doc_unknown")
        total  = len(raw_chunks)

        return [
            {
                "chunk_id":     f"{doc_id}_c{i}",
                "doc_id":       doc_id,
                "chunk_index":  i,
                "total_chunks": total,
                "chunk_text":   chunk,
                # Inherited metadata
                "source":    doc.get("source",    ""),
                "url":       doc.get("url",       ""),
                "title":     doc.get("title",     ""),
                "date":      doc.get("date",      ""),
                "company":   doc.get("company",   config.COMPANY),
                "type":      doc.get("type",      ""),
                "publisher": doc.get("publisher", ""),
            }
            for i, chunk in enumerate(raw_chunks)
        ]

    def chunk_all(self, documents: list[dict]) -> list[dict]:
        """Chunk every document; return flat list of all chunks."""
        all_chunks = []
        skipped    = 0

        for doc in documents:
            chunks = self.chunk_document(doc)
            if chunks:
                all_chunks.extend(chunks)
            else:
                skipped += 1

        logger.info(
            f"Chunking complete: {len(all_chunks)} chunks from "
            f"{len(documents) - skipped} docs ({skipped} skipped) "
            f"[size={self.chunk_size}, overlap={self.overlap}]"
        )
        return all_chunks


# ===========================================================================
# Full Pipeline Entry Point
# ===========================================================================

def run_processing_pipeline(
    documents: list[dict],
    chunk_size:  int = None,
    chunk_overlap: int = None,
) -> list[dict]:
    """
    Run all three processing stages in order: clean → dedupe → chunk.

    Args:
        documents:     Raw Document dicts from storage.build_documents()
        chunk_size:    Override config.CHUNK_SIZE for this run
        chunk_overlap: Override config.CHUNK_OVERLAP for this run

    Returns:
        Flat list of Chunk dicts ready for embedder.generate_embeddings()
    """
    logger.info(f"Processing pipeline: {len(documents)} documents in")

    cleaner = TextCleaner()
    deduper = Deduplicator()
    chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)

    cleaned = cleaner.clean_all(documents)
    unique  = deduper.deduplicate(cleaned)
    chunks  = chunker.chunk_all(unique)

    logger.info(f"Pipeline complete → {len(chunks)} chunks ready for embedding")
    return chunks


# ===========================================================================
# Stats Helper
# ===========================================================================

def chunk_stats(chunks: list[dict]) -> dict:
    if not chunks:
        return {"total_chunks": 0}
    lengths   = [len(c["chunk_text"]) for c in chunks]
    by_source = {}
    for c in chunks:
        src = c.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
    return {
        "total_chunks":    len(chunks),
        "avg_chunk_chars": round(sum(lengths) / len(lengths)),
        "min_chunk_chars": min(lengths),
        "max_chunk_chars": max(lengths),
        "by_source":       by_source,
    }


# ===========================================================================
# Entry point (standalone test)
# ===========================================================================

if __name__ == "__main__":
    import json
    from clean_storage import load_dataset

    docs   = load_dataset()
    chunks = run_processing_pipeline(docs)
    print(json.dumps(chunk_stats(chunks), indent=2))

    if chunks:
        c = chunks[0]
        print(f"\n--- First Chunk ---")
        print(f"ID:     {c['chunk_id']}")
        print(f"Source: {c['source']}  |  Type: {c['type']}")
        print(f"Title:  {c['title']}")
        print(f"Chars:  {len(c['chunk_text'])}")
        print(f"Text:   {c['chunk_text'][:300]}...")
