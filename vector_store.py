# =============================================================================
# vector_store.py — ChromaDB Cloud vector store (single source of truth)
#
# Consolidated from vector_store.py + vectordb.py.
#
# Responsibilities:
#   Push  — upsert_embeddings(), add(), add_in_batches()
#   Query — search(), search_by_text()
#   Fetch — get_documents(), get_stats(), count()
#   Manage — delete_collection(), reset()
#
# Config (config.py):
#   CHROMA_API_KEY         — cloud API key
#   CHROMA_TENANT          — tenant UUID
#   CHROMA_DATABASE        — database name
#   CHROMA_COLLECTION_NAME — collection name
#   CHROMA_DISTANCE_METRIC — "cosine" | "l2" | "ip"
#   CHROMA_TOP_K           — default results per query
#
# Typical usage:
#   from vector_store import ChromaVectorStore
#
#   store = ChromaVectorStore()
#
#   # Push — after running embedder.generate_embeddings()
#   store.upsert_embeddings(embedded_chunks)
#
#   # Query — semantic search
#   results = store.search_by_text("Lufthansa fleet expansion", embedder)
#
#   # Fetch — inspect stored records
#   docs  = store.get_documents(limit=10)
#   stats = store.get_stats()
# =============================================================================

import logging

import config

logger = logging.getLogger("vector_store")


class ChromaVectorStore:
    """
    Full-featured ChromaDB Cloud client for the Lufthansa RAG pipeline.

    Handles push, query, fetch, and management in one place.
    """

    def __init__(
        self,
        collection_name: str = None,
        distance_metric: str = None,
    ):
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        self.distance_metric = distance_metric or config.CHROMA_DISTANCE_METRIC
        self.database        = config.CHROMA_DATABASE
        self.tenant          = config.CHROMA_TENANT

        import chromadb
        from chromadb.config import Settings

        logger.info(
            f"Connecting to ChromaDB Cloud "
            f"(tenant={self.tenant}, db={self.database})"
        )
        # self._client = chromadb.HttpClient(
        #     host="api.trychroma.com",
        #     ssl=True,
        #     headers={
        #         "x-chroma-token":    config.CHROMA_API_KEY,
        #         "X-Chroma-Tenant":   self.tenant,
        #         "X-Chroma-Database": self.database,
        #     },
        #     settings=Settings(anonymized_telemetry=False),
        # )

        self._client = chromadb.CloudClient(
  api_key= "ck-42Hn2rzqUrXu5k3GtwmDaJtDmfNh6xa8MUs39d9EanM4",
  tenant='3cd57535-7db4-499f-a4c4-91022866ca80',
  database='ragsystem'
)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )
        logger.info(
            f"Collection '{self.collection_name}' ready "
            f"({self._collection.count()} vectors already indexed)"
        )

    # =========================================================================
    # PUSH — store embeddings in ChromaDB
    # =========================================================================

    def upsert_embeddings(self, embedded_chunks: list[dict]) -> int:
        """
        Primary pipeline method: upsert EmbeddedChunk dicts directly from
        embedder.generate_embeddings().

        Each dict must contain "chunk_id", "chunk_text", and "embedding".
        Splits into chunks + embeddings lists then delegates to add_in_batches().

        Args:
            embedded_chunks: Output of embedder.generate_embeddings()

        Returns:
            Number of chunks upserted.
        """
        if not embedded_chunks:
            logger.warning("upsert_embeddings called with empty list")
            return 0

        chunks     = embedded_chunks                                 # full dicts
        embeddings = [c["embedding"] for c in embedded_chunks]       # extract vectors

        return self.add_in_batches(chunks, embeddings)

    def add(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        """
        Low-level upsert: separate chunk dicts and embedding vectors.
        Safe to call multiple times — existing IDs are overwritten, not duplicated.

        Args:
            chunks:     Chunk dicts (must have chunk_id, chunk_text)
            embeddings: Corresponding embedding vectors (same length as chunks)

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            logger.warning("add() called with empty chunk list — nothing to index")
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                f"must be the same length"
            )

        self._collection.upsert(
            ids        = [c["chunk_id"]   for c in chunks],
            embeddings = embeddings,
            documents  = [c["chunk_text"] for c in chunks],
            metadatas  = [_build_metadata(c) for c in chunks],
        )

        total_now = self._collection.count()
        logger.info(f"Upserted {len(chunks)} chunks. Collection total: {total_now}")
        return len(chunks)

    def add_in_batches(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        batch_size: int = 250,
    ) -> int:
        """
        Same as add() but splits work into batches — safe for large datasets.

        Args:
            chunks:     Chunk dicts
            embeddings: Corresponding vectors
            batch_size: Max items per ChromaDB upsert call (default 500)

        Returns:
            Total chunks upserted.
        """
        total = 0
        for start in range(0, len(chunks), batch_size):
            end      = start + batch_size
            total   += self.add(chunks[start:end], embeddings[start:end])
            logger.info(f"  Batch {start}–{end}: indexed {min(end, len(chunks)) - start} chunks")
        return total

    # =========================================================================
    # QUERY — semantic search
    # =========================================================================

    def search(
        self,
        query_vector: list[float],
        top_k: int = None,
        filters: dict = None,
    ) -> list[dict]:
        """
        Semantic search by embedding vector.

        Args:
            query_vector: Embedding from embedder.embed_query()
            top_k:        Number of results (default config.CHROMA_TOP_K)
            filters:      ChromaDB metadata filter, e.g.
                          {"source": {"$eq": "reddit"}}
                          {"date":   {"$gte": "2026-01-01"}}

        Returns:
            List of result dicts:
            {
              chunk_id, doc_id, chunk_text, title, url,
              source, type, date, company, publisher,
              score   # cosine similarity 0–1 (higher = more similar)
            }
        """
        top_k = top_k or config.CHROMA_TOP_K
        n     = min(top_k, self._collection.count() or 1)

        kwargs = {
            "query_embeddings": [query_vector],
            "n_results":        n,
            "include":          ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        raw       = self._collection.query(**kwargs)
        ids       = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        results = []
        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            results.append({
                "chunk_id":   chunk_id,
                "doc_id":     meta.get("doc_id",    ""),
                "chunk_text": text,
                "title":      meta.get("title",     ""),
                "url":        meta.get("url",        ""),
                "source":     meta.get("source",     ""),
                "type":       meta.get("type",       ""),
                "date":       meta.get("date",       ""),
                "company":    meta.get("company",    ""),
                "publisher":  meta.get("publisher",  ""),
                "score":      round(1 - dist, 4),    # cosine distance → similarity
            })
        return results

    def search_by_text(
        self,
        query_text: str,
        embedder,
        top_k: int = None,
        filters: dict = None,
    ) -> list[dict]:
        """
        Convenience wrapper: embed the query text then call search().

        Args:
            query_text: Raw query string
            embedder:   EmbeddingGenerator instance (from embedder.py)
            top_k:      Number of results
            filters:    Metadata filter dict

        Returns:
            List of result dicts (same as search())
        """
        query_vector = embedder.embed_query(query_text)
        return self.search(query_vector, top_k=top_k, filters=filters)

    # =========================================================================
    # FETCH — read stored records from ChromaDB
    # =========================================================================

    def get_documents(
        self,
        limit: int = 100,
        filters: dict = None,
    ) -> list[dict]:
        """
        Fetch stored chunk records from ChromaDB (no vector search — direct get).

        Useful for inspecting what's in the DB, auditing, or building a
        re-ranking layer on top of a pre-filtered set.

        Args:
            limit:   Max records to return (default 100)
            filters: ChromaDB metadata filter dict (same syntax as search())

        Returns:
            List of dicts: {chunk_id, chunk_text, <all metadata fields>}
        """
        kwargs = {
            "limit":   limit,
            "include": ["documents", "metadatas"],
        }
        if filters:
            kwargs["where"] = filters

        raw       = self._collection.get(**kwargs)
        ids       = raw.get("ids",       [])
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])

        results = []
        for chunk_id, text, meta in zip(ids, documents, metadatas):
            results.append({
                "chunk_id":   chunk_id,
                "chunk_text": text,
                **meta,
            })
        return results

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        return self._collection.count()

    def get_stats(self) -> dict:
        """
        Return a summary of the collection: total vectors + breakdown by
        source and document type. Samples up to 1000 records.
        """
        total = self._collection.count()
        if total == 0:
            return {
                "collection":    self.collection_name,
                "total_vectors": 0,
            }

        sample    = self._collection.get(
            limit=min(total, 1000),
            include=["metadatas"],
        )
        metadatas = sample.get("metadatas", [])
        by_source = {}
        by_type   = {}

        for m in metadatas:
            src = m.get("source", "unknown")
            typ = m.get("type",   "unknown")
            by_source[src] = by_source.get(src, 0) + 1
            by_type[typ]   = by_type.get(typ, 0) + 1

        return {
            "collection":    self.collection_name,
            "database":      self.database,
            "distance":      self.distance_metric,
            "total_vectors": total,
            "by_source":     by_source,
            "by_type":       by_type,
        }

    # =========================================================================
    # MANAGEMENT — collection lifecycle
    # =========================================================================

    def delete_collection(self) -> None:
        """
        Permanently delete the entire collection and all its vectors.
        Use when you want to re-index from scratch.
        """
        self._client.delete_collection(self.collection_name)
        logger.warning(f"Collection '{self.collection_name}' deleted")

    def reset(self) -> None:
        """
        Delete and immediately recreate the collection (clean slate).
        All existing vectors are wiped; the collection is ready for fresh inserts.
        """
        self.delete_collection()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )
        logger.info(f"Collection '{self.collection_name}' reset (empty)")


# ===========================================================================
# Internal helpers
# ===========================================================================

def _build_metadata(chunk: dict) -> dict:
    """
    Build the metadata dict for a ChromaDB upsert.
    ChromaDB requires metadata values to be str | int | float | bool — no None.
    """
    return {
        "doc_id":       str(chunk.get("doc_id",       "")),
        "source":       str(chunk.get("source",       "")),
        "url":          str(chunk.get("url",           "")),
        "title":        str(chunk.get("title",         ""))[:500],  # cap long titles
        "chunk_index":  int(chunk.get("chunk_index",   0)),
        "total_chunks": int(chunk.get("total_chunks",  1)),
        "date":         str(chunk.get("date",          "")),
        "company":      str(chunk.get("company",       "")),
        "type":         str(chunk.get("type",          "")),
        "publisher":    str(chunk.get("publisher",     "")),
        "model":        str(chunk.get("embedding_model", "")),
    }


# ===========================================================================
# Singleton (optional convenience for pipeline scripts)
# ===========================================================================

_store_instance: ChromaVectorStore | None = None


def get_store() -> ChromaVectorStore:
    """Return the shared ChromaVectorStore instance (lazy init, cached)."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaVectorStore()
    return _store_instance


# ===========================================================================
# Pipeline helper
# ===========================================================================

def store_all(embedded_chunks: list[dict]) -> dict:
    """
    One-call store: connect → upsert → return stats.

    Args:
        embedded_chunks: Output of embedder.generate_embeddings()

    Returns:
        {"chroma_chunks": N, "total_in_db": M}
    """
    store  = ChromaVectorStore()
    saved  = store.upsert_embeddings(embedded_chunks)
    total  = store.count()
    logger.info(f"store_all: {saved} chunks upserted | {total} total in DB")
    return {"chroma_chunks": saved, "total_in_db": total}


# ===========================================================================
# Entry point (standalone test)
# ===========================================================================

if __name__ == "__main__":
    import json
    from clean_storage import load_dataset
    from processor     import run_processing_pipeline
    from embedder      import EmbeddingGenerator

    docs     = load_dataset()
    chunks   = run_processing_pipeline(docs)

    if not chunks:
        print("No chunks found. Run the scraping pipeline first.")
    else:
        embedder = EmbeddingGenerator()
        embedded = generate_embeddings = embedder.embed_chunks(chunks[:20])  # small test batch

        store = ChromaVectorStore()

        print("\n── Push ─────────────────────────────────────────────────")
        store.upsert_embeddings(embedded)

        print("\n── Stats ────────────────────────────────────────────────")
        print(json.dumps(store.get_stats(), indent=2))

        print("\n── Fetch (first 3 records) ──────────────────────────────")
        docs_in_db = store.get_documents(limit=3)
        for d in docs_in_db:
            print(f"  [{d['chunk_id']}] {d.get('title','')[:60]}")

        print("\n── Query ────────────────────────────────────────────────")
        query   = "Lufthansa fleet expansion and sustainability strategy"
        results = store.search_by_text(query, embedder, top_k=3)
        print(f"Top {len(results)} results for: '{query}'\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] score={r['score']:.4f}  {r['title'][:60]}")
            print(f"     {r['chunk_text'][:150]}...")
            print()
