# =============================================================================
# vectordb.py — ChromaDB vector store for the Lufthansa Strategic Intelligence Agent
#
# Responsibilities:
#   - Create / load a persistent ChromaDB collection
#   - Index chunk dicts + their embeddings (upsert-safe: no duplicate IDs)
#   - Semantic search by query vector
#   - Collection stats and management helpers
#
# Config knobs (config.py):
#   CHROMA_PERSIST_DIR     — folder where ChromaDB stores its data
#   CHROMA_COLLECTION_NAME — collection name (default: "lufthansa_intel")
#   CHROMA_DISTANCE_METRIC — "cosine" | "l2" | "ip"
#   CHROMA_TOP_K           — default number of results per query
#
# Install:
#   pip install chromadb --break-system-packages
# =============================================================================

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

import config

logger = logging.getLogger("vectordb")


# ===========================================================================
# ChromaDB Client & Collection
# ===========================================================================

class VectorStore:
    """
    Wraps a persistent ChromaDB collection.

    Usage:
        store = VectorStore()
        store.add(chunks, embeddings)
        results = store.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        persist_dir: Path = None,
        collection_name: str = None,
        distance_metric: str = None,
        chroma_api: str = None,
        chroma_tenant: str = None,
    ):
        self.persist_dir       = persist_dir       or config.CHROMA_PERSIST_DIR
        self.database          = collection_name   or config.CHROMA_DATABASE
        self.collection_name   = collection_name   or config.CHROMA_COLLECTION_NAME
        self.distance_metric   = distance_metric   or config.CHROMA_DISTANCE_METRIC
        self.chroma_api        = chroma_api        or config.CHROMA_API_KEY
        self.chroma_tenant     = chroma_tenant     or config.CHROMA_TENANT

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Connecting to ChromaDB at: {self.persist_dir}")
        self._client = chromadb.CloudClient(
            cloud_port=443,
            cloud_host='europe-west1.gcp.trychroma.com',
            api_key=self.chroma_api,
            tenant=self.chroma_tenant,
            database=self.database,            
            )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )

        logger.info(
            f"Collection '{self.collection_name}' ready "
            f"({self._collection.count()} vectors already indexed)"
        )

    # -----------------------------------------------------------------------
    # Indexing
    # -----------------------------------------------------------------------

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        """
        Add chunk dicts and their pre-computed embeddings to ChromaDB.
        Uses upsert semantics — safe to run multiple times without creating duplicates.

        Args:
            chunks:     Chunk dicts from processor.process_documents()
            embeddings: Corresponding embedding vectors from embedder.embed_chunks()
                        Must be the same length as chunks.

        Returns:
            Number of new chunks actually added (duplicates are silently updated).
        """
        if not chunks:
            logger.warning("add() called with empty chunk list — nothing to index")
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have equal length"
            )

        ids        = [c["chunk_id"]   for c in chunks]
        documents  = [c["chunk_text"] for c in chunks]
        metadatas  = [_build_metadata(c) for c in chunks]

        # ChromaDB upsert: insert new, overwrite existing same-ID records
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        total_now = self._collection.count()
        logger.info(f"Upserted {len(chunks)} chunks. Collection total: {total_now}")
        return len(chunks)

    def add_in_batches(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        batch_size: int = 500,
    ) -> int:
        """
        Same as add() but splits into batches to avoid memory issues with
        very large datasets (10 000+ chunks).

        Args:
            chunks:     Chunk dicts
            embeddings: Corresponding vectors
            batch_size: Max items per ChromaDB upsert call

        Returns:
            Total chunks upserted.
        """
        print("Started add_in_batches")
        total = 0
        for start in range(0, len(chunks), batch_size):
            print(f"Trying to push the batch: {start}")
            end         = start + batch_size
            batch_c     = chunks[start:end]
            batch_e     = embeddings[start:end]
            total      += self.add(batch_c, batch_e)
            logger.info(f"Batch {start}–{end}: indexed {len(batch_c)} chunks")
        return total

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = None,
        filters: dict = None,
    ) -> list[dict]:
        """
        Semantic search: find the top-k most similar chunks to a query vector.

        Args:
            query_vector: Embedding of the query (from embedder.embed_query())
            top_k:        Number of results (default from config.CHROMA_TOP_K)
            filters:      Optional ChromaDB metadata filter dict.
                          Example: {"source": "reddit"} or {"type": "press_release"}

        Returns:
            List of result dicts, each containing:
              - chunk_id, doc_id, chunk_text, title, url, source, type,
                date, company, distance (lower = more similar for cosine)
        """
        top_k = top_k or config.CHROMA_TOP_K

        query_kwargs = {
            "query_embeddings": [query_vector],
            "n_results": min(top_k, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query_kwargs["where"] = filters

        raw = self._collection.query(**query_kwargs)

        results = []
        ids        = raw["ids"][0]
        documents  = raw["documents"][0]
        metadatas  = raw["metadatas"][0]
        distances  = raw["distances"][0]

        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            results.append({
                "chunk_id":   chunk_id,
                "doc_id":     meta.get("doc_id", ""),
                "chunk_text": text,
                "title":      meta.get("title", ""),
                "url":        meta.get("url", ""),
                "source":     meta.get("source", ""),
                "type":       meta.get("type", ""),
                "date":       meta.get("date", ""),
                "company":    meta.get("company", ""),
                "publisher":  meta.get("publisher", ""),
                "distance":   round(dist, 4),
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
            embedder:   An embedder instance (from embedder.get_embedder())
            top_k:      Number of results
            filters:    Metadata filter dict

        Returns:
            List of result dicts (same as search())
        """
        query_vector = embedder.embed_query(query_text)
        return self.search(query_vector, top_k=top_k, filters=filters)

    # -----------------------------------------------------------------------
    # Management Helpers
    # -----------------------------------------------------------------------

    def count(self) -> int:
        """Return number of vectors currently in the collection."""
        return self._collection.count()

    def get_stats(self) -> dict:
        """
        Return a summary of the collection contents.
        Samples up to 1000 records to compute source/type breakdowns.
        """
        total = self._collection.count()
        if total == 0:
            return {"total_vectors": 0, "collection": self.collection_name}

        # Sample metadata to build breakdowns
        sample = self._collection.get(
            limit=min(total, 1000),
            include=["metadatas"],
        )
        metadatas  = sample.get("metadatas", [])
        by_source  = {}
        by_type    = {}

        for m in metadatas:
            src  = m.get("source", "unknown")
            typ  = m.get("type", "unknown")
            by_source[src] = by_source.get(src, 0) + 1
            by_type[typ]   = by_type.get(typ, 0) + 1

        return {
            "collection":    self.collection_name,
            "persist_dir":   str(self.persist_dir),
            "distance":      self.distance_metric,
            "total_vectors": total,
            "by_source":     by_source,
            "by_type":       by_type,
        }

    def delete_collection(self) -> None:
        """
        Permanently delete the entire collection and all its vectors.
        Use when you want to re-index from scratch.
        """
        self._client.delete_collection(self.collection_name)
        logger.warning(f"Collection '{self.collection_name}' deleted")

    def reset(self) -> None:
        """
        Delete and immediately recreate the collection (fresh start).
        """
        self.delete_collection()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )
        logger.info(f"Collection '{self.collection_name}' reset (empty)")


# ===========================================================================
# Metadata Builder (internal)
# ===========================================================================

def _build_metadata(chunk: dict) -> dict:
    """
    Extract the fields to store as ChromaDB metadata.
    ChromaDB metadata values must be str | int | float | bool.
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
    }


# ===========================================================================
# Module-Level Singleton (optional convenience)
# ===========================================================================

_store_instance: VectorStore | None = None


def get_store() -> VectorStore:
    """
    Return the shared VectorStore instance (lazy init, cached).
    Use this when you want a single store across the whole pipeline.
    """
    print("Trying to return the VectorStore instance")
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore()
    return _store_instance


# ===========================================================================
# Entry point (standalone test)
# ===========================================================================

if __name__ == "__main__":
    import json
    from clean_storage   import load_dataset
    from processor import process_documents
    from embedder  import get_embedder

    # Load data
    docs   = load_dataset()
    chunks = process_documents(docs)

    if not chunks:
        print("No chunks found. Run the scraping pipeline first.")
    else:
        # Embed
        embedder   = get_embedder()
        embeddings = embedder.embed_chunks(chunks)

        # Index
        store = get_store()
        store.add_in_batches(chunks, embeddings)

        # Stats
        stats = store.get_stats()
        print(json.dumps(stats, indent=2))

        # Test search
        query   = "What is Lufthansa's strategy for sustainable aviation?"
        results = store.search_by_text(query, embedder, top_k=3)

        print(f"\nTop {len(results)} results for: '{query}'\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] {r['title'][:80]}")
            print(f"     Source: {r['source']} | Date: {r['date']} | Distance: {r['distance']}")
            print(f"     {r['chunk_text'][:200]}...")
            print()
