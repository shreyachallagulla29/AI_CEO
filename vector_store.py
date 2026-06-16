# =============================================================================
# vector_store.py — Cloud Database Layer
#
# Two cloud backends, both configured in config.py:
#
#   SupabaseDocumentStore  — stores raw Document dicts in Supabase (PostgreSQL)
#                            Sign up free at https://supabase.com
#                            Run setup_supabase_table() once to create the table.
#
#   ChromaVectorStore      — stores embeddings in ChromaDB Cloud
#                            Sign up free at https://trychroma.com
#                            Supports upsert, semantic search, metadata filtering.
#
# Typical usage:
#   from vector_store import SupabaseDocumentStore, ChromaVectorStore
#
#   # Save raw documents
#   supa = SupabaseDocumentStore()
#   supa.upsert_documents(documents)
#
#   # Save embeddings
#   chroma = ChromaVectorStore()
#   chroma.upsert_embeddings(embedded_chunks)
#
#   # Retrieve at query time
#   gen    = EmbeddingGenerator()
#   q_vec  = gen.embed_query("Lufthansa fleet expansion plans")
#   results = chroma.query(q_vec, top_k=5)
# =============================================================================

import logging

import config

logger = logging.getLogger("vector_store")

# ===========================================================================
# ChromaDB Cloud — Vector Embedding Store
# ===========================================================================

class ChromaVectorStore:
    """
    Stores chunk embeddings in ChromaDB Cloud and supports semantic retrieval.

    ChromaDB Cloud (trychroma.com) — free tier available.
    Uses chromadb.HttpClient pointing to the cloud API.

    Collection: config.CHROMA_COLLECTION_NAME  (default: "lufthansa_embeddings")
    Distance:   config.CHROMA_DISTANCE_METRIC  (default: "cosine")

    Setup:
        1. Sign up at https://trychroma.com
        2. Create a database called "lufthansa_intel"
        3. Copy your API key, tenant ID, and database name into config.py
           (or set CHROMA_API_KEY / CHROMA_TENANT / CHROMA_DATABASE env vars)
    """

    def __init__(self):
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.HttpClient(
            host="api.trychroma.com",
            ssl=True,
            headers={
                "x-chroma-token": config.CHROMA_API_KEY,
                "X-Chroma-Tenant": config.CHROMA_TENANT,
                "X-Chroma-Database": config.CHROMA_DATABASE,
            },
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": config.CHROMA_DISTANCE_METRIC},
        )
        logger.info(
            f"ChromaDB Cloud connected — collection: '{config.CHROMA_COLLECTION_NAME}' "
            f"({self.collection.count()} existing vectors)"
        )

    def upsert_embeddings(self, embedded_chunks: list[dict]) -> int:
        """
        Upsert embedded chunks into ChromaDB.
        Existing chunk_ids are updated; new ones are inserted.

        Args:
            embedded_chunks: EmbeddedChunk dicts from embedder.generate_embeddings()

        Returns:
            Number of chunks upserted.
        """
        if not embedded_chunks:
            logger.warning("upsert_embeddings called with empty list")
            return 0

        # ChromaDB upsert in batches of 500
        batch_size = 500
        total = 0

        for i in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[i : i + batch_size]

            ids        = [c["chunk_id"]   for c in batch]
            embeddings = [c["embedding"]  for c in batch]
            documents  = [c["chunk_text"] for c in batch]
            metadatas  = [
                {
                    "doc_id":      c.get("doc_id",      ""),
                    "source":      c.get("source",      ""),
                    "url":         c.get("url",         ""),
                    "title":       c.get("title",       ""),
                    "date":        c.get("date",        ""),
                    "company":     c.get("company",     config.COMPANY),
                    "type":        c.get("type",        ""),
                    "publisher":   c.get("publisher",   ""),
                    "chunk_index": str(c.get("chunk_index", 0)),
                    "model":       c.get("embedding_model", ""),
                }
                for c in batch
            ]

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            total += len(batch)
            logger.info(f"ChromaDB: upserted {total}/{len(embedded_chunks)} chunks")

        logger.info(f"ChromaDB upsert complete — {total} chunks in '{config.CHROMA_COLLECTION_NAME}'")
        return total

    def query(
        self,
        query_embedding: list[float],
        top_k: int = None,
        filters: dict = None,
    ) -> list[dict]:
        """
        Semantic search — returns the top-k most similar chunks.

        Args:
            query_embedding: Embedded query vector from EmbeddingGenerator.embed_query()
            top_k:           Number of results (defaults to config.CHROMA_TOP_K)
            filters:         Optional ChromaDB where-filter, e.g.
                             {"source": {"$eq": "reddit"}}
                             {"date": {"$gte": "2026-01-01"}}

        Returns:
            List of result dicts:
            {
              "chunk_id":  "doc_abc_c0",
              "text":      "chunk content...",
              "score":     0.91,          # cosine similarity (higher = more similar)
              "metadata":  {...},
            }
        """
        top_k = top_k or config.CHROMA_TOP_K

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results":        top_k,
            "include":          ["documents", "metadatas", "distances"],
        }
        if filters:
            query_kwargs["where"] = filters

        raw = self.collection.query(**query_kwargs)

        results = []
        ids        = raw.get("ids",        [[]])[0]
        documents  = raw.get("documents",  [[]])[0]
        metadatas  = raw.get("metadatas",  [[]])[0]
        distances  = raw.get("distances",  [[]])[0]

        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
            similarity = round(1 - dist, 4)
            results.append({
                "chunk_id": chunk_id,
                "text":     text,
                "score":    similarity,
                "metadata": meta,
            })

        return results

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        return self.collection.count()

    def delete_collection(self) -> None:
        """Delete and recreate the collection (wipes all data). Use with caution."""
        self.client.delete_collection(config.CHROMA_COLLECTION_NAME)
        logger.warning(f"ChromaDB collection '{config.CHROMA_COLLECTION_NAME}' deleted")


# ===========================================================================
# Full Store Pipeline
# ===========================================================================

def store_all(
    documents: list[dict],
    embedded_chunks: list[dict],
) -> dict:
    """
    Save everything to both cloud backends in one call.

    Args:
        documents:       Raw Document dicts (from storage.build_documents())
        embedded_chunks: EmbeddedChunk dicts (from embedder.generate_embeddings())

    Returns:
        {"supabase_docs": N, "chroma_chunks": M}
    """
    chroma_store = ChromaVectorStore()
    chunks_saved = chroma_store.upsert_embeddings(embedded_chunks)

    logger.info(f"store_all  {chunks_saved} chunks in ChromaDB")
    return {"chroma_chunks": chunks_saved}


# ===========================================================================
# Entry point (standalone test)
# ===========================================================================

if __name__ == "__main__":
    from clean_storage import load_dataset
    from processor import run_processing_pipeline
    from embedder import generate_embeddings, EmbeddingGenerator

    docs           = load_dataset()
    chunks         = run_processing_pipeline(docs)
    embedded       = generate_embeddings(chunks[:20])   # small test batch

    print("\n--- Storing to ChromaDB Cloud ---")
    chroma = ChromaVectorStore()
    chroma.upsert_embeddings(embedded)
    print(f"Total vectors in ChromaDB: {chroma.count()}")

    print("\n--- Semantic Search Test ---")
    gen = EmbeddingGenerator()
    q_vec = gen.embed_query("Lufthansa fleet expansion and sustainability")
    results = chroma.query(q_vec, top_k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['metadata'].get('title','')[:60]}")
        print(f"           {r['text'][:120]}...")
