# =============================================================================
# embedder.py — Embedding Generation Layer
#
# Supports two models, switchable via config.EMBEDDING_MODEL:
#   "bge"    → BAAI/bge-base-en-v1.5   (768-dim, best quality for retrieval)
#   "minilm" → all-MiniLM-L6-v2        (384-dim, faster, lower memory)
#
# Input:  list of Chunk dicts (output of processor.run_processing_pipeline())
# Output: list of EmbeddedChunk dicts — same as Chunk but with "embedding" field
#
# EmbeddedChunk adds to Chunk:
# {
#   "embedding":       [0.021, -0.134, ...],   # list[float], length = model dim
#   "embedding_model": "BAAI/bge-base-en-v1.5",
# }
# =============================================================================

import logging
import time

import config

logger = logging.getLogger("embedder")


# ===========================================================================
# Model Loader
# ===========================================================================

def load_embedding_model(model_key: str = None):
    """
    Load and return a SentenceTransformer model.

    Args:
        model_key: "bge" | "minilm"  (defaults to config.EMBEDDING_MODEL)

    Returns:
        (SentenceTransformer instance, model_name string)
    """
    from sentence_transformers import SentenceTransformer

    key        = (model_key or config.EMBEDDING_MODEL).lower()
    model_name = config.EMBEDDING_MODELS.get(key)

    if not model_name:
        raise ValueError(
            f"Unknown model key '{key}'. Choose from: {list(config.EMBEDDING_MODELS.keys())}"
        )

    logger.info(f"Loading embedding model: {model_name}  (device={config.EMBEDDING_DEVICE})")
    model = SentenceTransformer(model_name, device=config.EMBEDDING_DEVICE)
    logger.info(f"Model ready — output dim: {model.get_embedding_dimension()}")
    return model, model_name


# ===========================================================================
# Embedding Generator
# ===========================================================================

class EmbeddingGenerator:
    """
    Generates embeddings for Chunk dicts in batches.

    Usage:
        gen = EmbeddingGenerator()              # uses config.EMBEDDING_MODEL
        gen = EmbeddingGenerator("bge")         # explicit BGE
        gen = EmbeddingGenerator("minilm")      # explicit MiniLM

        embedded_chunks = gen.embed_chunks(chunks)
        query_vec       = gen.embed_query("What are Lufthansa's expansion plans?")
    """

    def __init__(self, model_key: str = None):
        self.model_key  = (model_key or config.EMBEDDING_MODEL).lower()
        self.model, self.model_name = load_embedding_model(self.model_key)

    def _prepare_text(self, chunk: dict) -> str:
        """
        BGE performs better when the document title is prepended to the chunk.
        MiniLM uses the chunk text as-is.
        """
        text  = chunk.get("chunk_text", "").strip()
        title = chunk.get("title", "").strip()
        if self.model_key == "bge" and title:
            text = f"{title}. {text}"
        return text

    def embed_chunks(
        self,
        chunks: list[dict],
        batch_size: int = None,
        show_progress: bool = True,
    ) -> list[dict]:
        """
        Generate embeddings for all chunks in batches.

        Args:
            chunks:        Chunk dicts from processor.run_processing_pipeline()
            batch_size:    Override config.EMBEDDING_BATCH_SIZE
            show_progress: Log batch progress

        Returns:
            EmbeddedChunk dicts — each chunk dict with "embedding" and
            "embedding_model" fields added.
        """
        if not chunks:
            logger.warning("embed_chunks called with empty list")
            return []

        batch_size     = batch_size or config.EMBEDDING_BATCH_SIZE
        texts          = [self._prepare_text(c) for c in chunks]
        total          = len(texts)
        all_embeddings = []

        logger.info(f"Embedding {total} chunks | model={self.model_name} | batch={batch_size}")
        t_start = time.time()

        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            vecs = self.model.encode(
                batch_texts,
                normalize_embeddings=True,   # unit-norm vectors → cosine via dot product
                show_progress_bar=False,
            )
            all_embeddings.extend(vecs.tolist())
            if show_progress:
                logger.info(f"  [{min(i + batch_size, total)}/{total}] chunks embedded")

        elapsed = round(time.time() - t_start, 1)
        dim     = len(all_embeddings[0])
        logger.info(f"Done in {elapsed}s — {total} vectors, dim={dim}")

        return [
            {**chunk, "embedding": vec, "embedding_model": self.model_name}
            for chunk, vec in zip(chunks, all_embeddings)
        ]

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single retrieval query.

        BGE requires a query-time prefix (not used during indexing).
        MiniLM uses the raw query string.

        Returns:
            Unit-normalised embedding as list[float].
        """
        if self.model_key == "bge":
            query = config.BGE_QUERY_PREFIX + query

        vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0].tolist()


# ===========================================================================
# Convenience function
# ===========================================================================

def generate_embeddings(
    chunks: list[dict],
    model_key: str = None,
    batch_size: int = None,
) -> list[dict]:
    """
    One-call interface: load model → embed all chunks → return EmbeddedChunks.

    Args:
        chunks:    Chunk dicts from processor.run_processing_pipeline()
        model_key: "bge" | "minilm"  (defaults to config.EMBEDDING_MODEL)
        batch_size: Override config.EMBEDDING_BATCH_SIZE

    Returns:
        List of EmbeddedChunk dicts.
    """
    gen = EmbeddingGenerator(model_key=model_key)
    return gen.embed_chunks(chunks, batch_size=batch_size)


# ===========================================================================
# Entry point (standalone test)
# ===========================================================================

if __name__ == "__main__":
    from clean_storage import load_dataset
    from processor import run_processing_pipeline

    docs   = load_dataset()
    chunks = run_processing_pipeline(docs)

    print(f"\nEmbedding first 10 chunks with model='{config.EMBEDDING_MODEL}'")
    embedded = generate_embeddings(chunks[:10])

    s = embedded[0]
    print(f"\nSample:")
    print(f"  chunk_id:        {s['chunk_id']}")
    print(f"  embedding_model: {s['embedding_model']}")
    print(f"  embedding dim:   {len(s['embedding'])}")
    print(f"  first 5 values:  {[round(v,4) for v in s['embedding'][:5]]}")
