# =============================================================================
# rag_query.py — Production RAG query pipeline
#
# Steps:
#   1. load_queries   — read query objects from JSON file
#   2. retrieve       — embed query + ChromaDB similarity search
#   3. save_retrieved — write retrieved docs to /outputs/retrieved_docs/
#   4. build_prompt   — assemble system + user prompt with context block
#   5. call_llm       — Together AI via HuggingFace InferenceClient (w/ retry)
#   6. save_result    — write LLM output + metadata to /outputs/llm_results/
#
# Run:
#   python rag_query.py                              # uses config.QUERIES_JSON_FILE
#   python rag_query.py --queries queries.json
#   python rag_query.py --top-k 7 --temperature 0.1
# =============================================================================

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-15s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rag_query")

# ---------------------------------------------------------------------------
# Singleton EmbeddingGenerator — load once, reuse across all queries
# ---------------------------------------------------------------------------
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from embedder import EmbeddingGenerator
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _embedder = EmbeddingGenerator(config.EMBEDDING_MODEL)
    return _embedder


# ---------------------------------------------------------------------------
# Singleton LLM pipeline — loaded once, reused across all queries
# ---------------------------------------------------------------------------
_llm_pipeline = None


def _get_llm_pipeline():
    global _llm_pipeline
    if _llm_pipeline is None:
        import torch
        from transformers import pipeline, BitsAndBytesConfig
        logger.info(f"Loading LLM: {config.HF_MODEL_NAME} (8-bit quantised)")
        _llm_pipeline = pipeline(
            task="text-generation",
            model=config.HF_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            model_kwargs={
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True)
            },
        )
        logger.info("LLM pipeline ready")
    return _llm_pipeline


# ===========================================================================
# STEP 1 — Load queries
# ===========================================================================

def load_queries(path: Path) -> list[dict]:
    """
    Load a JSON file containing a list of query objects.

    Expected schema per object:
    {
        "query_id":   "q001",
        "query":      "What is Lufthansa's sustainability strategy?",
        "prompt":     "Summarise Lufthansa's sustainability initiatives...",
        "collection": "Lufthansa_Knowledge",
        "company":    "Lufthansa",       # optional filter
        "top_k":      5,                 # optional override
        "temperature": 0.2               # optional override
    }
    """
    if not path.exists():
        logger.error(f"Queries file not found: {path}")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        queries = json.load(f)

    if not isinstance(queries, list) or not queries:
        logger.error("Queries file must be a non-empty JSON array.")
        sys.exit(1)

    logger.info(f"Loaded {len(queries)} queries from {path}")
    return queries


# ===========================================================================
# STEP 2 — Retrieve relevant documents
# ===========================================================================

def retrieve(
    query_text: str,
    collection_name: str,
    top_k: int = None,
    company_filter: str = None,
) -> list[dict]:
    """
    Embed `query_text` and run a ChromaDB similarity search.

    Returns a list of result dicts:
    [{chunk_id, doc_id, chunk_text, title, url, source, type, date, company, publisher, score}]
    """
    from vector_store import ChromaVectorStore

    embedder = _get_embedder()
    top_k    = top_k or config.CHROMA_TOP_K

    filters = {}
    if company_filter:
        filters["company"] = company_filter

    store   = ChromaVectorStore(collection_name=collection_name)
    results = store.search_by_text(
        query_text=query_text,
        embedder=embedder,
        top_k=top_k,
        filters=filters or None,
    )

    logger.info(
        f"Retrieved {len(results)} docs from '{collection_name}' "
        f"(top_k={top_k}, filter={filters or 'none'})"
    )
    return results


# ===========================================================================
# STEP 3 — Save retrieved documents for auditing
# ===========================================================================

def save_retrieved(query_id: str, query: str, results: list[dict]) -> Path:
    """Write retrieved docs to /outputs/retrieved_docs/{query_id}.json."""
    config.RETRIEVED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RETRIEVED_DOCS_DIR / f"{query_id}.json"

    payload = {
        "query_id":    query_id,
        "query":       query,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "num_results": len(results),
        "results": [
            {
                "rank":       rank + 1,
                "score":      r.get("score"),
                "chunk_id":   r.get("chunk_id"),
                "chunk_text": r.get("chunk_text", ""),
                "title":      r.get("title", ""),
                "url":        r.get("url", ""),
                "source":     r.get("source", ""),
                "type":       r.get("type", ""),
                "date":       r.get("date", ""),
                "company":    r.get("company", ""),
                "publisher":  r.get("publisher", ""),
            }
            for rank, r in enumerate(results)
        ],
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Retrieved docs saved → {out_path}")
    return out_path


# ===========================================================================
# STEP 4 — Build LLM prompt
# ===========================================================================

SYSTEM_PROMPT = (
    "You are a strategic intelligence analyst specialising in the global aviation industry. "
    "You provide accurate, evidence-based analysis drawing only on the retrieved context below. "
    "If the context does not contain enough information to answer confidently, say so explicitly. "
    "Do not fabricate facts or cite sources not present in the context."
)


def build_user_prompt(prompt: str, results: list[dict]) -> str:
    """
    Combine the user's analytical prompt with the retrieved context.

    Format:
        <prompt>

        ---
        RETRIEVED CONTEXT
        [1] Title | Source | Date
        <chunk_text>
        ...
        ---
    """
    context_lines = ["---", "RETRIEVED CONTEXT", ""]
    for i, r in enumerate(results, 1):
        header = (
            f"[{i}] {r.get('title', 'Untitled')} | "
            f"{r.get('source', '?')} | "
            f"{r.get('date', '?')} | "
            f"score={r.get('score', 0):.3f}"
        )
        context_lines.append(header)
        context_lines.append(r.get("chunk_text", "").strip())
        context_lines.append("")

    context_lines.append("---")
    context_block = "\n".join(context_lines)

    return f"{prompt}\n\n{context_block}"


# ===========================================================================
# STEP 5 — Call LLM via HuggingFace InferenceClient (Together AI)
# ===========================================================================

def call_llm(
    system_prompt: str,
    user_prompt:   str,
    temperature:   float = None,
    max_new_tokens: int  = None,
    retries:       int   = None,
) -> str:
    """
    Run inference using a local transformers pipeline (Qwen3, 8-bit quantised).

    The model is loaded once as a module-level singleton (_get_llm_pipeline)
    and reused across all queries to avoid repeated GPU loading overhead.

    Retries on unexpected runtime errors with exponential backoff.
    """
    temperature    = temperature    if temperature    is not None else config.LLM_TEMPERATURE
    max_new_tokens = max_new_tokens if max_new_tokens is not None else config.LLM_MAX_NEW_TOKENS
    retries        = retries        if retries        is not None else config.LLM_RETRIES

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
        {"""You are an AI Strategic Intelligence Analyst.

Your task is to analyze ONLY the provided retrieved documents.

Rules:

Do NOT explain your reasoning.
Do NOT include thinking steps.
Do NOT include markdown.
Do NOT include text before or after JSON.
Return valid JSON only.
If evidence is unavailable, return an empty array.
Every finding must be supported by evidence from the retrieved documents.
Confidence score must be between 0 and 100.

Output must be directly usable by a dashboard frontend."""}
    ]

    llm = _get_llm_pipeline()
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                f"LLM inference attempt {attempt}/{retries} — "
                f"model={config.HF_MODEL_NAME}  temp={temperature}  max_new_tokens={max_new_tokens}"
            )
            outputs = llm(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=config.LLM_TOP_P,
                do_sample=temperature > 0,
            )
            # transformers returns: [{"generated_text": [..., {"role": "assistant", "content": "..."}]}]
            text = outputs[0]["generated_text"][-1]["content"]
            logger.info(f"LLM response received ({len(text)} chars)")
            return text

        except Exception as exc:
            last_exc = exc
            wait = config.LLM_RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning(f"LLM inference failed (attempt {attempt}): {exc} — retrying in {wait}s")
            if attempt < retries:
                time.sleep(wait)

    raise RuntimeError(f"LLM inference failed after {retries} attempts: {last_exc}") from last_exc


# ===========================================================================
# STEP 6 — Save LLM result
# ===========================================================================

def save_result(
    query_id:      str,
    query:         str,
    prompt:        str,
    results:       list[dict],
    llm_output:    str,
    collection:    str,
    temperature:   float,
) -> Path:
    """Write full RAG result to /outputs/llm_results/{query_id}.json."""
    config.LLM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.LLM_RESULTS_DIR / f"{query_id}.json"

    payload = {
        "query_id":    query_id,
        "query":       query,
        "prompt":      prompt,
        "collection":  collection,
        "model":       config.HF_MODEL_NAME,
        "temperature": temperature,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_context_docs": len(results),
        "retrieved_context": [
            {
                "rank":       i + 1,
                "chunk_text": r.get("chunk_text", ""),
                "title":      r.get("title", ""),
                "url":        r.get("url", ""),
                "score":      r.get("score"),
                "date":       r.get("date", ""),
                "company":    r.get("company", ""),
            }
            for i, r in enumerate(results)
        ],
        "llm_output": llm_output,
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"LLM result saved → {out_path}")
    return out_path


# ===========================================================================
# Main pipeline loop
# ===========================================================================

def run_rag_pipeline(
    queries_path: Path = None,
    top_k:        int  = None,
    temperature:  float = None,
) -> list[dict]:
    """
    Run the full 6-step RAG pipeline for every query in the queries file.

    Returns a list of summary dicts (one per query).
    """
    queries_path = queries_path or config.QUERIES_JSON_FILE
    queries      = load_queries(queries_path)
    summaries    = []

    for q in queries:
        query_id   = q.get("query_id",   f"q_{int(time.time())}")
        query_text = q.get("query",      "")
        prompt     = q.get("prompt",     query_text)
        collection = q.get("collection", config.CHROMA_COLLECTION_NAME)
        company    = q.get("company",    None)
        q_top_k    = top_k       or q.get("top_k",       config.CHROMA_TOP_K)
        q_temp     = temperature if temperature is not None else q.get("temperature", config.LLM_TEMPERATURE)

        logger.info(f"═══ Query: {query_id} ═══")
        logger.info(f"    Text:       {query_text[:100]}")
        logger.info(f"    Collection: {collection}")

        try:
            # STEP 2: Retrieve
            results = retrieve(query_text, collection, top_k=q_top_k, company_filter=company)

            # STEP 3: Save retrieved docs
            save_retrieved(query_id, query_text, results)

            # STEP 4: Build prompt
            user_prompt = build_user_prompt(prompt, results)

            # STEP 5: Call LLM
            llm_output = call_llm(SYSTEM_PROMPT, user_prompt, temperature=q_temp)

            # STEP 6: Save result
            out_path = save_result(
                query_id=query_id,
                query=query_text,
                prompt=prompt,
                results=results,
                llm_output=llm_output,
                collection=collection,
                temperature=q_temp,
            )

            summaries.append({
                "query_id":  query_id,
                "status":    "ok",
                "docs":      len(results),
                "output":    str(out_path),
            })

        except Exception as exc:
            logger.error(f"Query {query_id} failed: {exc}", exc_info=True)
            summaries.append({"query_id": query_id, "status": "error", "error": str(exc)})
        break

    ok  = sum(1 for s in summaries if s["status"] == "ok")
    err = len(summaries) - ok
    logger.info(f"Pipeline complete — {ok} succeeded, {err} failed")
    return summaries


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG query pipeline — ChromaDB + Together AI")
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help=f"Path to queries JSON file (default: {config.QUERIES_JSON_FILE})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Number of docs to retrieve per query (default: {config.CHROMA_TOP_K})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"LLM temperature (default: {config.LLM_TEMPERATURE})",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries) if args.queries else None
    results = run_rag_pipeline(
        queries_path=queries_path,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    print(f"\n{'═' * 60}")
    print(f"Results:\n{json.dumps(results, indent=2)}")
