# AI Strategic Intelligence Agent
**RAG System for Airline Competitive Intelligence**

> **Course:** NLP Final Project — SRH  University of Applied Sciences Heidelberg|

---

## System Architecture

Two independent phases — build the knowledge base once, query it any time.

```
WEB SOURCES          PROCESSING           VECTOR DB
──────────────       ──────────────       ──────────────────────
Lufthansa            TextCleaner          Lufthansa_Knowledge
Air India      ───▶  Deduplicator  ───▶   AirIndia_Knowledge
United               TextChunker          UnitedAirlines_Knowledge
Delta                                     DeltaAirlines_Knowledge
American             BGE Embedder         AmericanAirlines_Knowledge
Skytrax Reviews      (768-dim vectors)    Reviews_Knowledge
(Hyperbrowser)                                    │
(Apify)                                           │
                                                  ▼
USER QUERY ──▶ embed_query() ──▶ HNSW Search ──▶ Qwen 3 LLM ──▶ OUTPUT
               (BGE, same model)   top-K chunks    35B, 8-bit     llm_results/
```

---

## Data Flow

```
links.json ──▶ scraper_1.py ──▶ processor.py ──▶ embedder.py ──▶ vector_store.py
               _fetch_page()    clean_all()       embed_chunks()   upsert_embeddings()
               _to_raw()        deduplicate()     768-dim vector   ChromaDB Cloud
                                chunk_all()       per chunk
                                1000-char windows
                                150-char overlap
                                         │
                              ───────────┘ (stored)
                                         │
queries.json ──▶ retrieve() ──▶ build_user_prompt() ──▶ call_llm() ──▶ save_result()
                 embed query     prompt + context        Qwen 3 35B     outputs/
                 cosine search   chunks injected         temp=0.2       llm_results/
                 top-K chunks    as evidence
```

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Web Scraping | Hyperbrowser (headless Chrome) | Airline sites are JS-rendered SPAs — plain requests returns empty HTML |
| Review Scraping | Apify (Skytrax actor) | Handles bot detection and pagination on review sites |
| Text Cleaning | BeautifulSoup4 + regex | Strip HTML, remove boilerplate, NFKC normalisation |
| Embedding Model | BAAI/bge-base-en-v1.5 (768-dim) | Trained for asymmetric retrieval — short query vs long document |
| Embedding Fallback | all-MiniLM-L6-v2 (384-dim) | Faster on CPU, switchable via `config.EMBEDDING_MODEL` |
| Vector Database | ChromaDB Cloud | Hosted HNSW index, cosine similarity, one collection per airline |
| LLM | Qwen 3 35B (8-bit quantised) | Decoder-only, strong instruction following, runs locally |
| LLM Runtime | HuggingFace Transformers + BitsAndBytes | 8-bit quant reduces 140GB → 35GB VRAM |
| Config | `config.py` | Single source of truth for all API keys, paths, model settings |

---

## Design Decisions

**One collection per airline** — Separate ChromaDB collections prevent competitor chunks from appearing in a Lufthansa-specific query. Each airline's data can be rebuilt independently.

**Character-based chunking (1000 chars)** — Token counts are model-dependent. 1000 chars ≈ 250 tokens, safely within BGE's 512-token limit without importing any tokenizer at processing time.

**BGE over BERT/MiniLM** — BGE was trained specifically for retrieval on (query, document) pairs. Its query prefix (`"Represent this sentence for searching relevant passages: "`) shifts the query vector into the document's subspace — no other free local model has this by design.

**Cosine similarity** — Measures angle between vectors, ignoring magnitude. A short focused chunk and a long article about the same topic score equally if their direction matches. With unit-normalised vectors (normalize_embeddings=True), cosine = dot product — the fastest possible computation.

**Upsert over insert** — Re-running the pipeline overwrites existing vectors by chunk_id rather than creating duplicates. Pipeline is safe to re-run after scraping updates.

**Split audit outputs** — `retrieved_docs/` saved before LLM call, `llm_results/` after. If an answer is wrong, you can check whether the failure was retrieval (wrong chunks) or generation (right chunks, LLM misused them).

---

## AI Pipeline

```
INDEXING (run_pipeline.py)          RETRIEVAL (rag_query.py)

chunk_text                          query string
     │                                   │
_prepare_text()                     embed_query()
"{title}. {chunk_text}"            BGE_QUERY_PREFIX + query
     │                                   │
model.encode()                      model.encode()
normalize=True                      normalize=True
     │                                   │
768-dim unit vector    ◀──────────▶ 768-dim unit vector
stored in ChromaDB      cosine sim   compared via HNSW
                        score = 1 - distance
```

**Why asymmetric?** BGE uses different subspaces for documents vs queries. The query prefix shifts the query vector to align with document vectors. Using the prefix at indexing time would break this alignment.

**Prompt assembly:**
```
[Analytical prompt from queries.json]
---
RETRIEVED CONTEXT
[1] Title | Source | Date | score=0.91
    chunk_text (~250 tokens)
[2] ...
---
```
The LLM is instructed to answer only from this context. It cannot hallucinate — every claim must trace to a retrieved chunk.

---

## Project Structure

```
├── config.py            ← All settings: API keys, chunk size, model names
├── run_pipeline.py      ← Entry point: scrape → process → embed → store
├── rag_query.py         ← Entry point: retrieve → prompt → LLM → save
├── scraper_1.py         ← Hyperbrowser BFS scraper
├── processor.py         ← Clean → deduplicate → chunk
├── embedder.py          ← BGE/MiniLM embedding
├── vector_store.py      ← ChromaDB wrapper
├── review_scraper.py    ← Skytrax reviews via Apify
├── links.json           ← URLs to scrape per company
├── queries.json         ← Strategic questions for RAG
├── data/                ← Raw scraped JSON files
└── outputs/
    ├── retrieved_docs/  ← Retrieval audit (one JSON per query)
    └── llm_results/     ← LLM answers (one JSON per query)
```

---

## Setup & Usage

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Build knowledge base
python run_pipeline.py

# Skip scraping, re-embed existing data
python run_pipeline.py --skip-scrape

# Run all queries
python rag_query.py

# Run single query
python rag_query.py --queries single_query.json --top-k 5
```

---

