# agent.py — AI Strategic Intelligence Agent Orchestrator
# Flow: CEO Goal → Embed Goal → Search All Companies → Rank by Similarity
#       → Analyze Top Chunks → Recommend → Validate

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import config
from embedder import EmbeddingGenerator
from vector_store import ChromaVectorStore

log = logging.getLogger("agent")


# ── TOOLS ─────────────────────────────────────────────────────────────────────

def tool_search_knowledge(company: str, query: str, top_k: int = 5) -> list[dict]:
    """Tool 1: Search a company's knowledge base in ChromaDB."""
    collection = config.COLLECTION_MAP.get(company)
    if not collection:
        log.warning(f"No collection for company: {company}")
        return []
    embedder = EmbeddingGenerator(config.EMBEDDING_MODEL)
    store = ChromaVectorStore(collection_name=collection)
    results = store.search_by_text(query_text=query, embedder=embedder, top_k=top_k)
    log.info(f"[TOOL] search_knowledge({company}, '{query[:50]}') → {len(results)} results")
    return results


def tool_compare_companies(companies: list[str], aspect: str) -> dict[str, list]:
    """Tool 2: Cross-company comparison on a specific aspect."""
    comparison = {}
    for company in companies:
        results = tool_search_knowledge(company, aspect, top_k=3)
        comparison[company] = [r.get("chunk_text", "")[:300] for r in results]
        log.info(f"[TOOL] compare_companies → {company}: {len(results)} chunks")
    return comparison


def tool_validate_recommendation(recommendation: str, evidence: list[str], confidence_score: int) -> dict:
    """Tool 3: Validate recommendation before presenting to CEO."""
    issues = []
    passed = True

    if confidence_score < 40:
        issues.append(f"Low confidence ({confidence_score}/100) — insufficient evidence")
        passed = False
    if len(evidence) < 2:
        issues.append("Too few evidence points — retrieval may have failed")
        passed = False
    if len(recommendation.strip()) < 20:
        issues.append("Recommendation too vague")
        passed = False

    result = {
        "passed": passed,
        "confidence_score": confidence_score,
        "evidence_count": len(evidence),
        "issues": issues,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    log.info(f"[TOOL] validate_recommendation → passed={passed}, issues={issues}")
    return result

def tool_analyze_sentiment(chunks: list[str]) -> dict:
    """Simple sentiment scoring on retrieved text chunks."""
    from textblob import TextBlob
    scores = []
    for chunk in chunks:
        blob = TextBlob(chunk[:500] if isinstance(chunk, str) else chunk.get("chunk_text","")[:500])
        scores.append(blob.sentiment.polarity)  # -1 to +1
    avg = sum(scores) / len(scores) if scores else 0
    label = "Positive" if avg > 0.1 else "Negative" if avg < -0.1 else "Neutral"
    return {
        "average_score": round(avg, 3),
        "label": label,
        "news_sentiment": label,
        "public_sentiment": label,
        "chunk_scores": scores[:10]
    }


# ── RETRIEVAL ─────────────────────────────────────────────────────────────────

def retrieve_by_goal(ceo_goal: str, top_k_per_company: int = 5) -> dict:
    """
    STEP 1: Embed the CEO goal and search ALL company collections directly.
    Companies and topics are selected by cosine similarity.
    Returns: {company_name: [chunk_dicts sorted by similarity score]}
    """
    embedder = EmbeddingGenerator(config.EMBEDDING_MODEL)
    retrieved = {}

    for company, collection_name in config.COLLECTION_MAP.items():
        store = ChromaVectorStore(collection_name=collection_name)
        results = store.search_by_text(
            query_text=ceo_goal,
            embedder=embedder,
            top_k=top_k_per_company,
        )
        if results:
            retrieved[company] = results
            top_score = results[0].get("score", 0)
            log.info(f"[RETRIEVE] {company}: {len(results)} chunks, top score: {top_score:.4f}")
        else:
            log.info(f"[RETRIEVE] {company}: no results")

    # Derive focus companies — those with highest top-chunk similarity score
    scored = sorted(
        retrieved.items(),
        key=lambda x: x[1][0].get("score", 0) if x[1] else 0,
        reverse=True,
    )
    focus_companies = [company for company, _ in scored if scored]

    log.info(f"[RETRIEVE] Companies ranked by relevance: {focus_companies}")
    return retrieved, focus_companies


# ── ANALYSIS ──────────────────────────────────────────────────────────────────

def analyze(retrieved_evidence: dict, llm_pipeline) -> dict:
    """STEP 3: Analyze — LLM analyzes retrieved evidence, extracts risks and opportunities."""
    context_lines = []
    for key, chunks in retrieved_evidence.items():
        context_lines.append(f"=== {key} ===")
        for chunk in chunks:
            context_lines.append(
                chunk[:500] if isinstance(chunk, str) else chunk.get("chunk_text", "")[:500]
            )
    context = "\n".join(context_lines)[:6000]

    system_prompt = "You are a strategic analyst. Analyze evidence and return structured JSON only. No markdown, no commentary."

    user_prompt = f"""Analyze this evidence and return ONLY this JSON:
{{
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"],
  "risks": ["<risk 1>", "<risk 2>"],
  "opportunities": ["<opportunity 1>", "<opportunity 2>"],
  "decision": "<invest | divest | monitor | act_immediately>",
  "confidence_score": <integer 0-100>
}}

EVIDENCE:
{context}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    output = llm_pipeline(messages, max_new_tokens=1024, temperature=0.1, do_sample=False)
    raw = output[0]["generated_text"]
    raw = re.sub(r'<think>[\s\S]*?</think>', '', raw).strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            analysis = json.loads(m.group())
        except json.JSONDecodeError:
            analysis = None
    else:
        analysis = None

    if not analysis:
        analysis = {
            "key_findings": [],
            "risks": [],
            "opportunities": [],
            "decision": "monitor",
            "confidence_score": 0,
        }

    log.info(f"[ANALYZE] Decision: {analysis.get('decision')}, Confidence: {analysis.get('confidence_score')}")
    return analysis


# ── RECOMMENDATION ────────────────────────────────────────────────────────────

def recommend(ceo_goal: str, analysis: dict, llm_pipeline) -> dict:
    """STEP 4: Decide + Recommend — generate a concrete CEO recommendation."""
    system_prompt = "You are the AI Strategic Advisor to the CEO. Return ONLY valid JSON. No markdown, no commentary."

    user_prompt = f"""CEO GOAL: {ceo_goal}

ANALYSIS FINDINGS:
{json.dumps(analysis, indent=2)}

Return ONLY this JSON:
{{
  "recommendation": "<one clear actionable sentence>",
  "priority": "<High | Medium | Low>",
  "expected_impact": "<2 sentences on business impact>",
  "supporting_evidence": ["<evidence 1>", "<evidence 2>", "<evidence 3>"],
  "risk_if_ignored": "<1 sentence on consequence of inaction>",
  "confidence_score": <integer 0-100>
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    output = llm_pipeline(messages, max_new_tokens=1024, temperature=0.15, do_sample=True)
    raw = output[0]["generated_text"]
    raw = re.sub(r'<think>[\s\S]*?</think>', '', raw).strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            rec = json.loads(m.group())
        except json.JSONDecodeError:
            rec = {}
    else:
        rec = {}

    log.info(f"[RECOMMEND] Priority: {rec.get('priority')}")
    return rec


# ── MAIN AGENT LOOP ───────────────────────────────────────────────────────────

def run_agent(ceo_goal: str, output_path: Path = None) -> dict:
    """Full agent loop: Goal → Plan → Retrieve → Analyze → Decide → Recommend → Validate"""
    import os
    import requests

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable not set.")

    HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
    MODEL_ID = "Qwen/Qwen3-8B:featherless-ai"

    def llm(messages, max_new_tokens=1024, temperature=0.1, **kwargs):
        payload = {
            "model": MODEL_ID,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }

        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()

        data = resp.json()
        msg = data["choices"][0]["message"]
        # Qwen3-8B is a thinking model — actual answer may be in reasoning_content
        content = msg.get("content") or msg.get("reasoning_content") or ""

        return [{"generated_text": content}]

    log.info("HuggingFace Router API ready")
    log.info(f"═══ AGENT START ═══")
    log.info(f"Goal: {ceo_goal}")

    # Input validation guard
    if len(ceo_goal.strip()) < 20:
        log.warning("Goal too short — not a valid strategic question")
        return {"error": "Please provide a specific strategic goal or question."}

    # STEP 1: EMBED GOAL + RETRIEVE by similarity across all companies
    log.info("── Step 1: Embedding goal and retrieving by similarity ──")
    retrieved_evidence, focus_companies = retrieve_by_goal(ceo_goal, top_k_per_company=5)

    # Derive investigation topics — extract meaningful bigrams from CEO goal
    _stop = {
        'what', 'how', 'should', 'the', 'a', 'an', 'and', 'or', 'in', 'of',
        'to', 'for', 'is', 'are', 'be', 'with', 'given', 'current', 'about',
        'do', 'does', 'its', 'its', 'their', 'our', 'we', 'i', 'that', 'this',
        'if', 'would', 'could', 'can', 'will', 'which', 'on', 'at', 'by',
        'from', 'into', 'than', 'as', 'its', 'any', 'vs', 'versus'
    }
    _words = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', ceo_goal) if w.lower() not in _stop]
    # Build bigrams from meaningful words
    _bigrams = [f"{_words[i]} {_words[i+1]}" for i in range(len(_words) - 1)]
    investigation_topics = (_bigrams[:3] if len(_bigrams) >= 3 else _bigrams + _words)[:3]
    if not investigation_topics:
        investigation_topics = [ceo_goal[:50]]

    # Classify primary concern from goal keywords
    _goal_lower = ceo_goal.lower()
    if any(w in _goal_lower for w in ["risk", "threat", "danger", "loss", "decline", "problem", "challenge"]):
        primary_concern = "risk"
    elif any(w in _goal_lower for w in ["invest", "expand", "grow", "opportunity", "potential", "partner", "acqui"]):
        primary_concern = "opportunity"
    elif any(w in _goal_lower for w in ["competitor", "competition", "rival", "positioning", "market share"]):
        primary_concern = "competitive"
    else:
        primary_concern = "operational"

    retrieval_summary = {
        "focus_companies": focus_companies[:3],
        "investigation_topics": investigation_topics,
        "primary_concern": primary_concern,
        "rationale": (
            f"Embedding similarity search identified {focus_companies[0] if focus_companies else 'relevant companies'} "
            f"as most relevant to the CEO goal. Analysis focuses on {primary_concern} aspects."
        ),
        "retrieval_method": "embedding_similarity",
        "total_chunks": sum(len(v) for v in retrieved_evidence.values()),
    }

    # STEP 2: SENTIMENT across all retrieved chunks
    log.info("── Step 2: Sentiment analysis ──")
    all_chunks = [c.get("chunk_text", "") if isinstance(c, dict) else c
                  for chunks in retrieved_evidence.values() for c in chunks]
    sentiment = tool_analyze_sentiment(all_chunks)

    # STEP 3: ANALYZE
    log.info("── Step 3: Analyzing evidence ──")
    analysis = analyze(retrieved_evidence, llm)

    # STEP 4: RECOMMEND
    log.info("── Step 4: Generating recommendation ──")
    recommendation = recommend(ceo_goal, analysis, llm)

    # STEP 5: VALIDATE
    log.info("── Step 5: Validating recommendation ──")
    validation = tool_validate_recommendation(
        recommendation=recommendation.get("recommendation", ""),
        evidence=recommendation.get("supporting_evidence", []),
        confidence_score=recommendation.get("confidence_score", 0),
    )

    if not validation["passed"]:
        log.warning(f"[VALIDATE] Failed validation: {validation['issues']}")
        recommendation["validation_warning"] = validation["issues"]
    else:
        log.info("[VALIDATE] Recommendation passed validation ✓")

    agent_output = {
        "agent_run_id": f"agent_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "ceo_goal": ceo_goal,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_summary": retrieval_summary,
        "analysis": analysis,
        "recommendation": recommendation,
        "validation": validation,
        "sentiment": sentiment,
    }

    out_path = output_path or Path("outputs/agent_results") / f"{agent_output['agent_run_id']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(agent_output, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"═══ AGENT COMPLETE → {out_path} ═══")

    return agent_output


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser(description="AI Strategic Intelligence Agent")
    parser.add_argument("--goal", type=str, required=True, help="CEO strategic goal or question")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    result = run_agent(
        ceo_goal=args.goal,
        output_path=Path(args.output) if args.output else None,
    )
    print(json.dumps(result, indent=2))