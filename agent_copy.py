# agent.py — AI Strategic Intelligence Agent Orchestrator
# Wraps existing rag_query.py + vector_store.py
# into an explicit Goal → Plan → Retrieve → Analyze → Decide → Recommend → Validate loop

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


# ── PLANNING ──────────────────────────────────────────────────────────────────

def plan(ceo_goal: str, llm_pipeline) -> dict:
    """STEP 1: Plan — LLM decides which companies and topics to investigate."""
    system_prompt = """You are an AI strategic planning agent.
Given a CEO's strategic goal, produce a JSON investigation plan.
Return ONLY valid JSON, no commentary, no markdown."""

    user_prompt = f"""CEO GOAL: {ceo_goal}

Available companies: Lufthansa, Air India, United Airlines, Delta Air Lines, American Airlines

Return this JSON:
{{
  "focus_companies": ["<company1>", "<company2>"],
  "investigation_topics": ["<topic1>", "<topic2>", "<topic3>"],
  "primary_concern": "<risk | opportunity | competitive | operational>",
  "rationale": "<one sentence why this plan addresses the CEO goal>"
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    output = llm_pipeline(messages, max_new_tokens=512, temperature=0.1, do_sample=False)
    raw = output[0]["generated_text"]
    raw = re.sub(r'<think>[\s\S]*?</think>', '', raw).strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            plan_dict = json.loads(m.group())
        except json.JSONDecodeError:
            plan_dict = None
    else:
        plan_dict = None

    if not plan_dict:
        plan_dict = {
            "focus_companies": ["Lufthansa"],
            "investigation_topics": [ceo_goal],
            "primary_concern": "competitive",
            "rationale": "Fallback plan — LLM did not return structured output",
        }

    log.info(f"[PLAN] Focus: {plan_dict['focus_companies']}, Topics: {plan_dict['investigation_topics']}")
    return plan_dict


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
    MODEL_ID = "Qwen/Qwen3-8B"

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

    # STEP 1: PLAN
    log.info("── Step 1: Planning ──")
    agent_plan = plan(ceo_goal, llm)

    # STEP 2: RETRIEVE
    log.info("── Step 2: Retrieving evidence ──")
    retrieved_evidence = {}
    for company in agent_plan["focus_companies"]:
        for topic in agent_plan["investigation_topics"]:
            key = f"{company}::{topic}"
            chunks = tool_search_knowledge(company, topic, top_k=5)
            retrieved_evidence[key] = chunks

    if len(agent_plan["focus_companies"]) > 1:
        for topic in agent_plan["investigation_topics"][:2]:
            comparison = tool_compare_companies(agent_plan["focus_companies"], topic)
            for company, texts in comparison.items():
                retrieved_evidence[f"COMPARISON::{company}::{topic}"] = texts

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
        "plan": agent_plan,
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