#!/usr/bin/env python3
"""
dashboard_builder.py — Standalone dashboard content pipeline
============================================================

Reads Qwen LLM result JSONs from  outputs/llm_results/
→ sends each section's content to local Qwen model for structuring
→ validates JSON (retries up to 3x, falls back to raw text on failure)
→ writes  outputs/dashboard/section_X.json  per section
→ synthesises section_7 from sections 2–6
→ merges everything into  outputs/dashboard/dashboard_payload.json

Usage:
    python dashboard_builder.py                        # all sections
    python dashboard_builder.py --section section_2    # one section only
    python dashboard_builder.py --dry-run              # preview inputs, no LLM calls
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.modules["torchvision"] = None
sys.modules["torchaudio"] = None
# =============================================================================
# ── CONFIGURATION  (all settings live here) ──────────────────────────────────
# =============================================================================

LLM_MODEL       = "Qwen/Qwen3.6-35B-A3B"     # local model via transformers
TEMPERATURE     = 0.1          # low = more consistent JSON
MAX_NEW_TOKENS  = 2048
MAX_RETRIES     = 3
RETRY_DELAY     = 2            # seconds (exponential: 2 → 4 → 8)

BASE_DIR         = Path(__file__).parent
LLM_RESULTS_DIR  = BASE_DIR / "outputs" / "llm_results"
DASHBOARD_DIR    = BASE_DIR / "outputs" / "dashboard"

# Which query IDs feed each section
SECTION_MAP: dict[str, list[str]] = {
    "section_2": ["q001", "q0021", "q0022", "q0023", "q0024",
                  "q0031", "q0032", "q0033", "q0034", "q0035", "q004"],
    "section_3": ["q005", "q006"],
    "section_4": ["q0071", "q0072", "q0073", "q0074", "q0075", "q0076", "q008", "q009"],
    "section_5": ["q010", "q011"],
    "section_6": ["q012"],
}

# =============================================================================
# ── LOGGING ───────────────────────────────────────────────────────────────────
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("dashboard_builder")

# =============================================================================
# ── SECTION 1 — HARDCODED COMPANY OVERVIEW (static content) ──────────────────
# =============================================================================

def build_section_1() -> dict:
    """
    Section 1: Company Overview — static, hardcoded.
    Displays: company name, industry, num documents, num sources, last update.
    num_documents is counted dynamically from whatever q*.json files exist.
    """
    # Count available result files for a live document count
    num_docs = len(list(LLM_RESULTS_DIR.glob("q*.json"))) if LLM_RESULTS_DIR.exists() else 0

    payload = {
        "section_id":   "section_1",
        "title":        "Company Overview",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "company_name":           "Lufthansa Group",
            "industry":               "Aviation / Commercial Airline",
            "headquarters":           "Cologne, Germany",
            "founded":                1953,
            "stock_ticker":           "LHA.XE",
            "competitors_monitored":  ["Air India", "United Airlines", "Delta Air Lines", "American Airlines"],
            "num_collected_documents": num_docs,
            "num_data_sources":       5,
            "data_sources": [
                "Lufthansa Newsroom (newsroom.lufthansagroup.com)",
                "Investor Relations (investor-relations.lufthansagroup.com)",
                "Skytrax Customer Reviews (airlinequality.com)",
                "Competitor Websites",
                "Industry & Finance News",
            ],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        },
    }

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / "section_1.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"  Section 1 (static) saved → {out_path}")
    return payload


# =============================================================================
# ── SECTION SCHEMAS & PROMPTS (sections 2–6) ─────────────────────────────────
# =============================================================================

SECTION_SYSTEM_PROMPT = """\
You are a JSON formatter for an aviation executive intelligence dashboard.
You receive raw analytical text and must convert it into clean, structured JSON
that a frontend React/Vue application can render directly.

STRICT RULES:
- Respond with VALID JSON only. No markdown fences, no commentary, no extra text.
- If the source text lacks information for a field, use null.
- All string values must be properly escaped.
- Arrays must always be actual JSON arrays, never plain strings.
"""

SECTION_USER_TEMPLATES: dict[str, str] = {

    # ── Section 2: Market Intelligence ───────────────────────────────────────
    # Dashboard displays: recent news, competitor activities, emerging technologies,
    # important company announcements.
    "section_2": """\
You are formatting Market Intelligence data for an executive dashboard.
Extract from the source text and return ONLY this JSON (no other text):

{{
  "query_id": "{query_id}",
  "company": "{company}",
  "category": "<one of: recent_news | competitor_activity | emerging_technology | company_announcement>",
  "headline": "<one sentence headline of the most important market development>",
  "summary": "<2–3 sentences summarising the market intelligence finding>",
  "key_developments": [
    "<specific development 1>",
    "<specific development 2>",
    "<specific development 3>"
  ],
  "analysis_date": "{analysis_date}",
  "num_sources": {num_sources}
}}

SOURCE TEXT:
{llm_output}
""",

    # ── Section 3: Opportunity Monitor ───────────────────────────────────────
    # Dashboard displays: opportunity title, impact level, evidence, confidence score.
    "section_3": """\
You are formatting an Opportunity Monitor entry for an executive dashboard.
Extract a single business opportunity from the source text and return ONLY this JSON:

{{
  "query_id": "{query_id}",
  "company": "{company}",
  "opportunity_title": "<short title of the opportunity>",
  "description": "<2–3 sentences describing the opportunity and why it matters>",
  "impact_level": "<high | medium | low>",
  "evidence": [
    "<specific evidence point 1 from the text>",
    "<specific evidence point 2 from the text>",
    "<specific evidence point 3 from the text>"
  ],
  "confidence_score": <integer 0–100 reflecting how well-supported this opportunity is>,
  "analysis_date": "{analysis_date}",
  "num_sources": {num_sources}
}}

SOURCE TEXT:
{llm_output}
""",

    # ── Section 4: Risk Monitor ───────────────────────────────────────────────
    # Dashboard displays: risk title, risk category, severity level, evidence, confidence score.
    "section_4": """\
You are formatting a Risk Monitor entry for an executive dashboard.
Extract a single business risk from the source text and return ONLY this JSON:

{{
  "query_id": "{query_id}",
  "company": "{company}",
  "risk_title": "<short title of the risk>",
  "risk_category": "<one of: competitive | regulatory | financial | operational | reputational | supply_chain | technology>",
  "severity_level": "<high | medium | low>",
  "evidence": [
    "<specific evidence point 1 from the text>",
    "<specific evidence point 2 from the text>",
    "<specific evidence point 3 from the text>"
  ],
  "confidence_score": <integer 0–100 reflecting how well-supported this risk assessment is>,
  "analysis_date": "{analysis_date}",
  "num_sources": {num_sources}
}}

SOURCE TEXT:
{llm_output}
""",

    # ── Section 5: Sentiment Analysis ────────────────────────────────────────
    # Dashboard displays: news sentiment, public sentiment, sentiment trends.
    # Include scores for visualisations.
    "section_5": """\
You are formatting a Sentiment Analysis entry for an executive dashboard.
Analyse the sentiment in the source text and return ONLY this JSON:

{{
  "query_id": "{query_id}",
  "company": "{company}",
  "news_sentiment": "<positive | neutral | negative>",
  "news_sentiment_score": <float -1.0 to 1.0, where 1.0 = very positive>,
  "public_sentiment": "<positive | neutral | negative>",
  "public_sentiment_score": <float -1.0 to 1.0>,
  "sentiment_summary": "<2–3 sentences describing the overall sentiment landscape>",
  "sentiment_trends": [
    "<trend observation 1>",
    "<trend observation 2>",
    "<trend observation 3>"
  ],
  "key_themes": ["<theme driving sentiment 1>", "<theme 2>", "<theme 3>"],
  "analysis_date": "{analysis_date}",
  "num_sources": {num_sources}
}}

SOURCE TEXT:
{llm_output}
""",

    # ── Section 6: Strategic Recommendations ─────────────────────────────────
    # Dashboard displays: recommendation, priority (High/Medium/Low),
    # supporting evidence, expected impact, risk level.
    "section_6": """\
You are formatting Strategic Recommendations for an executive dashboard.
Extract ONE actionable strategic recommendation from the source text and return ONLY this JSON:

{{
  "query_id": "{query_id}",
  "company": "{company}",
  "recommendation": "<clear, actionable recommendation in one sentence>",
  "priority": "<High | Medium | Low>",
  "supporting_evidence": [
    "<evidence point 1 from the text>",
    "<evidence point 2 from the text>",
    "<evidence point 3 from the text>"
  ],
  "expected_impact": "<2–3 sentences on the anticipated business impact if this recommendation is followed>",
  "risk_level": "<High | Medium | Low>",
  "analysis_date": "{analysis_date}",
  "num_sources": {num_sources}
}}

SOURCE TEXT:
{llm_output}
""",
}

# ── Section 7: CEO Briefing ───────────────────────────────────────────────────
# Answers three questions: What happened? Why does it matter? What should management do next?

SECTION_7_SYSTEM_PROMPT = """\
You are the AI Strategic Intelligence Advisor briefing the CEO of Lufthansa Group.
You receive structured intelligence from five analysis sections and must produce a concise
executive briefing in JSON format.

STRICT RULES:
- Respond with VALID JSON only. No markdown fences, no extra text.
- what_happened: 3–4 sentences covering the most critical recent developments across all sections.
- why_it_matters: 3–4 sentences explaining the business significance and strategic implications.
- management_actions: list of 3–5 specific, prioritised action items for the CEO.
- Each action item must be a clear imperative sentence (e.g. "Accelerate the Starlink rollout...").
"""

SECTION_7_USER_TEMPLATE = """\
Based on the following intelligence sections, generate the CEO Briefing JSON:

{{
  "what_happened": "<3–4 sentence summary of the most critical events and findings across all sections>",
  "why_it_matters": "<3–4 sentence explanation of business significance and strategic implications>",
  "management_actions": [
    "<prioritised action 1 — most urgent>",
    "<action 2>",
    "<action 3>",
    "<action 4 — optional>",
    "<action 5 — optional>"
  ]
}}

INTELLIGENCE SECTIONS:
{sections_text}
"""

# =============================================================================
# ── HELPERS ──────────────────────────────────────────────────────────────────
# =============================================================================

def _load_llm_result(query_id: str) -> dict | None:
    """Load a single q*.json from outputs/llm_results/. Returns None if missing."""
    path = LLM_RESULTS_DIR / f"{query_id}.json"
    if not path.exists():
        log.warning(f"  [{query_id}] File not found — skipping: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_company(result: dict) -> str:
    """Best-effort company extraction from retrieved_context."""
    ctx = result.get("retrieved_context", [])
    for item in ctx:
        c = item.get("company", "").strip()
        if c:
            return c
    return "Unknown"


def _extract_json_from_text(text: str) -> dict | list | None:
    """
    Try to pull valid JSON out of a response that may contain surrounding text.
    Tries: 1) direct parse, 2) first {...} block, 3) first [...] block.
    """
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. First {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # 3. First [...] block
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return None


def _is_valid_entry(parsed: dict | list, required_keys: list[str]) -> bool:
    """Return True if parsed is a dict containing all required_keys."""
    if not isinstance(parsed, dict):
        return False
    return all(k in parsed for k in required_keys)


# =============================================================================
# ── LLM CALL ─────────────────────────────────────────────────────────────────
# =============================================================================

# =============================================================================
# ── LOCAL LLM SINGLETON ───────────────────────────────────────────────────────
# =============================================================================

_llm_pipeline = None


def _get_pipeline():
    global _llm_pipeline
    if _llm_pipeline is None:
        import torch
        from transformers import pipeline, BitsAndBytesConfig
        log.info(f"Loading local LLM: {LLM_MODEL} (8-bit quantised) — this takes a minute...")
        _llm_pipeline = pipeline(
            task="text-generation",
            model=LLM_MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_8bit=True)},
        )
        log.info("LLM pipeline ready")
    return _llm_pipeline


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Run inference with local Qwen pipeline. Raises on failure."""
    llm = _get_pipeline()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    outputs = llm(
        messages,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.9,
        do_sample=TEMPERATURE > 0,
    )
    # transformers output: [{"generated_text": [..., {"role": "assistant", "content": "..."}]}]
    return outputs[0]["generated_text"][-1]["content"]


# =============================================================================
# ── ENTRY STRUCTURING ────────────────────────────────────────────────────────
# =============================================================================

def structure_entry(
    query_id: str,
    result: dict,
    section_id: str,
    required_keys: list[str],
    dry_run: bool = False,
) -> dict:
    """
    For one query result, call Llama to produce a structured dict.
    Retries up to MAX_RETRIES times, then falls back to a raw-text entry.
    """
    company       = _extract_company(result)
    llm_output    = result.get("llm_output", "")
    analysis_date = result.get("generated_at", "")[:10]
    num_sources   = result.get("num_context_docs", 0)

    template = SECTION_USER_TEMPLATES[section_id]
    user_prompt = template.format(
        query_id=query_id,
        company=company,
        llm_output=llm_output[:6000],   # cap to avoid token overflow
        analysis_date=analysis_date,
        num_sources=num_sources,
    )

    if dry_run:
        log.info(f"  [DRY RUN] Would call LLM for {query_id} ({section_id})")
        return {"query_id": query_id, "company": company, "dry_run": True}

    last_raw = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"  [{query_id}] LLM call attempt {attempt}/{MAX_RETRIES}")
            raw = call_llm(SECTION_SYSTEM_PROMPT, user_prompt)
            last_raw = raw
            parsed = _extract_json_from_text(raw)

            if parsed and _is_valid_entry(parsed, required_keys):
                log.info(f"  [{query_id}] ✓ Valid JSON on attempt {attempt}")
                return parsed
            else:
                log.warning(
                    f"  [{query_id}] Attempt {attempt}: JSON invalid or missing keys {required_keys}. "
                    f"Raw (first 300 chars): {raw[:300]!r}"
                )

        except Exception as e:
            last_raw = f"ERROR on attempt {attempt}: {e}"
            log.warning(f"  [{query_id}] Attempt {attempt} error: {e}", exc_info=True)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))

    # Fallback — return raw text so frontend can at least show something
    log.error(f"  [{query_id}] All {MAX_RETRIES} attempts failed — using raw fallback")
    return {
        "query_id":   query_id,
        "company":    company,
        "status":     "structuring_failed",
        "error":      last_raw[:500],   # actual error or truncated bad response
    }


# =============================================================================
# ── PER-SECTION PIPELINE ─────────────────────────────────────────────────────
# =============================================================================

# Keys that must be present for a structured entry to be considered valid
REQUIRED_KEYS: dict[str, list[str]] = {
    "section_2": ["query_id", "headline", "summary", "key_developments", "category"],
    "section_3": ["query_id", "opportunity_title", "impact_level", "evidence", "confidence_score"],
    "section_4": ["query_id", "risk_title", "risk_category", "severity_level", "evidence", "confidence_score"],
    "section_5": ["query_id", "news_sentiment", "public_sentiment", "sentiment_trends"],
    "section_6": ["query_id", "recommendation", "priority", "supporting_evidence", "expected_impact", "risk_level"],
}


def build_section(section_id: str, dry_run: bool = False) -> dict:
    """
    Process all query IDs for a section, call Llama for each,
    and return a section payload dict.
    """
    query_ids    = SECTION_MAP[section_id]
    required     = REQUIRED_KEYS[section_id]
    entries      = []
    skipped      = []

    log.info(f"─── {section_id}: {len(query_ids)} queries ───────────────────")

    for qid in query_ids:
        result = _load_llm_result(qid)
        if result is None:
            skipped.append(qid)
            continue

        entry = structure_entry(qid, result, section_id, required, dry_run=dry_run)
        entries.append(entry)

    section_payload = {
        "section_id":   section_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status":       "complete" if not skipped else "partial",
        "num_entries":  len(entries),
        "skipped_ids":  skipped,
        "entries":      entries,
    }

    # Save individual section file
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / f"{section_id}.json"
    out_path.write_text(
        json.dumps(section_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"  Saved → {out_path}  ({len(entries)} entries, {len(skipped)} skipped)")
    return section_payload


# =============================================================================
# ── SECTION 7 SYNTHESIS ──────────────────────────────────────────────────────
# =============================================================================

def build_section_7(sections: dict[str, dict], dry_run: bool = False) -> dict:
    """
    Synthesise sections 2–6 into an executive summary (section 7).
    Reads from already-built section payloads.
    """
    log.info("─── section_7: synthesising from sections 2–6 ─────────────")

    # Condense each section to a text block for the LLM prompt
    # Uses the primary field for each section as the "headline" fallback
    SECTION_HEADLINE_FIELD = {
        "section_2": "headline",
        "section_3": "opportunity_title",
        "section_4": "risk_title",
        "section_5": "sentiment_summary",
        "section_6": "recommendation",
    }
    SECTION_DETAIL_FIELD = {
        "section_2": "key_developments",
        "section_3": "evidence",
        "section_4": "evidence",
        "section_5": "sentiment_trends",
        "section_6": "supporting_evidence",
    }

    condensed_parts = []
    for sid, payload in sections.items():
        if sid == "section_1":
            continue   # static overview, not useful for synthesis
        condensed_parts.append(f"=== {sid.upper()} ===")
        h_field = SECTION_HEADLINE_FIELD.get(sid, "headline")
        d_field = SECTION_DETAIL_FIELD.get(sid, "key_developments")
        for entry in payload.get("entries", []):
            qid     = entry.get("query_id", "?")
            company = entry.get("company",  "?")
            title   = entry.get(h_field, entry.get("error", "no data"))
            details = entry.get(d_field, [])
            if isinstance(details, list) and details:
                condensed_parts.append(
                    f"[{qid} | {company}] {title}\n  • " + "\n  • ".join(str(d) for d in details[:3])
                )
            else:
                condensed_parts.append(f"[{qid} | {company}] {title}")
        condensed_parts.append("")

    sections_text = "\n".join(condensed_parts)

    if dry_run:
        log.info("  [DRY RUN] Would synthesise section_7")
        section_7 = {"section_id": "section_7", "dry_run": True}
    else:
        user_prompt = SECTION_7_USER_TEMPLATE.format(sections_text=sections_text[:8000])
        last_raw = ""
        section_7 = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                log.info(f"  [section_7] LLM synthesis attempt {attempt}/{MAX_RETRIES}")
                raw = call_llm(SECTION_7_SYSTEM_PROMPT, user_prompt)
                last_raw = raw
                parsed = _extract_json_from_text(raw)

                if parsed and isinstance(parsed, dict) and "what_happened" in parsed and "management_actions" in parsed:
                    log.info(f"  [section_7] ✓ Valid on attempt {attempt}")
                    section_7 = parsed
                    break
                else:
                    log.warning(f"  [section_7] Attempt {attempt}: missing required CEO briefing keys")

            except Exception as e:
                last_raw = f"ERROR on attempt {attempt}: {e}"
                log.warning(f"  [section_7] Attempt {attempt} error: {e}", exc_info=True)

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))

        if section_7 is None:
            log.error("  [section_7] All attempts failed — raw fallback")
            section_7 = {"status": "synthesis_failed", "error": last_raw[:500]}

    section_7["section_id"]   = "section_7"
    section_7["generated_at"] = datetime.now(timezone.utc).isoformat()

    out_path = DASHBOARD_DIR / "section_7.json"
    out_path.write_text(
        json.dumps(section_7, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"  Saved → {out_path}")
    return section_7


# =============================================================================
# ── FINAL MERGE ──────────────────────────────────────────────────────────────
# =============================================================================

def merge_dashboard(all_sections: dict[str, dict]) -> Path:
    """Merge all section payloads into one dashboard_payload.json."""
    payload = {
        "dashboard_version": "1.0",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "sections": all_sections,
    }
    out_path = DASHBOARD_DIR / "dashboard_payload.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"Dashboard payload saved → {out_path}")
    return out_path


# =============================================================================
# ── ENTRY POINT ──────────────────────────────────────────────────────────────
# =============================================================================

def main():
    ALL_SECTIONS = ["section_1"] + list(SECTION_MAP.keys()) + ["section_7"]

    parser = argparse.ArgumentParser(description="Build dashboard JSON from LLM result files")
    parser.add_argument(
        "--section",
        choices=ALL_SECTIONS,
        default=None,
        help="Run only one section (omit to run all sections)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load files and preview inputs without making LLM calls",
    )
    args = parser.parse_args()

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    all_sections: dict[str, dict] = {}

    if args.section == "section_1":
        all_sections["section_1"] = build_section_1()

    elif args.section == "section_7":
        # Load existing section JSONs to synthesise from
        for sid in SECTION_MAP:
            p = DASHBOARD_DIR / f"{sid}.json"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    all_sections[sid] = json.load(f)
            else:
                log.warning(f"  {sid}.json not found — section_7 synthesis may be incomplete")
        all_sections["section_7"] = build_section_7(all_sections, dry_run=args.dry_run)

    elif args.section:
        # Single LLM section
        payload = build_section(args.section, dry_run=args.dry_run)
        all_sections[args.section] = payload

    else:
        # Full run — section_1 (static) → sections 2–6 (LLM) → section_7 (synthesis)
        all_sections["section_1"] = build_section_1()
        for section_id in SECTION_MAP:
            payload = build_section(section_id, dry_run=args.dry_run)
            all_sections[section_id] = payload
        all_sections["section_7"] = build_section_7(all_sections, dry_run=args.dry_run)

    out_path = merge_dashboard(all_sections)

    log.info("═══ Done ═══════════════════════════════════════════════════")
    log.info(f"Output: {out_path}")


if __name__ == "__main__":
    main()
