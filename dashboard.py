import streamlit as st
import json
import glob
from pathlib import Path
import agent

st.set_page_config(page_title="AI CEO Dashboard", layout="wide")
st.title("🛫 AI CEO: Strategic Intelligence Dashboard")

# Sidebar — run agent or load existing result
with st.sidebar:
    st.header("Run Agent")
    goal = st.text_area("CEO Goal", placeholder="What should Lufthansa do about low-cost carrier competition?")
    if st.button("Run Agent", type="primary"):
        with st.spinner("Agent thinking..."):
            result = agent.run_agent(goal)
            st.session_state["result"] = result

    st.divider()
    st.header("Or Load Previous Run")
    files = sorted(glob.glob("outputs/agent_results/*.json"), reverse=True)
    if files:
        selected = st.selectbox("Select run", files)
        if st.button("Load"):
            st.session_state["result"] = json.loads(Path(selected).read_text())

if "result" not in st.session_state:
    st.info("Enter a CEO goal and click Run Agent, or load a previous result.")
    st.stop()

r = st.session_state["result"]
retrieval_summary = r.get("retrieval_summary", {})
analysis = r.get("analysis", {})
rec = r.get("recommendation", {})
validation = r.get("validation", {})
sentiment = r.get("sentiment", {})

# ── Section 1: Company Overview ──────────────────────────────
st.header("1. Company Overview")
col1, col2, col3, col4 = st.columns(4)
focus = retrieval_summary.get("focus_companies", ["Lufthansa"])
col1.metric("Top Company", focus[0] if focus else "—")
col2.metric("Industry", "Aviation")
col3.metric("Confidence Score", f"{rec.get('confidence_score', 0)}/100")
col4.metric("Chunks Retrieved", retrieval_summary.get("total_chunks", 0))

# ── Section 2: Market Intelligence ───────────────────────────
st.header("2. Market Intelligence")
col1, col2 = st.columns(2)
with col1:
    st.write("**Focus Companies:**")
    for c in retrieval_summary.get("focus_companies", []):
        st.write(f"• {c}")
    st.write(f"**Primary Concern:** `{retrieval_summary.get('primary_concern','—').upper()}`")
with col2:
    st.write("**Investigation Topics:**")
    for t in retrieval_summary.get("investigation_topics", []):
        st.write(f"• {t}")
    st.write(f"**Chunks Retrieved:** {retrieval_summary.get('total_chunks', 0)}")
st.info(retrieval_summary.get("rationale", ""))

# ── Section 3: Opportunity Monitor ───────────────────────────
st.header("3. Opportunity Monitor")
for opp in analysis.get("opportunities", []):
    with st.expander(f"🟢 {opp[:80]}"):
        st.write(opp)
        st.progress(analysis.get("confidence_score", 0) / 100)
        st.caption(f"Confidence: {analysis.get('confidence_score', 0)}/100")

# ── Section 4: Risk Monitor ───────────────────────────────────
st.header("4. Risk Monitor")
for risk in analysis.get("risks", []):
    with st.expander(f"🔴 {risk[:80]}"):
        st.write(risk)
        st.caption(f"Agent Decision: {analysis.get('decision', '').upper()}")

# ── Section 5: Sentiment Analysis ────────────────────────────
st.header("5. Sentiment Analysis")
col1, col2, col3 = st.columns(3)
label = sentiment.get("label", "N/A")
score = sentiment.get("average_score", 0)
color = "🟢" if label == "Positive" else "🔴" if label == "Negative" else "🟡"
col1.metric("News Sentiment", f"{color} {label}")
col2.metric("Public Sentiment", f"{color} {label}")
col3.metric("Sentiment Score", score)
if sentiment.get("chunk_scores"):
    import pandas as pd
    st.bar_chart(pd.DataFrame({"Sentiment": sentiment["chunk_scores"]}))

# ── Section 6: Strategic Recommendations ─────────────────────
st.header("6. Strategic Recommendations")
priority = rec.get("priority", "Medium")
color_map = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
st.subheader(f"{color_map.get(priority, '🟡')} Priority: {priority}")
st.write(f"**Recommendation:** {rec.get('recommendation', '')}")
st.write(f"**Expected Impact:** {rec.get('expected_impact', '')}")
st.write(f"**Risk if Ignored:** {rec.get('risk_if_ignored', '')}")
st.write("**Supporting Evidence:**")
for ev in rec.get("supporting_evidence", []):
    st.write(f"• {ev}")

# ── Section 7: CEO Briefing ───────────────────────────────────
st.header("7. CEO Briefing")
st.subheader("What happened?")
for f in analysis.get("key_findings", []):
    st.write(f"• {f}")
st.subheader("Why does it matter?")
st.write(rec.get("risk_if_ignored", ""))
st.subheader("What should management do next?")
st.success(rec.get("recommendation", ""))
decision = analysis.get("decision", "").upper()
st.metric("Agent Decision", decision)

# Validation badge
if validation.get("passed"):
    st.success("✅ Recommendation passed validation")
else:
    st.warning(f"⚠️ Validation issues: {', '.join(validation.get('issues', []))}")
