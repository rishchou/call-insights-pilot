# Call Insights Desk — OpenAI-only Prototype (Streamlit)
# -----------------------------------------------------
# What this app does
# • Upload multiple call recordings (mp3/wav/m4a/ogg/aac/flac)
# • Transcribe with OpenAI Whisper (whisper-1) — segments include timestamps
# • Index segments (utterance-like) with embeddings for retrieval
# • Ask multi-part questions with presets (RCA, Destinations/Products, Refund Commitments,
#   Customer Requirements, Voice of Customer)
# • Get structured JSON answers + evidence (file, timestamp, quote)
# • Export evidence CSV
#
# How to run locally:
#   1) pip install streamlit openai numpy pandas tqdm
#   2) streamlit run streamlit_openai_call_insights_app.py
#
# How to deploy on Streamlit Cloud (free):
#   • Put this file in a GitHub repo, then deploy at https://share.streamlit.io
#   • Set OPENAI_API_KEY in the app sidebar at runtime (or via Streamlit secrets)

import io
import os
import json
import time
import math
import queue
import base64
import numpy as np
import pandas as pd
import streamlit as st
from tqdm import tqdm
from typing import List, Dict, Any

from openai import OpenAI

# ----------------------------
# App Config & Session State
# ----------------------------
st.set_page_config(page_title="Call Insights Desk", layout="wide")

if "records" not in st.session_state:
    # records: list of { filename, audio_bytes, segments:[{start,end,text}], embed_vectors: np.ndarray }
    st.session_state["records"] = []
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"
if "embed_model" not in st.session_state:
    st.session_state["embed_model"] = "text-embedding-3-large"
if "top_k" not in st.session_state:
    st.session_state["top_k"] = 8

# ----------------------------
# Sidebar: API Key & Settings
# ----------------------------
st.sidebar.title("🔐 Keys & Settings")
OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    oai = OpenAI(api_key=OPENAI_API_KEY)
else:
    oai = None

st.sidebar.markdown("---")
st.sidebar.subheader("Analysis Model")
st.session_state["openai_model"] = st.sidebar.selectbox(
    "Reasoning model",
    ["gpt-4o-mini", "gpt-4o"],
    index=0,
)

st.sidebar.subheader("Embedding Model")
st.session_state["embed_model"] = st.sidebar.selectbox(
    "Embeddings",
    ["text-embedding-3-large", "text-embedding-3-small"],
    index=0,
)

st.sidebar.subheader("Retrieval")
st.session_state["top_k"] = st.sidebar.slider("Top-K segments per insight", 3, 20, st.session_state["top_k"]) 

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear all data"):
    st.session_state["records"] = []
    st.success("Cleared transcripts & index from memory.")

# ----------------------------
# Helpers
# ----------------------------

def embed_texts(oai_client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 3072), dtype=np.float32)
    resp = oai_client.embeddings.create(model=model, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    an = a / (np.linalg.norm(a) + 1e-9)
    bn = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(an, bn))

def transcribe_file(oai_client: OpenAI, filename: str, raw_bytes: bytes) -> Dict[str, Any]:
    """Transcribe using OpenAI Whisper (verbose_json for segments)."""
    # Save to temp for API (file-like needed)
    tmp_path = os.path.join("/tmp", filename)
    with open(tmp_path, "wb") as f:
        f.write(raw_bytes)
    with open(tmp_path, "rb") as f:
        r = oai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"
        )
    # r.model_dump() -> dict-like
    data = r.model_dump()
    segments = []
    for seg in data.get("segments", []) or []:
        segments.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": seg.get("text", "").strip()
        })
    full_text = data.get("text", "").strip()
    return {"text": full_text, "segments": segments}

PRESET_HINTS = {
    "RCA": "Root Cause Analysis focusing on what went wrong, contributing factors, and process gaps.",
    "Destinations/Products": "Extract destinations and product/service mentions; normalize names (DXB->Dubai).",
    "Refund Commitments": "Detect binding refund commitments made by the agent (not the customer).",
    "Requirements": "Identify explicit and implicit customer requirements and urgency.",
    "VoC": "Voice of Customer: themes with counts and representative quotes with sentiment."
}

INSIGHTS = list(PRESET_HINTS.keys())

def retrieve_segments(oai_client: OpenAI, query: str, hint: str, top_k: int) -> List[Dict[str, Any]]:
    """Return top-k segment dicts across all files for a specific subtask."""
    # Build a query embedding
    q = f"{query}\nTask: {hint}"
    q_vec = embed_texts(oai_client, [q], st.session_state["embed_model"])[0]

    # Score all segments
    scored = []
    for rec in st.session_state["records"]:
        segs = rec.get("segments", [])
        vecs = rec.get("embed_vectors", np.zeros((0, 3072), dtype=np.float32))
        for i, seg in enumerate(segs):
            sim = cosine_sim(q_vec, vecs[i]) if i < len(vecs) else 0.0
            scored.append((sim, rec["filename"], seg))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    out = []
    for sim, fname, seg in top:
        out.append({
            "filename": fname,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "score": float(sim)
        })
    return out

SYSTEM_PROMPT = (
    "You analyze customer service call transcripts. If any content is not English, first translate it in your reasoning and present outputs in English. "
    "Use precise, professional language. Base claims only on provided context and cite FILE names with timestamps in parentheses. "
    "Return ONLY valid JSON as instructed for each task."
)

SCHEMAS = {
    "RCA": {
        "instruction": (
            "Produce: {summary, what_went_wrong[], immediate_fixes[], preventive_actions[], evidence[]}. "
            "Each evidence: {file, start_s, end_s, quote}."
        ),
        "keys": ["summary", "what_went_wrong", "immediate_fixes", "preventive_actions", "evidence"],
    },
    "Destinations/Products": {
        "instruction": (
            "Extract destinations/products with normalization. Return: {items:[{type:'destination'|'product', value, synonyms[], evidence:[{file,start_s,end_s,quote}]}]}."
        ),
        "keys": ["items"],
    },
    "Refund Commitments": {
        "instruction": (
            "Determine if an AGENT made a binding refund commitment (not customer demand). Return: {answer:'YES'|'NO', commitments:[{type:'full'|'partial'|'conditional', file, start_s, end_s, quote, confidence:0-1}], notes}."
        ),
        "keys": ["answer", "commitments", "notes"],
    },
    "Requirements": {
        "instruction": (
            "List customer requirements: {requirements:[{requirement, urgency:'low'|'medium'|'high', evidence:[{file,start_s,end_s,quote}]}]}."
        ),
        "keys": ["requirements"],
    },
    "VoC": {
        "instruction": (
            "Voice of Customer themes: {themes:[{theme, count, sentiment:'positive'|'negative'|'neutral', quotes:[{file,start_s,end_s,quote}]}], summary}."
        ),
        "keys": ["themes", "summary"],
    },
}

def format_context_block(segments: List[Dict[str, Any]], max_chars: int = 9000) -> str:
    """Render a context string from top segments, trimmed to avoid context blow-up."""
    lines = []
    total = 0
    for s in segments:
        line = f"FILE: {s['filename']} [{s['start']:.1f}-{s['end']:.1f}s]\n{s['text']}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n---\n".join(lines)

def ask_llm(oai_client: OpenAI, task: str, user_query: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema = SCHEMAS[task]
    context_block = format_context_block(segments)
    prompt = (
        f"Task: {task}. {PRESET_HINTS[task]}\nDesired JSON schema: {schema['instruction']}\n\n"
        f"User query (may be empty): {user_query}\n\nContext:\n{context_block}\n\n"
        f"Return ONLY valid compact JSON with keys: {', '.join(schema['keys'])}."
    )
    resp = oai_client.chat.completions.create(
        model=st.session_state["openai_model"],
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"_raw": resp.choices[0].message.content}

# ----------------------------
# Header
# ----------------------------
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🎧 Call Insights Desk — OpenAI-only")
    st.caption("Upload calls → Transcribe (Whisper) → Ask → Get evidence-backed answers")
with col2:
    st.write("")
    st.write("")
    st.markdown("**Mode:** Prototype")

if not oai:
    st.warning("Enter your OpenAI API key in the sidebar to begin.")

# ----------------------------
# 1) Upload & Transcribe
# ----------------------------
st.header("1) Upload & Transcribe")
files = st.file_uploader(
    "Drop multiple audio files",
    type=["mp3","wav","m4a","ogg","aac","flac"],
    accept_multiple_files=True,
)

transcribe_btn = st.button("Transcribe with Whisper (whisper-1)", disabled=not (files and oai))

if transcribe_btn and oai:
    new_records = []
    progress = st.progress(0)
    for i, f in enumerate(files):
        try:
            tr = transcribe_file(oai, f.name, f.getvalue())
            segs = tr["segments"] if tr else []
            # Embed segment texts
            vecs = embed_texts(oai, [s["text"] for s in segs], st.session_state["embed_model"]) if segs else np.zeros((0,3072), dtype=np.float32)
            new_records.append({
                "filename": f.name,
                "audio_bytes": f.getvalue(),
                "segments": segs,
                "embed_vectors": vecs,
            })
        except Exception as e:
            st.error(f"Failed to transcribe {f.name}: {e}")
        progress.progress(int((i+1)/max(len(files),1)*100))
    st.session_state["records"].extend(new_records)
    st.success(f"Transcribed & indexed {len(new_records)} file(s). Total indexed: {len(st.session_state['records'])}")

# Quick summary table
if st.session_state["records"]:
    st.subheader("Indexed files")
    df_idx = pd.DataFrame([
        {"filename": r["filename"], "segments": len(r["segments"]) } for r in st.session_state["records"]
    ])
    st.dataframe(df_idx, use_container_width=True, hide_index=True)

# ----------------------------
# 2) Ask the Calls
# ----------------------------
st.header("2) Ask the Calls")
colA, colB = st.columns([0.65, 0.35])
with colA:
    chosen = st.multiselect("Choose insights", options=INSIGHTS, default=["RCA", "Destinations/Products", "Refund Commitments"]) 
    user_query = st.text_area("Your question (optional)", placeholder="e.g., 'Why did voucher delivery fail and which destinations were mentioned?'", height=80)
with colB:
    st.metric("Segments in index", sum(len(r.get("segments", [])) for r in st.session_state["records"]))
    st.metric("Files indexed", len(st.session_state["records"]))

analyze_btn = st.button("Analyze", disabled=not (oai and st.session_state["records"] and chosen))

results = {}
if analyze_btn:
    with st.spinner("Thinking…"):
        for task in chosen:
            top_segments = retrieve_segments(oai, user_query or task, PRESET_HINTS[task], st.session_state["top_k"]) 
            results[task] = {
                "segments": top_segments,
                "answer": ask_llm(oai, task, user_query, top_segments)
            }
    st.success("Analysis complete.")

# ----------------------------
# 3) Results (tabs)
# ----------------------------
if results:
    tabs = st.tabs(chosen)
    for (task, tab) in zip(chosen, tabs):
        with tab:
            st.subheader(task)
            ans = results[task]["answer"]
            st.markdown("**Answer (JSON):**")
            st.code(json.dumps(ans, ensure_ascii=False, indent=2))

            # Build an evidence table if present
            evidence_rows = []
            def add_evidence(file, start_s, end_s, quote, extra: Dict[str, Any] | None = None):
                row = {
                    "file": file,
                    "start_s": start_s,
                    "end_s": end_s,
                    "quote": quote,
                }
                if extra:
                    row.update(extra)
                evidence_rows.append(row)

            # Parse per schema
            try:
                if task == "RCA":
                    for e in ans.get("evidence", []) or []:
                        add_evidence(e.get("file",""), e.get("start_s",0), e.get("end_s",0), e.get("quote",""))
                elif task == "Destinations/Products":
                    for item in ans.get("items", []) or []:
                        for e in item.get("evidence", []) or []:
                            add_evidence(e.get("file",""), e.get("start_s",0), e.get("end_s",0), e.get("quote",""),
                                         {"type": item.get("type",""), "value": item.get("value","")})
                elif task == "Refund Commitments":
                    for c in ans.get("commitments", []) or []:
                        add_evidence(c.get("file",""), c.get("start_s",0), c.get("end_s",0), c.get("quote",""),
                                     {"commitment_type": c.get("type",""), "confidence": c.get("confidence",0)})
                elif task == "Requirements":
                    for r in ans.get("requirements", []) or []:
                        for e in r.get("evidence", []) or []:
                            add_evidence(e.get("file",""), e.get("start_s",0), e.get("end_s",0), e.get("quote",""),
                                         {"requirement": r.get("requirement",""), "urgency": r.get("urgency","")})
                elif task == "VoC":
                    for t in ans.get("themes", []) or []:
                        for q in t.get("quotes", []) or []:
                            add_evidence(q.get("file",""), q.get("start_s",0), q.get("end_s",0), q.get("quote",""),
                                         {"theme": t.get("theme",""), "sentiment": t.get("sentiment",""), "count": t.get("count",0)})
            except Exception as e:
                st.warning(f"Could not parse evidence for {task}: {e}")

            if evidence_rows:
                st.markdown("**Evidence** (click the time in your player to verify):")
                df_ev = pd.DataFrame(evidence_rows)
                st.dataframe(df_ev, use_container_width=True, hide_index=True)
                csv = df_ev.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download evidence CSV",
                    data=csv,
                    file_name=f"evidence_{task.replace(' ','_').lower()}.csv",
                    mime="text/csv"
                )

# ----------------------------
# 4) Simple Audio Browser
# ----------------------------
if st.session_state["records"]:
    st.header("4) Audio Player (manual seek)")
    sel = st.selectbox("Choose a file to play", options=[r["filename"] for r in st.session_state["records"]])
    chosen_rec = next((r for r in st.session_state["records"] if r["filename"] == sel), None)
    if chosen_rec:
        st.audio(chosen_rec["audio_bytes"], format="audio/mpeg")
        st.caption("Use the evidence timestamps above to seek manually in the player.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("PII redaction is not applied in this prototype. Use internal data only. © Your Company")
