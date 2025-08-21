# Call Insights Desk — Neutral UI + Custom Analyst
# ------------------------------------------------
# Users can:
# • Upload multiple audio files
# • Transcribe (AI) to segments with timestamps
# • Select specific files (search, select all/clear)
# • Ask Insights: RCA, Destinations/Products, Refund Commitments, Requirements, VoC
# • Ask free-form "Custom Analyst" questions over selected calls (e.g., 5-call summary)
# • Get friendly sections + Evidence (file, timestamp, quote) and download CSV

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional
from openai import OpenAI

# =========================
# Hidden config (no UI)
# =========================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # set in Streamlit Secrets
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
oai = OpenAI(api_key=OPENAI_API_KEY)

# Internal defaults (edit here only)
REASONING_MODEL = "gpt-4o-mini"
EMBED_MODEL     = "text-embedding-3-large"
TOP_K_DEFAULT   = 8           # total segments per insight (general)
TOP_K_PER_FILE  = 6           # segments per file (Custom Analyst mode)

# =========================
# Page config & state
# =========================
st.set_page_config(page_title="Call Insights Desk", layout="wide")

if "records" not in st.session_state:
    # records: [{ filename, audio_bytes, segments:[{start,end,text}], embed_vectors: np.ndarray }]
    st.session_state["records"] = []
if "selected_files" not in st.session_state:
    st.session_state["selected_files"] = set()

# =========================
# Helpers
# =========================
def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 3072), dtype=np.float32)
    resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    an = a / (np.linalg.norm(a) + 1e-9)
    bn = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(an, bn))

def transcribe_file(filename: str, raw_bytes: bytes) -> Dict[str, Any]:
    """Transcribe using AI with segments & timestamps."""
    tmp_path = os.path.join("/tmp", filename)
    with open(tmp_path, "wb") as f:
        f.write(raw_bytes)
    with open(tmp_path, "rb") as f:
        r = oai.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"
        )
    data = r.model_dump()
    segments = []
    for seg in data.get("segments", []) or []:
        segments.append({
            "start": float(seg.get("start", 0.0)),
            "end":   float(seg.get("end",   0.0)),
            "text": (seg.get("text") or "").strip()
        })
    full_text = (data.get("text") or "").strip()
    return {"text": full_text, "segments": segments}

# Insight presets and schemas
PRESET_HINTS = {
    "RCA": "Root Cause Analysis focusing on what went wrong, contributing factors, and process gaps.",
    "Destinations/Products": "Extract destinations and product/service mentions; normalize names (DXB->Dubai, KSA->Saudi Arabia).",
    "Refund Commitments": "Detect binding refund commitments made by the agent (not the customer).",
    "Requirements": "Identify explicit and implicit customer requirements and their urgency.",
    "VoC": "Voice of Customer: themes with counts and representative quotes with sentiment.",
    "Custom Analyst": "Answer the user's free-form question across ONLY the selected calls; return a cross-call summary with timeline and actions."
}
INSIGHTS = ["RCA", "Destinations/Products", "Refund Commitments", "Requirements", "VoC", "Custom Analyst"]

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
            "Extract destinations/products with normalization. Return: "
            "{items:[{type:'destination'|'product', value, synonyms[], evidence:[{file,start_s,end_s,quote}]}]}."
        ),
        "keys": ["items"],
    },
    "Refund Commitments": {
        "instruction": (
            "Determine if an AGENT made a binding refund commitment (not customer demand). "
            "Return: {answer:'YES'|'NO', commitments:[{type:'full'|'partial'|'conditional', file, start_s, end_s, quote, confidence:0-1}], notes}."
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
    "Custom Analyst": {
        "instruction": (
            "Be a customer-service analyst across ONLY the selected calls provided in context. "
            "Return JSON: {executive_summary, per_call:[{file, bullets[]}], "
            "timeline:[{when, file, start_s, end_s, event}], action_items[], "
            "evidence:[{file,start_s,end_s,quote}]}."
        ),
        "keys": ["executive_summary", "per_call", "timeline", "action_items", "evidence"],
    },
}

SYSTEM_PROMPT = (
    "You analyze customer service call transcripts. "
    "If any content is not English, translate internally and present outputs in English. "
    "Be precise and evidence-based; cite FILE names with timestamps in parentheses in quotes. "
    "Return ONLY valid JSON for the requested schema."
)

def format_context_block(segments: List[Dict[str, Any]], max_chars: int = 9000) -> str:
    lines, total = [], 0
    for s in segments:
        line = f"FILE: {s['filename']} [{s['start']:.1f}-{s['end']:.1f}s]\n{s['text']}"
        if total + len(line) > max_chars: break
        lines.append(line); total += len(line)
    return "\n---\n".join(lines)

def ask_llm(task: str, user_query: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema = SCHEMAS[task]
    context_block = format_context_block(segments)
    prompt = (
        f"Task: {task}. {PRESET_HINTS[task]}\n"
        f"Desired JSON schema: {schema['instruction']}\n\n"
        f"User query (may be empty): {user_query}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Return ONLY valid compact JSON with keys: {', '.join(schema['keys'])}."
    )
    resp = oai.chat.completions.create(
        model=REASONING_MODEL,
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

# ---------- Retrieval ----------
def retrieve_segments_general(user_query: str, hint: str, top_k: int,
                              allowed_filenames: Optional[set] = None) -> List[Dict[str, Any]]:
    """Top-K globally (used by preset insights except Custom Analyst)."""
    q = f"{user_query}\nTask: {hint}".strip()
    q_vec = embed_texts([q])[0]
    scored = []
    for rec in st.session_state["records"]:
        if allowed_filenames and rec["filename"] not in allowed_filenames:
            continue
        segs = rec.get("segments", [])
        vecs = rec.get("embed_vectors", np.zeros((0, 3072), dtype=np.float32))
        for i, seg in enumerate(segs):
            sim = cosine_sim(q_vec, vecs[i]) if i < len(vecs) else 0.0
            scored.append((sim, rec["filename"], seg))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    return [{
        "filename": fname,
        "start": seg["start"],
        "end": seg["end"],
        "text": seg["text"],
        "score": float(sim)
    } for (sim, fname, seg) in top]

def retrieve_segments_round_robin(user_query: str, hint: str, per_file_k: int,
                                  allowed_filenames: set) -> List[Dict[str, Any]]:
    """Ensure coverage across selected files: take top-N per file."""
    q = f"{user_query}\nTask: {hint}".strip()
    q_vec = embed_texts([q])[0]
    per_file_lists = []
    for rec in st.session_state["records"]:
        if rec["filename"] not in allowed_filenames:
            continue
        local = []
        segs = rec.get("segments", [])
        vecs = rec.get("embed_vectors", np.zeros((0, 3072), dtype=np.float32))
        for i, seg in enumerate(segs):
            sim = cosine_sim(q_vec, vecs[i]) if i < len(vecs) else 0.0
            local.append((sim, rec["filename"], seg))
        local.sort(key=lambda x: x[0], reverse=True)
        per_file_lists.append(local[:per_file_k])

    # flatten (we could do strict round-robin; here simple concat is okay)
    out = []
    for local in per_file_lists:
        for sim, fname, seg in local:
            out.append({
                "filename": fname,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "score": float(sim)
            })
    return out

# =========================
# Header
# =========================
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🎧 Call Insights Desk")
    st.caption("Upload calls → Transcribe → Select → Ask → Get evidence-backed answers")
with col2:
    st.markdown("**Mode:** Internal Pilot")

# =========================
# 1) Upload & Transcribe
# =========================
st.header("1) Upload & Transcribe")
files = st.file_uploader(
    "Drop multiple audio files",
    type=["mp3","wav","m4a","ogg","aac","flac"],
    accept_multiple_files=True,
)

if st.button("Transcribe", disabled=not files):
    new_records = []
    progress = st.progress(0)
    for i, f in enumerate(files):
        try:
            tr = transcribe_file(f.name, f.getvalue())
            segs = tr["segments"] if tr else []
            vecs = embed_texts([s["text"] for s in segs]) if segs else np.zeros((0,3072), dtype=np.float32)
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
    st.success(f"Transcribed & indexed {len(new_records)} file(s). Total: {len(st.session_state['records'])}")

# Quick summary
if st.session_state["records"]:
    st.subheader("Indexed files")
    df_idx = pd.DataFrame([{"filename": r["filename"], "segments": len(r["segments"])} for r in st.session_state["records"]])
    st.dataframe(df_idx, use_container_width=True, hide_index=True)

# =========================
# 2) Select Calls (filter)
# =========================
if st.session_state["records"]:
    st.header("2) Select Calls")
    all_filenames = [r["filename"] for r in st.session_state["records"]]
    search = st.text_input("Search by filename / customer / order (matches filename text)")

    if search:
        filtered = [fn for fn in all_filenames if search.lower() in fn.lower()]
    else:
        filtered = all_filenames

    cols = st.columns([0.6, 0.4])
    with cols[0]:
        chosen = st.multiselect("Choose from indexed files", options=filtered, default=sorted(st.session_state["selected_files"] & set(filtered)))
    with cols[1]:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Select all (filtered)"):
                st.session_state["selected_files"] = set(filtered)
                chosen = filtered
        with c2:
            if st.button("Clear selection"):
                st.session_state["selected_files"] = set()
                chosen = []

    st.session_state["selected_files"] = set(chosen)
    st.caption(f"Selected: {len(st.session_state['selected_files'])} / {len(all_filenames)} files")

# =========================
# 3) Ask the Calls
# =========================
st.header("3) Ask the Calls")
colA, colB = st.columns([0.65, 0.35])
with colA:
    selected_insights = st.multiselect(
        "Choose insights",
        options=INSIGHTS,
        default=["RCA", "Destinations/Products", "Refund Commitments"]
    )
    user_query = st.text_area(
        "Your question (for Custom Analyst or to refine other insights)",
        placeholder="e.g., 'Summarize these 5 calls and give a timeline + action items.'",
        height=80
    )
with colB:
    st.metric("Segments in index", sum(len(r.get("segments", [])) for r in st.session_state["records"]))
    st.metric("Files indexed", len(st.session_state["records"]))
    st.metric("Files selected", len(st.session_state["selected_files"]))

can_analyze = bool(st.session_state["records"]) and bool(selected_insights) and (("Custom Analyst" not in selected_insights) or st.session_state["selected_files"])
if st.button("Analyze", disabled=not can_analyze):
    with st.spinner("Analyzing…"):
        results = {}
        allowed = set(st.session_state["selected_files"]) if st.session_state["selected_files"] else None

        for task in selected_insights:
            if task == "Custom Analyst":
                if not allowed:
                    st.warning("Select at least one file for Custom Analyst.")
                    continue
                # Ensure coverage per selected file
                top_segments = retrieve_segments_round_robin(
                    user_query or "Summarize the selected calls", PRESET_HINTS[task],
                    per_file_k=TOP_K_PER_FILE, allowed_filenames=allowed
                )
            else:
                top_segments = retrieve_segments_general(
                    user_query or task, PRESET_HINTS[task], TOP_K_DEFAULT,
                    allowed_filenames=allowed  # if user selected files, constrain
                )
            results[task] = {
                "segments": top_segments,
                "answer": ask_llm(task, user_query, top_segments)
            }
        st.session_state["last_results"] = results
        st.success("Analysis complete.")

# =========================
# 4) Results (tabs)
# =========================
def render_rca(ans: Dict[str, Any]):
    st.markdown("**Summary**")
    st.write(ans.get("summary", "—"))
    st.markdown("**What went wrong**")
    for item in ans.get("what_went_wrong", []) or []:
        st.write(f"- {item}")
    st.markdown("**Immediate fixes**")
    for item in ans.get("immediate_fixes", []) or []:
        st.write(f"- {item}")
    st.markdown("**Preventive actions**")
    for item in ans.get("preventive_actions", []) or []:
        st.write(f"- {item}")

def render_destinations(ans: Dict[str, Any]):
    items = ans.get("items", []) or []
    if not items:
        st.write("—")
        return
    for it in items:
        badge = "Destination" if it.get("type") == "destination" else "Product"
        st.write(f"- **{badge}:** {it.get('value','—')}")

def render_commitments(ans: Dict[str, Any]):
    st.write(f"**Answer:** {ans.get('answer','—')}")
    notes = ans.get("notes")
    if notes: st.caption(notes)

def render_requirements(ans: Dict[str, Any]):
    for r in ans.get("requirements", []) or []:
        st.write(f"- **{r.get('requirement','—')}** (urgency: {r.get('urgency','—')})")

def render_voc(ans: Dict[str, Any]):
    st.markdown("**Summary**")
    st.write(ans.get("summary", "—"))
    st.markdown("**Themes**")
    for t in ans.get("themes", []) or []:
        st.write(f"- {t.get('theme','—')} (count: {t.get('count',0)}, sentiment: {t.get('sentiment','—')})")

def render_custom_analyst(ans: Dict[str, Any]):
    st.markdown("**Executive Summary**")
    st.write(ans.get("executive_summary", "—"))
    st.markdown("**Per-call bullets**")
    for pc in ans.get("per_call", []) or []:
        st.write(f"- **{pc.get('file','—')}**")
        for b in pc.get("bullets", []) or []:
            st.write(f"   • {b}")
    st.markdown("**Timeline**")
    for t in ans.get("timeline", []) or []:
        when = t.get("when","")
        file = t.get("file","")
        st.write(f"- {when}: {file} [{t.get('start_s',0)}–{t.get('end_s',0)}s] — {t.get('event','')}")
    st.markdown("**Action items**")
    for a in ans.get("action_items", []) or []:
        st.write(f"- {a}")

def build_evidence(task: str, ans: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    def add(file, s, e, q, extra=None):
        row = {"file": file, "start_s": s, "end_s": e, "quote": q}
        if extra: row.update(extra)
        rows.append(row)

    try:
        if task == "RCA":
            for ev in ans.get("evidence", []) or []:
                add(ev.get("file",""), ev.get("start_s",0), ev.get("end_s",0), ev.get("quote",""))
        elif task == "Destinations/Products":
            for it in ans.get("items", []) or []:
                for ev in it.get("evidence", []) or []:
                    add(ev.get("file",""), ev.get("start_s",0), ev.get("end_s",0), ev.get("quote",""),
                        {"type": it.get("type",""), "value": it.get("value","")})
        elif task == "Refund Commitments":
            for c in ans.get("commitments", []) or []:
                add(c.get("file",""), c.get("start_s",0), c.get("end_s",0), c.get("quote",""),
                    {"commitment_type": c.get("type",""), "confidence": c.get("confidence",0)})
        elif task == "Requirements":
            for r in ans.get("requirements", []) or []:
                for ev in r.get("evidence", []) or []:
                    add(ev.get("file",""), ev.get("start_s",0), ev.get("end_s",0), ev.get("quote",""),
                        {"requirement": r.get("requirement",""), "urgency": r.get("urgency","")})
        elif task == "VoC":
            for t in ans.get("themes", []) or []:
                for q in t.get("quotes", []) or []:
                    add(q.get("file",""), q.get("start_s",0), q.get("end_s",0), q.get("quote",""),
                        {"theme": t.get("theme",""), "sentiment": t.get("sentiment",""), "count": t.get("count",0)})
        elif task == "Custom Analyst":
            for ev in ans.get("evidence", []) or []:
                add(ev.get("file",""), ev.get("start_s",0), ev.get("end_s",0), ev.get("quote",""))
    except Exception:
        pass
    return pd.DataFrame(rows)

results = st.session_state.get("last_results", {})
if results:
    tabs = st.tabs(list(results.keys()))
    for (task, tab) in zip(results.keys(), tabs):
        with tab:
            st.subheader(task)
            ans = results[task]["answer"] or {}

            # Friendly renderers
            if task == "RCA": render_rca(ans)
            elif task == "Destinations/Products": render_destinations(ans)
            elif task == "Refund Commitments": render_commitments(ans)
            elif task == "Requirements": render_requirements(ans)
            elif task == "VoC": render_voc(ans)
            elif task == "Custom Analyst": render_custom_analyst(ans)

            # Evidence
            df_ev = build_evidence(task, ans)
            st.markdown("**Evidence**")
            if df_ev.empty:
                st.write("—")
            else:
                st.dataframe(df_ev, use_container_width=True, hide_index=True)
                csv = df_ev.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download evidence CSV",
                    data=csv,
                    file_name=f"evidence_{task.replace(' ','_').lower()}.csv",
                    mime="text/csv"
                )

# =========================
# 5) Audio Player
# =========================
if st.session_state["records"]:
    st.header("4) Audio Player")
    sel = st.selectbox("Choose a file to play", options=[r["filename"] for r in st.session_state["records"]])
    rec = next((r for r in st.session_state["records"] if r["filename"] == sel), None)
    if rec:
        st.audio(rec["audio_bytes"], format="audio/mpeg")
        st.caption("Use timestamps from Evidence to seek in your player.")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("For internal use only. This pilot does not apply PII redaction. © Your Company")
