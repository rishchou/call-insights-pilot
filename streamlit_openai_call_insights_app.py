# Call Insights Desk — CS General Audit • RCA • VoC
# End-user UI (clean) with Agent Name, N/A, Fatal, CSV template, Summary table, Dashboards, Exports

import os, io, re, json, hashlib
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Plotly optional
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

from openai import OpenAI
import google.generativeai as genai

# =========================
# Config / Secrets
# =========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not OPENAI_API_KEY or not GEMINI_API_KEY:
    st.error("Missing OPENAI_API_KEY or GEMINI_API_KEY in environment or Streamlit secrets.")
    st.stop()

oai = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

_GEMINI_MODEL_NAME = "gemini-1.5-pro"
gemini_model = genai.GenerativeModel(
    _GEMINI_MODEL_NAME,
    generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
    system_instruction=(
        "You analyze customer service call transcripts.\n"
        "Translate internally to English if needed. Be precise and evidence-based.\n"
        "Always return STRICT JSON for the requested schema only, in English."
    ),
)

_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_CSV_PATH = "/mnt/data/CS QA parameters.csv"  # optional default rubric if present

# =========================
# Page + CSS
# =========================
st.set_page_config(page_title="Call Insights Desk", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; }
.card { background:#fff;border:1px solid #eee;border-radius:14px;padding:16px 18px; }
.muted { color:#667085; }
.kpi { background:#fafafa;border:1px solid #eee;border-radius:12px;padding:12px 16px;text-align:center; }
.kpi .big { font-size:22px;font-weight:700; }
.rowpad { margin-top: 0.75rem; }
.badge { border-radius:8px;padding:2px 8px;font-size:12px;color:white; }
.badge-high { background:#D92D20; }
.badge-medium { background:#F79009; }
.badge-low { background:#12B76A; }
</style>
""", unsafe_allow_html=True)

# =========================
# Session State
# =========================
if "records" not in st.session_state:
    # [{filename, hash, audio_bytes, language, text_orig, text_en, segments, embed_vectors, agent_name}]
    st.session_state["records"] = []
if "selected_files" not in st.session_state:
    st.session_state["selected_files"] = set()
if "cs_rubric_df" not in st.session_state:
    st.session_state["cs_rubric_df"] = None
if "last_results" not in st.session_state:
    st.session_state["last_results"] = {}

# =========================
# Helpers: general
# =========================
def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _mmss(sec: float) -> str:
    m = int((sec or 0)//60); s = int((sec or 0)%60); return f"{m:02d}:{s:02d}"

def _get_rubric_df() -> pd.DataFrame:
    """
    Return a valid rubric DataFrame (uploaded/edited one if available, else the template).
    """
    rub = st.session_state.get("cs_rubric_df")
    if isinstance(rub, pd.DataFrame) and not rub.empty:
        return rub
    return _rubric_template_df()

def _extract_agent_name_from_filename(filename: str) -> str:
    """Extract agent name from patterns like [Name], (Name), or ' - Name'."""
    if not filename:
        return ""
    m = re.search(r"\[([^\[\]]+)\]", filename)
    if m: return m.group(1).strip()
    m = re.search(r"\(([^\(\)]+)\)", filename)
    if m:
        cand = m.group(1).strip()
        if not re.search(r"\d", cand) and len(cand.split()) <= 4:
            return cand
    m = re.search(r"-\s*([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+){0,3})", filename)
    if m:
        cand = m.group(1).strip()
        if not re.search(r"\d", cand):
            return cand
    return ""

# =========================
# Embeddings / Retrieval
# =========================
def _embed_texts(texts: List[str]) -> np.ndarray:
    if not texts: return np.zeros((0, 3072), dtype=np.float32)
    resp = oai.embeddings.create(model=_EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0: return 0.0
    an = a/(np.linalg.norm(a)+1e-9); bn = b/(np.linalg.norm(b)+1e-9)
    return float(np.dot(an, bn))

# =========================
# Whisper Transcribe + Translate (correct endpoints)
# =========================
@st.cache_data(show_spinner=False)
def _transcribe_file(filename: str, raw_bytes: bytes) -> Dict[str, Any]:
    """
    Pass 1: transcription (original) via audio.transcriptions.create
    Pass 2: translation (English) via audio.translations.create
    """
    tmp = os.path.join("/tmp", filename)
    with open(tmp, "wb") as f: f.write(raw_bytes)

    with open(tmp, "rb") as f1:
        r_orig = oai.audio.transcriptions.create(
            model="whisper-1", file=f1, response_format="verbose_json", temperature=0
        )
    d_orig = r_orig.model_dump() if hasattr(r_orig, "model_dump") else json.loads(r_orig.json())

    with open(tmp, "rb") as f2:
        r_en = oai.audio.translations.create(
            model="whisper-1", file=f2, response_format="verbose_json", temperature=0
        )
    d_en = r_en.model_dump() if hasattr(r_en, "model_dump") else json.loads(r_en.json())

    language = d_orig.get("language") or "unknown"
    segs_o = d_orig.get("segments", []) or []
    segs_e = d_en.get("segments", []) or []
    max_len = max(len(segs_o), len(segs_e))
    segments = []
    for i in range(max_len):
        so = segs_o[i] if i < len(segs_o) else {}
        se = segs_e[i] if i < len(segs_e) else {}
        segments.append({
            "start": float(so.get("start", se.get("start", 0.0))),
            "end":   float(so.get("end",   se.get("end",   0.0))),
            "text_orig": (so.get("text") or "").strip(),
            "text_en":   (se.get("text") or "").strip(),
        })

    # synthetic single segment if none
    if not segments:
        full = (d_en.get("text") or d_orig.get("text") or "").strip()
        if full:
            segments = [{"start":0.0,"end":0.0,"text_orig":full[:4000],"text_en":full[:4000]}]

    return {
        "language": language,
        "text_orig": (d_orig.get("text") or "").strip(),
        "text_en":   (d_en.get("text") or "").strip(),
        "segments": segments
    }

# =========================
# Speaker Attribution (AI)
# =========================
def _label_speakers_via_gemini(segments: List[Dict[str, Any]], filename: str) -> List[str]:
    items = []; used = 0; max_chars = 8000
    for i, s in enumerate(segments):
        txt = (s.get("text_en") or s.get("text_orig") or "").strip()
        if not txt: continue
        chunk = txt[:400]
        add = len(chunk) + 15
        if used + add > max_chars: break
        used += add
        items.append({"i": i, "text": chunk})

    payload = {
        "task": "Classify call segments by speaker role",
        "file": filename,
        "instructions": (
            "For each segment, output 'AGENT' or 'CUSTOMER'. "
            "Use cues like greeting scripts, verification, policy explanations (AGENT) "
            "vs. problems, requests, complaints (CUSTOMER). "
            "Return JSON: {labels:[{'i': <index>, 'speaker':'AGENT'|'CUSTOMER'}]}."
        ),
        "segments": items
    }
    resp = gemini_model.generate_content(json.dumps(payload, ensure_ascii=False))
    try:
        data = json.loads(resp.text)
        lab_map = {d["i"]: d.get("speaker","") for d in data.get("labels", []) if isinstance(d, dict)}
    except Exception:
        lab_map = {}
    out = []
    for i in range(len(segments)):
        out.append(lab_map.get(i, ""))
    return out

def _build_transcripts_df(selected_only: bool = True, use_ai_speaker: bool = True) -> pd.DataFrame:
    chosen = set(st.session_state.get("selected_files", set())) if selected_only else None
    rows = []
    for rec in st.session_state["records"]:
        if chosen and rec["filename"] not in chosen: continue
        segs = rec.get("segments", []) or []
        speakers = []
        if use_ai_speaker and segs:
            try:
                speakers = _label_speakers_via_gemini(segs, rec["filename"])
            except Exception:
                speakers = [""] * len(segs)
        else:
            speakers = [""] * len(segs)

        for i, s in enumerate(segs):
            rows.append({
                "file": rec["filename"],
                "agent_name": rec.get("agent_name",""),
                "t": _mmss(float(s.get("start",0.0))),
                "start_s": float(s.get("start",0.0)),
                "end_s": float(s.get("end",0.0)),
                "speaker": speakers[i] if i < len(speakers) else "",
                "text_orig": (s.get("text_orig") or "").strip(),
                "text_en": (s.get("text_en") or "").strip(),
            })
    return pd.DataFrame(rows)

# =========================
# Rubric: template, strict validation, tools
# =========================
RUBRIC_REQUIRED_COLS = ["parameter", "weight", "max_score"]  # percent regime
OPTIONAL_FATAL_COL = "fatal"  # yes/no (optional)

def _rubric_template_df() -> pd.DataFrame:
    return pd.DataFrame({
        "parameter": ["Call Greetings", "Active Listening", "Hold Procedure", "Resolution", "Closure & Recap"],
        "weight":    [10, 15, 20, 30, 25],  # must sum to 100
        "max_score": [100,100,100,100,100],
        "fatal":     ["no","no","no","no","no"]
    })

def _download_template_csv_bytes() -> bytes:
    return _rubric_template_df().to_csv(index=False).encode("utf-8")

def _validate_rubric_strict(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    issues = []
    for c in RUBRIC_REQUIRED_COLS:
        if c not in df.columns:
            issues.append(f"Missing column: {c}")
    if issues:
        return df, issues

    out = df.copy()
    out["parameter"] = out["parameter"].astype(str).str.strip()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out["max_score"] = pd.to_numeric(out["max_score"], errors="coerce")
    if OPTIONAL_FATAL_COL in out.columns:
        out[OPTIONAL_FATAL_COL] = out[OPTIONAL_FATAL_COL].astype(str).str.lower().str.strip()
    else:
        out[OPTIONAL_FATAL_COL] = "no"

    out = out[out["parameter"]!=""].reset_index(drop=True)

    for i, row in out.iterrows():
        w = row["weight"]; m = row["max_score"]
        if pd.isna(w): issues.append(f"Row {i+1} '{row['parameter']}': weight is empty")
        if pd.isna(m): issues.append(f"Row {i+1} '{row['parameter']}': max_score is empty")
        if not pd.isna(w) and (w < 0 or w > 100): issues.append(f"Row {i+1} '{row['parameter']}': weight must be 0–100")
        if not pd.isna(m) and (m <= 0 or m > 100): issues.append(f"Row {i+1} '{row['parameter']}': max_score must be 1–100")

    if "weight" in out.columns and out["weight"].notna().all():
        total_w = float(out["weight"].sum())
        if abs(total_w - 100.0) > 0.1:
            issues.append(f"Weights must sum to 100. Current total: {total_w:.2f}")

    return out, issues

def _rubric_remaining_weight(df: pd.DataFrame) -> float:
    used = pd.to_numeric(df["weight"], errors="coerce").fillna(0).sum() if "weight" in df.columns else 0.0
    return 100.0 - float(used)

def _rebalance_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    w = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    total = float(w.sum())
    if total <= 0:
        n = len(out)
        out["weight"] = 100.0 / n if n else 0.0
    else:
        out["weight"] = (w / total) * 100.0
    return out.round({"weight": 2})

# normalized rubric for modeling + maps
def _rubric_norm_and_maps(df: pd.DataFrame):
    r = df.copy()
    total = float(pd.to_numeric(r["weight"], errors="coerce").fillna(0).sum()) or 1.0
    r["norm_w"] = (pd.to_numeric(r["weight"], errors="coerce").fillna(0) / total) * 100.0
    r["max_score"] = pd.to_numeric(r["max_score"], errors="coerce").fillna(100.0).clip(lower=1.0, upper=100.0)
    r[OPTIONAL_FATAL_COL] = r[OPTIONAL_FATAL_COL].astype(str).str.lower().str.strip()
    rubric_json = r[["parameter","norm_w","max_score", OPTIONAL_FATAL_COL]].to_dict(orient="records")
    weight_map = {row["parameter"]: float(row["norm_w"]) for _, row in r.iterrows()}
    max_map    = {row["parameter"]: float(row["max_score"]) for _, row in r.iterrows()}
    fatal_map  = {row["parameter"]: (row.get(OPTIONAL_FATAL_COL,"no")=="yes") for _, row in r.iterrows()}
    return rubric_json, weight_map, max_map, fatal_map

# =========================
# Retrieval (grouped per file to guarantee coverage)
# =========================
def _retrieve_by_file(user_query: str, hint: str, per_file_k: int,
                        allowed_filenames: set) -> Dict[str, List[Dict[str, Any]]]:
    q_vec = _embed_texts([f"{user_query}\nTask: {hint}"])[0]
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in st.session_state["records"]:
        if rec["filename"] not in allowed_filenames: continue
        segs = rec.get("segments", [])
        vecs = rec.get("embed_vectors", np.zeros((0,3072), dtype=np.float32))
        scored_local = []
        for i, seg in enumerate(segs):
            sim = _cosine_sim(q_vec, vecs[i]) if i < len(vecs) else 0.0
            scored_local.append((sim, {
                "filename": rec["filename"],
                "start": seg["start"], "end": seg["end"],
                "text": (seg.get("text_en") or seg.get("text_orig") or "").strip(),
                "score": float(sim)
            }))
        if not scored_local:
            full = (rec.get("text_en") or rec.get("text_orig") or "").strip()
            if full:
                scored_local = [(0.0, {"filename": rec["filename"], "start":0.0,"end":0.0,"text":full[:4000],"score":0.0})]
        scored_local.sort(key=lambda x: x[0], reverse=True)
        keep = [s for _, s in scored_local[:max(1, per_file_k)]]
        grouped[rec["filename"]] = keep
    return grouped

# =========================
# JSON guard
# =========================
def _ensure_json_dict(resp_text: str, keys: List[str]) -> Dict[str, Any]:
    try:
        obj = json.loads(resp_text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    fix_payload = {"task":"Repair into JSON", "desired_keys": keys, "raw": resp_text}
    r = gemini_model.generate_content(json.dumps(fix_payload))
    try:
        return json.loads(r.text)
    except Exception:
        return {"_raw": (resp_text or "").strip()}

# =========================
# CS General Audit prompt (per-file output + NA + fatal)
# =========================
def _ask_gemini_cs_audit(user_query: str,
                         grouped_segments: Dict[str, List[Dict[str, Any]]],
                         rubric_json: List[Dict[str, Any]]) -> Dict[str, Any]:
    blocks = []
    for fname, segs in grouped_segments.items():
        lines = [f"# FILE: {fname}"]
        for s in segs:
            lines.append(f"[{s['start']:.1f}-{s['end']:.1f}s] {s['text']}")
        blocks.append("\n".join(lines))
    context_block = "\n\n---\n\n".join(blocks)

    instruction = (
        "Return JSON: {per_call:[{file, agent_name, purpose, category, summary, "
        "severity:'low'|'medium'|'high', sentiment:{start,mid,end}, "
        "parameters:[{parameter, score, max_score, na:boolean, fatal_triggered:boolean, justification, "
        "evidence:[{file,start_s,end_s,quote}]}], overall_weighted_score}], coaching_opportunities[]}.\n"
        "You MUST include exactly one object for EVERY file in the context (even if evidence is thin). "
        "Use rubric weights (norm_w) and max_score. If a parameter does not apply, set na=true.\n"
        "If a parameter is marked 'fatal' in rubric and it occurred, set fatal_triggered=true.\n"
        "For agent_name, infer from context if present in text (e.g., 'This is <name>'), else leave empty; UI may inject it from filename.\n"
        "Compute overall_weighted_score as a percentage 0..100 (if NA, redistribute weights over applicable params; if any fatal_triggered=true, overall=0)."
    )
    prompt = (
        "Task: CS General Audit with rubric scoring and call context extraction.\n"
        "Also produce purpose, category (Query/Complaint/Follow-up/Escalation/Sales), severity, and sentiment trend.\n\n"
        f"Rubric (weights normalized to 100): {json.dumps(rubric_json, ensure_ascii=False)}\n"
        f"Desired JSON schema: {instruction}\n"
        f"User instruction (optional): {user_query or ''}\n\n"
        f"Context (grouped by file):\n{context_block}\n\n"
        f"Return ONLY compact JSON with keys: per_call, coaching_opportunities."
    )
    resp = gemini_model.generate_content(prompt)
    return _ensure_json_dict(resp.text or "", ["per_call", "coaching_opportunities"])

# =========================
# Post-process: recompute %s, NA handling, fatal override, inject agent
# =========================
def _recompute_and_inject(ans: Dict[str, Any],
                          weight_map: Dict[str, float],
                          max_map: Dict[str, float],
                          fatal_map: Dict[str, bool]) -> Dict[str, Any]:
    per_call = ans.get("per_call", []) or []
    out = []
    for c in per_call:
        fname = c.get("file","")
        # inject agent from filename if empty
        agent = c.get("agent_name") or ""
        if not agent:
            rec = next((r for r in st.session_state["records"] if r["filename"] == fname), None)
            if rec: agent = rec.get("agent_name","")
        params = c.get("parameters", []) or []

        # compute pct for each param and check NA/fatal
        p_rows = []
        fatal_hit = False
        applicable_weights = 0.0
        for p in params:
            name = p.get("parameter","")
            score = float(pd.to_numeric(p.get("score", 0), errors="coerce") or 0.0)
            m = float(pd.to_numeric(p.get("max_score", max_map.get(name, 100.0)), errors="coerce") or 100.0)
            na = bool(p.get("na", False))
            fatal_triggered = bool(p.get("fatal_triggered", False))
            if fatal_map.get(name, False) and fatal_triggered:
                fatal_hit = True
            pct = 0.0 if m <= 0 else (score / m) * 100.0
            pct = max(0.0, min(100.0, pct))
            p_rows.append({
                "parameter": name,
                "pct": pct,
                "na": na,
                "fatal_triggered": fatal_triggered,
                "justification": p.get("justification",""),
                "evidence": p.get("evidence", [])
            })
            if not na:
                applicable_weights += float(weight_map.get(name, 0.0))

        # recompute overall
        if fatal_hit:
            overall = 0.0
        else:
            overall = 0.0
            denom = applicable_weights if applicable_weights > 0 else 1.0
            for r in p_rows:
                if r["na"]: continue
                w = float(weight_map.get(r["parameter"], 0.0))
                overall += r["pct"] * (w / denom)
        overall = round(float(overall), 2)

        out.append({
            "file": fname,
            "agent_name": agent,
            "purpose": c.get("purpose",""),
            "category": c.get("category",""),
            "summary": c.get("summary",""),
            "severity": c.get("severity",""),
            "sentiment": c.get("sentiment", {"start":"","mid":"","end":""}),
            "parameters": p_rows,
            "overall_weighted_score": overall,
            "fatal_overridden": fatal_hit
        })
    return {"per_call": out, "coaching_opportunities": ans.get("coaching_opportunities", [])}

# =========================
# Simple RCA / VoC (kept minimal)
# =========================
_PRESET_HINTS = {
    "RCA": "Root Cause Analysis focusing on what went wrong, contributing factors, and process gaps.",
    "VoC": "Voice of Customer: themes with counts and representative quotes with sentiment; also a concise summary."
}

def _format_context_flat(segs: List[Dict[str, Any]], max_chars=9000) -> str:
    lines=[]; total=0
    for s in segs:
        line = f"FILE: {s['filename']} [{s['start']:.1f}-{s['end']:.1f}s]\n{s['text']}"
        if total + len(line) > max_chars: break
        lines.append(line); total += len(line)
    return "\n---\n".join(lines)

def _retrieve_general(user_query: str, hint: str, top_k: int, allowed: Optional[set]) -> List[Dict[str,Any]]:
    q_vec = _embed_texts([f"{user_query}\nTask: {hint}"])[0]
    scored=[]
    for rec in st.session_state["records"]:
        if allowed and rec["filename"] not in allowed: continue
        segs = rec.get("segments", []); vecs = rec.get("embed_vectors", np.zeros((0,3072), dtype=np.float32))
        for i, seg in enumerate(segs):
            sim = _cosine_sim(q_vec, vecs[i]) if i < len(vecs) else 0.0
            scored.append((sim, rec["filename"], {"start":seg["start"],"end":seg["end"],
                                                  "text":(seg.get("text_en") or seg.get("text_orig") or "").strip()}))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    return [{"filename":fn,"start":s["start"],"end":s["end"],"text":s["text"],"score":float(sim)} for (sim,fn,s) in top]

def _ask_gemini_simple(task:str, user_query:str, segs: List[Dict[str,Any]]) -> Dict[str,Any]:
    if task == "RCA":
        instr = "Return JSON: {summary, what_went_wrong[], immediate_fixes[], preventive_actions[], evidence[]}. Each evidence: {file,start_s,end_s,quote}."
        keys = ["summary","what_went_wrong","immediate_fixes","preventive_actions","evidence"]
    else:
        instr = "Return JSON: {themes:[{theme,count,sentiment:'positive'|'negative'|'neutral',quotes:[{file,start_s,end_s,quote}]}], summary}."
        keys = ["themes","summary"]
    prompt = (
        f"Task: {task}. {_PRESET_HINTS[task]}\n"
        f"Desired JSON schema: {instr}\n"
        f"User instruction (optional): {user_query or ''}\n\n"
        f"Context:\n{_format_context_flat(segs)}\n\n"
        f"Return ONLY compact JSON with keys: {', '.join(keys)}."
    )
    resp = gemini_model.generate_content(prompt)
    return _ensure_json_dict(resp.text or "", keys)

# =========================
# Follow-up Q&A on any result
# =========================
def _followup_any(mode_name: str, question: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"task":"Follow-up Q&A on prior analysis output", "mode":mode_name,
               "question":question, "context":analysis_result}
    resp = gemini_model.generate_content(json.dumps(payload, ensure_ascii=False))
    try:
        return json.loads(resp.text)
    except Exception:
        return {"_raw": (resp.text or "").strip()}

# =========================
# Header
# =========================
st.markdown("## Call Insights Desk")
st.caption("Upload calls → Select → Configure & Ask → Summary & Dashboards → Follow-ups & Exports")

# Advanced retrieval (hidden by default)
with st.expander("Advanced (Retrieval & Runtime)", expanded=False):
    colx, coly = st.columns([0.5,0.5])
    with colx: k_general = st.slider("Top-K (RCA/VoC)", 4, 24, 8, 2)
    with coly: k_per_file = st.slider("Top-K per file (CS Audit)", 2, 12, 6, 1)

# =========================
# 1) Upload
# =========================
st.markdown("### 1) Upload Calls")
st.write('<div class="card">', unsafe_allow_html=True)
files = st.file_uploader("Drag & drop audio files", type=["mp3","wav","m4a","ogg","aac","flac"], accept_multiple_files=True)
st.write('</div>', unsafe_allow_html=True)

def _process_new_files(uploaded_files):
    if not uploaded_files: return
    with st.status("Processing uploaded calls…", expanded=True) as status:
        for f in uploaded_files:
            raw = f.getvalue(); h = _sha256(raw)
            if any(r.get("hash")==h for r in st.session_state["records"]):
                st.info(f"Skipping (already processed): {f.name}"); continue
            try:
                tr = _transcribe_file(f.name, raw)
                segs = tr["segments"] if tr else []
                if not segs:
                    full = (tr.get("text_en") or tr.get("text_orig") or "").strip()
                    if full:
                        segs = [{"start":0.0,"end":0.0,"text_orig":full[:4000],"text_en":full[:4000]}]
                embed_source = [(s.get("text_en") or s.get("text_orig") or "") for s in segs]
                vecs = _embed_texts(embed_source) if segs else np.zeros((0,3072), dtype=np.float32)
                agent_name = _extract_agent_name_from_filename(f.name)

                st.session_state["records"].append({
                    "filename": f.name, "hash": h, "audio_bytes": raw,
                    "language": tr.get("language","unknown"),
                    "text_orig": tr.get("text_orig",""), "text_en": tr.get("text_en",""),
                    "segments": segs, "embed_vectors": vecs,
                    "agent_name": agent_name
                })
                st.write(f"• {f.name} — OK")
            except Exception as e:
                st.error(f"Failed processing {f.name}: {e}")
        status.update(label=f"✅ Calls ready: {len(st.session_state['records'])} indexed.", state="complete")

if files: _process_new_files(files)

if st.session_state["records"]:
    df_idx = pd.DataFrame([{
        "filename": r["filename"],
        "agent": r.get("agent_name",""),
        "language": r.get("language","?"),
        "segments": len(r.get("segments", []))
    } for r in st.session_state["records"]])
    st.dataframe(df_idx, use_container_width=True, hide_index=True)

# =========================
# 2) Select Calls
# =========================
if st.session_state["records"]:
    st.markdown("### 2) Select Calls")
    st.write('<div class="card">', unsafe_allow_html=True)
    all_names = [r["filename"] for r in st.session_state["records"]]
    search = st.text_input("Search by filename / customer / order")
    filtered = [fn for fn in all_names if (search.lower() in fn.lower())] if search else all_names

    c0, c1 = st.columns([0.7,0.3])
    with c0:
        chosen = st.multiselect("Choose from indexed files", options=filtered, default=sorted(st.session_state["selected_files"] & set(filtered)))
    with c1:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Select all (filtered)"):
                st.session_state["selected_files"] = set(filtered); chosen = filtered
        with b2:
            if st.button("Clear selection"):
                st.session_state["selected_files"] = set(); chosen = []
        st.metric("Files selected", len(chosen))
    st.session_state["selected_files"] = set(chosen)
    st.write('</div>', unsafe_allow_html=True)

# =========================
# 3) Configure & Ask
# =========================
st.markdown("### 3) Configure & Ask")
st.write('<div class="card">', unsafe_allow_html=True)
colA, colB = st.columns([0.6, 0.4])
with colA:
    selected_insights = st.multiselect("Choose insights", options=["CS General Audit","RCA","VoC"], default=["CS General Audit","RCA","VoC"])
    user_query = st.text_area("Optional: add a specific instruction or question",
                              placeholder="E.g., 'In RCA, check if agent confirmed travel dates before booking.'", height=80)
with colB:
    st.caption("Run context")
    st.metric("Files indexed", len(st.session_state["records"]))
    st.metric("Segments in index", sum(len(r.get("segments", [])) for r in st.session_state["records"]))
    st.metric("Files selected", len(st.session_state["selected_files"]))

# ---- CS General Audit — Rubric (strict CSV template + validation) ----
rubric_ok = True
rubric_issues: List[str] = []

if "CS General Audit" in selected_insights:
    st.markdown("#### CS General Audit — Rubric")
    if st.session_state.get("cs_rubric_df") is None:
        try:
            if os.path.exists(DEFAULT_CSV_PATH):
                st.session_state["cs_rubric_df"] = pd.read_csv(DEFAULT_CSV_PATH)
            else:
                st.session_state["cs_rubric_df"] = _rubric_template_df()
        except Exception:
            st.session_state["cs_rubric_df"] = _rubric_template_df()

    c_top1, c_top2, c_top3 = st.columns([0.35,0.35,0.30])
    with c_top1:
        st.download_button("⬇️ Download rubric template (CSV)",
                          data=_download_template_csv_bytes(),
                          file_name="cs_audit_rubric_template.csv", mime="text/csv",
                          help="Columns: parameter, weight, max_score, fatal (yes/no). Weights must sum to 100.")
    with c_top2:
        up = st.file_uploader("Upload rubric CSV", type=["csv"], key="cs_rubric_upload")
        if up is not None:
            try:
                st.session_state["cs_rubric_df"] = pd.read_csv(up)
                st.success("Rubric CSV loaded.")
            except Exception as e:
                st.error(f"Invalid rubric CSV: {e}")

    edited = st.data_editor(
        st.session_state["cs_rubric_df"], use_container_width=True, num_rows="dynamic", key="cs_rubric_editor",
        column_config={
            "parameter": st.column_config.TextColumn("Parameter", required=True),
            "weight": st.column_config.NumberColumn("Weight %", min_value=0.0, max_value=100.0),
            "max_score": st.column_config.NumberColumn("Max %", min_value=1.0, max_value=100.0),
            OPTIONAL_FATAL_COL: st.column_config.TextColumn("Fatal (yes/no)")
        }
    )
    clean, rubric_issues = _validate_rubric_strict(edited)
    st.session_state["cs_rubric_df"] = clean

    rem = _rubric_remaining_weight(clean)
    c_mid1, c_mid2, c_mid3 = st.columns([0.33,0.33,0.34])
    with c_mid1: st.metric("Remaining weight", f"{rem:.2f} %")
    with c_mid2:
        if st.button("Normalize to 100%"):
            st.session_state["cs_rubric_df"] = _rebalance_weights(clean); st.experimental_rerun()
    with c_mid3:
        if st.button("Evenly distribute"):
            df = clean.copy(); n = len(df) if len(df) > 0 else 1
            df["weight"] = 100.0 / n; st.session_state["cs_rubric_df"] = df.round({"weight":2}); st.experimental_rerun()

    c_norm1, c_norm2 = st.columns([0.48,0.52])
    with c_norm1:
        st.dataframe(clean[["parameter","weight","max_score", OPTIONAL_FATAL_COL]], use_container_width=True, hide_index=True)
    with c_norm2:
        if _HAS_PLOTLY and not clean.empty:
            figw = px.pie(clean, names="parameter", values="weight", title="Weight distribution (entered)")
            st.plotly_chart(figw, use_container_width=True)
        else:
            st.info("Plotly not installed; showing table instead of chart.")

    if rubric_issues:
        rubric_ok = False
        st.error("Please fix these rubric issues before analyzing:")
        for msg in rubric_issues: st.write(f"- {msg}")
    elif abs(rem) > 0.1:
        st.warning("Weights should sum to 100. Click Normalize to 100% or adjust manually.")

st.write('</div>', unsafe_allow_html=True)

# =========================
# 4) Analyze
# =========================
can_analyze = bool(st.session_state["records"]) and bool(selected_insights) and rubric_ok
if st.button("Analyze", disabled=not can_analyze, use_container_width=True):
    with st.spinner("Analyzing…"):
        results = {}
        allowed = set(st.session_state["selected_files"]) if st.session_state["selected_files"] else {r["filename"] for r in st.session_state["records"]}

        # CS General Audit (grouped per file)
        if "CS General Audit" in selected_insights:
            grouped = _retrieve_by_file(
                user_query or "Customer Service general audit rubric scoring.",
                "Score Customer Service inbound calls using a weighted rubric.",
                per_file_k=6,
                allowed_filenames=allowed
            )
            # IMPORTANT: these two lines must be indented under the CS Audit block
            rub = _get_rubric_df()
            rubric_json, weight_map, max_map, fatal_map = _rubric_norm_and_maps(rub)

            ans = _ask_gemini_cs_audit(user_query, grouped, rubric_json)
            ans = _recompute_and_inject(ans, weight_map, max_map, fatal_map)  # NA handling + fatal + agent injection

            # (optional) keep flattened segments for evidence table
            flat_segments = []
            for fname, segs in grouped.items():
                for s in segs:
                    flat_segments.append({"filename": fname, **s})

            results["CS General Audit"] = {"segments": flat_segments, "answer": ans, "rubric": rubric_json}

        # RCA (optional)
        if "RCA" in selected_insights:
            segs = _retrieve_general(user_query or "RCA", _PRESET_HINTS["RCA"], k_general, allowed)
            results["RCA"] = {"segments": segs, "answer": _ask_gemini_simple("RCA", user_query, segs)}

        # VoC (optional)
        if "VoC" in selected_insights:
            segs = _retrieve_general(user_query or "VoC", _PRESET_HINTS["VoC"], k_general, allowed)
            results["VoC"] = {"segments": segs, "answer": _ask_gemini_simple("VoC", user_query, segs)}

        st.session_state["last_results"] = results
        st.success("Analysis complete.")


# =========================
# 5) Results & Exports
# =========================
def _cs_build_summary_table(cs_ans: Dict[str, Any]) -> pd.DataFrame:
    rows=[]
    for c in cs_ans.get("per_call", []) or []:
        fname = c.get("file","")
        agent = c.get("agent_name","")
        overall = float(c.get("overall_weighted_score", 0.0))
        sev = (c.get("severity") or "").title()
        # weakest param
        p = pd.DataFrame(c.get("parameters", []))
        weak = ""
        if not p.empty:
            px = p[~p["na"]].copy() if "na" in p.columns else p.copy()
            if not px.empty:
                px = px.sort_values("pct", ascending=True)
                weak = str(px.iloc[0]["parameter"])
        sent = c.get("sentiment", {})
        rows.append({
            "File": fname,
            "Agent": agent,
            "Purpose": c.get("purpose",""),
            "Category": c.get("category",""),
            "Severity": sev or "",
            "Overall %": round(overall,2),
            "Weakest Param": weak,
            "Sentiment Start": sent.get("start",""),
            "Sentiment End": sent.get("end",""),
            "Fatal": "Yes" if c.get("fatal_overridden", False) else "No"
        })
    return pd.DataFrame(rows)

def _render_cs_audit(cs_ans: Dict[str, Any]):
    pc = cs_ans.get("per_call", []) or []
    # KPIs
    all_scores = [float(x.get("overall_weighted_score", 0)) for x in pc]
    avg = float(np.mean(all_scores)) if all_scores else 0.0
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f'<div class="kpi"><div class="muted">Average Overall Score</div><div class="big">{avg:.2f}%</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi"><div class="muted">Calls Scored</div><div class="big">{len(pc)}</div></div>', unsafe_allow_html=True)
    with k3:
        rub = _get_rubric_df()
        st.markdown(
            f'<div class="kpi"><div class="muted">Parameters</div><div class="big">{len(rub)}</div></div>',
            unsafe_allow_html=True
        )

    # Summary table with filters/search/pagination
    st.markdown("#### Summary (all calls)")
    df_sum = _cs_build_summary_table(cs_ans)
    if df_sum.empty:
        st.info("No calls to display.")
        return

    # Filters
    fcol1, fcol2, fcol3, fcol4 = st.columns([0.25, 0.25, 0.25, 0.25])
    with fcol1:
        sev_filter = st.multiselect(
            "Severity",
            options=sorted(df_sum["Severity"].unique()),
            default=list(sorted(df_sum["Severity"].unique()))
        )
    with fcol2:
        cat_filter = st.multiselect(
            "Category",
            options=sorted(df_sum["Category"].unique()),
            default=list(sorted(df_sum["Category"].unique()))
        )
    with fcol3:
        min_score = st.number_input("Min Overall %", 0.0, 100.0, 0.0, 1.0)
    with fcol4:
        search = st.text_input("Search (file/agent/purpose)")

    filt = df_sum.copy()
    if sev_filter:
        filt = filt[filt["Severity"].isin(sev_filter)]
    if cat_filter:
        filt = filt[filt["Category"].isin(cat_filter)]
    if min_score > 0:
        filt = filt[filt["Overall %"] >= min_score]
    if search:
        s = search.lower()
        mask = (
            filt["File"].str.lower().str.contains(s)
            | filt["Agent"].str.lower().str.contains(s)
            | filt["Purpose"].str.lower().str.contains(s)
        )
        filt = filt[mask]

    # Pagination
    page_size = st.selectbox("Rows per page", options=[10, 25, 50, 100], index=1)
    total_rows = len(filt)
    total_pages = max(1, int(np.ceil(total_rows / page_size)))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = start + page_size
    st.dataframe(filt.iloc[start:end], use_container_width=True, hide_index=True)
    st.caption(f"Showing {start+1}-{min(end, total_rows)} of {total_rows} rows")

    # Expand for details per call
    st.markdown("#### Call Details")
    for c in pc:
        fname = c.get("file", "")
        if_row = filt["File"].isin([fname]).any()
        if not if_row:
            continue

        with st.expander(f"{fname} — {c.get('purpose','')}"):
            agent = c.get("agent_name", "")
            sev = (c.get("severity", "") or "").lower()
            sev_cls = (
                "badge-low" if sev == "low"
                else "badge-medium" if sev == "medium"
                else "badge-high" if sev == "high"
                else ""
            )
            sev_badge = f'<span class="badge {sev_cls}">{sev.upper()}</span>' if sev else ""
            st.markdown(
                f"**Agent:** {agent} &nbsp;&nbsp; **Overall:** {c.get('overall_weighted_score', 0)}% &nbsp;&nbsp; {sev_badge}",
                unsafe_allow_html=True
            )
            st.write(f"**Category:** {c.get('category','—')}")
            st.write(f"**Summary:** {c.get('summary','—')}")
            sent = c.get("sentiment", {})
            st.write(f"**Sentiment:** start={sent.get('start','—')}, mid={sent.get('mid','—')}, end={sent.get('end','—')}")

            # Parameter table (percent + NA)
            params = c.get("parameters", []) or []
            dfp = pd.DataFrame(params)
            if not dfp.empty:
                dfp["Score %"] = dfp["pct"].round(2)
                dfp["NA"] = dfp["na"].map(lambda x: "Yes" if x else "No")
                dfp["Fatal"] = dfp["fatal_triggered"].map(lambda x: "Yes" if x else "No")
                st.dataframe(
                    dfp[["parameter", "Score %", "NA", "Fatal", "justification"]],
                    use_container_width=True,
                    hide_index=True
                )

            # Evidence flatten
            ev_rows = []
            for p in params:
                for ev in p.get("evidence", []) or []:
                    ev_rows.append({
                        "parameter": p.get("parameter", ""),
                        "file": ev.get("file", ""),
                        "t": _mmss(ev.get("start_s", 0)),
                        "start_s": ev.get("start_s", 0),
                        "end_s": ev.get("end_s", 0),
                        "quote": ev.get("quote", "")
                    })
            df_ev = pd.DataFrame(ev_rows)
            st.markdown("**Evidence**")
            if df_ev.empty:
                st.write("—")
            else:
                st.dataframe(df_ev, use_container_width=True, hide_index=True)

    # Dashboards
    st.markdown("### Dashboards")

    # Parameter performance
    param_rows = []
    for c in pc:
        for p in c.get("parameters", []) or []:
            if p.get("na"):
                continue
            param_rows.append({"parameter": p.get("parameter", ""), "pct": p.get("pct", 0.0)})
    dfa = pd.DataFrame(param_rows)
    if not dfa.empty and _HAS_PLOTLY:
        perf = dfa.groupby("parameter", as_index=False)["pct"].mean().sort_values("pct", ascending=True)
        figp = px.bar(perf, x="pct", y="parameter", orientation="h", title="Average Parameter Performance (% of max)")
        st.plotly_chart(figp, use_container_width=True)
    elif not dfa.empty:
        st.dataframe(dfa.groupby("parameter", as_index=False)["pct"].mean(), use_container_width=True)

    # Severity distribution
    sev_df = pd.DataFrame([{"Severity": (c.get("severity", "") or "").title()} for c in pc])
    if not sev_df.empty and _HAS_PLOTLY:
        sev_ct = sev_df.value_counts("Severity").reset_index(name="count")
        fig = px.pie(sev_ct, names="Severity", values="count", title="Severity Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Score histogram
    sc_df = pd.DataFrame([{"overall": float(c.get("overall_weighted_score", 0))} for c in pc])
    if not sc_df.empty and _HAS_PLOTLY:
        figh = px.histogram(sc_df, x="overall", nbins=10, title="Overall Score Distribution")
        st.plotly_chart(figh, use_container_width=True)

    # Agent charts
    calls_table = _cs_build_summary_table(cs_ans)
    if not calls_table.empty and _HAS_PLOTLY:
        by_agent = calls_table.groupby("Agent", as_index=False)["Overall %"].mean().sort_values("Overall %")
        figA = px.bar(by_agent, x="Overall %", y="Agent", orientation="h", title="Average Score by Agent")
        st.plotly_chart(figA, use_container_width=True)

        vol = calls_table["Agent"].value_counts().reset_index()
        vol.columns = ["Agent", "Calls"]
        figV = px.bar(vol, x="Calls", y="Agent", orientation="h", title="Calls per Agent")
        st.plotly_chart(figV, use_container_width=True)

    # Downloads
    st.markdown("### Downloads")
    colD1, colD2, colD3 = st.columns(3)

    # a) audit_results.csv (flat)
    with colD1:
        flat = []
        for c in pc:
            fname = c.get("file", "")
            agent = c.get("agent_name", "")
            for p in c.get("parameters", []) or []:
                ev = (p.get("evidence") or [{}])[0]
                flat.append({
                    "file": fname,
                    "agent_name": agent,
                    "purpose": c.get("purpose", ""),
                    "category": c.get("category", ""),
                    "severity": c.get("severity", ""),
                    "overall_pct": c.get("overall_weighted_score", 0),
                    "parameter": p.get("parameter", ""),
                    "score_pct": round(float(p.get("pct", 0.0)), 2),
                    "na": p.get("na", False),
                    "fatal_triggered": p.get("fatal_triggered", False),
                    "justification": p.get("justification", ""),
                    "ev_file": ev.get("file", ""),
                    "ev_start_s": ev.get("start_s", 0),
                    "ev_quote": ev.get("quote", "")
                })
        df_flat = pd.DataFrame(flat)
        if not df_flat.empty:
            st.download_button(
                "⬇️ Download audit_results.csv",
                data=df_flat.to_csv(index=False).encode("utf-8"),
                file_name="audit_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No audit rows to export yet.")

    # b) transcripts_with_speaker.csv
    with colD2:
        if st.button("Prepare transcripts (selected) with speaker"):
            dft = _build_transcripts_df(selected_only=True, use_ai_speaker=True)
            if dft.empty:
                st.info("No transcripts for selected calls.")
            else:
                st.download_button(
                    "⬇️ transcripts_with_speaker.csv",
                    data=dft.to_csv(index=False).encode("utf-8"),
                    file_name="transcripts_with_speaker.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # c) calls_overview.csv
    with colD3:
        if not calls_table.empty:
            st.download_button(
                "⬇️ calls_overview.csv",
                data=calls_table.to_csv(index=False).encode("utf-8"),
                file_name="calls_overview.csv",
                mime="text/csv",
                use_container_width=True
            )


# RCA / VoC tabs and follow-ups
results = st.session_state.get("last_results", {})
if results:
    st.markdown("### 6) Results (RCA / VoC) & Follow-ups")
    tabs = [t for t in ["RCA","VoC"] if t in results]
    if tabs:
        t_objs = st.tabs(tabs)
        for task, tab in zip(tabs, t_objs):
            with tab:
                ans = results[task]["answer"] or {}
                if task == "RCA":
                    st.markdown("**Summary**"); st.write(ans.get("summary","—"))
                    st.markdown("**What went wrong**"); [st.write(f"- {x}") for x in (ans.get("what_went_wrong") or [])]
                    st.markdown("**Immediate fixes**"); [st.write(f"- {x}") for x in (ans.get("immediate_fixes") or [])]
                    st.markdown("**Preventive actions**"); [st.write(f"- {x}") for x in (ans.get("preventive_actions") or [])]
                else:
                    st.markdown("**Summary**"); st.write(ans.get("summary","—"))
                    st.markdown("**Themes**")
                    for t in ans.get("themes", []) or []:
                        st.write(f"- {t.get('theme','—')} (count: {t.get('count',0)}, sentiment: {t.get('sentiment','—')})")

                st.markdown("#### Follow-up Q&A")
                fu = st.text_input(f"Ask a follow-up about {task}", key=f"fu_{task.lower()}")
                if st.button(f"Ask about {task}", key=f"fu_btn_{task.lower()}") and fu.strip():
                    base = results[task]["answer"]
                    ans_fu = _followup_any(task, fu.strip(), base if isinstance(base, dict) else {"_raw": base})
                    st.markdown("**Answer**")
                    if isinstance(ans_fu, dict) and "_raw" not in ans_fu:
                        st.json(ans_fu)
                    else:
                        st.write(ans_fu.get("_raw", ans_fu))

# =========================
# 7) Audio Player
# =========================
if st.session_state["records"]:
    st.markdown("### 7) Audio Player")
    st.write('<div class="card">', unsafe_allow_html=True)
    sel = st.selectbox("Choose a file", options=[r["filename"] for r in st.session_state["records"]])
    rec = next((r for r in st.session_state["records"] if r["filename"]==sel), None)
    if rec: st.audio(rec["audio_bytes"], format="audio/mpeg")
    st.caption("Use timestamps from Evidence (mm:ss) to seek.")
    st.write('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("For QA pilot use. No PII redaction. © Your Company")
