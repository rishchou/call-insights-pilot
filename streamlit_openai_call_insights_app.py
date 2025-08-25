# Call Insights Desk — Auto-STT + Gemini Analysis (RCA • CS General Audit • VoC)
# ------------------------------------------------------------------------------
# Flow: Upload → Select Calls → Pick Insight(s) → Analyze → Evidence & Dashboard → Follow-up Q&A
# Under the hood:
#   • OpenAI Whisper for transcription (original + English translation)
#   • OpenAI embeddings (English) for retrieval
#   • Google Gemini for analysis (STRICT JSON; auto-repair if needed)
#
# Env/Secrets needed:
#   OPENAI_API_KEY="sk-..."   GEMINI_API_KEY="AIza..."
#
# requirements.txt:
#   streamlit
#   openai
#   google-generativeai
#   numpy
#   pandas
#   plotly

import os
import io
import json
import hashlib
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# --- OpenAI (STT + embeddings) ---
from openai import OpenAI

# --- Gemini (analysis) ---
import google.generativeai as genai

# =========================
# Hidden config (no UI)
# =========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not OPENAI_API_KEY or not GEMINI_API_KEY:
    st.error("Missing OPENAI_API_KEY or GEMINI_API_KEY. Set as env vars or in Streamlit secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
oai = OpenAI(api_key=OPENAI_API_KEY)

genai.configure(api_key=GEMINI_API_KEY)
_GEMINI_MODEL_NAME = "gemini-1.5-pro"   # or "gemini-1.5-flash" for cheaper/faster
gemini_model = genai.GenerativeModel(
    _GEMINI_MODEL_NAME,
    generation_config={
        "temperature": 0.2,
        "response_mime_type": "application/json",
    },
    system_instruction=(
        "You analyze customer service call transcripts. If transcripts are multilingual, rely on the English translation. "
        "Be precise and evidence-based; cite FILE names with timestamps (mm:ss) in quotes you extract. "
        "Always return STRICT JSON for the requested schema only, in English."
    ),
)

# Internal defaults
_EMBED_MODEL = "text-embedding-3-large"
_TOP_K_GENERAL_DEFAULT = 8
_TOP_K_PER_FILE_DEFAULT = 6
DEFAULT_CSV_PATH = "/mnt/data/CS QA parameters.csv"

# =========================
# Page & session state
# =========================
st.set_page_config(page_title="Call Insights Desk", layout="wide")

if "records" not in st.session_state:
    # records: [{ filename, hash, audio_bytes, language, text_orig, text_en, segments:[{start,end,text_orig,text_en}], embed_vectors: np.ndarray }]
    st.session_state["records"] = []

if "selected_files" not in st.session_state:
    st.session_state["selected_files"] = set()

# =========================
# Helpers
# =========================
def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 3072), dtype=np.float32)
    resp = oai.embeddings.create(model=_EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    an = a / (np.linalg.norm(a) + 1e-9)
    bn = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(an, bn))

@st.cache_data(show_spinner=False)
def _transcribe_file(filename: str, raw_bytes: bytes) -> Dict[str, Any]:
    """
    Whisper pass 1: original-language transcript (verbose_json)
    Whisper pass 2: English translation (translate=True)
    Returns: {text_orig, text_en, segments:[{start,end,text_orig,text_en}], language}
    """
    tmp = os.path.join("/tmp", filename)
    with open(tmp, "wb") as f:
        f.write(raw_bytes)

    # Pass 1: original
    with open(tmp, "rb") as f1:
        r_orig = oai.audio.transcriptions.create(
            model="whisper-1",
            file=f1,
            response_format="verbose_json",
            temperature=0
        )
    d_orig = r_orig.model_dump() if hasattr(r_orig, "model_dump") else json.loads(r_orig.json())

    # Pass 2: English translation
    with open(tmp, "rb") as f2:
        r_en = oai.audio.transcriptions.create(
            model="whisper-1",
            file=f2,
            response_format="verbose_json",
            temperature=0,
            translate=True
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

    return {
        "language": language,
        "text_orig": (d_orig.get("text") or "").strip(),
        "text_en":   (d_en.get("text") or "").strip(),
        "segments": segments
    }

def _mmss(sec: float) -> str:
    m = int((sec or 0) // 60)
    s = int((sec or 0) % 60)
    return f"{m:02d}:{s:02d}"

# --- Insights we KEEP ---
_PRESET_HINTS = {
    "RCA": "Root Cause Analysis focusing on what went wrong, contributing factors, and process gaps. Return a short summary, what went wrong, immediate fixes, preventive actions, and evidence quotes with timestamps.",
    "CS General Audit": "Score Customer Service inbound calls using a weighted rubric. Score each parameter 0..max_score with a short justification and 1–2 evidence quotes. Compute overall weighted score per call.",
    "VoC": "Voice of Customer: themes with counts and representative quotes with sentiment; also a concise summary."
}
_INSIGHTS = ["RCA", "CS General Audit", "VoC"]

_SCHEMAS = {
    "RCA": {
        "instruction": (
            "Produce: {summary, what_went_wrong[], immediate_fixes[], preventive_actions[], evidence[]}. "
            "Each evidence: {file, start_s, end_s, quote}."
        ),
        "keys": ["summary", "what_went_wrong", "immediate_fixes", "preventive_actions", "evidence"],
    },
    "CS General Audit": {
        "instruction": (
            "Return JSON: {per_call:[{file, parameters:[{parameter, score, max_score, justification, "
            "evidence:[{file,start_s,end_s,quote}]}], overall_weighted_score}], coaching_opportunities[]}."
        ),
        "keys": ["per_call", "coaching_opportunities"]
    },
    "VoC": {
        "instruction": (
            "Voice of Customer themes: {themes:[{theme, count, sentiment:'positive'|'negative'|'neutral', "
            "quotes:[{file,start_s,end_s,quote}]}], summary}."
        ),
        "keys": ["themes", "summary"],
    }
}

def _format_context(segments: List[Dict[str, Any]], max_chars: int = 9000) -> str:
    lines, total = [], 0
    for s in segments:
        line = f"FILE: {s['filename']} [{s['start']:.1f}-{s['end']:.1f}s]\n{s['text']}"
        if total + len(line) > max_chars:
            break
        lines.append(line); total += len(line)
    return "\n---\n".join(lines)

def _ensure_json_dict(resp_text: str, desired_keys: List[str]) -> Dict[str, Any]:
    """Try to parse JSON; if it fails, ask Gemini to repair to valid JSON."""
    try:
        obj = json.loads(resp_text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    fix_prompt = json.dumps({
        "task": "Repair model output into valid compact JSON",
        "desired_keys": desired_keys,
        "raw": resp_text
    }, ensure_ascii=False)
    r = gemini_model.generate_content(fix_prompt)
    try:
        return json.loads(r.text)
    except Exception:
        return {"_raw": (resp_text or "").strip()}

def _ask_gemini(task: str, user_query: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema = _SCHEMAS[task]
    extra = ""

    # Inject normalized rubric for CS General Audit
    if task == "CS General Audit":
        rub = st.session_state.get("cs_rubric_df")
        if rub is not None and not rub.empty:
            rub_v = _validate_rubric(rub)
            rub_n = _normalize_weights(rub_v)
            rubric_json = rub_n[["parameter","norm_w","max_score"]].to_dict(orient="records")
            extra = f"\nRubric (weights normalized to 100): {json.dumps(rubric_json, ensure_ascii=False)}\n"

    prompt = (
        f"Task: {task}. {_PRESET_HINTS[task]}\n"
        f"Desired JSON schema: {schema['instruction']}\n"
        f"{extra}"
        f"User query (may be empty): {user_query}\n\n"
        f"Context:\n{_format_context(segments)}\n\n"
        f"Return ONLY valid compact JSON with keys: {', '.join(schema['keys'])}."
    )
    resp = gemini_model.generate_content(prompt)
    return _ensure_json_dict(resp.text or "", _SCHEMAS[task]["keys"])

# ---------- Retrieval ----------
def _embed_source_text(seg: Dict[str, Any]) -> str:
    return (seg.get("text_en") or seg.get("text_orig") or "").strip()

def _retrieve_general(user_query: str, hint: str, top_k: int,
                      allowed_filenames: Optional[set] = None) -> List[Dict[str, Any]]:
    """Top-K across (optionally) restricted files for a task."""
    q = f"{user_query}\nTask: {hint}".strip()
    q_vec = _embed_texts([q])[0]
    scored = []
    for rec in st.session_state["records"]:
        if allowed_filenames and rec["filename"] not in allowed_filenames:
            continue
        segs = rec.get("segments", [])
        vecs = rec.get("embed_vectors", np.zeros((0, 3072), dtype=np.float32))
        for i, seg in enumerate(segs):
            sim = _cosine_sim(q_vec, vecs[i]) if i < len(vecs) else 0.0
            scored.append((sim, rec["filename"], {
                "start": seg["start"],
                "end": seg["end"],
                "text": _embed_source_text(seg)
            }))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    return [{
        "filename": fname,
        "start": seg["start"],
        "end": seg["end"],
        "text": seg["text"],
        "score": float(sim)
    } for (sim, fname, seg) in top]

def _retrieve_round_robin(user_query: str, hint: str, per_file_k: int,
                          allowed_filenames: set) -> List[Dict[str, Any]]:
    """Guarantee coverage: top-N segments per selected file."""
    q = f"{user_query}\nTask: {hint}".strip()
    q_vec = _embed_texts([q])[0]
    out = []
    for rec in st.session_state["records"]:
        if rec["filename"] not in allowed_filenames:
            continue
        local = []
        segs = rec.get("segments", [])
        vecs = rec.get("embed_vectors", np.zeros((0, 3072), dtype=np.float32))
        for i, seg in enumerate(segs):
            sim = _cosine_sim(q_vec, vecs[i]) if i < len(vecs) else 0.0
            local.append((sim, rec["filename"], {
                "start": seg["start"],
                "end": seg["end"],
                "text": _embed_source_text(seg)
            }))
        local.sort(key=lambda x: x[0], reverse=True)
        for sim, fname, seg in local[:per_file_k]:
            out.append({
                "filename": fname,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "score": float(sim)
            })
    return out

# =========================
# Rubric helpers (CS General Audit)
# =========================
def _load_default_cs_rubric() -> pd.DataFrame:
    cols = ["parameter", "weight", "max_score"]
    try:
        if os.path.exists(DEFAULT_CSV_PATH):
            df = pd.read_csv(DEFAULT_CSV_PATH)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in default CSV: {missing}")
            return df[cols]
    except Exception as e:
        st.warning(f"Could not load default rubric CSV: {e}")

    # Fallback template
    return pd.DataFrame({
        "parameter": ["Greeting & ID","Empathy","Policy Adherence","Resolution","Closure & Recap"],
        "weight":    [10,15,25,30,20],
        "max_score": [5,5,5,5,5]
    })

def _validate_rubric(df: pd.DataFrame) -> pd.DataFrame:
    req = ["parameter","weight","max_score"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Rubric missing column: {c}")
    out = df.copy()
    out["parameter"] = out["parameter"].fillna("").astype(str).str.strip()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["max_score"] = pd.to_numeric(out["max_score"], errors="coerce").fillna(1.0).clip(lower=0.1)
    out = out[out["parameter"] != ""].reset_index(drop=True)
    if out.empty:
        raise ValueError("Rubric has no valid parameters.")
    return out

def _normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    total = float(out["weight"].sum())
    if total <= 0:
        out["norm_w"] = 100.0 / len(out)
    else:
        out["norm_w"] = (out["weight"] / total) * 100.0
    return out

# =========================
# Header
# =========================
left, right = st.columns([0.8, 0.2])
with left:
    st.title("🎧 Call Insights Desk")
    st.caption("Upload calls → Select → Analyze → Evidence & Dashboard → Ask follow-ups")
with right:
    st.markdown("**Mode:** Internal Pilot")

# Sidebar: Retrieval tuning
with st.sidebar:
    st.header("Retrieval")
    _k_general = st.slider("Top-K (general)", 4, 24, _TOP_K_GENERAL_DEFAULT, 2)
    _k_per_file = st.slider("Top-K per file", 2, 12, _TOP_K_PER_FILE_DEFAULT, 1)

# =========================
# 1) Upload (auto processing)
# =========================
st.header("1) Upload Calls")
files = st.file_uploader(
    "Drop multiple audio files",
    type=["mp3","wav","m4a","ogg","aac","flac"],
    accept_multiple_files=True,
)

def _process_new_files(uploaded_files):
    if not uploaded_files:
        return
    with st.status("Processing uploaded calls…", expanded=True) as status:
        for f in uploaded_files:
            raw = f.getvalue()
            h = _sha256(raw)
            if any(r.get("hash") == h for r in st.session_state["records"]):
                st.info(f"Skipping (already processed): {f.name}")
                continue
            try:
                tr = _transcribe_file(f.name, raw)
                segs = tr["segments"] if tr else []
                embed_source = [(_s.get("text_en") or _s.get("text_orig") or "") for _s in segs]
                vecs = _embed_texts(embed_source) if segs else np.zeros((0,3072), dtype=np.float32)
                st.session_state["records"].append({
                    "filename": f.name,
                    "hash": h,
                    "audio_bytes": raw,
                    "language": tr.get("language","unknown"),
                    "text_orig": tr.get("text_orig",""),
                    "text_en": tr.get("text_en",""),
                    "segments": segs,
                    "embed_vectors": vecs,
                })
                st.write(f"• {f.name} — OK")
            except Exception as e:
                st.error(f"Failed processing {f.name}: {e}")
        status.update(label=f"✅ Calls ready: {len(st.session_state['records'])} indexed.", state="complete")

if files:
    _process_new_files(files)

# Quick summary table
if st.session_state["records"]:
    st.subheader("Indexed files")
    df_idx = pd.DataFrame([{
        "filename": r["filename"],
        "language": r.get("language","?"),
        "segments": len(r["segments"])
    } for r in st.session_state["records"]])
    st.dataframe(df_idx, use_container_width=True, hide_index=True)

# =========================
# 2) Select Calls (filter)
# =========================
if st.session_state["records"]:
    st.header("2) Select Calls")
    all_names = [r["filename"] for r in st.session_state["records"]]
    search = st.text_input("Search by filename / customer / order (matches filename text)")
    filtered = [fn for fn in all_names if (search.lower() in fn.lower())] if search else all_names

    c0, c1 = st.columns([0.6, 0.4])
    with c0:
        chosen = st.multiselect(
            "Choose from indexed files",
            options=filtered,
            default=sorted(st.session_state["selected_files"] & set(filtered))
        )
    with c1:
        a, b = st.columns(2)
        with a:
            if st.button("Select all (filtered)"):
                st.session_state["selected_files"] = set(filtered)
                chosen = filtered
        with b:
            if st.button("Clear selection"):
                st.session_state["selected_files"] = set()
                chosen = []
    st.session_state["selected_files"] = set(chosen)
    st.caption(f"Selected: {len(st.session_state['selected_files'])} / {len(all_names)} files")

# =========================
# 3) Configure Insights & Rubric (CS Audit)
# =========================
st.header("3) Configure & Ask")
colA, colB = st.columns([0.65, 0.35])

with colA:
    selected_insights = st.multiselect(
        "Choose insights",
        options=_INSIGHTS,
        default=["RCA", "CS General Audit", "VoC"]
    )
    user_query = st.text_area(
        "Optional: add a specific question or instruction",
        placeholder="E.g., 'In RCA, check if agent confirmed travel dates before booking.'",
        height=80
    )

with colB:
    st.metric("Segments in index", sum(len(r.get("segments", [])) for r in st.session_state["records"]))
    st.metric("Files indexed", len(st.session_state["records"]))
    st.metric("Files selected", len(st.session_state["selected_files"]))

    # CS General Audit — Rubric Manager
    if "CS General Audit" in selected_insights:
        st.markdown("### CS General Audit — Rubric")
        if "cs_rubric_df" not in st.session_state:
            st.session_state["cs_rubric_df"] = _load_default_cs_rubric()

        up = st.file_uploader("Upload rubric CSV (parameter, weight, max_score)", type=["csv"], key="cs_rubric_upload")
        if up is not None:
            try:
                df_up = pd.read_csv(up)
                df_up = _validate_rubric(df_up)
                st.session_state["cs_rubric_df"] = df_up
                st.success("Rubric CSV loaded.")
            except Exception as e:
                st.error(f"Invalid rubric CSV: {e}")

        edited = st.data_editor(
            st.session_state["cs_rubric_df"],
            use_container_width=True,
            num_rows="dynamic",
            key="cs_rubric_editor",
            column_config={
                "parameter": st.column_config.TextColumn("Parameter", required=True),
                "weight": st.column_config.NumberColumn("Weight", help="Relative importance (normalized to 100)"),
                "max_score": st.column_config.NumberColumn("Max Score", help="Max points for this parameter"),
            }
        )
        try:
            edited = _validate_rubric(edited)
            st.session_state["cs_rubric_df"] = edited
            norm = _normalize_weights(edited)
            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(norm[["parameter","norm_w","max_score"]], use_container_width=True, hide_index=True)
            with c2:
                figw = px.pie(norm, names="parameter", values="norm_w", title="Weight distribution (normalized)")
                st.plotly_chart(figw, use_container_width=True)

            st.download_button(
                "⬇️ Download current rubric CSV",
                data=edited.to_csv(index=False).encode("utf-8"),
                file_name="cs_general_audit_rubric.csv",
                mime="text/csv"
            )
            if st.button("Reset to default rubric"):
                st.session_state["cs_rubric_df"] = _load_default_cs_rubric()
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Rubric error: {e}")

# =========================
# 4) Analyze
# =========================
can_analyze = bool(st.session_state["records"]) and bool(selected_insights)
if st.button("Analyze", disabled=not can_analyze):
    with st.spinner("Analyzing…"):
        results = {}
        allowed = set(st.session_state["selected_files"]) if st.session_state["selected_files"] else None

        for task in selected_insights:
            # Retrieval strategy: for RCA & VoC use global Top-K; for CS Audit emphasize coverage
            if task == "CS General Audit":
                if not allowed:
                    allowed = set([r["filename"] for r in st.session_state["records"]])  # audit can run on all if none picked
                top_segments = _retrieve_round_robin(
                    user_query or "Customer Service general audit rubric scoring.",
                    _PRESET_HINTS[task],
                    per_file_k=_k_per_file,
                    allowed_filenames=allowed
                )
            else:
                top_segments = _retrieve_general(
                    user_query or task,
                    _PRESET_HINTS[task],
                    _k_general,
                    allowed_filenames=allowed
                )

            results[task] = {
                "segments": top_segments,
                "answer": _ask_gemini(task, user_query, top_segments)
            }

        st.session_state["last_results"] = results
        st.success("Analysis complete.")

# =========================
# 5) Render Results (tabs) + Follow-ups
# =========================
def _render_rca(ans: Dict[str, Any]):
    st.markdown("**Summary**"); st.write(ans.get("summary", "—"))
    st.markdown("**What went wrong**")
    for x in ans.get("what_went_wrong", []) or []: st.write(f"- {x}")
    st.markdown("**Immediate fixes**")
    for x in ans.get("immediate_fixes", []) or []: st.write(f"- {x}")
    st.markdown("**Preventive actions**")
    for x in ans.get("preventive_actions", []) or []: st.write(f"- {x}")

def _render_voc(ans: Dict[str, Any]):
    st.markdown("**Summary**"); st.write(ans.get("summary", "—"))
    st.markdown("**Themes**")
    for t in ans.get("themes", []) or []:
        st.write(f"- {t.get('theme','—')} (count: {t.get('count',0)}, sentiment: {t.get('sentiment','—')})")

def _build_evidence(task: str, ans: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    def add(file, s, e, q, extra=None):
        row = {"file": file, "start_s": s, "end_s": e, "quote": q}
        if extra: row.update(extra)
        rows.append(row)
    try:
        if task == "RCA":
            for ev in ans.get("evidence", []) or []:
                add(ev.get("file",""), ev.get("start_s",0), ev.get("end_s",0), ev.get("quote",""))
        elif task == "CS General Audit":
            for c in ans.get("per_call", []) or []:
                for p in c.get("parameters", []) or []:
                    for ev in p.get("evidence", []) or []:
                        add(ev.get("file",""), ev.get("start_s",0), ev.get("end_s",0), ev.get("quote",""),
                            {"parameter": p.get("parameter",""), "file_call": c.get("file","")})
        elif task == "VoC":
            for t in ans.get("themes", []) or []:
                for q in t.get("quotes", []) or []:
                    add(q.get("file",""), q.get("start_s",0), q.get("end_s",0), q.get("quote",""),
                        {"theme": t.get("theme",""), "sentiment": t.get("sentiment",""), "count": t.get("count",0)})
    except Exception:
        pass
    df = pd.DataFrame(rows)
    if not df.empty:
        df["t"] = df["start_s"].apply(_mmss)
        # reorder common columns first
        lead = [c for c in ["file","t","start_s","end_s","quote","parameter","file_call","theme","sentiment","count"] if c in df.columns]
        df = df[lead + [c for c in df.columns if c not in lead]]
    return df

def _render_cs_audit(ans: Dict[str, Any]):
    pc = ans.get("per_call", []) or []

    # KPI row
    all_scores = [c.get("overall_weighted_score", 0) for c in pc if isinstance(c, dict)]
    avg_score = float(np.mean(all_scores)) if all_scores else 0.0
    k1, k2 = st.columns(2)
    k1.metric("Average Overall Score", f"{avg_score:.2f}")
    k2.metric("Calls Scored", len(pc))

    # Per-call table and cross-call aggregation
    param_rows = []
    for c in pc:
        fname = c.get("file","(file)")
        st.markdown(f"**📞 {fname} — Overall:** {c.get('overall_weighted_score',0)}")
        params = c.get("parameters", []) or []
        dfp = pd.DataFrame(params)
        if not dfp.empty:
            for rrow in params:
                param_rows.append({
                    "file": fname,
                    "parameter": rrow.get("parameter",""),
                    "score": rrow.get("score",0),
                    "max_score": rrow.get("max_score",1)
                })
            st.dataframe(dfp[["parameter","score","max_score","justification"]], use_container_width=True, hide_index=True)

    st.markdown("### Dashboard")
    if param_rows:
        dfa = pd.DataFrame(param_rows)
        dfa["pct"] = (dfa["score"] / dfa["max_score"]).replace([np.inf,-np.inf], 0.0) * 100.0
        perf = dfa.groupby("parameter", as_index=False)["pct"].mean().sort_values("pct", ascending=True)
        figp = px.bar(perf, x="pct", y="parameter", orientation="h", title="Average Parameter Performance (% of max)")
        st.plotly_chart(figp, use_container_width=True)
    else:
        st.info("No parameter data to chart yet.")

    if ans.get("coaching_opportunities"):
        st.markdown("### Coaching Opportunities")
        for x in ans["coaching_opportunities"]:
            st.write(f"- {x}")

# Universal follow-up helper
def _followup_any(mode_name: str, question: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "task": "Follow-up Q&A on prior analysis output",
        "mode": mode_name,
        "question": question,
        "context": analysis_result
    }
    prompt = json.dumps(payload, ensure_ascii=False)
    resp = gemini_model.generate_content(prompt)
    try:
        return json.loads(resp.text)
    except Exception:
        return {"_raw": (resp.text or "").strip()}

# Render tabs
results = st.session_state.get("last_results", {})
if results:
    tabs = st.tabs(list(results.keys()))
    for (task, tab) in zip(results.keys(), tabs):
        with tab:
            st.subheader(task)
            ans = results[task]["answer"] or {}

            if task == "RCA":
                _render_rca(ans)
            elif task == "CS General Audit":
                _render_cs_audit(ans)
            elif task == "VoC":
                _render_voc(ans)

            # Evidence
            df_ev = _build_evidence(task, ans)
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

            # Universal Follow-up Q&A
            st.divider()
            st.subheader("Follow-up Q&A")
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
# 6) Audio Player
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
