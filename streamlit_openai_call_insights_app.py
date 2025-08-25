# Call Insights Desk — RCA • CS General Audit • VoC
# Clean UI version (no sidebar, roomy dashboard, universal follow-ups)

import os, io, json, hashlib
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Plotly is optional but recommended for charts
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

from openai import OpenAI
import google.generativeai as genai

# === Config / Secrets ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not OPENAI_API_KEY or not GEMINI_API_KEY:
    st.error("Missing OPENAI_API_KEY or GEMINI_API_KEY. Add them to env vars or Streamlit secrets.")
    st.stop()

oai = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

_GEMINI_MODEL_NAME = "gemini-1.5-pro"   # or gemini-1.5-flash
gemini_model = genai.GenerativeModel(
    _GEMINI_MODEL_NAME,
    generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
    system_instruction=(
        "You analyze customer service call transcripts. If transcripts are multilingual, rely on the English translation. "
        "Be precise and evidence-based; cite FILE names with timestamps (mm:ss) in quotes you extract. "
        "Always return STRICT JSON for the requested schema only, in English."
    ),
)

# Embeddings config
_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_CSV_PATH = "/mnt/data/CS QA parameters.csv"

# === Page ===
st.set_page_config(page_title="Call Insights Desk", layout="wide")

# Minimal CSS polish
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-weight: 700; }
    .card { background: #ffffff; border: 1px solid #eee; border-radius: 14px; padding: 16px 18px; }
    .muted { color: #667085; }
    .kpi { background:#fafafa;border:1px solid #eee;border-radius:12px;padding:12px 16px;text-align:center; }
    .kpi .big { font-size: 22px; font-weight: 700; }
    .section { margin-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# === Session ===
if "records" not in st.session_state:
    st.session_state["records"] = []  # [{ filename, hash, audio_bytes, language, text_orig, text_en, segments, embed_vectors }]
if "selected_files" not in st.session_state:
    st.session_state["selected_files"] = set()
if "cs_rubric_df" not in st.session_state:
    st.session_state["cs_rubric_df"] = None

# === Helpers ===
def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _embed_texts(texts: List[str]) -> np.ndarray:
    if not texts: return np.zeros((0, 3072), dtype=np.float32)
    resp = oai.embeddings.create(model=_EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0: return 0.0
    an = a / (np.linalg.norm(a) + 1e-9)
    bn = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(an, bn))

@st.cache_data(show_spinner=False)
def _transcribe_file(filename: str, raw_bytes: bytes) -> Dict[str, Any]:
    """
    Whisper pass 1: original-language transcript (verbose_json)
    Whisper pass 2: English translation via translations.create (verbose_json)
    Returns: {text_orig, text_en, segments:[{start,end,text_orig,text_en}], language}
    """
    tmp = os.path.join("/tmp", filename)
    with open(tmp, "wb") as f:
        f.write(raw_bytes)

    # Pass 1: original-language transcription
    with open(tmp, "rb") as f1:
        r_orig = oai.audio.transcriptions.create(
            model="whisper-1",
            file=f1,
            response_format="verbose_json",
            temperature=0
        )
    d_orig = r_orig.model_dump() if hasattr(r_orig, "model_dump") else json.loads(r_orig.json())

    # Pass 2: English translation (NOTE: use translations.create — no 'translate' arg)
    with open(tmp, "rb") as f2:
        r_en = oai.audio.translations.create(
            model="whisper-1",
            file=f2,
            response_format="verbose_json",
            temperature=0
        )
    d_en = r_en.model_dump() if hasattr(r_en, "model_dump") else json.loads(r_en.json())

    language = d_orig.get("language") or "unknown"

    # Align segments best-effort by index (both verbose_json responses usually include segments)
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
    m = int((sec or 0)//60); s = int((sec or 0)%60); return f"{m:02d}:{s:02d}"

# Insights we keep
_PRESET_HINTS = {
    "RCA": "Root Cause Analysis focusing on what went wrong, contributing factors, and process gaps. Return a short summary, what went wrong, immediate fixes, preventive actions, and evidence quotes with timestamps.",
    "CS General Audit": "Score Customer Service inbound calls using a weighted rubric. Score each parameter 0..max_score with a short justification and 1–2 evidence quotes. Compute overall weighted score per call.",
    "VoC": "Voice of Customer: themes with counts and representative quotes with sentiment; also a concise summary."
}
_INSIGHTS = ["RCA", "CS General Audit", "VoC"]
_SCHEMAS = {
    "RCA": {
        "instruction": "Produce: {summary, what_went_wrong[], immediate_fixes[], preventive_actions[], evidence[]}. Each evidence: {file, start_s, end_s, quote}.",
        "keys": ["summary", "what_went_wrong", "immediate_fixes", "preventive_actions", "evidence"],
    },
    "CS General Audit": {
        "instruction": "Return JSON: {per_call:[{file, parameters:[{parameter, score, max_score, justification, evidence:[{file,start_s,end_s,quote}]}], overall_weighted_score}], coaching_opportunities[]}.",
        "keys": ["per_call", "coaching_opportunities"]
    },
    "VoC": {
        "instruction": "Voice of Customer themes: {themes:[{theme, count, sentiment:'positive'|'negative'|'neutral', quotes:[{file,start_s,end_s,quote}]}], summary}.",
        "keys": ["themes", "summary"],
    }
}

def _format_context(segments: List[Dict[str, Any]], max_chars: int = 9000) -> str:
    lines, total = [], 0
    for s in segments:
        line = f"FILE: {s['filename']} [{s['start']:.1f}-{s['end']:.1f}s]\n{s['text']}"
        if total + len(line) > max_chars: break
        lines.append(line); total += len(line)
    return "\n---\n".join(lines)

def _ensure_json_dict(resp_text: str, desired_keys: List[str]) -> Dict[str, Any]:
    try:
        obj = json.loads(resp_text)
        if isinstance(obj, dict): return obj
    except Exception:
        pass
    # ask Gemini to repair
    fix_prompt = json.dumps({"task": "Repair model output into valid compact JSON",
                             "desired_keys": desired_keys, "raw": resp_text}, ensure_ascii=False)
    r = gemini_model.generate_content(fix_prompt)
    try:
        return json.loads(r.text)
    except Exception:
        return {"_raw": (resp_text or "").strip()}

def _validate_rubric(df: pd.DataFrame) -> pd.DataFrame:
    req = ["parameter","weight","max_score"]
    for c in req:
        if c not in df.columns: raise ValueError(f"Rubric missing column: {c}")
    out = df.copy()
    out["parameter"] = out["parameter"].fillna("").astype(str).str.strip()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["max_score"] = pd.to_numeric(out["max_score"], errors="coerce").fillna(1.0).clip(lower=0.1)
    out = out[out["parameter"]!=""].reset_index(drop=True)
    if out.empty: raise ValueError("Rubric has no valid parameters.")
    return out

def _normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    total = float(out["weight"].sum())
    out["norm_w"] = (100.0/len(out)) if total <= 0 else (out["weight"]/total)*100.0
    return out

def _load_default_cs_rubric() -> pd.DataFrame:
    cols = ["parameter","weight","max_score"]
    try:
        if os.path.exists(DEFAULT_CSV_PATH):
            df = pd.read_csv(DEFAULT_CSV_PATH)
            miss = [c for c in cols if c not in df.columns]
            if miss: raise ValueError(f"Missing columns in default CSV: {miss}")
            return df[cols]
    except Exception as e:
        st.warning(f"Could not load default rubric CSV: {e}")
    return pd.DataFrame({
        "parameter": ["Greeting & ID","Empathy","Policy Adherence","Resolution","Closure & Recap"],
        "weight": [10,15,25,30,20],
        "max_score": [5,5,5,5,5]
    })

def _ask_gemini(task: str, user_query: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema = _SCHEMAS[task]
    extra = ""
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

def _embed_source_text(seg: Dict[str, Any]) -> str:
    return (seg.get("text_en") or seg.get("text_orig") or "").strip()

def _retrieve_general(user_query: str, hint: str, top_k: int,
                      allowed_filenames: Optional[set] = None) -> List[Dict[str, Any]]:
    q_vec = _embed_texts([f"{user_query}\nTask: {hint}"])[0]
    scored = []
    for rec in st.session_state["records"]:
        if allowed_filenames and rec["filename"] not in allowed_filenames: continue
        segs = rec.get("segments", []); vecs = rec.get("embed_vectors", np.zeros((0,3072), dtype=np.float32))
        for i, seg in enumerate(segs):
            sim = _cosine_sim(q_vec, vecs[i]) if i < len(vecs) else 0.0
            scored.append((sim, rec["filename"], {"start": seg["start"], "end": seg["end"], "text": _embed_source_text(seg)}))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    return [{"filename": fn, "start": s["start"], "end": s["end"], "text": s["text"], "score": float(sim)} for (sim, fn, s) in top]

def _retrieve_round_robin(user_query: str, hint: str, per_file_k: int, allowed_filenames: set) -> List[Dict[str, Any]]:
    q_vec = _embed_texts([f"{user_query}\nTask: {hint}"])[0]; out=[]
    for rec in st.session_state["records"]:
        if rec["filename"] not in allowed_filenames: continue
        local=[]
        segs = rec.get("segments", []); vecs = rec.get("embed_vectors", np.zeros((0,3072), dtype=np.float32))
        for i, seg in enumerate(segs):
            sim = _cosine_sim(q_vec, vecs[i]) if i < len(vecs) else 0.0
            local.append((sim, rec["filename"], {"start": seg["start"], "end": seg["end"], "text": _embed_source_text(seg)}))
        local.sort(key=lambda x: x[0], reverse=True)
        for sim, fn, s in local[:per_file_k]:
            out.append({"filename": fn, "start": s["start"], "end": s["end"], "text": s["text"], "score": float(sim)})
    return out

def _followup_any(mode_name: str, question: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"task": "Follow-up Q&A on prior analysis output", "mode": mode_name, "question": question, "context": analysis_result}
    resp = gemini_model.generate_content(json.dumps(payload, ensure_ascii=False))
    try:
        return json.loads(resp.text)
    except Exception:
        return {"_raw": (resp.text or "").strip()}

# === Header ===
st.markdown("<h1>Call Insights Desk</h1>", unsafe_allow_html=True)
st.caption("Upload calls → Select → Configure & Ask → Results & Dashboard → Follow-ups")

# === Retrieval controls (moved out of sidebar) ===
with st.expander("Advanced (Retrieval & Runtime)", expanded=False):
    colx, coly, colz = st.columns([0.4,0.4,0.2])
    with colx:
        k_general = st.slider("Top-K (general)", 4, 24, 8, 2)
    with coly:
        k_per_file = st.slider("Top-K per file (coverage)", 2, 12, 6, 1)
    with colz:
        st.write("") ; st.write("")
        st.caption("Tune for bigger datasets")

# === 1) Upload ===
st.markdown("## 1) Upload Calls")
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
                embed_source = [(_s.get("text_en") or _s.get("text_orig") or "") for _s in segs]
                vecs = _embed_texts(embed_source) if segs else np.zeros((0,3072), dtype=np.float32)
                st.session_state["records"].append({
                    "filename": f.name, "hash": h, "audio_bytes": raw,
                    "language": tr.get("language","unknown"),
                    "text_orig": tr.get("text_orig",""), "text_en": tr.get("text_en",""),
                    "segments": segs, "embed_vectors": vecs,
                })
                st.write(f"• {f.name} — OK")
            except Exception as e:
                st.error(f"Failed processing {f.name}: {e}")
        status.update(label=f"✅ Calls ready: {len(st.session_state['records'])} indexed.", state="complete")

if files: _process_new_files(files)

if st.session_state["records"]:
    df_idx = pd.DataFrame([{"filename": r["filename"], "language": r.get("language","?"), "segments": len(r["segments"])} for r in st.session_state["records"]])
    st.dataframe(df_idx, use_container_width=True, hide_index=True)

# === 2) Select Calls ===
if st.session_state["records"]:
    st.markdown("## 2) Select Calls")
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
            if st.button("Select all (filtered)"): st.session_state["selected_files"] = set(filtered); chosen = filtered
        with b2:
            if st.button("Clear selection"): st.session_state["selected_files"] = set(); chosen = []
        st.metric("Files selected", len(chosen))
    st.session_state["selected_files"] = set(chosen)
    st.write('</div>', unsafe_allow_html=True)

# === 3) Configure & Ask ===
st.markdown("## 3) Configure & Ask")
st.write('<div class="card">', unsafe_allow_html=True)
colA, colB = st.columns([0.6, 0.4])
with colA:
    selected_insights = st.multiselect("Choose insights", options=["RCA","CS General Audit","VoC"], default=["RCA","CS General Audit","VoC"])
    user_query = st.text_area("Optional: add a specific question or instruction", placeholder="E.g., 'In RCA, check if agent confirmed travel dates before booking.'", height=80)

with colB:
    st.caption("Run context")
    st.metric("Files indexed", len(st.session_state["records"]))
    st.metric("Segments in index", sum(len(r.get("segments", [])) for r in st.session_state["records"]))
    st.metric("Files selected", len(st.session_state["selected_files"]))

# CS General Audit — Rubric (full width below)
if "CS General Audit" in selected_insights:
    st.markdown("#### CS General Audit — Rubric")
    if st.session_state["cs_rubric_df"] is None:
        st.session_state["cs_rubric_df"] = _load_default_cs_rubric()

    up = st.file_uploader("Upload rubric CSV (parameter, weight, max_score)", type=["csv"], key="cs_rubric_upload")
    if up is not None:
        try:
            df_up = _validate_rubric(pd.read_csv(up))
            st.session_state["cs_rubric_df"] = df_up
            st.success("Rubric CSV loaded.")
        except Exception as e:
            st.error(f"Invalid rubric CSV: {e}")

    edited = st.data_editor(
        st.session_state["cs_rubric_df"],
        use_container_width=True, num_rows="dynamic", key="cs_rubric_editor",
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

        c_norm1, c_norm2 = st.columns([0.48, 0.52])
        with c_norm1:
            st.dataframe(norm[["parameter","norm_w","max_score"]], use_container_width=True, hide_index=True)
        with c_norm2:
            if _HAS_PLOTLY:
                figw = px.pie(norm, names="parameter", values="norm_w", title="Weight distribution (normalized)")
                st.plotly_chart(figw, use_container_width=True)
            else:
                st.info("Plotly not installed; showing table instead of chart.")

        st.download_button("⬇️ Download current rubric CSV", data=edited.to_csv(index=False).encode("utf-8"),
                           file_name="cs_general_audit_rubric.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Rubric error: {e}")

st.write('</div>', unsafe_allow_html=True)

# === 4) Analyze ===
can_analyze = bool(st.session_state["records"]) and bool(selected_insights)
if st.button("Analyze", disabled=not can_analyze, use_container_width=True):
    with st.spinner("Analyzing…"):
        results = {}
        allowed = set(st.session_state["selected_files"]) if st.session_state["selected_files"] else None

        for task in selected_insights:
            if task == "CS General Audit":
                if not allowed:
                    allowed = set([r["filename"] for r in st.session_state["records"]])
                top_segments = _retrieve_round_robin(
                    user_query or "Customer Service general audit rubric scoring.",
                    _PRESET_HINTS[task],
                    per_file_k=6,  # default; advanced sliders are in expander
                    allowed_filenames=allowed
                )
            else:
                top_segments = _retrieve_general(
                    user_query or task,
                    _PRESET_HINTS[task],
                    8,
                    allowed_filenames=allowed
                )
            results[task] = {"segments": top_segments, "answer": _ask_gemini(task, user_query, top_segments)}
        st.session_state["last_results"] = results
        st.success("Analysis complete.")

# === 5) Results ===
def _render_rca(ans: Dict[str, Any]):
    st.markdown("**Summary**"); st.write(ans.get("summary","—"))
    st.markdown("**What went wrong**"); [st.write(f"- {x}") for x in (ans.get("what_went_wrong") or [])]
    st.markdown("**Immediate fixes**"); [st.write(f"- {x}") for x in (ans.get("immediate_fixes") or [])]
    st.markdown("**Preventive actions**"); [st.write(f"- {x}") for x in (ans.get("preventive_actions") or [])]

def _render_voc(ans: Dict[str, Any]):
    st.markdown("**Summary**"); st.write(ans.get("summary","—"))
    st.markdown("**Themes**")
    for t in ans.get("themes", []) or []:
        st.write(f"- {t.get('theme','—')} (count: {t.get('count',0)}, sentiment: {t.get('sentiment','—')})")

def _build_evidence(task: str, ans: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    def add(file,s,e,q,extra=None):
        row={"file":file,"start_s":s,"end_s":e,"quote":q}
        if extra: row.update(extra)
        rows.append(row)
    try:
        if task=="RCA":
            for ev in ans.get("evidence",[]) or []: add(ev.get("file",""),ev.get("start_s",0),ev.get("end_s",0),ev.get("quote",""))
        elif task=="CS General Audit":
            for c in ans.get("per_call",[]) or []:
                for p in c.get("parameters",[]) or []:
                    for ev in p.get("evidence",[]) or []:
                        add(ev.get("file",""),ev.get("start_s",0),ev.get("end_s",0),ev.get("quote",""),
                            {"parameter":p.get("parameter",""),"file_call":c.get("file","")})
        elif task=="VoC":
            for t in ans.get("themes",[]) or []:
                for q in t.get("quotes",[]) or []:
                    add(q.get("file",""),q.get("start_s",0),q.get("end_s",0),q.get("quote",""),
                        {"theme":t.get("theme",""),"sentiment":t.get("sentiment",""),"count":t.get("count",0)})
    except Exception: pass
    df = pd.DataFrame(rows)
    if not df.empty:
        df["t"]=df["start_s"].apply(_mmss)
        cols = [c for c in ["file","t","start_s","end_s","quote","parameter","file_call","theme","sentiment","count"] if c in df.columns]
        df = df[cols + [c for c in df.columns if c not in cols]]
    return df

def _render_cs_audit(ans: Dict[str, Any]):
    pc = ans.get("per_call", []) or []
    # KPIs
    all_scores = [c.get("overall_weighted_score",0) for c in pc if isinstance(c,dict)]
    avg = float(np.mean(all_scores)) if all_scores else 0.0
    k1, k2, k3 = st.columns(3)
    with k1: st.markdown('<div class="kpi"><div class="muted">Average Overall Score</div><div class="big">{:.2f}</div></div>'.format(avg), unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="kpi"><div class="muted">Calls Scored</div><div class="big">{len(pc)}</div></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="kpi"><div class="muted">Parameters</div><div class="big">{len(st.session_state.get("cs_rubric_df") or [])}</div></div>', unsafe_allow_html=True)

    # Tables + cross-call aggregation
    param_rows=[]
    for c in pc:
        fname=c.get("file","(file)")
        st.markdown(f"**📞 {fname} — Overall:** {c.get('overall_weighted_score',0)}")
        params=c.get("parameters",[]) or []
        dfp=pd.DataFrame(params)
        if not dfp.empty:
            for rrow in params:
                param_rows.append({"file":fname,"parameter":rrow.get("parameter",""),"score":rrow.get("score",0),"max_score":rrow.get("max_score",1)})
            st.dataframe(dfp[["parameter","score","max_score","justification"]], use_container_width=True, hide_index=True)

    st.markdown("### Dashboard")
    if param_rows:
        dfa=pd.DataFrame(param_rows)
        dfa["pct"]=(dfa["score"]/dfa["max_score"]).replace([np.inf,-np.inf],0.0)*100.0
        perf=dfa.groupby("parameter",as_index=False)["pct"].mean().sort_values("pct",ascending=True)
        if _HAS_PLOTLY and not perf.empty:
            figp=px.bar(perf,x="pct",y="parameter",orientation="h",title="Average Parameter Performance (% of max)")
            st.plotly_chart(figp, use_container_width=True)
        else:
            st.dataframe(perf, use_container_width=True)
    else:
        st.info("No parameter data to chart yet.")

    if ans.get("coaching_opportunities"):
        st.markdown("### Coaching Opportunities")
        for x in ans["coaching_opportunities"]:
            st.write(f"- {x}")

# Results tabs
results = st.session_state.get("last_results", {})
if results:
    st.markdown("## 4) Results & Evidence")
    tabs = st.tabs(list(results.keys()))
    for (task, tab) in zip(results.keys(), tabs):
        with tab:
            ans = results[task]["answer"] or {}
            if task=="RCA": _render_rca(ans)
            elif task=="CS General Audit": _render_cs_audit(ans)
            elif task=="VoC": _render_voc(ans)

            st.markdown("#### Evidence")
            df_ev = _build_evidence(task, ans)
            if df_ev.empty:
                st.write("—")
            else:
                st.dataframe(df_ev, use_container_width=True, hide_index=True)
                st.download_button("Download evidence CSV", data=df_ev.to_csv(index=False).encode("utf-8"),
                                   file_name=f"evidence_{task.replace(' ','_').lower()}.csv", mime="text/csv")

            st.divider()
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

# === 6) Player ===
if st.session_state["records"]:
    st.markdown("## 5) Audio Player")
    st.write('<div class="card">', unsafe_allow_html=True)
    sel = st.selectbox("Choose a file", options=[r["filename"] for r in st.session_state["records"]])
    rec = next((r for r in st.session_state["records"] if r["filename"]==sel), None)
    if rec: st.audio(rec["audio_bytes"], format="audio/mpeg")
    st.caption("Use timestamps from Evidence (mm:ss) to seek.")
    st.write('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("For QA pilot use. No PII redaction. © Your Company")
