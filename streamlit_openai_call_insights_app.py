# main_app.py
import pandas as pd
import streamlit as st
from typing import Dict, List

# Your modules
import audio_processing
import ai_engine
import exports  # <-- the helper we added for CSV building

# =============================================================================
# PAGE CONFIG & LIGHT POLISH
# =============================================================================

st.set_page_config(
    page_title="Call Insights Desk",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional tiny CSS polish (safe to remove)
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
[data-testid="stMetricValue"] { font-size: 22px; }
</style>
""", unsafe_allow_html=True)

BADGE_COLORS = {
    "ready": "#16a34a",      # green
    "warning": "#f59e0b",    # amber
    "failed": "#ef4444",     # red
    "info": "#64748b"        # slate
}

def badge(label: str, kind: str = "info") -> str:
    color = BADGE_COLORS.get(kind, "#64748b")
    return f"<span style='background:{color};color:white;padding:3px 8px;border-radius:10px;font-size:12px'>{label}</span>"

def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div style="
            border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;
            background: white;">
            <div style="font-size:12px;color:#6b7280">{label}</div>
            <div style="font-size:20px;font-weight:600;margin-top:4px;color:#0f172a">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# SESSION STATE
# =============================================================================

def initialize_session_state():
    if "files_metadata" not in st.session_state:
        st.session_state.files_metadata = {}  # filename -> {size_mb, format, ...}
    if "transcription_results" not in st.session_state:
        st.session_state.transcription_results = {}  # filename -> transcript bundle
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}  # filename -> {ab_result|legacy_result, transcript_data, ...}
    if "run_history" not in st.session_state:
        st.session_state.run_history = []
    if "processing_capabilities" not in st.session_state:
        st.session_state.processing_capabilities = audio_processing.get_processing_capabilities()

initialize_session_state()

# =============================================================================
# UTILS
# =============================================================================

def validate_transcript_quality(transcript_data: Dict) -> Dict:
    v = {"valid": True, "warnings": [], "errors": []}
    if transcript_data.get("status") != "success":
        v["valid"] = False
        v["errors"].append(f"Transcription failed: {transcript_data.get('error', 'Unknown error')}")
        return v
    transcript_text = transcript_data.get("english_transcript", "")
    if not transcript_text or len(transcript_text.strip()) < 10:
        v["valid"] = False
        v["errors"].append("Transcript is empty or too short for meaningful analysis")
        return v
    if len(transcript_text.split()) < 20:
        v["warnings"].append("Very short transcript - analysis may be limited")
    if not transcript_data.get("segments"):
        v["warnings"].append("No segment data available - speaker analysis may be limited")
    return v

def get_analysis_ready_files() -> List[str]:
    ready = []
    for filename, transcript_data in st.session_state.transcription_results.items():
        if validate_transcript_quality(transcript_data)["valid"]:
            ready.append(filename)
    return ready

def cleanup_failed_uploads():
    failed = [fn for fn, td in st.session_state.transcription_results.items() if td.get("status") != "success"]
    for fn in failed:
        st.session_state.files_metadata.pop(fn, None)
        st.session_state.transcription_results.pop(fn, None)
    if failed:
        st.warning(f"Removed {len(failed)} failed uploads: {', '.join(failed)}")

# ---- Dashboard helpers made robust to handle both legacy and A/B result shapes

def _create_summary_df(analysis_results: dict) -> pd.DataFrame:
    summary_data = []

    def compute_avg_from_ab(ab_result: dict, variant: str = "A") -> float:
        # Try to average parameter scores from stages.parameter_scores[variant]
        try:
            stages = {s.get("name"): s for s in ab_result.get("stages", [])}
            params = (stages.get("parameter_scores", {}) or {}).get(variant, {}) or {}
            scores = [d.get("score", 0) for d in params.values() if isinstance(d, dict) and "score" in d]
            return sum(scores) / len(scores) if scores else 0.0
        except Exception:
            return 0.0

    for file_name, bundle in analysis_results.items():
        # Prefer A/B structure if available
        ab = bundle.get("ab_result")
        triage, outcome, avg_score = {}, {}, 0.0

        if isinstance(ab, dict) and ("stages" in ab or "overall" in ab):
            stages = {s.get("name"): s for s in ab.get("stages", [])}
            triage_map = {x.get("label"): x.get("result", {}) for x in stages.get("triage", {}).get("results", [])}
            outcome_map = {x.get("label"): x.get("result", {}) for x in stages.get("business_outcome", {}).get("results", [])}
            triage = triage_map.get("A", {}) or triage_map.get("B", {}) or {}
            outcome = outcome_map.get("A", {}) or outcome_map.get("B", {}) or {}
            avg_score = compute_avg_from_ab(ab, "A") or compute_avg_from_ab(ab, "B")
        else:
            # Legacy shape support (triage/outcome at top-level + scores list)
            results = bundle
            triage = results.get('triage', {}) if isinstance(results, dict) else {}
            outcome = results.get('outcome', {}) if isinstance(results, dict) else {}
            scores = [s['details'].get('score', 0)
                      for s in results.get('scores', []) if 'details' in s and 'score' in s['details']]
            avg_score = sum(scores) / len(scores) if scores else 0.0

        summary_data.append({
            "File Name": file_name,
            "Category": triage.get('category', 'N/A'),
            "Call Purpose": triage.get('purpose', 'N/A'),
            "Outcome": (outcome.get('business_outcome') if isinstance(outcome, dict) else outcome) or "N/A",
            "Average Score": f"{avg_score:.1f}" if avg_score > 0 else "N/A",
            "Risk Identified": bool(outcome.get('risk_identified')) if isinstance(outcome, dict) else False
        })

    return pd.DataFrame(summary_data)

def _create_detailed_df(analysis_results: dict) -> pd.DataFrame:
    detailed = []

    # A/B structure
    for file_name, bundle in analysis_results.items():
        ab = bundle.get("ab_result")
        if isinstance(ab, dict) and "stages" in ab:
            stages = {s.get("name"): s for s in ab.get("stages", [])}
            param_scores = (stages.get("parameter_scores", {}) or {})
            for variant in ("A", "B"):
                pset = param_scores.get(variant, {}) or {}
                for pname, details in pset.items():
                    if not isinstance(details, dict) or "score" not in details or "error" in details:
                        continue
                    detailed.append({
                        "File Name": file_name,
                        "Variant": variant,
                        "Parameter": pname,
                        "Score": details.get("score"),
                        "Confidence": details.get("confidence", "N/A"),
                        "Justification": details.get("justification"),
                        "Primary Evidence": details.get("primary_evidence"),
                        "Coaching Opportunity": details.get("coaching_opportunity")
                    })
            continue

        # Legacy shape fallback
        results = bundle
        if isinstance(results, dict):
            for score_item in results.get('scores', []):
                details = score_item.get('details', {})
                if 'score' in details:
                    detailed.append({
                        "File Name": file_name,
                        "Variant": "A",  # default tag
                        "Parameter": score_item.get('parameter'),
                        "Score": details.get('score'),
                        "Confidence": details.get('confidence', 'N/A'),
                        "Justification": details.get('justification'),
                        "Primary Evidence": details.get('primary_evidence'),
                        "Coaching Opportunity": details.get('coaching_opportunity')
                    })

    return pd.DataFrame(detailed)

# =============================================================================
# UI PAGES
# =============================================================================

def page_call_analysis():
    st.title("🔍 Call Analysis")
    st.caption("Upload calls, review the queue, configure analysis, then export results (A/B).")

    # Sidebar quick system status
    caps = st.session_state.processing_capabilities
    st.sidebar.markdown("**System Status**")
    st.sidebar.markdown(f"Transcription: {'✅' if caps['transcription_available'] else '❌'}")
    st.sidebar.markdown(f"Speaker ID: {'✅' if caps['speaker_labeling_available'] else '❌'}")
    st.sidebar.markdown("---")

    # Summary ribbon
    total_files = len(st.session_state.files_metadata)
    ready_files = len(get_analysis_ready_files())
    analyzed_files = len(st.session_state.analysis_results)

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Uploaded", str(total_files))
    with c2: metric_card("Ready for Analysis", str(ready_files))
    with c3: metric_card("Analyzed (this session)", str(analyzed_files))

    st.markdown("---")

    tab_upload, tab_queue, tab_analyze, tab_export = st.tabs(["Upload", "Queue", "Analyze", "Exports"])

    # -------------------------
    # Tab 1: Upload
    # -------------------------
    with tab_upload:
        st.subheader("1) Upload Audio")
        files = st.file_uploader(
            "Drop audio files here (mp3, wav, m4a, ogg, flac, webm)",
            type=["mp3", "wav", "m4a", "ogg", "flac", "webm"],
            accept_multiple_files=True
        )

        if files:
            st.info(f"Processing {len(files)} new file(s)…")
            for file in files:
                if file.name in st.session_state.files_metadata:
                    st.warning(f"Skipping {file.name} (already in queue)")
                    continue

                file_content = file.getvalue()
                validation = audio_processing.validate_file_before_upload(file.name, file_content)
                if not validation["valid"]:
                    st.error(f"{file.name}: {', '.join(validation['errors'])}")
                    continue
                if validation["warnings"]:
                    st.warning(f"{file.name}: {', '.join(validation['warnings'])}")

                st.session_state.files_metadata[file.name] = {**validation["file_info"]}

                with st.expander(f"🔊 Preview: {file.name}", expanded=False):
                    st.audio(file_content)
                    st.caption(f"Size: {validation['file_info'].get('size_mb',0):.1f} MB • Format: {validation['file_info'].get('format','?')}")

                # Process with progress
                with st.status(f"Transcribing {file.name}…", expanded=True) as status:
                    t0 = audio_processing.process_audio_with_progress(file.name, file_content)
                    st.session_state.transcription_results[file.name] = t0
                    if t0.get("status") == "success":
                        status.update(label=f"✅ {file.name} processed", state="complete")
                    else:
                        status.update(label=f"❌ {file.name} failed", state="error")
                        st.error(t0.get("error", "Unknown error"))

        st.success("Upload step complete. Switch to the **Queue** tab to review.")

    # -------------------------
    # Tab 2: Queue
    # -------------------------
    with tab_queue:
        st.subheader("2) Queue & Status")
        if not st.session_state.files_metadata:
            st.info("No files in queue yet. Upload in the previous tab.")
        else:
            data = []
            for filename, meta in st.session_state.files_metadata.items():
                tr = st.session_state.transcription_results.get(filename, {})
                check = validate_transcript_quality(tr)
                if check["valid"]:
                    status_html = badge("Ready", "ready") if not check["warnings"] else badge("Ready (with warnings)", "warning")
                else:
                    status_html = badge("Failed", "failed")
                dur = tr.get("duration", 0)
                lang = tr.get("detected_language", "Unknown")
                spk = len(set(s.get("speaker","?") for s in tr.get("segments", []) or []))
                data.append([filename, f"{meta.get('size_mb',0):.1f} MB", f"{dur:.1f}s" if dur else "—", lang, spk, status_html])

            df = pd.DataFrame(data, columns=["File","Size","Duration","Language","Speakers","#Status"])
            st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

            cA, cB = st.columns(2)
            with cA:
                if st.button("🗑️ Clean up failed files"):
                    cleanup_failed_uploads()
                    st.rerun()
            with cB:
                if st.button("🧹 Clear ALL files"):
                    st.session_state.files_metadata.clear()
                    st.session_state.transcription_results.clear()
                    st.session_state.analysis_results.clear()
                    st.success("Cleared all files.")
                    st.rerun()

    # -------------------------
    # Tab 3: Analyze
    # -------------------------
    with tab_analyze:
        st.subheader("3) Configure & Run Analysis")

        ready = get_analysis_ready_files()
        if not ready:
            st.info("No ready files. Please process files in the previous tabs.")
        else:
            left, right = st.columns([1,1])

            with left:
                # Depth and custom rubric (fallback to dicts on ai_engine)
                depth_options = list(getattr(ai_engine, "ANALYSIS_PARAMETERS", {"Standard Analysis": []}).keys())
                default_idx = depth_options.index("Standard Analysis") if "Standard Analysis" in depth_options else 0
                selected_depth = st.selectbox("Analysis Depth", options=depth_options, index=default_idx)

                rubric_options = list(getattr(ai_engine, "CUSTOM_PARAMETERS", {}).keys())
                selected_rubric = st.selectbox("Custom Rubric (optional)", options=["None"] + rubric_options)
                selected_rubric = None if selected_rubric == "None" else selected_rubric

                selected_files = st.multiselect("Files to analyze", options=ready, default=ready)

                run_clicked = st.button("🚀 Analyze Selected Files", type="primary", use_container_width=True)

            with right:
                st.markdown("**Transcript Preview**")
                if ready:
                    preview_file = st.selectbox("Preview:", options=ready, key="preview_file_select")
                    tdata = st.session_state.transcription_results.get(preview_file, {})
                    segs = tdata.get("segments", []) or []
                    if segs:
                        for s in segs[:20]:
                            st.markdown(f"**{s.get('speaker','?')}:** {s.get('text','')}")
                        if len(segs) > 20:
                            st.info("Showing first 20 segments…")
                    else:
                        st.write(tdata.get("english_transcript", "No transcript available."))

            if run_clicked:
                if not selected_files:
                    st.error("Please select at least one file.")
                else:
                    with st.spinner("Running comprehensive analysis…"):
                        all_results = {}
                        prog = st.progress(0.0)
                        for i, fname in enumerate(selected_files):
                            st.write(f"Analyzing **{fname}** …")
                            tdata = st.session_state.transcription_results[fname]
                            text = tdata.get("english_transcript", "")

                            try:
                                result = ai_engine.run_comprehensive_analysis(
                                    transcript=text,
                                    depth=selected_depth,
                                    custom_rubric=selected_rubric
                                )
                                # Detect A/B-shaped result
                                ab_result = result if (isinstance(result, dict) and ("stages" in result or "overall" in result)) else None

                                all_results[fname] = {
                                    "ab_result": ab_result,
                                    "legacy_result": None if ab_result else result,
                                    "transcript_data": tdata,
                                    "analysis_metadata": {
                                        "depth": selected_depth,
                                        "custom_rubric": selected_rubric,
                                        "timestamp": pd.Timestamp.now().isoformat()
                                    }
                                }
                            except Exception as e:
                                all_results[fname] = {"error": str(e), "transcript_data": tdata}
                                st.error(f"{fname}: {e}")

                            prog.progress((i+1)/len(selected_files))

                        st.session_state.analysis_results.update(all_results)
                        st.session_state.run_history.append({
                            "run_id": f"Analysis - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            "depth": selected_depth,
                            "custom_rubric": selected_rubric,
                            "files_analyzed": len(selected_files),
                            "files": selected_files.copy(),
                            "results": all_results.copy()
                        })
                        st.success("✅ Analysis complete!")

    # -------------------------
    # Tab 4: Exports (A/B CSV)
    # -------------------------
    with tab_export:
        st.subheader("4) Exports")
        st.caption("Download per-variant (A/B) audit CSV across the last analysis run’s files.")

        if not st.session_state.analysis_results:
            st.info("No analysis data yet. Run analysis in the previous tab.")
        else:
            dfs_A, dfs_B = [], []
            # Build from any entries that have ab_result
            file_keys = [k for k,v in st.session_state.analysis_results.items() if v.get("ab_result")]
            if not file_keys:
                st.warning("No A/B-structured results found to export. Ensure your AI engine returns A/B variants.")
            for fname in file_keys:
                bundle = st.session_state.analysis_results[fname]
                ab = bundle.get("ab_result")
                tdata = bundle.get("transcript_data") or {}
                try:
                    dfA = exports.build_variant_dataframe(fname, ab, tdata, "A")
                    if not dfA.empty: dfs_A.append(dfA)
                except Exception as e:
                    st.warning(f"CSV build failed for {fname} (A): {e}")
                try:
                    dfB = exports.build_variant_dataframe(fname, ab, tdata, "B")
                    if not dfB.empty: dfs_B.append(dfB)
                except Exception as e:
                    st.warning(f"CSV build failed for {fname} (B): {e}")

            run_ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            csvA_bytes = pd.concat(dfs_A).to_csv(index=False).encode("utf-8") if dfs_A else b""
            csvB_bytes = pd.concat(dfs_B).to_csv(index=False).encode("utf-8") if dfs_B else b""

            cA, cB = st.columns(2)
            with cA:
                st.download_button(
                    "⬇️ Download Model A CSV",
                    data=csvA_bytes,
                    file_name=f"results_A_{run_ts}.csv",
                    mime="text/csv",
                    disabled=(not csvA_bytes),
                    use_container_width=True
                )
            with cB:
                st.download_button(
                    "⬇️ Download Model B CSV",
                    data=csvB_bytes,
                    file_name=f"results_B_{run_ts}.csv",
                    mime="text/csv",
                    disabled=(not csvB_bytes),
                    use_container_width=True
                )

# =============================================================================
# OTHER PAGES (kept minimal but functional)
# =============================================================================

def page_dashboard():
    st.title("📊 Dashboard & Results")

    if not st.session_state.analysis_results:
        st.info("Please run an analysis on the 'Call Analysis' page to see results here.")
        return

    st.subheader("Latest Analysis Summary")
    summary_df = _create_summary_df(st.session_state.analysis_results)

    tab1, tab2, tab3 = st.tabs(["Summary Table", "Visuals", "Detailed View"])

    with tab1:
        st.dataframe(summary_df, use_container_width=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            try:
                avg_scores = [float(x) for x in summary_df['Average Score'].replace('N/A', '0')]
                valid = [x for x in avg_scores if x > 0]
                overall_avg = sum(valid) / len(valid) if valid else 0
                st.metric("Overall Average Score", f"{overall_avg:.1f}")
            except Exception:
                st.metric("Overall Average Score", "N/A")
        with col2:
            risk_count = sum(1 for risk in summary_df['Risk Identified'] if risk)
            st.metric("Calls with Risk", risk_count)
        with col3:
            categories = summary_df['Category'].value_counts()
            st.metric("Most Common Category", categories.index[0] if not categories.empty else "N/A")
        with col4:
            st.metric("Total Calls Analyzed", len(summary_df))

    with tab2:
        df_chart = summary_df.copy()
        df_chart['Average Score'] = pd.to_numeric(df_chart['Average Score'].replace('N/A', '0'))
        df_chart = df_chart[df_chart['Average Score'] > 0]
        if not df_chart.empty:
            st.markdown("**Average Scores by File**")
            st.bar_chart(df_chart.set_index('File Name')['Average Score'])
            st.markdown("**Call Categories Distribution**")
            st.bar_chart(summary_df['Category'].value_counts())
        else:
            st.info("No valid scores available for visualization.")

    with tab3:
        for file_name, bundle in st.session_state.analysis_results.items():
            with st.expander(f"📞 {file_name}"):
                tdata = bundle.get("transcript_data", {})
                segs = tdata.get("segments", []) or []
                st.markdown("**🗣️ Conversation Flow**")
                if segs:
                    for s in segs[:200]:  # cap
                        st.markdown(f"**{s.get('speaker','?')}:** {s.get('text','')}")
                    if len(segs) > 200:
                        st.info("Showing first 200 segments…")
                else:
                    st.write(tdata.get("english_transcript", "No transcript available."))

                st.markdown("**📋 Analysis (raw)**")
                if bundle.get("ab_result"):
                    st.json(bundle["ab_result"])
                elif bundle.get("legacy_result"):
                    st.json(bundle["legacy_result"])
                else:
                    st.warning("No analysis payload found for this call.")

def page_rubric_editor():
    st.title("📝 Rubric Editor")
    st.info("Rubric creation and editing UI (coming soon).")
    # You can list available custom rubrics for now:
    rubrics = list(getattr(ai_engine, "CUSTOM_PARAMETERS", {}).keys())
    if rubrics:
        st.markdown("**Available Custom Rubrics:**")
        for r in rubrics:
            st.write(f"- {r}")
    else:
        st.write("No custom rubrics defined yet.")

def page_run_history():
    st.title("🗂️ Run History")
    if not st.session_state.run_history:
        st.info("No analysis runs recorded yet.")
        return

    for i, run in enumerate(reversed(st.session_state.run_history)):
        run_number = len(st.session_state.run_history) - i
        with st.expander(f"Run #{run_number}: {run['run_id']} — {run.get('depth','Unknown')} ({run['files_analyzed']} files)"):
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Files Analyzed", run['files_analyzed'])
            with c2: st.metric("Analysis Depth", run.get('depth', 'Unknown'))
            with c3: st.metric("Rubric Used", run.get('custom_rubric') or 'Standard')
            files_analyzed = run.get('files', [])
            if files_analyzed:
                st.markdown("**Files:** " + ", ".join(files_analyzed))
            run_results = run.get('results', {})
            if run_results:
                st.markdown("**Summary of this run:**")
                st.dataframe(_create_summary_df(run_results), use_container_width=True)
                if st.button(f"🔄 Restore Run #{run_number} as Current", key=f"restore_{i}"):
                    st.session_state.analysis_results = run_results.copy()
                    st.success(f"Run #{run_number} restored!")
                    st.rerun()
            else:
                st.warning("No results data available in this run.")

# =============================================================================
# ROUTER
# =============================================================================

def main():
    st.sidebar.title("Call Insights Desk")
    pages = {
        "Call Analysis": page_call_analysis,
        "Dashboard": page_dashboard,
        "Rubric Editor": page_rubric_editor,
        "Run History": page_run_history,
    }
    page_name = st.sidebar.radio("Navigation", list(pages.keys()))
    try:
        pages[page_name]()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
