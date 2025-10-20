# main_app.py
import pandas as pd
import streamlit as st
from typing import Dict, List
import html

# Your modules
import audio_whisper_gemini
import ai_engine
import exports

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title="Call Insights Desk",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
[data-testid="stMetricValue"] { font-size: 22px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE & HELPERS
# =============================================================================

def initialize_session_state():
    if "files_metadata" not in st.session_state:
        st.session_state.files_metadata = {}
    if "transcription_results" not in st.session_state:
        st.session_state.transcription_results = {}
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}
    if "run_history" not in st.session_state:
        st.session_state.run_history = []

initialize_session_state()

def metric_card(label: str, value: str):
    st.markdown(f"""
    <div style="border:1px solid #e5e7eb; border-radius:12px; padding:14px 16px; background:white;">
        <div style="font-size:12px; color:#6b7280">{label}</div>
        <div style="font-size:20px; font-weight:600; margin-top:4px; color:#0f172a">{value}</div>
    </div>""", unsafe_allow_html=True)

def get_analysis_ready_files() -> List[str]:
    ready_files = set()
    for key, result in st.session_state.transcription_results.items():
        if result.get("status") == "success":
            filename = key.split("::")[0]
            ready_files.add(filename)
    return sorted(list(ready_files))

def _create_summary_df(analysis_results: dict) -> pd.DataFrame:
    # This function creates a summary DataFrame from analysis results
    # For brevity, its full logic is omitted here, but your original version was good.
    # You should keep your original working function here.
    summary_data = []
    for file_name, bundle in analysis_results.items():
        # A simplified placeholder logic
        summary_data.append({
            "File Name": file_name, "Category": "N/A", "Call Purpose": "N/A",
            "Outcome": "N/A", "Average Score": "N/A", "Risk Identified": False
        })
    return pd.DataFrame(summary_data)

# =============================================================================
# UI PAGES
# =============================================================================

def page_call_analysis(selected_engines: List[str]):
    st.title("🔍 Call Analysis & Benchmarking")
    st.caption("Upload calls, process them with one or more engines, and analyze the results.")

    total_files = len(st.session_state.files_metadata)
    ready_files = len(get_analysis_ready_files())
    analyzed_files = len(st.session_state.analysis_results)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Unique Files Uploaded", str(total_files))
    with c2: metric_card("Files Ready for Analysis", str(ready_files))
    with c3: metric_card("Files Analyzed", str(analyzed_files))
    st.markdown("---")

    upload_tab, analyze_tab = st.tabs(["Upload & Process", "Analyze Results"])

    with upload_tab:
        st.subheader("1) Upload Audio")
        files = st.file_uploader("Drag & drop audio files here", type=audio_whisper_gemini.SUPPORTED_FORMATS, accept_multiple_files=True)

        if files:
            if not selected_engines:
                st.error("Please select at least one engine from the sidebar to process files.")
            else:
                for file in files:
                    if file.name in st.session_state.files_metadata:
                        continue
                    content = file.getvalue()
                    validation = audio_whisper_gemini._validate_audio_file(file.name, content)
                    if not validation["valid"]:
                        st.error(f"{file.name}: {', '.join(validation['errors'])}")
                        continue
                    st.session_state.files_metadata[file.name] = {**validation["file_info"]}
                    st.write(f"---")
                    st.write(f"Processing **{file.name}** with {len(selected_engines)} engine(s)...")
                    
                    for engine in selected_engines:
                        unique_key = f"{file.name}::{engine}"
                        if unique_key in st.session_state.transcription_results:
                            st.info(f"Skipping {engine} for {file.name} (already processed).")
                            continue
                        with st.spinner(f"Running engine: {engine}..."):
                            res = audio_whisper_gemini.process_audio(file.name, content)
                            st.session_state.transcription_results[unique_key] = res
                        if res.get("status") == "success":
                            st.success(f"✅ **{engine}:** Processed successfully.")
                        else:
                            st.error(f"❌ **{engine}:** Failed. {res.get('error_message', 'Unknown error')}")
        
        if st.session_state.files_metadata:
            st.markdown("---")
            st.subheader("Transcription Queue & Results")
            for filename in sorted(st.session_state.files_metadata.keys()):
                with st.expander(f"📄 **{filename}**"):
                    engine_results = {k.split("::")[1]: v for k, v in st.session_state.transcription_results.items() if k.startswith(filename)}
                    if not engine_results:
                        st.info("This file is in the queue but has not been processed.")
                        continue
                    cols = st.columns(len(engine_results))
                    for i, (engine_name, result) in enumerate(engine_results.items()):
                        with cols[i]:
                            st.markdown(f"##### Engine: `{engine_name}`")
                            if result.get("status") == "success":
                                st.success("Success")
                                st.text(f"Language: {result.get('detected_language', 'N/A')}")
                                st.text(f"Duration: {result.get('duration', 0):.1f}s")

                                # Transcript snippet
                                snippet_src = result.get("english_transcript") or result.get("original_transcript", "")
                                snippet = snippet_src[:200] + ("..." if len(snippet_src) > 200 else "")
                                st.text_area("Transcript Snippet", snippet, height=150, key=f"snippet_{filename}_{engine_name}")

                                # ---------------- Speakers (Diarization) ----------------
                                segs = result.get("segments") or []
                                with st.expander("🗣️ Speakers (Diarization)"):
                                    if segs:
                                        df_rows = [{
                                            "Start (s)": round(float(s.get("start", 0.0)), 2),
                                            "End (s)": round(float(s.get("end", 0.0)), 2),
                                            "Speaker": s.get("speaker", ""),
                                            "Text": (s.get("text", "")[:160] + ("…" if len(s.get("text", "")) > 160 else ""))
                                        } for s in segs]
                                        st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

                                        # quick metrics (if available)
                                        m = result.get("diarization_metrics") or {}
                                        if m:
                                            st.caption(
                                                f"Turns: {m.get('turns',0)} • "
                                                f"Interruptions: {m.get('interruptions',0)} • "
                                                f"Avg silence: {m.get('avg_silence',0.0):.2f}s"
                                            )
                                            # talk share progress bars
                                            for spk, ratio in (m.get("talk_ratio") or {}).items():
                                                # clamp 0..1 for safety
                                                ratio = max(0.0, min(1.0, float(ratio)))
                                                st.progress(ratio, text=f"{spk}: {ratio:.0%}")
                                    else:
                                        st.info("No diarization segments returned.")
                            else:
                                st.error(f"Error: {result.get('error', 'Unknown error')}")

        st.markdown("---")
        if st.button("🧹 Clear ALL Files & Results"):
            st.session_state.files_metadata.clear()
            st.session_state.transcription_results.clear()
            st.session_state.analysis_results.clear()
            st.rerun()

    with analyze_tab:
        st.subheader("2) Configure & Run Analysis")
        
        ready_files = get_analysis_ready_files()
        
        if not ready_files:
            st.warning("No files are ready for analysis. Please upload and process audio files first.")
            return
        
        # File selection
        selected_files = st.multiselect(
            "Select files to analyze",
            options=ready_files,
            default=ready_files,
            help="Choose which processed files to include in the analysis"
        )
        
        if not selected_files:
            st.info("Please select at least one file to analyze.")
            return
        
        st.markdown("---")
        
        # Analysis options
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Analysis Options**")
            include_sentiment = st.checkbox("Sentiment Analysis", value=True)
            include_summary = st.checkbox("Call Summary", value=True)
            include_key_points = st.checkbox("Key Points Extraction", value=True)
        
        with col2:
            st.markdown("**Call Center Metrics**")
            analyze_compliance = st.checkbox("Compliance Check", value=True)
            analyze_customer_satisfaction = st.checkbox("Customer Satisfaction", value=True)
            detect_issues = st.checkbox("Issue Detection", value=True)
        
        st.markdown("---")
        
        # Run Analysis button
        if st.button("🚀 Run AI Analysis", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_files = len(selected_files)
            
            for idx, filename in enumerate(selected_files):
                status_text.text(f"Analyzing {filename}...")
                progress_bar.progress((idx) / total_files)
                
                # Get the transcription result for this file
                file_results = {k: v for k, v in st.session_state.transcription_results.items() 
                               if k.startswith(filename) and v.get("status") == "success"}
                
                if not file_results:
                    continue
                
                # Use the first successful result (or combine if multiple engines)
                result = list(file_results.values())[0]
                
                # Prepare transcript for analysis
                transcript = result.get("english_transcript") or result.get("original_transcript", "")
                
                # Add speaker labels to transcript if available
                segments = result.get("segments", [])
                if segments:
                    formatted_transcript = "\n".join([
                        f"[{seg.get('speaker', 'UNKNOWN')}]: {seg.get('text', '')}"
                        for seg in segments
                    ])
                else:
                    formatted_transcript = transcript
                
                try:
                    # Run comprehensive AI analysis
                    analysis_result = ai_engine.run_comprehensive_analysis(
                        transcript=formatted_transcript,
                        depth="Standard Analysis",
                        custom_rubric=None,
                        max_retries=2,
                        admin_view=False
                    )
                    
                    # Enrich with additional data
                    analysis_result["file_name"] = filename
                    analysis_result["detected_language"] = result.get("detected_language", "unknown")
                    analysis_result["duration"] = result.get("duration", 0)
                    
                    st.session_state.analysis_results[filename] = analysis_result
                except Exception as e:
                    st.error(f"Analysis failed for {filename}: {str(e)}")
                    st.session_state.analysis_results[filename] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            st.success(f"✅ Successfully analyzed {len(selected_files)} file(s)")
            st.balloons()
        
        # Display existing analysis results
        if st.session_state.analysis_results:
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            for filename, analysis in st.session_state.analysis_results.items():
                if filename not in selected_files:
                    continue
                
                with st.expander(f"📋 {filename}", expanded=True):
                    if analysis.get("error"):
                        st.error(f"Analysis error: {analysis['error']}")
                        continue
                    
                    # Display metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Language", analysis.get("detected_language", "N/A"))
                    with col2:
                        st.metric("Duration", f"{analysis.get('duration', 0):.1f}s")
                    with col3:
                        st.metric("Run ID", analysis.get("run_id", "N/A"))
                    
                    st.markdown("---")
                    
                    # Display A/B comparison results
                    stages = analysis.get("stages", [])
                    overall = analysis.get("overall", {})
                    
                    # Show overall scores
                    if overall:
                        st.markdown("**📊 Overall Quality Scores (A/B Comparison)**")
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.markdown("**Model A**")
                            if "A" in overall:
                                metrics_a = overall["A"]
                                st.metric("Overall Score", f"{metrics_a.get('overall_score', 0):.1f}/10")
                                st.metric("Risk Level", metrics_a.get("risk_category", "N/A"))
                                st.metric("High Performers", metrics_a.get("count_high_performers", 0))
                                st.metric("Critical Concerns", metrics_a.get("count_critical_concerns", 0))
                        
                        with col_b:
                            st.markdown("**Model B**")
                            if "B" in overall:
                                metrics_b = overall["B"]
                                st.metric("Overall Score", f"{metrics_b.get('overall_score', 0):.1f}/10")
                                st.metric("Risk Level", metrics_b.get("risk_category", "N/A"))
                                st.metric("High Performers", metrics_b.get('count_high_performers', 0))
                                st.metric("Critical Concerns", metrics_b.get("count_critical_concerns", 0))
                    
                    # Display stage results
                    for stage in stages:
                        stage_name = stage.get("name", "Unknown")
                        
                        if stage_name == "triage":
                            st.markdown("**� Call Triage**")
                            results = stage.get("results", {})
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.markdown("**Model A**")
                                if "A" in results:
                                    triage_a = results["A"]
                                    st.write(f"**Category:** {triage_a.get('category', 'N/A')}")
                                    st.write(f"**Purpose:** {triage_a.get('call_purpose', 'N/A')}")
                                    st.write(f"**Sentiment:** {triage_a.get('sentiment', 'N/A')}")
                            
                            with col_b:
                                st.markdown("**Model B**")
                                if "B" in results:
                                    triage_b = results["B"]
                                    st.write(f"**Category:** {triage_b.get('category', 'N/A')}")
                                    st.write(f"**Purpose:** {triage_b.get('call_purpose', 'N/A')}")
                                    st.write(f"**Sentiment:** {triage_b.get('sentiment', 'N/A')}")
                        
                        elif stage_name == "business_outcome":
                            st.markdown("**💼 Business Outcome**")
                            results = stage.get("results", {})
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.markdown("**Model A**")
                                if "A" in results:
                                    outcome_a = results["A"]
                                    st.write(f"**Outcome:** {outcome_a.get('outcome', 'N/A')}")
                                    st.write(f"**Reason:** {outcome_a.get('reason', 'N/A')}")
                            
                            with col_b:
                                st.markdown("**Model B**")
                                if "B" in results:
                                    outcome_b = results["B"]
                                    st.write(f"**Outcome:** {outcome_b.get('outcome', 'N/A')}")
                                    st.write(f"**Reason:** {outcome_b.get('reason', 'N/A')}")
                        
                        elif stage_name == "parameter_scores":
                            st.markdown("**� Parameter Scores**")
                            
                            # Create comparison table
                            param_a = stage.get("A", {})
                            param_b = stage.get("B", {})
                            
                            if param_a or param_b:
                                # Get all parameter names
                                all_params = set(param_a.keys()) | set(param_b.keys())
                                
                                comparison_data = []
                                for param in sorted(all_params):
                                    score_a = param_a.get(param, {}).get("score", "N/A")
                                    score_b = param_b.get(param, {}).get("score", "N/A")
                                    comparison_data.append({
                                        "Parameter": param,
                                        "Model A Score": score_a,
                                        "Model B Score": score_b,
                                        "Difference": abs(float(score_a) - float(score_b)) if isinstance(score_a, (int, float)) and isinstance(score_b, (int, float)) else "N/A"
                                    })
                                
                                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
            
            # Export options
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Export to Excel"):
                    try:
                        excel_data = exports.export_to_excel(st.session_state.analysis_results)
                        st.download_button(
                            label="Download Excel",
                            data=excel_data,
                            file_name="call_analysis_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
            
            with col2:
                if st.button("📄 Export to CSV"):
                    try:
                        csv_data = exports.export_to_csv(st.session_state.analysis_results)
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name="call_analysis_results.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
            
            with col3:
                if st.button("📊 Export to JSON"):
                    try:
                        json_data = exports.export_to_json(st.session_state.analysis_results)
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name="call_analysis_results.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
        
def page_dashboard():
    st.title("📊 Dashboard & Results")
    st.info("Dashboard will show results after an analysis is run.")
    if st.session_state.analysis_results:
        st.dataframe(_create_summary_df(st.session_state.analysis_results))

def page_rubric_editor():
    st.title("📝 Rubric Editor")
    st.info("Functionality to edit rubrics coming soon.")

def page_run_history():
    st.title("🗂️ Run History")
    st.info("Functionality to view past analysis runs coming soon.")

# =============================================================================
# ROUTER
# =============================================================================

def main():
    st.sidebar.title("Call Insights Desk")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Engine Configuration**")
    engine_options = ["whisper_gemini", "gladia", "assemblyai", "deepgram"]
    selected_engines = st.sidebar.multiselect(
        "Select Engine(s) to Run",
        options=engine_options,
        default=["whisper_gemini"],
        help="Select one for normal use, or multiple to run a benchmark comparison."
    )

    st.sidebar.markdown("---")
    pages = {
        "Call Analysis": lambda: page_call_analysis(selected_engines),
        "Dashboard": page_dashboard,
        "Rubric Editor": page_rubric_editor,
        "Run History": page_run_history,
    }
    page_name = st.sidebar.radio("Navigation", list(pages.keys()))
    
    pages[page_name]()

if __name__ == "__main__":
    main()
