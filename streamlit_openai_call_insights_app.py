import pandas as pd
import streamlit as st

# Import the new modules
import audio_processing
import ai_engine

# ======================================================================================
# CONFIGURATION
# ======================================================================================

st.set_page_config(
    page_title="Call Insights Desk",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================================
# SESSION STATE & DATA HANDLING
# ======================================================================================

if "records" not in st.session_state:
    st.session_state.records = {}

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

if "run_history" not in st.session_state:
    st.session_state["run_history"] = []
    
# ======================================================================================
# HELPER FUNCTIONS FOR DATA CONVERSION
# ======================================================================================

def _create_summary_df(analysis_results: dict) -> pd.DataFrame:
    """Converts the analysis results into a high-level summary DataFrame."""
    summary_data = []
    for file_name, results in analysis_results.items():
        scores = [s['details'].get('score', 0) for s in results.get('scores', [])]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        summary_data.append({
            "File Name": file_name,
            "Category": results.get('triage', {}).get('category', 'N/A'),
            "Call Purpose": results.get('triage', {}).get('purpose', 'N/A'),
            "Outcome": results.get('outcome', {}).get('business_outcome', {}).get('outcome', 'N/A'),
            "Average Score": f"{avg_score:.2f}",
            "Risk Identified": results.get('outcome', {}).get('risk_identified', {}).get('risk', False)
        })
    return pd.DataFrame(summary_data)

def _create_detailed_df(analysis_results: dict) -> pd.DataFrame:
    """Converts the analysis results into a detailed, row-by-row DataFrame."""
    detailed_data = []
    for file_name, results in analysis_results.items():
        for score_item in results.get('scores', []):
            details = score_item.get('details', {})
            detailed_data.append({
                "File Name": file_name,
                "Parameter": score_item.get('parameter'),
                "Score": details.get('score'),
                "Justification": details.get('justification'),
                "Primary Evidence": details.get('primary_evidence'),
                "Coaching Opportunity": details.get('coaching_opportunity')
            })
    return pd.DataFrame(detailed_data)

# ======================================================================================
# UI PAGES
# ======================================================================================

def page_call_analysis():
    st.title("🔎 Call Analysis")
    # ... (The rest of this page's code is unchanged) ...
    st.markdown("Upload audio files, configure your analysis, and run the AI engine.")

    with st.container(border=True):
        st.subheader("1. Upload Audio Files")
        files = st.file_uploader(
            "Select one or more audio files",
            type=["mp3", "wav", "m4a", "ogg"],
            accept_multiple_files=True
        )

        if files:
            for file in files:
                if file.name not in st.session_state.records:
                    st.session_state.records[file.name] = {
                        "content": file.getvalue(),
                        "transcript_data": {"english_transcript": "This is a placeholder transcript for the audio file."}
                    }
            st.success(f"{len(st.session_state.records)} file(s) are ready for analysis.")

    with st.container(border=True):
        st.subheader("2. Configure Analysis")
        
        if not st.session_state.records:
            st.info("Please upload audio files to begin.")
            st.stop()

        uploaded_files = list(st.session_state.records.keys())
        selected_files = st.multiselect("Select files to analyze", options=uploaded_files, default=uploaded_files)

        selected_model = st.selectbox(
            "Select AI Engine",
            options=list(ai_engine.AVAILABLE_MODELS.keys())
        )

    if st.button("Analyze Selected Files", type="primary", use_container_width=True):
        if not selected_files:
            st.error("Please select at least one file to analyze.")
        else:
            with st.spinner("Running full analysis..."):
                all_results = {}
                placeholder_rubric = {
                    "Call Greetings": "90-100: Professional greeting...",
                    "Active Listening": "90-100: Paraphrases customer issue..."
                }

                for file_name in selected_files:
                    record = st.session_state.records[file_name]
                    transcript_text = record["transcript_data"]["english_transcript"]
                    
                    st.info(f"Analyzing: {file_name}...")
                    
                    triage_results = ai_engine.run_initial_triage(transcript_text, selected_model)
                    outcome_results = ai_engine.run_business_outcome_analysis(transcript_text, selected_model)
                    
                    parameter_scores = []
                    for param, anchors in placeholder_rubric.items():
                        score_result = ai_engine.score_single_parameter(transcript_text, param, anchors, selected_model)
                        parameter_scores.append({"parameter": param, "details": score_result})
                    
                    all_results[file_name] = {
                        "triage": triage_results,
                        "outcome": outcome_results,
                        "scores": parameter_scores
                    }
                
                st.session_state.analysis_results = all_results
                st.session_state["run_history"].append({
                    "run_id": f"Analysis - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "model_used": selected_model,
                    "files_analyzed": len(selected_files),
                    "results": st.session_state.analysis_results
                })
                st.success("Analysis Complete!")
                st.balloons()


def page_dashboard():
    st.title("📊 Dashboard & Results")

    if not st.session_state.analysis_results:
        st.info("Please run an analysis on the 'Call Analysis' page to see results here.")
        return

    st.subheader("Latest Analysis Summary")

    tab1, tab2, tab3 = st.tabs(["Summary Table", "Visual Dashboards", "Downloads"])

    # Prepare data for the summary table
    summary_df = _create_summary_df(st.session_state.analysis_results)

    with tab1:
        st.markdown("#### Call Overview")
        st.dataframe(summary_df, use_container_width=True)
        
        for file_name, results in st.session_state.analysis_results.items():
            with st.expander(f"**Detailed Breakdown for: {file_name}**"):
                # ... (rest of this section is unchanged) ...
                st.write("**Triage & Summary:**")
                st.json(results.get('triage', {}))
                st.write("**Business Outcome:**")
                st.json(results.get('outcome', {}))
                st.write("**Parameter Scores:**")
                for score_item in results.get('scores', []):
                    st.write(f"---")
                    st.markdown(f"**Parameter:** {score_item.get('parameter')}")
                    st.json(score_item.get('details'))


    with tab2:
        st.markdown("#### Visual Dashboards")
        df_chart = summary_df.copy()
        df_chart['Average Score'] = pd.to_numeric(df_chart['Average Score'])
        st.bar_chart(df_chart.set_index('File Name')['Average Score'])

    with tab3:
        st.markdown("#### Download Reports")
        
        # --- NEW DOWNLOAD LOGIC ---
        detailed_df = _create_detailed_df(st.session_state.analysis_results)
        
        # Download Button 1: Summary Report
        st.download_button(
            label="⬇️ Download Summary Report (CSV)",
            data=summary_df.to_csv(index=False).encode('utf-8'),
            file_name="calls_summary_report.csv",
            mime="text/csv"
        )
        
        # Download Button 2: Detailed Report
        st.download_button(
            label="⬇️ Download Detailed Audit (CSV)",
            data=detailed_df.to_csv(index=False).encode('utf-8'),
            file_name="detailed_audit_report.csv",
            mime="text/csv"
        )


def page_rubric_editor():
    # ... (unchanged) ...
    st.title("📝 Rubric Editor")
    st.info("This is a placeholder for the rubric editor.")

def page_run_history():
    # ... (unchanged) ...
    st.title("🗂️ Run History")
    if not st.session_state["run_history"]:
        st.info("No analysis has been run yet.")
        return
    for run in reversed(st.session_state["run_history"]):
        with st.expander(f"**{run['run_id']}**"):
            st.json(run['results'])


# ======================================================================================
# SIDEBAR & MAIN APP LOGIC
# ======================================================================================

st.sidebar.title("Call Insights Desk")
pages = {
    "Call Analysis": page_call_analysis,
    "Dashboard": page_dashboard,
    "Rubric Editor": page_rubric_editor,
    "Run History": page_run_history,
}
page_name = st.sidebar.radio("Navigation", pages.keys())
pages[page_name]()
