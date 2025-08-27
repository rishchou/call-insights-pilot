import pandas as pd
import streamlit as st
import io

# Import the custom modules for audio processing and AI logic
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
# SESSION STATE INITIALIZATION
# ======================================================================================

# Using a dictionary for records makes it easier to access data by filename
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
    """Converts the full analysis results into a high-level summary DataFrame."""
    summary_data = []
    for file_name, results in analysis_results.items():
        # Calculate an average score from the 'scores' part of the results
        scores = [s['details'].get('score', 0) for s in results.get('scores', [])]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Safe outcome extraction
        def get_outcome_safely(results):
            outcome = results.get('outcome')
            if outcome is None:
                return 'N/A'
            elif isinstance(outcome, str):
                return outcome
            elif isinstance(outcome, dict):
                business_outcome = outcome.get('business_outcome')
                if isinstance(business_outcome, dict):
                    return business_outcome.get('outcome', 'N/A')
                else:
                    return outcome.get('outcome', 'N/A')
            else:
                return 'N/A'
        
        # Safe risk identification extraction
        def get_risk_safely(results):
            outcome = results.get('outcome')
            if isinstance(outcome, dict):
                risk_identified = outcome.get('risk_identified')
                if isinstance(risk_identified, dict):
                    return risk_identified.get('risk', False)
                else:
                    return False
            else:
                return False
        
        summary_data.append({
            "File Name": file_name,
            "Category": results.get('triage', {}).get('category', 'N/A'),
            "Call Purpose": results.get('triage', {}).get('purpose', 'N/A'),
            "Outcome": get_outcome_safely(results),
            "Average Score": f"{avg_score:.2f}",
            "Risk Identified": get_risk_safely(results)
        })
    return pd.DataFrame(summary_data)

def _create_detailed_df(analysis_results: dict) -> pd.DataFrame:
    """Converts the analysis results into a detailed, row-by-row DataFrame for audit."""
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
    """UI for the main analysis page."""
    st.title("🔎 Call Analysis")
    st.markdown("Upload audio files, configure your analysis, and run the AI engine.")

    with st.container(border=True):
        st.subheader("1. Upload Audio Files")
        files = st.file_uploader(
            "Select one or more audio files",
            type=["mp3", "wav", "m4a", "ogg"],
            accept_multiple_files=True
        )

        # This section now calls the real, multi-stage audio processing function
        if files:
            for file in files:
                if file.name not in st.session_state.records:
                    file_content = file.getvalue()
                    transcript_data = audio_processing.process_audio(file.name, file_content)
                    st.session_state.records[file.name] = {
                        "content": file_content,
                        "transcript_data": transcript_data
                    }
            st.success(f"{len(st.session_state.records)} file(s) are processed and ready for analysis.")
            
            # Display a preview of a transcript to confirm it worked
            if st.session_state.records:
                first_file_name = list(st.session_state.records.keys())[0]
                transcript_preview = st.session_state.records[first_file_name]["transcript_data"].get("english_transcript", "")
                with st.expander("Show Transcript Preview"):
                    st.write(transcript_preview)

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
            with st.spinner("Running full analysis... This may take several minutes."):
                all_results = {}
                # The rubric is a placeholder. In the future, this will be loaded from the Rubric Editor page.
                placeholder_rubric = {
                    "Call Greetings": "90-100: Professional greeting + agent ID + offer to help...",
                    "Active Listening": "90-100: Paraphrases customer issue and confirms understanding..."
                }

                for file_name in selected_files:
                    record = st.session_state.records[file_name]
                    transcript_text = record["transcript_data"].get("english_transcript")
                    
                    if not transcript_text or "error" in record["transcript_data"]:
                        st.error(f"Skipping {file_name} due to transcription error.")
                        continue
                    
                    st.info(f"Analyzing: {file_name}...")
                    
                    # This is the full, multi-stage AI analysis pipeline
                    all_results[file_name] = {
                        "triage": ai_engine.run_initial_triage(transcript_text, selected_model),
                        "outcome": ai_engine.run_business_outcome_analysis(transcript_text, selected_model),
                        "scores": [
                            {"parameter": param, "details": ai_engine.score_single_parameter(transcript_text, param, anchors, selected_model)}
                            for param, anchors in placeholder_rubric.items()
                        ],
                        "transcript_data": record["transcript_data"] # Include the rich transcript data in the final results
                    }
                
                # Save results and update history
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
    """UI for the main results dashboard."""
    st.title("📊 Dashboard & Results")

    if not st.session_state.analysis_results:
        st.info("Please run an analysis on the 'Call Analysis' page to see results here.")
        return

    st.subheader("Latest Analysis Summary")

    tab1, tab2, tab3 = st.tabs(["Summary Table", "Visual Dashboards", "Downloads"])

    summary_df = _create_summary_df(st.session_state.analysis_results)

    with tab1:
        st.markdown("#### Call Overview")
        st.dataframe(summary_df, use_container_width=True)
        
        # Display a detailed breakdown for each analyzed call
        for file_name, results in st.session_state.analysis_results.items():
            with st.expander(f"**Detailed Breakdown for: {file_name}**"):
                st.write("**Full Transcript (with speaker labels):**")
                # Display the structured, speaker-labeled transcript if available
                transcript_segments = results.get("transcript_data", {}).get("segments", [])
                if transcript_segments:
                    for segment in transcript_segments:
                        speaker = segment.get("speaker", "UNKNOWN")
                        text = segment.get("text", "")
                        st.markdown(f"**{speaker}:** {text}")
                else:
                    st.info("Detailed transcript segments not available.")

                st.write("**Triage & Summary:**"); st.json(results.get('triage', {}))
                st.write("**Business Outcome:**"); st.json(results.get('outcome', {}))
                st.write("**Parameter Scores:**")
                for score_item in results.get('scores', []):
                    st.markdown(f"--- \n **Parameter:** {score_item.get('parameter')}")
                    st.json(score_item.get('details'))

    with tab2:
        st.markdown("#### Visual Dashboards")
        df_chart = summary_df.copy()
        df_chart['Average Score'] = pd.to_numeric(df_chart['Average Score'])
        st.bar_chart(df_chart.set_index('File Name')['Average Score'])

    with tab3:
        st.markdown("#### Download Reports")
        
        detailed_df = _create_detailed_df(st.session_state.analysis_results)
        
        st.download_button(
            label="⬇️ Download Summary Report (CSV)",
            data=summary_df.to_csv(index=False).encode('utf-8'),
            file_name="calls_summary_report.csv", mime="text/csv"
        )
        
        st.download_button(
            label="⬇️ Download Detailed Audit (CSV)",
            data=detailed_df.to_csv(index=False).encode('utf-8'),
            file_name="detailed_audit_report.csv", mime="text/csv"
        )

def page_rubric_editor():
    """UI for the rubric editor page (placeholder)."""
    st.title("📝 Rubric Editor")
    st.info("This is a placeholder for the rubric editor. This is where you would build the interface to create and manage your custom scoring rubrics.")

def page_run_history():
    """UI for the run history page."""
    st.title("🗂️ Run History")
    if not st.session_state["run_history"]:
        st.info("No analysis has been run yet. History of past runs will appear here.")
        return
        
    st.markdown("Select a past run to view its summary.")
    for run in reversed(st.session_state["run_history"]):
        with st.expander(f"**{run['run_id']}** ({run['model_used']}, {run['files_analyzed']} files)"):
            st.write("---")
            run_summary_df = _create_summary_df(run)
            st.dataframe(run_summary_df)
            st.write("**Full JSON results for this run:**")
            st.json(run['results'])

# ======================================================================================
# SIDEBAR & MAIN APP LOGIC (PAGE ROUTING)
# ======================================================================================

st.sidebar.title("Call Insights Desk")
pages = {
    "Call Analysis": page_call_analysis,
    "Dashboard": page_dashboard,
    "Rubric Editor": page_rubric_editor,
    "Run History": page_run_history,
}
page_name = st.sidebar.radio("Navigation", pages.keys())

# Calls the function corresponding to the selected page
pages[page_name]()
