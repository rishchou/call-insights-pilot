import pandas as pd
import streamlit as st
import io
import hashlib

# Import the new modules
import audio_processing
import ai_engine
import database # Import the new database module

# ======================================================================================
# INITIALIZE DATABASE
# ======================================================================================
database.init_db()

# ======================================================================================
# CONFIGURATION
# (The rest of this section is unchanged)
# ======================================================================================

st.set_page_config(
    page_title="Call Insights Desk",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================================
# SESSION STATE & DATA HANDLING
# (This section is unchanged)
# ======================================================================================

if "records" not in st.session_state:
    st.session_state.records = {}

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

if "run_history" not in st.session_state:
    st.session_state["run_history"] = []

# ======================================================================================
# HELPER FUNCTIONS FOR DATA CONVERSION
# (This section is unchanged)
# ======================================================================================

def _create_summary_df(analysis_results: dict) -> pd.DataFrame:
    summary_data = []
    call_results = analysis_results.get('results', analysis_results)
    for file_name, results in call_results.items():
        scores = [s['details'].get('score', 0) for s in results.get('scores', [])]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        triage_info = results.get('triage', {})
        outcome_info = results.get('outcome', {})
        business_outcome_info = outcome_info.get('business_outcome', {})
        risk_info = outcome_info.get('risk_identified', {})
        
        summary_data.append({
            "File Name": file_name, "Category": triage_info.get('category', 'N/A'),
            "Call Purpose": triage_info.get('purpose', 'N/A'),
            "Outcome": business_outcome_info.get('outcome', 'N/A'),
            "Average Score": f"{avg_score:.2f}",
            "Risk Identified": risk_info.get('risk', False)
        })
    return pd.DataFrame(summary_data)

def _create_detailed_df(analysis_results: dict) -> pd.DataFrame:
    # (This function is unchanged)
    detailed_data = []
    call_results = analysis_results.get('results', analysis_results)
    for file_name, results in call_results.items():
        for score_item in results.get('scores', []):
            details = score_item.get('details', {})
            detailed_data.append({
                "File Name": file_name, "Parameter": score_item.get('parameter'),
                "Score": details.get('score'), "Justification": details.get('justification'),
                "Primary Evidence": details.get('primary_evidence'),
                "Coaching Opportunity": details.get('coaching_opportunity')
            })
    return pd.DataFrame(detailed_data)

def _create_transcript_df(analysis_results: dict) -> pd.DataFrame:
    # (This function is unchanged)
    transcript_data = []
    call_results = analysis_results.get('results', analysis_results)
    for file_name, results in call_results.items():
        segments = results.get("transcript_data", {}).get("segments", [])
        if segments:
            for seg in segments:
                transcript_data.append({
                    "File Name": file_name, "Timestamp (start)": f"{seg.get('start', 0):.2f}",
                    "Speaker": seg.get('speaker', 'UNKNOWN'), "Transcript": seg.get('text', '')
                })
    return pd.DataFrame(transcript_data)
# ======================================================================================
# UI PAGES
# ======================================================================================

def page_call_analysis():
    st.title("🔎 Call Analysis")
    st.markdown("Upload audio files, which will be transcribed and saved for future analysis.")

    with st.container(border=True):
        st.subheader("1. Upload Audio Files")
        files = st.file_uploader(
            "Select one or more audio files",
            type=["mp3", "wav", "m4a", "ogg"],
            accept_multiple_files=True
        )

        # --- UPDATED LOGIC TO USE DATABASE ---
        if files:
            for file in files:
                file_content = file.getvalue()
                file_hash = hashlib.sha256(file_content).hexdigest()

                # Check if transcript already exists in the database
                transcript_data = database.fetch_transcript(file_hash)
                
                if transcript_data:
                    st.success(f"'{file.name}' already processed. Loaded from history.")
                else:
                    with st.spinner(f"Processing '{file.name}' for the first time..."):
                        transcript_data = audio_processing.process_audio(file.name, file_content)
                        if "error" not in transcript_data:
                            database.save_transcript(file_hash, file.name, transcript_data)
                            st.success(f"'{file.name}' processed and saved to history.")
                        else:
                            st.error(f"Failed to process '{file.name}'.")
                
                # Add to the current session for analysis
                st.session_state.records[file.name] = {
                    "content": file_content,
                    "transcript_data": transcript_data
                }

    # --- THE REST OF THE APP IS LARGELY UNCHANGED ---
    with st.container(border=True):
        st.subheader("2. Configure Analysis")
        
        if not st.session_state.records:
            st.info("Please upload audio files to begin.")
            st.stop()
        # (The rest of the configuration and analysis button logic is unchanged)
        uploaded_files = list(st.session_state.records.keys())
        selected_files = st.multiselect("Select files to analyze", options=uploaded_files, default=uploaded_files)
        selected_model = st.selectbox("Select AI Engine", options=list(ai_engine.AVAILABLE_MODELS.keys()))
    
    if st.button("Analyze Selected Files", type="primary", use_container_width=True):
        # (The analysis loop is unchanged)
        if not selected_files:
            st.error("Please select at least one file to analyze.")
        else:
            # ... (analysis logic remains the same)
            with st.spinner("Running full analysis..."):
                # ... same logic as before to call AI engine ...
                st.success("Analysis Complete!")
                st.balloons()
                
# (The Dashboard, Rubric Editor, and Run History pages are unchanged)
def page_dashboard():
    # ...
    pass
def page_rubric_editor():
    # ...
    pass
def page_run_history():
    # ...
    pass

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

# To avoid errors, I've filled in the other page functions with placeholders.
# You should copy the full, working functions from your previous version.
if page_name == "Call Analysis":
    page_call_analysis()
else:
    st.title(page_name)
    st.info("This page is under construction in this code snippet.")
