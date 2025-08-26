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
    st.session_state.records = {} # Use a dict for easier access by file name

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

# ======================================================================================
# UI PAGES
# ======================================================================================

def page_call_analysis():
    st.title("🔎 Call Analysis")
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
                # Process each file and store it in session state if not already there
                if file.name not in st.session_state.records:
                    file_content = file.getvalue()
                    transcript_data = audio_processing.get_transcript(file.name, file_content)
                    st.session_state.records[file.name] = {
                        "content": file_content,
                        "transcript_data": transcript_data
                    }
            st.success(f"{len(st.session_state.records)} file(s) are ready for analysis.")

    with st.container(border=True):
        st.subheader("2. Configure Analysis")
        
        if not st.session_state.records:
            st.info("Please upload audio files to begin.")
            st.stop()

        # Let user select which of the uploaded files to analyze
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
                # In a real app, the rubric would come from the Rubric Editor page
                # For this example, we'll use a placeholder rubric.
                placeholder_rubric = {
                    "Call Greetings": "90-100: Professional greeting + agent ID + offer to help...",
                    "Active Listening": "90-100: Paraphrases customer issue and confirms understanding..."
                }

                for file_name in selected_files:
                    record = st.session_state.records[file_name]
                    transcript_text = record["transcript_data"].get("english_transcript", "")
                    
                    if not transcript_text:
                        all_results[file_name] = {"error": "Transcript not available."}
                        continue

                    st.info(f"Analyzing: {file_name}...")
                    
                    # Run the multi-stage analysis
                    triage_results = ai_engine.run_initial_triage(transcript_text, selected_model)
                    outcome_results = ai_engine.run_business_outcome_analysis(transcript_text, selected_model)
                    
                    # Loop through the rubric to score each parameter
                    parameter_scores = []
                    for param, anchors in placeholder_rubric.items():
                        score_result = ai_engine.score_single_parameter(transcript_text, param, anchors, selected_model)
                        parameter_scores.append({"parameter": param, "details": score_result})
                    
                    # Combine all results
                    all_results[file_name] = {
                        "triage": triage_results,
                        "outcome": outcome_results,
                        "scores": parameter_scores
                    }
                
                st.session_state.analysis_results = all_results
                st.success("Analysis Complete!")
                st.balloons()


def page_dashboard():
    st.title("📊 Dashboard & Results")

    if not st.session_state.analysis_results:
        st.info("Please run an analysis on the 'Call Analysis' page to see results here.")
        return

    st.subheader("Latest Analysis Results")
    
    for file_name, results in st.session_state.analysis_results.items():
        with st.expander(f"**Results for: {file_name}**"):
            st.json(results) # Display the full JSON result for inspection

def page_rubric_editor():
    st.title("📝 Rubric Editor")
    st.info("This is a placeholder for the rubric editor. In a full app, changes made here would be saved and used by the AI engine.")
    st.dataframe(pd.DataFrame({
        "Parameter": ["Call Greetings", "Active Listening"],
        "Behavioral Anchors": ["90-100: Professional...", "90-100: Paraphrases..."]
    }), use_container_width=True)

# ======================================================================================
# SIDEBAR & MAIN APP LOGIC
# ======================================================================================

st.sidebar.title("Call Insights Desk")
page = st.sidebar.radio(
    "Navigation",
    ["Call Analysis", "Dashboard", "Rubric Editor"]
)
st.sidebar.markdown("---")

if page == "Call Analysis":
    page_call_analysis()
elif page == "Dashboard":
    page_dashboard()
elif page == "Rubric Editor":
    page_rubric_editor()
