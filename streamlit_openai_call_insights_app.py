# Call Insights Desk - v2.0
# A multi-page Streamlit application for advanced call quality assurance and analysis.

import os
import json
import hashlib
import pandas as pd
import streamlit as st
from openai import OpenAI
import google.generativeai as genai

# ======================================================================================
# CONFIGURATION & SECRETS
# ======================================================================================

# --- Page Config ---
st.set_page_config(
    page_title="Call Insights Desk",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Keys ---
# Load API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# --- Model Definitions ---
# Define the available AI models for selection
AVAILABLE_MODELS = {
    "GPT-4o (OpenAI)": "gpt-4o",
    "Gemini 1.5 Pro (Google)": "gemini-1.5-pro"
}

# ======================================================================================
# SESSION STATE INITIALIZATION
# ======================================================================================

# Initialize session state variables to hold data across page reruns.
# This acts as a temporary memory for the user's session.

if "records" not in st.session_state:
    st.session_state["records"] = []  # To store uploaded call data

if "selected_files" not in st.session_state:
    st.session_state["selected_files"] = set()  # To track which files are selected for analysis

if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = {}  # To store the latest analysis output

if "run_history" not in st.session_state:
    st.session_state["run_history"] = []  # To store summaries of past analysis runs

# ======================================================================================
# CORE AI & HELPER FUNCTIONS
# ======================================================================================

# --- API Client Setup ---
# Setup API clients if keys are available.
if OPENAI_API_KEY:
    oai_client = OpenAI(api_key=OPENAI_API_KEY)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_model(model_name="gemini-1.5-pro"):
    """Returns a configured Gemini model instance."""
    return genai.GenerativeModel(
        model_name,
        generation_config={"temperature": 0.2, "response_mime_type": "application/json"}
    )

# --- General Helpers ---
def _sha256(b: bytes) -> str:
    """Calculates the SHA256 hash of a byte string."""
    return hashlib.sha256(b).hexdigest()

def _json_guard(text_response: str) -> dict:
    """Ensures the AI response is a valid JSON object, attempting to fix it if not."""
    try:
        # Find the first '{' and the last '}' to handle potential markdown ```json formatting
        start = text_response.find('{')
        end = text_response.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text_response[start:end+1])
    except (json.JSONDecodeError, IndexError):
        # If parsing fails, return an error structure
        return {"error": "Failed to parse AI response as JSON.", "raw_response": text_response}
    return {"error": "No valid JSON object found in the response.", "raw_response": text_response}

# --- Main AI Call Function ---
def call_ai_engine(prompt: str, selected_model: str) -> dict:
    """
    Calls the selected AI engine (GPT or Gemini) with a given prompt
    and returns a structured dictionary.
    """
    if "Gemini" in selected_model:
        if not GEMINI_API_KEY:
            return {"error": "Gemini API key is not configured."}
        model = get_gemini_model(AVAILABLE_MODELS[selected_model])
        response = model.generate_content(prompt)
        return _json_guard(response.text)
    elif "GPT" in selected_model:
        if not OPENAI_API_KEY:
            return {"error": "OpenAI API key is not configured."}
        response = oai_client.chat.completions.create(
            model=AVAILABLE_MODELS[selected_model],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return _json_guard(response.choices[0].message.content)
    else:
        return {"error": "Invalid model selected."}

# ======================================================================================
# UI PAGES
# ======================================================================================

def page_call_analysis():
    """Main page for uploading, configuring, and running the analysis."""
    st.title("🔎 Call Analysis")
    st.markdown("Upload audio files, configure your analysis, and run the AI engine.")

    # --- 1. File Uploader ---
    with st.container(border=True):
        st.subheader("1. Upload Audio Files")
        files = st.file_uploader(
            "Select one or more audio files",
            type=["mp3", "wav", "m4a", "ogg"],
            accept_multiple_files=True
        )
        if files:
            # Placeholder for file processing logic
            st.success(f"{len(files)} file(s) uploaded. Processing would happen automatically here.")
            # In a real implementation, you would trigger transcription and add records to st.session_state["records"]

    # --- 2. Configuration ---
    with st.container(border=True):
        st.subheader("2. Configure Analysis")
        # In a real app, you would list uploaded files here for selection
        st.info("File selection and processing logic would appear here.")

        # AI Engine Selection
        selected_model = st.selectbox(
            "Select AI Engine",
            options=list(AVAILABLE_MODELS.keys()),
            help="Choose the AI model to perform the analysis."
        )

    # --- 3. Run Analysis Button ---
    if st.button("Analyze", type="primary", use_container_width=True):
        with st.spinner("Analysis in progress... This may take a few moments."):
            # This is where the multi-stage AI analysis pipeline would be called.
            # For this example, we'll create placeholder results.
            st.session_state["analysis_results"] = {
                "summary": f"Analysis completed with {selected_model}.",
                "details": "This is a placeholder for the detailed analysis output."
            }
            # Add to run history
            st.session_state["run_history"].append({
                "run_id": f"Analysis - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "model_used": selected_model,
                "files_analyzed": len(files) if files else 0,
                "results": st.session_state["analysis_results"]
            })
            st.success("Analysis Complete!")
            st.balloons()

def page_dashboard():
    """Page for displaying the dashboards and visuals from the latest run."""
    st.title("📊 Dashboard & Results")

    if not st.session_state["analysis_results"]:
        st.info("Please run an analysis on the 'Call Analysis' page to see results here.")
        return

    st.subheader("Latest Analysis Summary")
    st.success(st.session_state["analysis_results"].get("summary", "No summary available."))

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Summary Table", "Visual Dashboards", "Downloads"])

    with tab1:
        st.markdown("#### Detailed Results")
        st.write("This area would display the detailed, filterable summary table of all analyzed calls, including scores, categories, and business outcomes.")
        # Placeholder for the detailed DataFrame
        st.dataframe(pd.DataFrame({
            "Agent": ["Sarah", "John"],
            "Overall Score": [85, 92],
            "Business Outcome": ["Issue_Resolved_First_Call", "Sale_Completed"]
        }))

    with tab2:
        st.markdown("#### Visual Dashboards")
        st.write("This area would contain the interactive charts for parameter performance, agent leaderboards, and sentiment analysis.")
        # Placeholder for charts
        st.bar_chart(pd.DataFrame({"Agent": ["Sarah", "John"], "Score": [85, 92]}).set_index("Agent"))

    with tab3:
        st.markdown("#### Download Reports")
        st.write("This area provides links to download the comprehensive Excel reports.")
        # Placeholder for download button
        st.download_button(
            "Download Comprehensive Report (Excel)",
            data="placeholder content",
            file_name="Comprehensive_Analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def page_rubric_editor():
    """Page for creating, editing, and managing custom QA rubrics."""
    st.title("📝 Rubric Editor")
    st.markdown("Define the parameters and behavioral anchors for your custom analysis.")
    st.info("This is a placeholder for the rubric editor interface. Here, users would be able to add, edit, and save their custom scoring criteria, which would then be used dynamically in the AI prompts.")

    # Placeholder for a data editor for the rubric
    st.dataframe(pd.DataFrame({
        "Parameter": ["Call Greetings", "Hold Procedure"],
        "Weight": [10, 15],
        "Behavioral Anchors": ["90-100: Professional greeting...", "90-100: Asks permission..."]
    }), use_container_width=True)

def page_run_history():
    """Page for viewing and comparing past analysis runs."""
    st.title("🗂️ Run History")
    st.markdown("Review and compare results from previous analysis batches.")

    if not st.session_state["run_history"]:
        st.info("No analysis has been run yet. The history will appear here after your first run.")
        return

    st.write("Select a past run to view its detailed results.")
    # Display the list of past runs
    for run in reversed(st.session_state["run_history"]):
        with st.expander(f"**{run['run_id']}** ({run['model_used']}, {run['files_analyzed']} files)"):
            st.write(run['results'])

# ======================================================================================
# SIDEBAR & PAGE ROUTING
# ======================================================================================

# --- Sidebar Navigation ---
st.sidebar.title("Call Insights Desk")
page = st.sidebar.radio(
    "Navigation",
    ["Call Analysis", "Dashboard", "Rubric Editor", "Run History"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This is an advanced QA tool powered by generative AI. "
    "Navigate through the pages to analyze calls, view dashboards, "
    "and configure your settings."
)

# --- Page Routing Logic ---
if page == "Call Analysis":
    page_call_analysis()
elif page == "Dashboard":
    page_dashboard()
elif page == "Rubric Editor":
    page_rubric_editor()
elif page == "Run History":
    page_run_history()
