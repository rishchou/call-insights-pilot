# main_app.py
import pandas as pd
import streamlit as st
from typing import Dict, List
import html

# Your modules
import audio_processing
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
# SESSION STATE & HELPERS (Mostly Unchanged)
# =============================================================================

def initialize_session_state():
    if "files_metadata" not in st.session_state:
        st.session_state.files_metadata = {}  # filename -> {size_mb, format}
    if "transcription_results" not in st.session_state:
        st.session_state.transcription_results = {}  # unique_key (filename::engine) -> result
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}  # filename -> analysis_bundle
    if "run_history" not in st.session_state:
        st.session_state.run_history = []

initialize_session_state()

def get_analysis_ready_files() -> List[str]:
    """Gets a list of unique filenames that have at least one successful transcription."""
    ready_files = set()
    for key, result in st.session_state.transcription_results.items():
        if result.get("status") == "success":
            filename = key.split("::")[0]
            ready_files.add(filename)
    return sorted(list(ready_files))

# ... (Your other helper functions like _create_summary_df, metric_card, badge, etc. are good and can remain) ...
# For brevity, they are omitted here, but you should keep them in your file.

# =============================================================================
# UI PAGES
# =============================================================================

def page_call_analysis(selected_engines: List[str]):
    st.title("🔍 Call Analysis & Benchmarking")
    st.caption("Upload calls, process them with one or more engines, and analyze the results.")

    # --- Summary Ribbon ---
    total_files = len(st.session_state.files_metadata)
    ready_files = len(get_analysis_ready_files())
    analyzed_files = len(st.session_state.analysis_results)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Unique Files Uploaded", str(total_files))
    with c2: st.metric("Files Ready for Analysis", str(ready_files))
    with c3: st.metric("Files Analyzed", str(analyzed_files))
    st.markdown("---")

    upload_tab, analyze_tab = st.tabs(["Upload & Process", "Analyze Results"])

    # =========================
    # UPLOAD & PROCESS TAB
    # =========================
    with upload_tab:
        st.subheader("1) Upload Audio")
        files = st.file_uploader(
            "Drag & drop audio files here",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True
        )

        # === CHANGED BLOCK: FILE PROCESSING LOGIC ===
        if files:
            if not selected_engines:
                st.error("Please select at least one engine from the sidebar to process files.")
            else:
                for file in files:
                    if file.name in st.session_state.files_metadata:
                        continue # Skip if base file metadata is already there

                    content = file.getvalue()
                    validation = audio_processing._validate_audio_file(file.name, content)
                    if not validation["valid"]:
                        st.error(f"{file.name}: {', '.join(validation['errors'])}")
                        continue
                    
                    st.session_state.files_metadata[file.name] = {**validation["file_info"]}

                    st.write(f"---")
                    st.write(f"Processing **{file.name}** with {len(selected_engines)} engine(s)...")
                    
                    # Loop through and process with each selected engine
                    for engine in selected_engines:
                        unique_key = f"{file.name}::{engine}"
                        
                        if unique_key in st.session_state.transcription_results:
                            st.info(f"Skipping {engine} for {file.name} (already processed).")
                            continue

                        # We use the non-progress version here for a cleaner multi-file UI
                        # The spinner provides good feedback
                        with st.spinner(f"Running engine: {engine}..."):
                            res = audio_processing.process_audio(file.name, content, engine=engine)
                            st.session_state.transcription_results[unique_key] = res
                        
                        if res.get("status") == "success":
                            st.success(f"✅ **{engine}:** Processed successfully.")
                        else:
                            st.error(f"❌ **{engine}:** Failed. {res.get('error_message', 'Unknown error')}")
        # === END OF CHANGED BLOCK ===
        
        # === CHANGED BLOCK: RESULTS DISPLAY ===
        if st.session_state.files_metadata:
            st.markdown("---")
            st.subheader("Transcription Queue & Results")

            files_to_show = sorted(st.session_state.files_metadata.keys())
            
            for filename in files_to_show:
                with st.expander(f"📄 **{filename}**"):
                    engine_results = {
                        k.split("::")[1]: v 
                        for k, v in st.session_state.transcription_results.items() 
                        if k.startswith(filename)
                    }
                    
                    if not engine_results:
                        st.info("This file is in the queue but has not been processed.")
                        continue

                    # Create columns for each engine result to show side-by-side
                    cols = st.columns(len(engine_results))
                    
                    for i, (engine_name, result) in enumerate(engine_results.items()):
                        with cols[i]:
                            st.markdown(f"##### Engine: `{engine_name}`")
                            if result.get("status") == "success":
                                st.success("Success")
                                st.text(f"Language: {result.get('language', 'N/A')}")
                                st.text(f"Duration: {result.get('duration', 0):.1f}s")
                                
                                transcript_snippet = result.get('english_text') or result.get('original_text', '')
                                st.text_area(
                                    "Transcript Snippet", 
                                    transcript_snippet[:200] + "...", 
                                    height=150, 
                                    key=f"snippet_{filename}_{engine_name}"
                                )
                            else:
                                st.error("Failed")
                                st.caption(result.get('error_message'))

            # Cleanup Buttons
            st.markdown("---")
            colA, colB = st.columns(2)
            with colA:
                if st.button("🗑️ Remove failed transcriptions"):
                    # Logic to find and remove failed runs
                    pass # Add cleanup logic if needed
            with colB:
                if st.button("🧹 Clear ALL Files & Results"):
                    st.session_state.files_metadata.clear()
                    st.session_state.transcription_results.clear()
                    st.session_state.analysis_results.clear()
                    st.rerun()
        # === END OF CHANGED BLOCK ===

    # =========================
    # ANALYZE RESULTS TAB
    # =========================
    with analyze_tab:
        st.subheader("2) Configure & Run Analysis")
        # ... (Your analysis tab logic is good and can remain the same) ...
        # It will correctly pick up the 'ready_files' and allow you to run the ai_engine on them.
        # For brevity, this section is omitted, but keep your original code here.
        pass

# ... (Your other page functions: page_dashboard, page_rubric_editor, page_run_history are good) ...
# For brevity, they are omitted, but you should keep them in your file.

# =============================================================================
# ROUTER
# =============================================================================

def main():
    st.sidebar.title("Call Insights Desk")

    # === CHANGED BLOCK: ENGINE SELECTOR ===
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Engine Configuration**")
    engine_options = ["whisper_gemini", "gladia", "assemblyai", "deepgram"]
    selected_engines = st.sidebar.multiselect(
        "Select Engine(s) to Run",
        options=engine_options,
        default=["whisper_gemini"],
        help="Select one for normal use, or multiple to run a benchmark comparison."
    )
    # === END OF CHANGED BLOCK ===

    # --- Page Navigation ---
    st.sidebar.markdown("---")
    pages = {
        "Call Analysis": lambda: page_call_analysis(selected_engines), # Pass selection to the page
        "Dashboard": page_dashboard,
        "Rubric Editor": page_rubric_editor,
        "Run History": page_run_history,
    }
    page_name = st.sidebar.radio("Navigation", list(pages.keys()))
    
    # Run the selected page function
    pages[page_name]()

if __name__ == "__main__":
    main()
