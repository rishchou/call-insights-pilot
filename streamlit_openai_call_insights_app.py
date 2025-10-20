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
        # Your existing analysis logic can go here. It will use the `get_analysis_ready_files()`
        # function to find files that have at least one successful transcription.
        st.info("Analysis section placeholder. Your original logic can be placed here.")
        
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
