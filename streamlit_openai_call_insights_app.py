# main_app.py
import pandas as pd
import streamlit as st
from typing import Dict, List
import html
import io
import hashlib

# Your modules
import stt_engines
import ai_engine
import csv_export

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
    if "selected_engine" not in st.session_state:
        st.session_state.selected_engine = "Whisper"
    if "comparison_cache" not in st.session_state:
        st.session_state.comparison_cache = {}
    
    # Production demo session state
    if "demo_screen" not in st.session_state:
        st.session_state.demo_screen = "welcome"
    if "demo_config" not in st.session_state:
        st.session_state.demo_config = {}
    if "demo_results" not in st.session_state:
        st.session_state.demo_results = {}
    if "demo_file" not in st.session_state:
        st.session_state.demo_file = None
    
    # Page navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Call Analysis"

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

def page_call_analysis():
    st.title("🔍 Call Analysis & QA with Multiple Transcription Engines")
    st.caption("Upload calls, process with your chosen transcription engine, and analyze with Gemini.")

    total_files = len(st.session_state.files_metadata)
    ready_files = len(get_analysis_ready_files())
    analyzed_files = len(st.session_state.analysis_results)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Unique Files Uploaded", str(total_files))
    with c2: metric_card("Files Ready for Analysis", str(ready_files))
    with c3: metric_card("Files Analyzed", str(analyzed_files))
    st.markdown("---")

    upload_tab, analyze_tab, export_tab = st.tabs([
        "Upload & Process", 
        "Analyze Results", 
        "Export CSV"
    ])

    with upload_tab:
        st.subheader("1) Select Transcription Engine")
        
        # Get available engines
        available_engines = stt_engines.get_available_engines()
        
        if not available_engines:
            st.error("No transcription engines available. Please configure API keys in secrets.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_engine = st.selectbox(
                "Choose transcription engine:",
                options=available_engines,
                index=0,
                help="Select the speech-to-text engine to use for transcription"
            )
            st.session_state.selected_engine = selected_engine
        
        with col2:
            # Get available analysis models
            available_models = []
            if st.secrets.get("GEMINI_API_KEY"):
                available_models.append("Gemini")
            if st.secrets.get("OPENAI_API_KEY"):
                available_models.append("GPT-4")
            if st.secrets.get("CLAUDE_API_KEY"):
                available_models.append("Claude")
            
            if not available_models:
                st.error("No analysis models available. Please configure API keys.")
                return
            
            selected_model = st.selectbox(
                "Choose analysis model:",
                options=available_models,
                index=0,
                help="Select the AI model to use for call quality analysis"
            )
            st.session_state.selected_model = selected_model
        
        st.info(f"**Selected:** {selected_engine} (transcription) + {selected_model} (analysis)")
        
        # Show engine info
        engine_info = {
            "Whisper": "OpenAI Whisper - Supports translation and multiple languages",
            "Gladia": "Gladia API - Fast async processing with translation",
            "Deepgram": "Deepgram Nova-2 Phonecall - Optimized for phone calls",
            "AssemblyAI": "AssemblyAI - Auto language detection with diarization"
        }
        
        model_info = {
            "Gemini": "Google Gemini 2.0 Flash - Fast and cost-effective",
            "GPT-4": "OpenAI GPT-4o - High accuracy and reasoning",
            "Claude": "Anthropic Claude Sonnet 4 - Excellent comprehension"
        }
        
        st.caption(engine_info.get(selected_engine, ""))
        st.caption("💡 Tip: You can upload the same file multiple times with different engines to compare transcription quality.")
        
        st.markdown("---")
        st.subheader("2) Upload Audio Files")
        files = st.file_uploader(
            "Drag & drop audio files here", 
            type=stt_engines.SUPPORTED_FORMATS, 
            accept_multiple_files=True
        )

        if files:
            for file in files:
                content = file.getvalue()
                unique_key = f"{file.name}::{selected_engine}"
                
                # Check if this specific file+engine combination was already processed
                if unique_key in st.session_state.transcription_results:
                    st.info(f"✅ {file.name} already processed with {selected_engine}")
                    result = st.session_state.transcription_results[unique_key]
                    if result.get("status") == "success":
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Duration", f"{result.get('duration', 0):.1f}s")
                        with col2:
                            st.metric("Language", result.get('language', 'unknown'))
                    continue
                
                # Validate file
                validation = stt_engines._validate_audio_file(file.name, content)
                
                if not validation["valid"]:
                    st.error(f"{file.name}: {', '.join(validation['errors'])}")
                    continue
                
                # Store file metadata if not already stored
                if file.name not in st.session_state.files_metadata:
                    st.session_state.files_metadata[file.name] = {**validation["file_info"]}
                
                st.write(f"---")
                st.write(f"Processing **{file.name}** with **{selected_engine}**...")
                
                with st.spinner(f"Running {selected_engine}..."):
                    res = stt_engines.process_audio(file.name, content, selected_engine)
                    st.session_state.transcription_results[unique_key] = res
                
                if res.get("status") == "success":
                    st.success(f"✅ Processed successfully with {selected_engine}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Duration", f"{res.get('duration', 0):.1f}s")
                    with col2:
                        st.metric("Language", res.get('language', 'unknown'))
                else:
                    st.error(f"❌ Failed: {res.get('error_message', 'Unknown error')}")
        
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

                                # Transcript snippet - handle "Not Available" placeholder
                                english_text = result.get("english_text", "")
                                original_text = result.get("original_text", "")
                                
                                if english_text and english_text != "Not Available":
                                    snippet_src = english_text
                                else:
                                    snippet_src = original_text
                                    
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
                                        st.dataframe(pd.DataFrame(df_rows), width='stretch', hide_index=True)

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
        
        # Show selected model
        selected_model = st.session_state.get("selected_model", "Gemini")
        st.info(f"**Analysis Model:** {selected_model}")
        
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
        
        # Analysis depth selection
        col1, col2 = st.columns(2)
        with col1:
            analysis_depth = st.selectbox(
                "Analysis Depth",
                options=["Quick Scan", "Standard Analysis", "Deep Dive"],
                index=1,
                help="Choose the level of detail for parameter scoring"
            )
        
        with col2:
            custom_rubric = st.selectbox(
                "Custom Rubric (Optional)",
                options=["None", "Sales Outbound", "Banking Support", "Technical Support"],
                index=0,
                help="Add industry-specific parameters"
            )
            custom_rubric = None if custom_rubric == "None" else custom_rubric
        
        st.markdown("---")
        
        # Run Analysis button
        if st.button(f"🚀 Run AI Analysis with {selected_model}", type="primary"):
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
                
                # Use the first successful result
                result = list(file_results.values())[0]
                
                # Prepare transcript for analysis
                segments = result.get("segments", [])
                if segments:
                    formatted_transcript = "\n".join([
                        f"[{seg.get('speaker', 'UNKNOWN')}]: {seg.get('text', '')}"
                        for seg in segments
                    ])
                else:
                    formatted_transcript = result.get("english_text") or result.get("original_text", "")
                
                try:
                    # Run comprehensive AI analysis with selected model
                    analysis_result = ai_engine.run_comprehensive_analysis(
                        transcript=formatted_transcript,
                        depth=analysis_depth,
                        custom_rubric=custom_rubric,
                        model=selected_model,
                        max_retries=2
                    )
                    
                    # Enrich with metadata
                    analysis_result["file_name"] = filename
                    analysis_result["detected_language"] = result.get("language", "unknown")
                    analysis_result["duration"] = result.get("duration", 0)
                    analysis_result["engine"] = result.get("engine", "")
                    
                    st.session_state.analysis_results[filename] = {
                        "stt_result": result,
                        "analysis": analysis_result,
                        "rubric": analysis_depth
                    }
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
            
            for filename, result in st.session_state.analysis_results.items():
                if filename not in selected_files:
                    continue
                
                with st.expander(f"📋 {filename}", expanded=False):
                    if result.get("error"):
                        st.error(f"Analysis error: {result['error']}")
                        continue
                    
                    analysis = result.get("analysis", {})
                    
                    # Display triage info
                    triage = analysis.get("triage", {})
                    if triage:
                        st.markdown("#### 📞 Call Triage")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Purpose", triage.get("purpose", "N/A"))
                        with col2:
                            st.metric("Category", triage.get("category", "N/A"))
                        with col3:
                            st.metric("Sentiment", triage.get("customer_sentiment", "N/A"))
                        st.caption(triage.get("summary", ""))
                    
                    # Display business outcome
                    outcome = analysis.get("business_outcome", {})
                    if outcome:
                        st.markdown("#### 💼 Business Outcome")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Outcome", outcome.get("business_outcome", "N/A"))
                            st.metric("Compliance", outcome.get("compliance_adherence", "N/A"))
                        with col2:
                            st.metric("Risk", outcome.get("risk_identified", "N/A"))
                            st.caption(outcome.get("justification", ""))
                    
                    # Display overall score
                    overall = analysis.get("overall", {})
                    if overall:
                        st.markdown("#### 🎯 Overall Quality Score")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            score = overall.get("overall_score", 0)
                            st.metric("Score", f"{score}/100")
                        with col2:
                            st.metric("Quality", overall.get("quality_bucket", "N/A"))
                        with col3:
                            st.metric("Parameters Scored", overall.get("total_parameters_scored", 0))
                        
                        # Show coaching opportunities
                        coaching = overall.get("coaching_opportunities", [])
                        if coaching:
                            st.markdown("**🎓 Coaching Opportunities:**")
                            for i, opp in enumerate(coaching[:3], 1):
                                st.info(f"{i}. **{opp.get('parameter')}**: {opp.get('coaching')}")
                    
                    # Show parameter scores
                    param_scores = analysis.get("parameter_scores", {})
                    if param_scores:
                        st.markdown("#### 📊 Parameter Scores")
                        param_df = pd.DataFrame([
                            {
                                "Parameter": name,
                                "Score": data.get("score", "N/A"),
                                "Confidence": data.get("confidence", "N/A"),
                                "Coaching": data.get("coaching_opportunity", "")[:100] + "..."
                            }
                            for name, data in param_scores.items()
                            if isinstance(data, dict) and "error" not in data
                        ])
                        st.dataframe(param_df, use_container_width=True)
    
    with export_tab:
        st.subheader("📥 Export Analysis Results to CSV")
        
        if not st.session_state.analysis_results:
            st.warning("No analysis results to export. Please run analysis first.")
            return
        
        st.info(f"**Ready to export:** {len(st.session_state.analysis_results)} analysis result(s)")
        
        # Summary export
        if st.button("📊 Export Summary CSV"):
            analyses = []
            for filename, result in st.session_state.analysis_results.items():
                if result.get("error"):
                    continue
                analyses.append({
                    "file_name": filename,
                    "stt_result": result.get("stt_result", {}),
                    "analysis": result.get("analysis", {})
                })
            
            summary_df = csv_export.create_summary_df(analyses)
            
            # Convert to CSV for download
            csv_buffer = io.StringIO()
            summary_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="⬇️ Download Summary CSV",
                data=csv_data,
                file_name="qa_summary.csv",
                mime="text/csv"
            )
            
            st.success("Summary CSV ready for download!")
            st.dataframe(summary_df, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed export with parameters
        if st.button("📋 Export Detailed Parameters CSV"):
            analyses = []
            for filename, result in st.session_state.analysis_results.items():
                if result.get("error"):
                    continue
                analyses.append({
                    "file_name": filename,
                    "stt_result": result.get("stt_result", {}),
                    "analysis": result.get("analysis", {}),
                    "rubric": result.get("rubric", "Standard Analysis")
                })
            
            all_rows = []
            for item in analyses:
                stt_ctx = csv_export.build_stt_context(
                    item["stt_result"], 
                    item["file_name"], 
                    rubric=item.get("rubric", "Standard Analysis")
                )
                rows = csv_export._param_rows_with_context(stt_ctx, item["analysis"], truncate_len=8000)
                all_rows.extend(rows)
            
            detailed_df = pd.DataFrame(all_rows)
            
            # Convert to CSV for download
            csv_buffer = io.StringIO()
            detailed_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="⬇️ Download Detailed Parameters CSV",
                data=csv_data,
                file_name="qa_params_detailed.csv",
                mime="text/csv"
            )
            
            st.success(f"Detailed CSV ready for download! ({len(all_rows)} parameter rows)")
            st.caption("Preview (first 100 rows):")
            st.dataframe(detailed_df.head(100), use_container_width=True)
        
        st.markdown("---")
        st.info("💡 **Tip**: Use the detailed CSV for in-depth analysis in Excel or other tools.")

def page_compare_models():
        st.subheader("🔬 Model Comparison Matrix")
        st.caption("Compare all combinations of transcription engines and analysis models")
        
        # Get available engines and models
        available_stt = stt_engines.get_available_engines()
        available_llm = []
        if st.secrets.get("GEMINI_API_KEY"):
            available_llm.append("Gemini")
        if st.secrets.get("OPENAI_API_KEY"):
            available_llm.append("GPT-4")
        if st.secrets.get("CLAUDE_API_KEY"):
            available_llm.append("Claude")
        
        if not available_stt or not available_llm:
            st.error("Need at least one transcription engine and one analysis model configured.")
            return
        
        st.info(f"**Available:** {len(available_stt)} STT engines × {len(available_llm)} analysis models = {len(available_stt) * len(available_llm)} combinations")
        
        # Upload file for comparison
        st.markdown("### 📤 Upload Audio File for Comparison")
        comparison_file = st.file_uploader(
            "Upload a single audio file to test all model combinations",
            type=stt_engines.SUPPORTED_FORMATS,
            key="comparison_file_uploader"
        )
        
        if not comparison_file:
            st.info("👆 Upload an audio file to start comparison")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            analysis_depth = st.selectbox(
                "Analysis Depth",
                options=["Quick Scan", "Standard Analysis", "Deep Dive"],
                index=0,
                help="Use Quick Scan for faster comparison"
            )
        with col2:
            custom_rubric = st.selectbox(
                "Custom Rubric (Optional)",
                options=["None", "Sales Outbound", "Banking Support", "Technical Support"],
                index=0
            )
            custom_rubric = None if custom_rubric == "None" else custom_rubric
        
        st.markdown("---")
        
        total_combinations = len(available_stt) * len(available_llm)
        
        # Generate cache key for this file + configuration
        file_content = comparison_file.getvalue()
        file_hash = hashlib.sha256(file_content).hexdigest()
        cache_key = f"{file_hash}_{analysis_depth}_{custom_rubric or 'None'}"
        
        # Check if results are cached
        cached = cache_key in st.session_state.comparison_cache
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"**{total_combinations} combinations** ({len(available_stt)} STT × {len(available_llm)} Analysis)")
        with col2:
            if cached:
                st.success("✅ Cached")
            else:
                st.warning("🔄 Not cached")
        
        # Show cached results info
        if cached:
            cached_data = st.session_state.comparison_cache[cache_key]
            st.info(f"💾 Using cached results from previous analysis of this file (saved {len(cached_data.get('data', []))} results)")
        
        run_button = st.button("🚀 Run Full Model Comparison", type="primary", disabled=False)
        
        if run_button:
            # Initialize results storage
            if "comparison_results" not in st.session_state:
                st.session_state.comparison_results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            comparison_data = []
            file_content = comparison_file.getvalue()
            file_name = comparison_file.name
            
            combo_idx = 0
            for stt_engine in available_stt:
                # Step 1: Transcribe with this STT engine
                status_text.text(f"[{combo_idx + 1}/{total_combinations}] Transcribing with {stt_engine}...")
                
                try:
                    stt_result = stt_engines.process_audio(file_name, file_content, stt_engine)
                    
                    if stt_result.get("status") != "success":
                        for llm_model in available_llm:
                            combo_idx += 1
                            comparison_data.append({
                                "STT Engine": stt_engine,
                                "Analysis Model": llm_model,
                                "Status": "❌ STT Failed",
                                "Error": stt_result.get("error_message", "Unknown error"),
                                "Overall Score": None,
                                "Quality": None
                            })
                            progress_bar.progress(combo_idx / total_combinations)
                        continue
                    
                    # Get transcript - handle "Not Available" string from some engines
                    english_text = stt_result.get("english_text", "")
                    original_text = stt_result.get("original_text", "")
                    
                    # Use english if available and not placeholder, otherwise use original
                    if english_text and english_text != "Not Available":
                        transcript = english_text
                    else:
                        transcript = original_text
                    
                    # Debug: Check if transcript is empty
                    if not transcript or not transcript.strip():
                        for llm_model in available_llm:
                            combo_idx += 1
                            comparison_data.append({
                                "STT Engine": stt_engine,
                                "Analysis Model": llm_model,
                                "Status": "❌ Empty Transcript",
                                "Error": f"Transcript is empty. Keys in result: {list(stt_result.keys())}",
                                "Overall Score": 0,
                                "Quality": "N/A"
                            })
                            progress_bar.progress(combo_idx / total_combinations)
                        continue
                    
                    # Step 2: Analyze with each LLM model
                    for llm_model in available_llm:
                        combo_idx += 1
                        status_text.text(f"[{combo_idx}/{total_combinations}] {stt_engine} + {llm_model}...")
                        progress_bar.progress(combo_idx / total_combinations)
                        
                        try:
                            analysis_result = ai_engine.run_comprehensive_analysis(
                                transcript=transcript,
                                depth=analysis_depth,
                                custom_rubric=custom_rubric,
                                model=llm_model,
                                max_retries=2
                            )
                            
                            if analysis_result.get("error"):
                                comparison_data.append({
                                    "STT Engine": stt_engine,
                                    "Analysis Model": llm_model,
                                    "Status": "❌ Analysis Failed",
                                    "Error": analysis_result.get("error"),
                                    "Overall Score": None,
                                    "Quality": None,
                                    "Language": stt_result.get("language", "N/A"),
                                    "Duration": f"{stt_result.get('duration', 0):.1f}s"
                                })
                            else:
                                overall = analysis_result.get("overall", {})
                                triage = analysis_result.get("triage", {})
                                business = analysis_result.get("business_outcome", {})
                                
                                comparison_data.append({
                                    "STT Engine": stt_engine,
                                    "Analysis Model": llm_model,
                                    "Status": "✅ Success",
                                    "Overall Score": overall.get("overall_score", 0),
                                    "Quality": overall.get("quality_bucket", "N/A"),
                                    "Category": triage.get("category", "N/A"),
                                    "Sentiment": triage.get("customer_sentiment", "N/A"),
                                    "Outcome": business.get("business_outcome", "N/A"),
                                    "Compliance": business.get("compliance_adherence", "N/A"),
                                    "Language": stt_result.get("language", "N/A"),
                                    "Duration": f"{stt_result.get('duration', 0):.1f}s",
                                    "Parameters Scored": overall.get("total_parameters_scored", 0),
                                    "Low Performers": overall.get("parameters_needing_attention", 0)
                                })
                        
                        except Exception as e:
                            comparison_data.append({
                                "STT Engine": stt_engine,
                                "Analysis Model": llm_model,
                                "Status": "❌ Exception",
                                "Error": str(e),
                                "Overall Score": None,
                                "Quality": None
                            })
                
                except Exception as e:
                    for llm_model in available_llm:
                        combo_idx += 1
                        comparison_data.append({
                            "STT Engine": stt_engine,
                            "Analysis Model": llm_model,
                            "Status": "❌ Exception",
                            "Error": str(e),
                            "Overall Score": None,
                            "Quality": None
                        })
                        progress_bar.progress(combo_idx / total_combinations)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Comparison complete!")
            
            # Store results in both session state and cache
            results_data = {
                "file": file_name,
                "depth": analysis_depth,
                "rubric": custom_rubric,
                "data": comparison_data
            }
            st.session_state.comparison_results = results_data
            st.session_state.comparison_cache[cache_key] = results_data
            
            st.success(f"💾 Results cached for future use!")
        
        # Load cached results if available and not just run
        elif cached and not run_button:
            st.session_state.comparison_results = st.session_state.comparison_cache[cache_key]
        
        # Display comparison results
        if "comparison_results" in st.session_state and st.session_state.comparison_results:
            st.markdown("---")
            st.subheader("📊 Comparison Results")
            
            comp_data = st.session_state.comparison_results.get("data", [])
            if not comp_data:
                st.info("No comparison data available yet.")
                return
            
            df = pd.DataFrame(comp_data)
            
            # Show summary of what happened
            st.markdown("#### 📊 Processing Summary")
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("Total Combinations", len(df))
            with summary_cols[1]:
                success_count = len(df[df["Status"] == "✅ Success"])
                st.metric("Successful", success_count)
            with summary_cols[2]:
                failed_count = len(df[df["Status"] != "✅ Success"])
                st.metric("Failed/Issues", failed_count)
            with summary_cols[3]:
                if len(df) > 0:
                    success_rate = (success_count / len(df)) * 100
                    st.metric("Success Rate", f"{success_rate:.0f}%")
            
            # Filter to successful runs only for detailed analysis
            success_df = df[df["Status"] == "✅ Success"].copy()
            
            if len(success_df) > 0:
                # Key metrics comparison
                st.markdown("#### 🎯 Overall Scores Comparison")
                
                # Create pivot table for heatmap
                pivot_scores = success_df.pivot_table(
                    values="Overall Score",
                    index="STT Engine",
                    columns="Analysis Model",
                    aggfunc="mean"
                )
                
                # Display pivot table with color gradient
                st.dataframe(
                    pivot_scores.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=100).format("{:.1f}"),
                    use_container_width=True
                )
                
                # Quality bucket distribution
                st.markdown("#### 📈 Quality Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    quality_counts = success_df.groupby(["Analysis Model", "Quality"]).size().unstack(fill_value=0)
                    st.bar_chart(quality_counts)
                
                with col2:
                    stt_quality = success_df.groupby(["STT Engine", "Quality"]).size().unstack(fill_value=0)
                    st.bar_chart(stt_quality)
                
                # Detailed comparison table
                st.markdown("#### 📋 Detailed Results")
                display_cols = ["STT Engine", "Analysis Model", "Overall Score", "Quality", 
                               "Category", "Sentiment", "Outcome", "Compliance", "Parameters Scored", "Low Performers"]
                
                # Only show columns that exist
                available_cols = [col for col in display_cols if col in success_df.columns]
                st.dataframe(success_df[available_cols].sort_values("Overall Score", ascending=False), 
                            use_container_width=True, hide_index=True)
                
                # Statistical summary
                st.markdown("#### 📊 Statistical Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Highest Score", f"{success_df['Overall Score'].max():.1f}")
                    best_combo = success_df.loc[success_df['Overall Score'].idxmax()]
                    st.caption(f"{best_combo['STT Engine']} + {best_combo['Analysis Model']}")
                
                with col2:
                    st.metric("Average Score", f"{success_df['Overall Score'].mean():.1f}")
                    st.caption(f"Std Dev: {success_df['Overall Score'].std():.1f}")
                
                with col3:
                    st.metric("Score Range", f"{success_df['Overall Score'].max() - success_df['Overall Score'].min():.1f}")
                    st.caption("Difference between best and worst")
                
                # Best performers by category
                st.markdown("#### 🏆 Best Combinations")
                
                best_by_stt = success_df.groupby("STT Engine")["Overall Score"].max()
                best_by_llm = success_df.groupby("Analysis Model")["Overall Score"].max()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Best by STT Engine:**")
                    for engine, score in best_by_stt.items():
                        best_llm = success_df[(success_df["STT Engine"] == engine) & 
                                             (success_df["Overall Score"] == score)]["Analysis Model"].iloc[0]
                        st.write(f"• {engine}: **{score:.1f}** (with {best_llm})")
                
                with col2:
                    st.markdown("**Best by Analysis Model:**")
                    for model, score in best_by_llm.items():
                        best_stt = success_df[(success_df["Analysis Model"] == model) & 
                                             (success_df["Overall Score"] == score)]["STT Engine"].iloc[0]
                        st.write(f"• {model}: **{score:.1f}** (with {best_stt})")
            
            # Show failed combinations if any
            failed_df = df[df["Status"] != "✅ Success"]
            if len(failed_df) > 0:
                st.warning(f"⚠️ {len(failed_df)} combinations had issues. Expand below for details.")
                with st.expander(f"⚠️ Failed/Problem Combinations ({len(failed_df)})", expanded=True):
                    display_cols = ["STT Engine", "Analysis Model", "Status"]
                    if "Error" in failed_df.columns:
                        display_cols.append("Error")
                    st.dataframe(failed_df[display_cols], 
                               use_container_width=True, hide_index=True)
            
            # Download comparison results
            st.markdown("---")
            col1, col2 = st.columns([4, 1])
            
            with col1:
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                csv_data = csv_buffer.getvalue()
                
                # Get filename from session state
                result_file_name = st.session_state.comparison_results.get("file", "comparison")
                
                st.download_button(
                    label="📥 Download Comparison Results CSV",
                    data=csv_data,
                    file_name=f"model_comparison_{result_file_name}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Clear cache button
                if st.button("🗑️ Clear Cache", help="Clear all cached comparison results"):
                    st.session_state.comparison_cache = {}
                    st.session_state.comparison_results = {}
                    st.success("Cache cleared!")
                    st.rerun()
        
        # Show cache info in sidebar
        if st.session_state.comparison_cache:
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 💾 Cache Info")
                st.info(f"**{len(st.session_state.comparison_cache)}** cached comparison(s)")
                if st.button("Clear All Cache", key="sidebar_clear_cache"):
                    st.session_state.comparison_cache = {}
                    st.session_state.comparison_results = {}
                    st.success("Cache cleared!")
                    st.rerun()

def page_production_demo():
        # Custom CSS for production demo styling
        st.markdown("""
        <style>
        /* Gradient background for demo section */
        .demo-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 40px;
            color: white;
            margin-bottom: 20px;
        }
        
        /* Welcome screen styling */
        .demo-title {
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 16px;
            background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .demo-subtitle {
            font-size: 1.5rem;
            text-align: center;
            opacity: 0.95;
            margin-bottom: 32px;
        }
        
        /* Feature cards */
        .feature-card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 24px;
            margin: 12px 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .feature-icon {
            font-size: 2rem;
            margin-bottom: 12px;
        }
        
        .feature-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .feature-desc {
            font-size: 1rem;
            opacity: 0.9;
        }
        
        /* Dashboard cards */
        .dash-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 16px;
        }
        
        .dash-card-title {
            color: #374151;
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .dash-card-value {
            color: #111827;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 4px;
        }
        
        .dash-card-trend {
            color: #10B981;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        /* Score badges */
        .score-badge-good {
            background: #10B981;
            color: white;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.875rem;
        }
        
        .score-badge-warning {
            background: #F59E0B;
            color: white;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.875rem;
        }
        
        /* Alert boxes */
        .alert-success {
            background: #D1FAE5;
            border-left: 4px solid #10B981;
            padding: 16px;
            border-radius: 8px;
            margin: 12px 0;
        }
        
        .alert-warning {
            background: #FEF3C7;
            border-left: 4px solid #F59E0B;
            padding: 16px;
            border-radius: 8px;
            margin: 12px 0;
        }
        
        .alert-title {
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 8px;
            color: #111827;
        }
        
        .alert-desc {
            font-size: 0.875rem;
            color: #374151;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Screen router
        screen = st.session_state.demo_screen
        
        if screen == "welcome":
            # Welcome screen
            st.markdown("""
            <div class="demo-container">
                <div class="demo-title">🎯 CallQA Pro</div>
                <div class="demo-subtitle">AI-Powered Call Quality Analysis Platform</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Feature highlights in 3 columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">🎙️</div>
                    <div class="feature-title">Advanced Transcription</div>
                    <div class="feature-desc">OpenAI Whisper with 99% accuracy across 50+ languages</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">🤖</div>
                    <div class="feature-title">AI-Driven Analysis</div>
                    <div class="feature-desc">Claude Sonnet 4 provides comprehensive quality assessment</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Actionable Insights</div>
                    <div class="feature-desc">Detailed scoring, sentiment analysis, and compliance tracking</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Get Started button centered
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🚀 Get Started", use_container_width=True, type="primary", key="demo_get_started"):
                    st.session_state.demo_screen = "setup"
                    st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Additional info
            st.info("💡 **For Investors:** This production demo showcases our flagship configuration with best-in-class AI models for maximum accuracy and insight quality.")
        
        elif screen == "setup":
            # Setup screen
            # Header with back button
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("← Back", key="demo_setup_back"):
                    st.session_state.demo_screen = "welcome"
                    st.rerun()
            with col2:
                st.title("📝 Setup Configuration")
            
            st.markdown("---")
            
            # Client Selection
            st.subheader("1️⃣ Select Client")
            clients = [
                "✈️ Delta Airlines - Customer Support",
                "🛒 Amazon - Order Support",
                "🏦 Providian Bank - Account Services",
                "🏨 Expedia - Travel Booking",
                "📱 T-Mobile - Technical Support"
            ]
            selected_client = st.selectbox("Client & Department", clients, key="demo_client")
            st.session_state.demo_config["client"] = selected_client
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Agent Information
            st.subheader("2️⃣ Agent Information")
            col1, col2 = st.columns(2)
            with col1:
                agent_name = st.text_input("Agent Name", value="John Doe", placeholder="Agent name", key="demo_agent_name")
                st.session_state.demo_config["agent_name"] = agent_name
            with col2:
                agent_id = st.text_input("Agent ID", value="A1234", placeholder="Agent ID", key="demo_agent_id")
                st.session_state.demo_config["agent_id"] = agent_id
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # QA Parameters
            st.subheader("3️⃣ QA Parameters to Evaluate")
            st.caption("Select the quality parameters to assess in this call:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Standard Parameters:**")
                greeting = st.checkbox("✅ Greeting & Opening", value=True, key="demo_param_greeting")
                empathy = st.checkbox("💝 Empathy & Active Listening", value=True, key="demo_param_empathy")
                problem_solving = st.checkbox("🔧 Problem Resolution", value=True, key="demo_param_problem")
                closing = st.checkbox("👋 Closing & Next Steps", value=True, key="demo_param_closing")
            
            with col2:
                st.markdown("**Advanced Parameters:**")
                compliance = st.checkbox("📋 Compliance & Policy Adherence", value=True, key="demo_param_compliance")
                communication = st.checkbox("💬 Clear Communication", value=True, key="demo_param_communication")
                professionalism = st.checkbox("👔 Professionalism", value=True, key="demo_param_professionalism")
                upsell = st.checkbox("💰 Upsell Opportunities", value=False, key="demo_param_upsell")
            
            # Store selected parameters
            st.session_state.demo_config["parameters"] = {
                "greeting": greeting,
                "empathy": empathy,
                "problem_solving": problem_solving,
                "closing": closing,
                "compliance": compliance,
                "communication": communication,
                "professionalism": professionalism,
                "upsell": upsell
            }
            
            st.markdown("---")
            
            # File Upload
            st.subheader("4️⃣ Upload Call Recording")
            uploaded_file = st.file_uploader(
                "Upload audio file (WAV, MP3, M4A, etc.)",
                type=stt_engines.SUPPORTED_FORMATS,
                help="Upload the call recording to analyze",
                key="demo_file_uploader"
            )
            
            if uploaded_file:
                st.session_state.demo_file = uploaded_file
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Process button
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("🎯 Analyze Call", use_container_width=True, type="primary", key="demo_analyze_btn"):
                        st.session_state.demo_screen = "processing"
                        st.rerun()
            else:
                st.info("📤 Please upload a call recording to proceed")
        
        elif screen == "processing":
            # Processing screen
            st.title("⚙️ Processing Call Analysis")
            st.markdown("---")
            
            # Check API keys
            if not st.secrets.get("OPENAI_API_KEY"):
                st.error("❌ OpenAI API key not configured. Whisper transcription requires OpenAI API access.")
                if st.button("← Back to Setup", key="demo_proc_back1"):
                    st.session_state.demo_screen = "setup"
                    st.rerun()
                st.stop()
            
            if not st.secrets.get("CLAUDE_API_KEY"):
                st.error("❌ Claude API key not configured. Analysis requires Anthropic API access.")
                if st.button("← Back to Setup", key="demo_proc_back2"):
                    st.session_state.demo_screen = "setup"
                    st.rerun()
                st.stop()
            
            # Check if we have the file
            if not st.session_state.demo_file:
                st.error("No file uploaded. Please go back to setup.")
                if st.button("← Back to Setup", key="demo_proc_back3"):
                    st.session_state.demo_screen = "setup"
                    st.rerun()
                st.stop()
            
            file = st.session_state.demo_file
            
            # Processing steps
            st.info("🔄 **Step 1/2:** Transcribing audio with OpenAI Whisper...")
            progress_bar_1 = st.progress(0)
            
            with st.spinner("Running Whisper transcription..."):
                # Transcribe with Whisper
                content = file.getvalue()
                transcription_result = stt_engines.process_audio(file.name, content, "Whisper")
                progress_bar_1.progress(100)
            
            if transcription_result.get("status") == "success":
                st.success(f"✅ Transcription complete! Detected language: {transcription_result.get('language', 'unknown')}")
                
                # Extract transcript
                english_text = transcription_result.get("english_text", "")
                original_text = transcription_result.get("original_text", "")
                
                if english_text and english_text != "Not Available":
                    transcript = english_text
                else:
                    transcript = original_text
                
                if not transcript or not transcript.strip():
                    st.error("❌ Transcription failed: Empty transcript")
                    if st.button("← Back to Setup", key="demo_proc_back4"):
                        st.session_state.demo_screen = "setup"
                        st.rerun()
                    st.stop()
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.info("🔄 **Step 2/2:** Analyzing call quality with Claude Sonnet 4...")
                progress_bar_2 = st.progress(0)
                
                with st.spinner("Running AI analysis..."):
                    # Analyze with Claude
                    analysis_result = ai_engine.run_comprehensive_analysis(
                        transcript=transcript,
                        depth="comprehensive",
                        custom_rubric=None,
                        model="Claude",
                        max_retries=2
                    )
                    progress_bar_2.progress(100)
                
                if analysis_result.get("error"):
                    st.error(f"❌ Analysis failed: {analysis_result.get('error')}")
                    if st.button("← Back to Setup", key="demo_proc_back5"):
                        st.session_state.demo_screen = "setup"
                        st.rerun()
                    st.stop()
                
                # Store results
                st.session_state.demo_results = {
                    "transcription": transcription_result,
                    "analysis": analysis_result,
                    "transcript": transcript
                }
                
                st.success("✅ Analysis complete!")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Navigate to dashboard
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("📊 View Dashboard", use_container_width=True, type="primary", key="demo_view_dashboard"):
                        st.session_state.demo_screen = "dashboard"
                        st.rerun()
            
            else:
                st.error(f"❌ Transcription failed: {transcription_result.get('error', 'Unknown error')}")
                if st.button("← Back to Setup", key="demo_proc_back6"):
                    st.session_state.demo_screen = "setup"
                    st.rerun()
        
        elif screen == "dashboard":
            # Dashboard screen
            # Header with back button
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("← Back", key="demo_dash_back"):
                    st.session_state.demo_screen = "setup"
                    st.rerun()
            with col2:
                st.title("📊 Call Quality Dashboard")
            
            st.markdown("---")
            
            # Get results
            if not st.session_state.demo_results:
                st.warning("No analysis results available. Please process a call first.")
                st.stop()
            
            analysis = st.session_state.demo_results.get("analysis", {})
            transcription = st.session_state.demo_results.get("transcription", {})
            config = st.session_state.demo_config
            
            # Client info banner
            st.markdown(f"""
            <div class="demo-container">
                <h3 style="margin: 0; font-size: 1.5rem;">
                    {config.get('client', 'Unknown Client')}
                </h3>
                <p style="margin: 8px 0 0 0; opacity: 0.9;">
                    Agent: {config.get('agent_name', 'Unknown')} ({config.get('agent_id', 'N/A')}) | 
                    Duration: {transcription.get('duration', 0):.1f}s | 
                    Language: {transcription.get('language', 'Unknown')}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Stats Grid
            overall = analysis.get("overall", {})
            triage = analysis.get("triage", {})
            business = analysis.get("business_outcome", {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                score = overall.get("overall_score", 0)
                st.markdown(f"""
                <div class="dash-card">
                    <div class="dash-card-title">Overall Score</div>
                    <div class="dash-card-value">{score}</div>
                    <div class="dash-card-trend">{"↗️ Above Average" if score >= 75 else "→ Needs Improvement"}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                quality = overall.get("quality_bucket", "Unknown")
                st.markdown(f"""
                <div class="dash-card">
                    <div class="dash-card-title">Quality Tier</div>
                    <div class="dash-card-value" style="font-size: 1.5rem;">{quality}</div>
                    <div class="dash-card-trend">Performance Rating</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                sentiment = triage.get("customer_sentiment", "Unknown")
                sentiment_emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}.get(sentiment, "")
                st.markdown(f"""
                <div class="dash-card">
                    <div class="dash-card-title">Customer Sentiment</div>
                    <div class="dash-card-value" style="font-size: 1.5rem;">{sentiment_emoji} {sentiment}</div>
                    <div class="dash-card-trend">Detected Emotion</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                outcome = business.get("business_outcome", "Unknown")
                st.markdown(f"""
                <div class="dash-card">
                    <div class="dash-card-title">Business Outcome</div>
                    <div class="dash-card-value" style="font-size: 1.5rem;">{outcome}</div>
                    <div class="dash-card-trend">Final Result</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Parameter Scores
            st.subheader("📊 Quality Parameter Breakdown")
            
            parameters = overall.get("parameters_summary", [])
            
            if parameters:
                for param in parameters:
                    param_name = param.get("name", "Unknown")
                    param_score = param.get("score", 0)
                    param_reasoning = param.get("reasoning", "No details provided")
                    
                    # Color code based on score
                    if param_score >= 80:
                        badge_class = "score-badge-good"
                    else:
                        badge_class = "score-badge-warning"
                    
                    st.markdown(f"""
                    <div class="dash-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <strong style="color: #111827; font-size: 1rem;">{param_name}</strong>
                            <span class="{badge_class}">{param_score}</span>
                        </div>
                        <div style="color: #6B7280; font-size: 0.875rem;">{param_reasoning}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No parameter scores available.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Key Observations & Alerts
            st.subheader("🚨 Key Observations")
            
            observations = overall.get("key_observations", [])
            strengths = overall.get("strengths", [])
            improvements = overall.get("areas_for_improvement", [])
            
            # Show strengths
            if strengths:
                st.markdown("""
                <div class="alert-success">
                    <div class="alert-title">✅ Strengths Identified</div>
                    <div class="alert-desc">""" + "<br>".join([f"• {s}" for s in strengths[:3]]) + """</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show improvements needed
            if improvements:
                st.markdown("""
                <div class="alert-warning">
                    <div class="alert-title">⚠️ Areas for Improvement</div>
                    <div class="alert-desc">""" + "<br>".join([f"• {i}" for i in improvements[:3]]) + """</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Action Buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 View Full Transcript", use_container_width=True, key="demo_view_transcript"):
                    st.session_state.demo_show_transcript = not st.session_state.get("demo_show_transcript", False)
            
            if st.session_state.get("demo_show_transcript", False):
                with st.expander("📝 Full Transcript", expanded=True):
                    st.text_area(
                        "Transcript",
                        st.session_state.demo_results.get("transcript", ""),
                        height=300,
                        key="demo_transcript_area"
                    )
            
            with col2:
                if st.button("📊 Export Report", use_container_width=True, key="demo_export_report"):
                    st.info("💡 Report export functionality coming soon!")
            
            with col3:
                if st.button("🔄 Analyze Another Call", use_container_width=True, key="demo_analyze_another"):
                    # Clear results and go back to setup
                    st.session_state.demo_results = {}
                    st.session_state.demo_file = None
                    st.session_state.demo_screen = "setup"
                    st.rerun()

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
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.markdown("### 📑 Navigation")
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .demo-subtitle {
        font-size: 1.5rem;
        text-align: center;
        opacity: 0.95;
        margin-bottom: 32px;
    }
    
    /* Feature cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 12px;
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .feature-desc {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Dashboard cards */
    .dash-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 16px;
    }
    
    .dash-card-title {
        color: #374151;
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .dash-card-value {
        color: #111827;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .dash-card-trend {
        color: #10B981;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Score badges */
    .score-badge-good {
        background: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .score-badge-warning {
        background: #F59E0B;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    /* Alert boxes */
    .alert-success {
        background: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
    }
    
    .alert-warning {
        background: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
    }
    
    .alert-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 8px;
        color: #111827;
    }
    
    .alert-desc {
        font-size: 0.875rem;
        color: #374151;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Screen router
    screen = st.session_state.demo_screen
    
    if screen == "welcome":
        render_welcome_screen()
    elif screen == "setup":
        render_setup_screen()
    elif screen == "processing":
        render_processing_screen()
    elif screen == "dashboard":
        render_dashboard_screen()

def render_welcome_screen():
    """Welcome screen with branding and Get Started button."""
    
    st.markdown("""
    <div class="demo-container">
        <div class="demo-title">🎯 CallQA Pro</div>
        <div class="demo-subtitle">AI-Powered Call Quality Analysis Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature highlights in 3 columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎙️</div>
            <div class="feature-title">Advanced Transcription</div>
            <div class="feature-desc">OpenAI Whisper with 99% accuracy across 50+ languages</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI-Driven Analysis</div>
            <div class="feature-desc">Claude Sonnet 4 provides comprehensive quality assessment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Actionable Insights</div>
            <div class="feature-desc">Detailed scoring, sentiment analysis, and compliance tracking</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Get Started button centered
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Get Started", use_container_width=True, type="primary"):
            st.session_state.demo_screen = "setup"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Additional info
    st.info("💡 **For Investors:** This production demo showcases our flagship configuration with best-in-class AI models for maximum accuracy and insight quality.")

def render_setup_screen():
    """Setup screen for client, agent info, and QA parameters."""
    
    # Header with back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back"):
            st.session_state.demo_screen = "welcome"
            st.rerun()
    with col2:
        st.title("📝 Setup Configuration")
    
    st.markdown("---")
    
    # Client Selection
    st.subheader("1️⃣ Select Client")
    clients = [
        "✈️ Delta Airlines - Customer Support",
        "🛒 Amazon - Order Support",
        "🏦 Providian Bank - Account Services",
        "🏨 Expedia - Travel Booking",
        "📱 T-Mobile - Technical Support"
    ]
    selected_client = st.selectbox("Client & Department", clients)
    st.session_state.demo_config["client"] = selected_client
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Agent Information
    st.subheader("2️⃣ Agent Information")
    col1, col2 = st.columns(2)
    with col1:
        agent_name = st.text_input("Agent Name", value="John Doe", placeholder="Agent name")
        st.session_state.demo_config["agent_name"] = agent_name
    with col2:
        agent_id = st.text_input("Agent ID", value="A1234", placeholder="Agent ID")
        st.session_state.demo_config["agent_id"] = agent_id
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # QA Parameters
    st.subheader("3️⃣ QA Parameters to Evaluate")
    st.caption("Select the quality parameters to assess in this call:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Standard Parameters:**")
        greeting = st.checkbox("✅ Greeting & Opening", value=True)
        empathy = st.checkbox("💝 Empathy & Active Listening", value=True)
        problem_solving = st.checkbox("🔧 Problem Resolution", value=True)
        closing = st.checkbox("👋 Closing & Next Steps", value=True)
    
    with col2:
        st.markdown("**Advanced Parameters:**")
        compliance = st.checkbox("📋 Compliance & Policy Adherence", value=True)
        communication = st.checkbox("💬 Clear Communication", value=True)
        professionalism = st.checkbox("👔 Professionalism", value=True)
        upsell = st.checkbox("💰 Upsell Opportunities", value=False)
    
    # Store selected parameters
    st.session_state.demo_config["parameters"] = {
        "greeting": greeting,
        "empathy": empathy,
        "problem_solving": problem_solving,
        "closing": closing,
        "compliance": compliance,
        "communication": communication,
        "professionalism": professionalism,
        "upsell": upsell
    }
    
    st.markdown("---")
    
    # File Upload
    st.subheader("4️⃣ Upload Call Recording")
    uploaded_file = st.file_uploader(
        "Upload audio file (WAV, MP3, M4A, etc.)",
        type=stt_engines.SUPPORTED_FORMATS,
        help="Upload the call recording to analyze"
    )
    
    if uploaded_file:
        st.session_state.demo_file = uploaded_file
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Process button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🎯 Analyze Call", use_container_width=True, type="primary"):
                st.session_state.demo_screen = "processing"
                st.rerun()
    else:
        st.info("📤 Please upload a call recording to proceed")

def render_processing_screen():
    """Processing screen with progress indicators."""
    
    st.title("⚙️ Processing Call Analysis")
    st.markdown("---")
    
    # Check API keys
    if not st.secrets.get("OPENAI_API_KEY"):
        st.error("❌ OpenAI API key not configured. Whisper transcription requires OpenAI API access.")
        if st.button("← Back to Setup"):
            st.session_state.demo_screen = "setup"
            st.rerun()
        return
    
    if not st.secrets.get("CLAUDE_API_KEY"):
        st.error("❌ Claude API key not configured. Analysis requires Anthropic API access.")
        if st.button("← Back to Setup"):
            st.session_state.demo_screen = "setup"
            st.rerun()
        return
    
    # Check if we have the file
    if not st.session_state.demo_file:
        st.error("No file uploaded. Please go back to setup.")
        if st.button("← Back to Setup"):
            st.session_state.demo_screen = "setup"
            st.rerun()
        return
    
    file = st.session_state.demo_file
    
    # Processing steps
    st.info("🔄 **Step 1/2:** Transcribing audio with OpenAI Whisper...")
    progress_bar_1 = st.progress(0)
    
    with st.spinner("Running Whisper transcription..."):
        # Transcribe with Whisper
        content = file.getvalue()
        transcription_result = stt_engines.process_audio(file.name, content, "Whisper")
        progress_bar_1.progress(100)
    
    if transcription_result.get("status") == "success":
        st.success(f"✅ Transcription complete! Detected language: {transcription_result.get('language', 'unknown')}")
        
        # Extract transcript
        english_text = transcription_result.get("english_text", "")
        original_text = transcription_result.get("original_text", "")
        
        if english_text and english_text != "Not Available":
            transcript = english_text
        else:
            transcript = original_text
        
        if not transcript or not transcript.strip():
            st.error("❌ Transcription failed: Empty transcript")
            if st.button("← Back to Setup"):
                st.session_state.demo_screen = "setup"
                st.rerun()
            return
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("🔄 **Step 2/2:** Analyzing call quality with Claude Sonnet 4...")
        progress_bar_2 = st.progress(0)
        
        with st.spinner("Running AI analysis..."):
            # Analyze with Claude
            analysis_result = ai_engine.run_comprehensive_analysis(
                transcript=transcript,
                depth="comprehensive",
                custom_rubric=None,
                model="Claude",
                max_retries=2
            )
            progress_bar_2.progress(100)
        
        if analysis_result.get("error"):
            st.error(f"❌ Analysis failed: {analysis_result.get('error')}")
            if st.button("← Back to Setup"):
                st.session_state.demo_screen = "setup"
                st.rerun()
            return
        
        # Store results
        st.session_state.demo_results = {
            "transcription": transcription_result,
            "analysis": analysis_result,
            "transcript": transcript
        }
        
        st.success("✅ Analysis complete!")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Navigate to dashboard
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📊 View Dashboard", use_container_width=True, type="primary"):
                st.session_state.demo_screen = "dashboard"
                st.rerun()
    
    else:
        st.error(f"❌ Transcription failed: {transcription_result.get('error', 'Unknown error')}")
        if st.button("← Back to Setup"):
            st.session_state.demo_screen = "setup"
            st.rerun()

def render_dashboard_screen():
    """Dashboard screen with results visualization."""
    
    # Header with back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back"):
            st.session_state.demo_screen = "setup"
            st.rerun()
    with col2:
        st.title("📊 Call Quality Dashboard")
    
    st.markdown("---")
    
    # Get results
    if not st.session_state.demo_results:
        st.warning("No analysis results available. Please process a call first.")
        return
    
    analysis = st.session_state.demo_results.get("analysis", {})
    transcription = st.session_state.demo_results.get("transcription", {})
    config = st.session_state.demo_config
    
    # Client info banner
    st.markdown(f"""
    <div class="demo-container">
        <h3 style="margin: 0; font-size: 1.5rem;">
            {config.get('client', 'Unknown Client')}
        </h3>
        <p style="margin: 8px 0 0 0; opacity: 0.9;">
            Agent: {config.get('agent_name', 'Unknown')} ({config.get('agent_id', 'N/A')}) | 
            Duration: {transcription.get('duration', 0):.1f}s | 
            Language: {transcription.get('language', 'Unknown')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stats Grid
    overall = analysis.get("overall", {})
    triage = analysis.get("triage", {})
    business = analysis.get("business_outcome", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = overall.get("overall_score", 0)
        st.markdown(f"""
        <div class="dash-card">
            <div class="dash-card-title">Overall Score</div>
            <div class="dash-card-value">{score}</div>
            <div class="dash-card-trend">{"↗️ Above Average" if score >= 75 else "→ Needs Improvement"}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        quality = overall.get("quality_bucket", "Unknown")
        st.markdown(f"""
        <div class="dash-card">
            <div class="dash-card-title">Quality Tier</div>
            <div class="dash-card-value" style="font-size: 1.5rem;">{quality}</div>
            <div class="dash-card-trend">Performance Rating</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sentiment = triage.get("customer_sentiment", "Unknown")
        sentiment_emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}.get(sentiment, "")
        st.markdown(f"""
        <div class="dash-card">
            <div class="dash-card-title">Customer Sentiment</div>
            <div class="dash-card-value" style="font-size: 1.5rem;">{sentiment_emoji} {sentiment}</div>
            <div class="dash-card-trend">Detected Emotion</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        outcome = business.get("business_outcome", "Unknown")
        st.markdown(f"""
        <div class="dash-card">
            <div class="dash-card-title">Business Outcome</div>
            <div class="dash-card-value" style="font-size: 1.5rem;">{outcome}</div>
            <div class="dash-card-trend">Final Result</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Parameter Scores
    st.subheader("📊 Quality Parameter Breakdown")
    
    parameters = overall.get("parameters_summary", [])
    
    if parameters:
        for param in parameters:
            param_name = param.get("name", "Unknown")
            param_score = param.get("score", 0)
            param_reasoning = param.get("reasoning", "No details provided")
            
            # Color code based on score
            if param_score >= 80:
                badge_class = "score-badge-good"
            else:
                badge_class = "score-badge-warning"
            
            st.markdown(f"""
            <div class="dash-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <strong style="color: #111827; font-size: 1rem;">{param_name}</strong>
                    <span class="{badge_class}">{param_score}</span>
                </div>
                <div style="color: #6B7280; font-size: 0.875rem;">{param_reasoning}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No parameter scores available.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Observations & Alerts
    st.subheader("🚨 Key Observations")
    
    observations = overall.get("key_observations", [])
    strengths = overall.get("strengths", [])
    improvements = overall.get("areas_for_improvement", [])
    
    # Show strengths
    if strengths:
        st.markdown("""
        <div class="alert-success">
            <div class="alert-title">✅ Strengths Identified</div>
            <div class="alert-desc">""" + "<br>".join([f"• {s}" for s in strengths[:3]]) + """</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show improvements needed
    if improvements:
        st.markdown("""
        <div class="alert-warning">
            <div class="alert-title">⚠️ Areas for Improvement</div>
            <div class="alert-desc">""" + "<br>".join([f"• {i}" for i in improvements[:3]]) + """</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Action Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 View Full Transcript", use_container_width=True):
            with st.expander("📝 Full Transcript", expanded=True):
                st.text_area(
                    "Transcript",
                    st.session_state.demo_results.get("transcript", ""),
                    height=300
                )
    
    with col2:
        if st.button("📊 Export Report", use_container_width=True):
            st.info("💡 Report export functionality coming soon!")
    
    with col3:
        if st.button("🔄 Analyze Another Call", use_container_width=True):
            # Clear results and go back to setup
            st.session_state.demo_results = {}
            st.session_state.demo_file = None
            st.session_state.demo_screen = "setup"
            st.rerun()

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
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.markdown("### 📑 Navigation")
        page_options = ["Call Analysis", "🔬 Compare Models", "🎯 Production Demo"]
        selected_page = st.radio(
            "Select Page:",
            page_options,
            index=page_options.index(st.session_state.current_page),
            label_visibility="collapsed"
        )
        
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()
        
        st.markdown("---")
        
        # Show available engines
        available_engines = stt_engines.get_available_engines()
        st.info(f"**Available Engines:** {', '.join(available_engines) if available_engines else 'None'}")
        
        # API key status
        st.markdown("### 🔑 API Keys Status")
        if st.secrets.get("OPENAI_API_KEY"):
            st.success("✅ OpenAI (Whisper)")
        if st.secrets.get("GEMINI_API_KEY"):
            st.success("✅ Gemini (Analysis & Diarization)")
        if st.secrets.get("GLADIA_API_KEY"):
            st.success("✅ Gladia")
        if st.secrets.get("DEEPGRAM_API_KEY"):
            st.success("✅ Deepgram")
        if st.secrets.get("ASSEMBLYAI_API_KEY"):
            st.success("✅ AssemblyAI")
        if st.secrets.get("CLAUDE_API_KEY"):
            st.success("✅ Claude")
        
        st.markdown("---")
        
        # Stats
        st.markdown("### 📊 Session Stats")
        st.metric("Files Uploaded", len(st.session_state.files_metadata))
        st.metric("Files Processed", len(st.session_state.transcription_results))
        st.metric("Files Analyzed", len(st.session_state.analysis_results))
        
        st.markdown("---")
        
        if st.button("🗑️ Clear All Data"):
            st.session_state.files_metadata.clear()
            st.session_state.transcription_results.clear()
            st.session_state.analysis_results.clear()
            st.rerun()
    
    # Main content - route to the selected page
    if st.session_state.current_page == "Call Analysis":
        page_call_analysis()
    elif st.session_state.current_page == "🔬 Compare Models":
        page_compare_models()
    elif st.session_state.current_page == "🎯 Production Demo":
        page_production_demo()


if __name__ == "__main__":
    main()

