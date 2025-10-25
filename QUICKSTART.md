# Quick Start Guide

## 1. Setup (5 minutes)

### Install Dependencies
```powershell
cd c:\Users\Public\Documents\QA_project\call-insights-pilot
pip install -r requirements.txt
```

### Configure API Keys
1. Copy the secrets template:
   ```powershell
   copy .streamlit\secrets.toml.template .streamlit\secrets.toml
   ```

2. Edit `.streamlit\secrets.toml` with your API keys:
   ```toml
   OPENAI_API_KEY = "sk-your-actual-key"
   GEMINI_API_KEY = "your-actual-key"
   ```

   Optional (for additional engines):
   ```toml
   GLADIA_API_KEY = "your-actual-key"
   DEEPGRAM_API_KEY = "your-actual-key"
   ASSEMBLYAI_API_KEY = "your-actual-key"
   ```

## 2. Run the App
```powershell
streamlit run streamlit_openai_call_insights_app.py
```

The app will open in your browser at `http://localhost:8501`

## 3. First Analysis (3 steps)

### Step 1: Upload & Process
1. Select transcription engine from dropdown (e.g., "Whisper")
2. Drag and drop an audio file (mp3, wav, m4a, etc.)
3. Wait for processing to complete ✅

### Step 2: Analyze Results
1. Switch to "Analyze Results" tab
2. Select your uploaded file
3. Choose analysis depth:
   - **Quick Scan**: Fast, 4 parameters
   - **Standard Analysis**: Balanced, 6 parameters (recommended)
   - **Deep Dive**: Comprehensive, 8 parameters
4. Click "🚀 Run AI Analysis with Gemini"
5. View results:
   - Call triage (purpose, category, sentiment)
   - Business outcome (result, compliance, risks)
   - Overall quality score
   - Top coaching opportunities
   - Detailed parameter scores

### Step 3: Export CSV
1. Switch to "Export CSV" tab
2. Click "📊 Export Summary CSV" for high-level overview
3. Click "📋 Export Detailed Parameters CSV" for full analysis
4. Download files appear in your browser

## 4. Understanding Results

### Call Triage
- **Purpose**: Why the customer called
- **Category**: Type of call (Query, Complaint, Sales, etc.)
- **Sentiment**: Customer mood (1-10 scale)
- **Summary**: 3-sentence overview

### Business Outcome
- **Outcome**: What happened (Issue_Resolved, Sale_Completed, etc.)
- **Compliance**: Did agent follow procedures?
- **Risk**: Any legal or reputational risks identified?

### Quality Score
- **Score**: 0-100 weighted average across all parameters
- **Quality Bucket**: Excellent (90+), Good (80-89), Average (70-79), etc.
- **Parameters Scored**: Number of parameters evaluated
- **Coaching Opportunities**: Top 3 actionable improvements

### Parameter Scores
Each parameter shows:
- **Score**: 0-100 based on behavioral anchors
- **Confidence**: 1-10 AI confidence in the score
- **Justification**: Why this score was given
- **Primary Evidence**: Exact quote from transcript
- **Coaching**: Specific improvement suggestion

## 5. CSV Exports

### Summary CSV
One row per call with:
- File name, engine, language, duration
- Call purpose, category, sentiment
- Business outcome, compliance
- Overall score, quality bucket
- Parameters needing attention

Perfect for Excel pivot tables and dashboards!

### Detailed Parameters CSV
One row per parameter per call with:
- All summary info
- Full transcripts (original, English, labeled)
- Parameter name and score
- Justification with evidence
- Context before and after evidence
- Coaching opportunity with impact
- Severity for low scores

Perfect for deep analysis and agent coaching!

## 6. Tips

### For Best Results
- Use clear audio with minimal background noise
- Ensure speakers don't talk over each other too much
- Longer calls (2+ minutes) give more analysis data
- Try different engines to compare quality

### Transcription Engine Comparison
- **Whisper**: Best for translation, supports 90+ languages
- **Gladia**: Fast async processing, good for high volume
- **Deepgram**: Best for phone call audio quality
- **AssemblyAI**: Strong language detection

### Analysis Depths
- **Quick Scan**: Use for rapid screening (< 30 sec per call)
- **Standard Analysis**: Balanced view (~ 60 sec per call)
- **Deep Dive**: Thorough evaluation (~ 90 sec per call)

### Custom Rubrics
Add industry-specific parameters:
- **Sales Outbound**: EMI offers, urgency, objections
- **Banking Support**: Identity verification, data protection
- **Technical Support**: Troubleshooting, technical accuracy

## 7. Troubleshooting

### "No engines available"
→ Check API keys in `.streamlit\secrets.toml`

### "Analysis failed"
→ Ensure GEMINI_API_KEY is configured
→ Check transcript is not empty

### "Polling timeout"
→ File may be too large or API is slow
→ Try a smaller file or wait and retry

### CSV is empty
→ Run analysis first before exporting
→ Check files are selected in analyze tab

## Need Help?

Check `README_REFACTOR.md` for detailed documentation.

Happy analyzing! 🎉
