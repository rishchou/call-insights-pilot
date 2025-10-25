# Call Insights Pilot - Refactored

## Overview
This refactored version supports multiple transcription engines with Gemini-2.0-flash-exp for QA analysis.

## Changes Made

### 1. New Transcription Engine Module (`stt_engines.py`)
- **Whisper (OpenAI)**: Original + English translation with Gemini speaker labeling
- **Gladia**: Async API with auto-detection, diarization, and translation
- **Deepgram**: Nova-2 Phonecall model optimized for call center audio
- **AssemblyAI**: Auto language detection with speaker diarization

All engines return a unified format with:
- `status`: "success" or "error"
- `engine`: Engine name
- `original_text`: Original transcript
- `english_text`: English translation (if available)
- `segments`: List of segments with speaker labels and timestamps
- `language`: Detected language
- `duration`: Audio duration in seconds

### 2. Simplified AI Engine (`ai_engine.py`)
- **Removed A/B testing**: Now uses only Gemini-2.0-flash-exp for all analysis
- **Streamlined functions**:
  - `run_initial_triage()`: Call purpose, category, sentiment
  - `run_business_outcome()`: Outcome, compliance, risks
  - `run_parameter_scoring()`: Score parameters based on rubric
  - `run_comprehensive_analysis()`: Full analysis with all stages

### 3. CSV Export Module (`csv_export.py`)
- **Detailed parameter rows**: One row per parameter with full context
- **Summary export**: High-level metrics across all calls
- **Multi-file export**: Batch export multiple analyses
- Includes:
  - Transcription context (file, engine, language, duration)
  - Full transcripts (original, English, labeled)
  - Parameter scores with justification and evidence
  - Coaching opportunities and improvement impact

### 4. Updated Streamlit UI (`streamlit_openai_call_insights_app.py`)
- **Transcription engine dropdown**: Select from available engines
- **Three tabs**:
  1. **Upload & Process**: Select engine, upload files, process audio
  2. **Analyze Results**: Configure depth, custom rubric, run Gemini analysis
  3. **Export CSV**: Download summary or detailed parameter CSV
- **Analysis depth options**: Quick Scan, Standard Analysis, Deep Dive
- **Custom rubrics**: Sales Outbound, Banking Support, Technical Support
- **Rich result display**: Triage, business outcome, scores, coaching

## Setup

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Configure API Keys
Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your-openai-key"
GEMINI_API_KEY = "your-gemini-key"
GLADIA_API_KEY = "your-gladia-key"  # Optional
DEEPGRAM_API_KEY = "your-deepgram-key"  # Optional
ASSEMBLYAI_API_KEY = "your-assemblyai-key"  # Optional
```

At minimum, you need:
- **OPENAI_API_KEY** (for Whisper)
- **GEMINI_API_KEY** (for analysis and speaker labeling)

### 3. Run the App
```powershell
cd call-insights-pilot
streamlit run streamlit_openai_call_insights_app.py
```

## Usage Flow

1. **Select Transcription Engine**
   - Choose from available engines in the dropdown
   - Engine availability depends on configured API keys

2. **Upload Audio Files**
   - Drag and drop audio files (mp3, wav, m4a, ogg, flac, webm)
   - Files are processed with the selected engine
   - Results show transcripts, speaker labels, and duration

3. **Run Analysis**
   - Select files to analyze
   - Choose analysis depth (Quick/Standard/Deep)
   - Optionally add custom rubric for industry-specific parameters
   - Click "Run AI Analysis with Gemini"

4. **View Results**
   - Expandable panels for each file
   - Triage info (purpose, category, sentiment)
   - Business outcome (result, compliance, risks)
   - Overall quality score and bucket
   - Top coaching opportunities
   - Detailed parameter scores

5. **Export CSV**
   - **Summary CSV**: High-level metrics for all files
   - **Detailed CSV**: Parameter rows with evidence and coaching

## Key Features

### Transcription
- **Multiple engines**: Compare results across providers
- **Auto language detection**: Supports multiple languages
- **Speaker diarization**: Agent vs Customer labeling
- **English translation**: Available for Whisper and Gladia

### Analysis (Gemini-2.0-flash-exp)
- **Call triage**: Purpose, category, sentiment, complexity
- **Business outcomes**: Result, compliance check, risk identification
- **Parameter scoring**: 0-100 scores with behavioral anchors
- **Coaching opportunities**: Actionable improvement suggestions
- **Overall metrics**: Weighted score and quality bucket

### Export
- **Summary view**: Quick overview of all calls
- **Detailed parameters**: Every score with full justification
- **Evidence-based**: Primary evidence with before/after context
- **Coaching-ready**: Specific, actionable recommendations

## Parameter Rubrics

### Quick Scan (4 parameters)
- Opening, Issue Identification, Resolution, Closing

### Standard Analysis (6 parameters)
- Greeting, Empathy, Resolution, Compliance, Professionalism, Communication Clarity

### Deep Dive (8 parameters)
- All standard + Active Listening, Product Knowledge, Call Control

### Custom Rubrics
- **Sales Outbound**: EMI offer, urgency creation, objection handling
- **Banking Support**: Identity verification, data protection, regulatory compliance
- **Technical Support**: Troubleshooting methodology, technical accuracy

## File Structure

```
call-insights-pilot/
├── stt_engines.py              # Unified transcription engine interface
├── ai_engine.py                # Gemini-based QA analysis
├── csv_export.py               # CSV export functionality
├── streamlit_openai_call_insights_app.py  # Main Streamlit UI
├── requirements.txt            # Python dependencies
└── README_REFACTOR.md          # This file
```

## API Key Requirements

| Feature | Required API Key |
|---------|------------------|
| Whisper transcription | OPENAI_API_KEY |
| Gladia transcription | GLADIA_API_KEY |
| Deepgram transcription | DEEPGRAM_API_KEY |
| AssemblyAI transcription | ASSEMBLYAI_API_KEY |
| QA Analysis | GEMINI_API_KEY |
| Speaker labeling | GEMINI_API_KEY |

## Notes

- **Caching**: Results are cached by file hash for faster re-processing
- **PII scrubbing**: Basic patterns for phone, email, card numbers
- **Error handling**: Graceful fallbacks with retry logic
- **Performance**: Parallel processing where possible

## Troubleshooting

### No engines available
- Check that at least one transcription API key is configured
- Verify API keys are valid and have sufficient credits

### Analysis fails
- Ensure GEMINI_API_KEY is configured
- Check that transcripts are not empty
- Try with a smaller file or shorter audio

### CSV export empty
- Make sure analysis has been run successfully
- Check that files are selected in the analyze tab

## Support

For issues or questions, please refer to the original codebase documentation.
