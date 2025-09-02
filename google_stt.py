import json
import streamlit as st
from google.cloud import speech

def get_google_stt_client():
    """Initialize Google STT client from Streamlit secrets."""
    google_credentials = json.loads(st.secrets["GOOGLE_STT_KEY"])
    return speech.SpeechClient.from_service_account_info(google_credentials)

def transcribe(file_name: str, file_content: bytes, language_code="en-US") -> dict:
    """Transcribe audio with Google STT and return consistent dict for main app."""
    try:
        client = get_google_stt_client()

        audio = speech.RecognitionAudio(content=file_content)
        config = speech.RecognitionConfig(
            language_code=language_code,
            enable_automatic_punctuation=True
            # Note: omit encoding/sample_rate for Google to auto-detect if possible
        )

        response = client.recognize(config=config, audio=audio)

        if not response.results:
            return {"status": "failed", "error": "No transcription returned"}

        transcript = " ".join([r.alternatives[0].transcript for r in response.results])

        return {
            "status": "success",
            "english_transcript": transcript,
            "segments": [],       # Google basic API doesn’t return segments
            "language": language_code,
            "duration": 0,        # Could fill in later if needed
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}
