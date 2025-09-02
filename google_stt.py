import json
import streamlit as st
from google.cloud import speech

def get_google_stt_client():
    """Initialize Google STT client from Streamlit secrets."""
    google_credentials = json.loads(st.secrets["GOOGLE_STT_KEY"])
    return speech.SpeechClient.from_service_account_info(google_credentials)

def transcribe(file_name: str, file_content: bytes, language_code="en-US") -> dict:
    """Transcribe audio with Google STT and return consistent dict."""
    client = get_google_stt_client()

    audio = speech.RecognitionAudio(content=file_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
    )

    response = client.recognize(config=config, audio=audio)

    if not response.results:
        return {"error": "No transcription returned"}

    text = " ".join([r.alternatives[0].transcript for r in response.results])
    return {
        "text": text,
        "segments": [],  # Google basic API doesn’t return detailed segments
        "language": language_code,
        "duration": 0,
    }
