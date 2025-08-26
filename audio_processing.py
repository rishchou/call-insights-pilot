import os
import json
import streamlit as st
from openai import OpenAI

# Initialize the OpenAI client
# It's good practice to handle the case where the key might be missing.
api_key = st.secrets.get("OPENAI_API_KEY")
if api_key:
    oai_client = OpenAI(api_key=api_key)
else:
    oai_client = None
    st.warning("OpenAI API key not found. Transcription will not work.")

@st.cache_data(show_spinner="Transcribing audio...")
def get_transcript(file_name: str, file_content: bytes) -> dict:
    """
    Transcribes and translates an audio file using OpenAI Whisper.
    Returns a dictionary with original, translated, and speaker-labeled transcripts.
    """
    if not oai_client:
        return {"error": "OpenAI client is not initialized."}

    try:
        # Create a temporary file-like object for the API
        audio_file = (file_name, file_content)

        # 1. Transcription in the original language
        transcript_orig = oai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )

        # 2. Translation to English
        transcript_en = oai_client.audio.translations.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
        
        # In a full implementation, speaker diarization would be called here.
        # For now, we return the raw transcription results.
        return {
            "status": "success",
            "original_transcript": transcript_orig.text,
            "english_transcript": transcript_en.text,
            "segments": transcript_en.segments # For detailed analysis
        }

    except Exception as e:
        return {"error": f"An error occurred during transcription: {e}"}
