import os
import json
import streamlit as st
from openai import OpenAI
import google.generativeai as genai

# --- Initialize API Clients ---
oai_client = None
if api_key := st.secrets.get("OPENAI_API_KEY"):
    oai_client = OpenAI(api_key=api_key)
else:
    st.warning("OpenAI API key not found. Transcription will not work.")

gemini_model = None
if api_key := st.secrets.get("GEMINI_API_KEY"):
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(
        "gemini-1.5-pro",
        generation_config={"response_mime_type": "application/json"}
    )
else:
    st.warning("Gemini API key not found. Speaker labeling will not work.")

# --- Helper function for JSON parsing ---
def _json_guard(text_response: str) -> dict:
    try:
        start = text_response.find('{')
        end = text_response.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text_response[start:end+1])
    except (json.JSONDecodeError, IndexError):
        return {"error": "Failed to parse AI response as JSON.", "raw_response": text_response}
    return {"error": "No valid JSON object found in the response.", "raw_response": text_response}

def _label_speakers(segments: list) -> list:
    """Uses Gemini to label speakers for each transcript segment."""
    if not gemini_model:
        return [{"speaker": "UNKNOWN", "id": seg.id, "text": seg.text} for seg in segments]

    # CORRECTED: Use attribute access (seg.id, seg.text) instead of dictionary access
    prompt_segments = [{"id": seg.id, "text": seg.text} for seg in segments]
    
    prompt = f"""
    You are a conversation analyst. Your task is to identify the speaker for each segment of a customer service call transcript.
    Label each segment as either 'AGENT' or 'CUSTOMER'.
    
    Use conversational cues to identify the speaker. For example:
    - The 'AGENT' usually starts with a greeting, asks for verification, explains policies, and provides solutions.
    - The 'CUSTOMER' usually states a problem, asks questions, provides personal information, and expresses opinions.

    Analyze the following list of transcript segments and return a JSON object with a key "labels", which is a list.
    Each item in the list should be an object with the "id" of the segment and the identified "speaker" ('AGENT' or 'CUSTOMER').

    Transcript Segments:
    {json.dumps(prompt_segments, indent=2)}

    Return ONLY the JSON object.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        labels_data = _json_guard(response.text)
        
        label_map = {item['id']: item['speaker'] for item in labels_data.get('labels', [])}
        
        labeled_segments = []
        for seg in segments:
            # CORRECTED: Create a dictionary from the object's attributes
            segment_dict = {
                "id": seg.id,
                "seek": seg.seek,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "tokens": seg.tokens,
                "temperature": seg.temperature,
                "avg_logprob": seg.avg_logprob,
                "compression_ratio": seg.compression_ratio,
                "no_speech_prob": seg.no_speech_prob,
                "speaker": label_map.get(seg.id, 'UNKNOWN')
            }
            labeled_segments.append(segment_dict)
        return labeled_segments
        
    except Exception as e:
        st.error(f"Speaker labeling failed: {e}")
        return [{"speaker": "ERROR", "id": seg.id, "text": seg.text} for seg in segments]


@st.cache_data(show_spinner="Processing audio...")
def process_audio(file_name: str, file_content: bytes) -> dict:
    """
    Performs the full audio processing pipeline:
    1. Transcribes in the original language.
    2. Translates to English.
    3. Performs speaker diarization on the English transcript.
    """
    if not oai_client:
        return {"error": "OpenAI client is not initialized."}

    try:
        audio_file = (file_name, file_content)

        st.write("Transcribing in original language...")
        transcript_orig_res = oai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
        original_transcript_text = transcript_orig_res.text

        st.write("Translating to English...")
        translation_res = oai_client.audio.translations.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
        english_transcript_text = translation_res.text
        english_segments = translation_res.segments

        st.write("Labeling speakers...")
        labeled_segments = _label_speakers(english_segments)

        return {
            "status": "success",
            "original_transcript": original_transcript_text,
            "english_transcript": english_transcript_text,
            "segments": labeled_segments
        }

    except Exception as e:
        st.error(f"An error occurred during audio processing: {e}")
        return {"error": f"Audio processing failed: {e}"}
