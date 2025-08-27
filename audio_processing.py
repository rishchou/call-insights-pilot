import os
import json
import streamlit as st
from openai import OpenAI
import google.generativeai as genai

# (The top part of the file with client initializations and helpers remains the same)
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

# (Helper functions _json_guard and _label_speakers remain the same)
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
    if not gemini_model:
        return [{"speaker": "UNKNOWN", "id": seg.id, "text": seg.text} for seg in segments]

    prompt_segments = [{"id": seg.id, "text": seg.text} for seg in segments]
    prompt = f"""
    You are a conversation analyst...
    (The rest of the speaker labeling prompt is unchanged)
    ...Return ONLY the JSON object.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        labels_data = _json_guard(response.text)
        label_map = {item['id']: item['speaker'] for item in labels_data.get('labels', [])}
        
        labeled_segments = []
        for seg in segments:
            segment_dict = {
                "id": seg.id, "seek": seg.seek, "start": seg.start, "end": seg.end,
                "text": seg.text, "tokens": seg.tokens, "temperature": seg.temperature,
                "avg_logprob": seg.avg_logprob, "compression_ratio": seg.compression_ratio,
                "no_speech_prob": seg.no_speech_prob, "speaker": label_map.get(seg.id, 'UNKNOWN')
            }
            labeled_segments.append(segment_dict)
        return labeled_segments
        
    except Exception as e:
        st.error(f"Speaker labeling failed: {e}")
        return [{"speaker": "ERROR", "id": seg.id, "text": seg.text} for seg in segments]

# --- Main processing function is now simpler ---
# REMOVED @st.cache_data decorator
def process_audio(file_name: str, file_content: bytes) -> dict:
    if not oai_client:
        return {"error": "OpenAI client is not initialized."}
    try:
        audio_file = (file_name, file_content)
        # (The rest of the function remains the same)
        st.write("Transcribing in original language...")
        transcript_orig_res = oai_client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="verbose_json"
        )
        st.write("Translating to English...")
        translation_res = oai_client.audio.translations.create(
            model="whisper-1", file=audio_file, response_format="verbose_json"
        )
        st.write("Labeling speakers...")
        labeled_segments = _label_speakers(translation_res.segments)

        return {
            "status": "success",
            "original_transcript": transcript_orig_res.text,
            "english_transcript": translation_res.text,
            "segments": labeled_segments
        }
    except Exception as e:
        st.error(f"An error occurred during audio processing: {e}")
        return {"error": f"Audio processing failed: {e}"}
