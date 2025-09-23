# audio_processing.py
import io
import json
import hashlib
import time
import re
from functools import lru_cache
from typing import Dict, List, Optional

import streamlit as st
from openai import OpenAI
import google.generativeai as genai

# SDKs for the new engines
# from gladia import GladiaClient
import assemblyai
from deepgram import DeepgramClient, PrerecordedOptions

# ======================================================================================
# CONSTANTS
# ======================================================================================
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
SUPPORTED_FORMATS = ['mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm']
WHISPER_MODEL = "whisper-1"
GEMINI_MODEL = "gemini-1.5-pro"
GEMINI_TIMEOUT = 30  # seconds

# ======================================================================================
# CLIENT INITIALIZATION & UTILITIES
# ======================================================================================
@lru_cache(maxsize=1)
def _get_openai_client() -> Optional[OpenAI]:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key: return None
    return OpenAI(api_key=api_key)

@lru_cache(maxsize=1)
def _get_gemini_model():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL, generation_config={"response_mime_type": "application/json"})

def _calculate_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()

def _scrub_pii(text: str) -> str:
    if not isinstance(text, str): return text
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD_REDACTED]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL_REDACTED]', text)
    return text

def _validate_audio_file(file_name: str, file_content: bytes) -> Dict[str, any]:
    vr = {"valid": True, "errors": [], "warnings": [], "file_info": {}}
    size = len(file_content)
    vr["file_info"]["size_mb"] = round(size / (1024 * 1024), 2)
    if size > MAX_FILE_SIZE:
        vr["valid"] = False
        vr["errors"].append(f"File size exceeds limit")
    return vr

# ======================================================================================
# ENGINE 1: Whisper + Gemini (Your Baseline)
# ======================================================================================
def _label_speakers_batch(segments: List) -> List[Dict]:
    """Labels speakers from text segments using Gemini."""
    model = _get_gemini_model()
    if not model:
        st.warning("Gemini model not available. Falling back to alternating speaker labels.")
        labeled_fallback = []
        for i, seg in enumerate(segments):
            labeled_fallback.append({
                "id": i, "speaker": "AGENT" if i % 2 == 0 else "CUSTOMER",
                "text": getattr(seg, 'text', ''), "start": getattr(seg, 'start', 0), "end": getattr(seg, 'end', 0)
            })
        return labeled_fallback

    all_labeled = []
    # (Full original logic for batching and prompting Gemini)
    for i in range(0, len(segments), 20):
        batch = segments[i:i + 20]
        prompt_segments = [{"id": i+j, "text": getattr(s, 'text', '')} for j, s in enumerate(batch)]
        prompt = f"""You are a conversation analyst. Identify the speaker for each segment as "AGENT" or "CUSTOMER".
        Rules: The first speaker is usually the AGENT. "Thank you for calling" indicates AGENT. "I have a problem" indicates CUSTOMER.
        Segments: {json.dumps(prompt_segments)}
        Return ONLY a single valid JSON object: {{"labels":[{{"id": <int>, "speaker": "AGENT|CUSTOMER"}}]}}"""
        
        label_map = {}
        try:
            resp = model.generate_content(prompt, request_options={"timeout": GEMINI_TIMEOUT})
            data = json.loads(getattr(resp, "text", "") or "{}")
            if "labels" in data:
                for item in data["labels"]:
                    label_map[item["id"]] = item["speaker"]
        except Exception as e:
            st.warning(f"Gemini speaker labeling failed for a batch: {e}")

        for j, seg in enumerate(batch):
            idx = i + j
            labeled_segment = {
                "id": idx,
                "speaker": label_map.get(idx, "AGENT" if idx % 2 == 0 else "CUSTOMER"),
                "text": getattr(seg, 'text', ''),
                "start": getattr(seg, 'start', 0),
                "end": getattr(seg, 'end', 0)
            }
            all_labeled.append(labeled_segment)
        return all_labeled

def _process_with_whisper_gemini(file_name: str, file_content: bytes) -> Dict:
    """Original pipeline using Whisper STT/Translate and Gemini diarization."""
    try:
        oai_client = _get_openai_client()
        if not oai_client: raise ValueError("OpenAI client not available.")
        
        audio_file = io.BytesIO(file_content); audio_file.name = file_name
        original_transcript = oai_client.audio.transcriptions.create(model=WHISPER_MODEL, file=audio_file, response_format="verbose_json")
        audio_file.seek(0)
        english_transcript = oai_client.audio.translations.create(model=WHISPER_MODEL, file=audio_file, response_format="verbose_json")
        
        labeled_segments = _label_speakers_batch(english_transcript.segments or [])
        
        return {
            "status": "success", "engine": "whisper_gemini",
            "original_text": original_transcript.text,
            "english_text": english_transcript.text,
            "segments": labeled_segments,
            "language": original_transcript.language,
            "duration": english_transcript.duration, "error_message": None
        }
    except Exception as e:
        return {"status": "error", "engine": "whisper_gemini", "error_message": str(e)}

# ======================================================================================
# ENGINE 2: Gladia (All-in-One)
# ======================================================================================
# def _process_with_gladia(file_name: str, file_content: bytes) -> Dict:
#    """Processes audio using Gladia's full potential."""
#    try:
#        api_key = st.secrets.get("GLADIA_API_KEY")
#        if not api_key: raise ValueError("Gladia API key not found.")
#        client = GladiaClient(api_key)
#        response = client.audio.transcription.create(audio_bytes=file_content, diarization=True, translation=True, target_translation_language="en")
#
#        segments_en = []
#        if response.translation and response.translation.utterances:
#            for utt in response.translation.utterances:
#               segments_en.append({"start": utt.start, "end": utt.end, "text": utt.transcription, "speaker": f"SPEAKER_{utt.speaker}"})
#        
#        return {
#            "status": "success", "engine": "gladia",
#            "original_text": response.transcription.full_transcript,
#            "english_text": response.translation.full_transcript,
#            "segments": segments_en,
#            "language": response.language, "duration": response.duration, "error_message": None
#        }
#    except Exception as e:
#        return {"status": "error", "engine": "gladia", "error_message": str(e)}

# ======================================================================================
# ENGINE 3: AssemblyAI (All-in-One)
# ======================================================================================
def _process_with_assemblyai(file_name: str, file_content: bytes) -> Dict:
    """Processes audio using AssemblyAI (transcription + diarization)."""
    try:
        api_key = st.secrets.get("ASSEMBLYAI_API_KEY")
        if not api_key: raise ValueError("AssemblyAI API key not found.")
        assemblyai.settings.api_key = api_key
        transcriber = assemblyai.Transcriber()
        config = assemblyai.TranscriptionConfig(speaker_labels=True, language_detection=True)
        transcript = transcriber.transcribe(io.BytesIO(file_content), config)

        if transcript.status == assemblyai.TranscriptStatus.error: raise Exception(transcript.error)
            
        segments = []
        if transcript.utterances:
            for utt in transcript.utterances:
                segments.append({"start": utt.start / 1000.0, "end": utt.end / 1000.0, "text": utt.text, "speaker": utt.speaker})
        
        return {
            "status": "success", "engine": "assemblyai",
            "original_text": transcript.text,
            "english_text": "N/A - Requires separate translation step",
            "segments": segments,
            "language": str(transcript.language_code), "duration": transcript.audio_duration, "error_message": None
        }
    except Exception as e:
        return {"status": "error", "engine": "assemblyai", "error_message": str(e)}

# ======================================================================================
# ENGINE 4: Deepgram (All-in-One)
# ======================================================================================
def _process_with_deepgram(file_name: str, file_content: bytes) -> Dict:
    """Processes audio using Deepgram (transcription, diarization, translation)."""
    try:
        api_key = st.secrets.get("DEEPGRAM_API_KEY")
        if not api_key: raise ValueError("Deepgram API key not found.")
        deepgram = DeepgramClient(api_key)
        payload = {"buffer": file_content}
        options = PrerecordedOptions(model="nova-2", smart_format=True, utterances=True, diarize=True, translation="en")
        response = deepgram.listen.prerecorded.v("1").transcribe_source(payload, options)
        
        result = response.to_dict()
        ch = result.get("results", {}).get("channels", [{}])[0]
        alt = ch.get("alternatives", [{}])[0]
        
        segments_en = []
        if result.get("results", {}).get("utterances"):
            for utt in result["results"]["utterances"]:
                translated_text = utt.get("translation", utt.get("transcript"))
                segments_en.append({"start": utt["start"], "end": utt["end"], "text": translated_text, "speaker": f"SPEAKER_{utt['speaker']}"})
        
        return {
            "status": "success", "engine": "deepgram",
            "original_text": "N/A - Translation replaces original in single call",
            "english_text": alt.get("transcript", ""),
            "segments": segments_en,
            "language": ch.get("detected_language", "unknown"),
            "duration": result.get("metadata", {}).get("duration", 0), "error_message": None
        }
    except Exception as e:
        return {"status": "error", "engine": "deepgram", "error_message": str(e)}

# ======================================================================================
# CACHED ROUTER FUNCTION
# ======================================================================================
@st.cache_data(show_spinner=False)
def _process_audio_cached(file_hash: str, file_name: str, file_content: bytes, engine: str) -> Dict:
    """Cached router that calls the correct end-to-end processing engine."""
    validation = _validate_audio_file(file_name, file_content)
    if not validation["valid"]:
        return {"status": "error", "error_message": ", ".join(validation["errors"])}

    engine_map = {
        "whisper_gemini": _process_with_whisper_gemini,
        # "gladia": _process_with_gladia,
        "assemblyai": _process_with_assemblyai,
        "deepgram": _process_with_deepgram
    }
    process_function = engine_map.get(engine)

    if not process_function:
        return {"status": "error", "engine": engine, "error_message": f"Unknown engine: {engine}"}
    
    result = process_function(file_name, file_content)
    
    if result.get("status") == "success":
        result["original_text"] = _scrub_pii(result.get("original_text", ""))
        result["english_text"] = _scrub_pii(result.get("english_text", ""))
        for segment in result.get("segments", []):
            segment["text"] = _scrub_pii(segment.get("text", ""))
    
    return result

# ======================================================================================
# PUBLIC API FUNCTIONS
# ======================================================================================
def process_audio(file_name: str, file_content: bytes, engine: str) -> Dict:
    """Main audio processing function."""
    unique_hash = _calculate_file_hash(file_content) + f"_{engine}"
    return _process_audio_cached(unique_hash, file_name, file_content, engine)

def process_audio_with_progress(file_name: str, file_content: bytes, engine: str) -> Dict:
    """Process audio with Streamlit progress indicators."""
    progress_bar = st.progress(0, text=f"[{engine}] Starting...")
    result = process_audio(file_name, file_content, engine)
    if result.get("status") == "success":
        progress_bar.progress(100, text=f"[{engine}] Complete!")
    else:
        st.error(f"[{engine}] Failed: {result.get('error_message', 'Unknown error')}")
        progress_bar.progress(100, text=f"[{engine}] Failed!")
    time.sleep(1)
    progress_bar.empty()
    return result
