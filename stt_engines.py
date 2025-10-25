# stt_engines.py - Unified STT Engine Interface with Multiple Providers
"""
This module provides a unified interface for multiple Speech-to-Text engines:
- Whisper (OpenAI)
- Gladia
- Deepgram
- AssemblyAI

All engines return a consistent format with speaker diarization using Gemini.
"""

import os
import io
import json
import time
import hashlib
import re
import requests
from typing import Dict, List, Optional
from functools import lru_cache

import streamlit as st
from openai import OpenAI
import google.generativeai as genai


# ======================================================================================
# CONSTANTS
# ======================================================================================

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
SUPPORTED_FORMATS = ['mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm']
GEMINI_TIMEOUT = 30
MAX_RETRIES = 3


# ======================================================================================
# API CLIENT INITIALIZATION
# ======================================================================================

@lru_cache(maxsize=1)
def _get_openai_client() -> Optional[OpenAI]:
    """Lazy initialization of OpenAI client."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("OpenAI API key not found.")
        return None
    return OpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def _get_gemini_client():
    """Lazy initialization of Gemini model for speaker labeling."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.warning("Gemini API key not found.")
        return None
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        "gemini-2.0-flash-exp",
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json"
        }
    )


def _get_api_key(service: str) -> Optional[str]:
    """Get API key for specified service from secrets."""
    key_mapping = {
        "gladia": "GLADIA_API_KEY",
        "deepgram": "DEEPGRAM_API_KEY",
        "assemblyai": "ASSEMBLYAI_API_KEY"
    }
    return st.secrets.get(key_mapping.get(service))


# ======================================================================================
# UTILITY FUNCTIONS
# ======================================================================================

def _calculate_file_hash(file_content: bytes) -> str:
    """Calculate hash of file content for caching."""
    return hashlib.md5(file_content).hexdigest()


def _scrub_pii(text: str) -> str:
    """Basic PII scrubbing."""
    if not isinstance(text, str):
        return text
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
    text = re.sub(r'\b784-\d{4}-\d{7}-\d{1}\b', '[EMIRATES_ID]', text)
    return text


def _json_guard(text: str) -> dict:
    """Safely parse JSON response."""
    if not text or not isinstance(text, str):
        return {"error": "Empty response"}
    s = text.strip().strip('`')
    try:
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s)
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(s[i:j+1])
    except Exception as e:
        return {"error": f"JSON parse error: {e}"}
    return {"error": "No JSON object found"}


def _validate_audio_file(file_name: str, file_content: bytes) -> Dict:
    """Validate audio file before processing."""
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "file_info": {}
    }
    
    file_size = len(file_content)
    validation_result["file_info"]["size_mb"] = round(file_size / (1024 * 1024), 2)
    
    if file_size > MAX_FILE_SIZE:
        validation_result["valid"] = False
        validation_result["errors"].append(
            f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit"
        )
    
    if file_size < 1024:
        validation_result["valid"] = False
        validation_result["errors"].append("File too small")
    
    ext = file_name.lower().split('.')[-1] if '.' in file_name else ''
    validation_result["file_info"]["format"] = ext
    
    if ext not in SUPPORTED_FORMATS:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Unsupported format: {ext}")
    
    return validation_result


# ======================================================================================
# SPEAKER LABELING WITH GEMINI
# ======================================================================================

def _label_speakers_with_gemini(segments: List, batch_size: int = 20) -> List[Dict]:
    """Label speakers using Gemini with batching."""
    model = _get_gemini_client()
    if not model:
        # Fallback to alternating pattern
        labeled = []
        for i, seg in enumerate(segments):
            text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
            start = seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0)
            end = seg.get("end", 0) if isinstance(seg, dict) else getattr(seg, "end", 0)
            labeled.append({
                "id": i,
                "speaker": "AGENT" if i % 2 == 0 else "CUSTOMER",
                "text": text,
                "start": start,
                "end": end
            })
        return labeled
    
    all_labeled = []
    n = len(segments)
    
    for batch_start in range(0, n, batch_size):
        batch = segments[batch_start:batch_start + batch_size]
        
        prompt_segments = []
        id_map = {}
        for j, seg in enumerate(batch):
            local_id = batch_start + j
            id_map[local_id] = batch_start + j
            text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
            prompt_segments.append({"id": local_id, "text": text})
        
        prompt = f"""You are a conversation analyst. Identify the speaker for each segment as "AGENT" or "CUSTOMER".
Rules:
- First speaker is usually AGENT
- "Thank you for calling" → AGENT
- "I have a problem" → CUSTOMER

Segments: {json.dumps(prompt_segments)}
Return ONLY valid JSON: {{"labels":[{{"id": 0, "speaker": "AGENT"}}]}}"""
        
        label_map = {}
        for attempt in range(2):
            try:
                resp = model.generate_content(prompt, request_options={"timeout": GEMINI_TIMEOUT})
                data = _json_guard(getattr(resp, "text", "") or "")
                if "error" not in data:
                    for item in data.get("labels", []):
                        if "id" in item and "speaker" in item:
                            label_map[int(item["id"])] = item["speaker"]
                    break
            except Exception:
                if attempt == 1:
                    break
                time.sleep(1)
        
        for local_id, global_idx in id_map.items():
            seg = segments[global_idx]
            text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
            start = seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0)
            end = seg.get("end", 0) if isinstance(seg, dict) else getattr(seg, "end", 0)
            speaker = label_map.get(local_id, "AGENT" if global_idx % 2 == 0 else "CUSTOMER")
            
            all_labeled.append({
                "id": local_id,
                "speaker": speaker,
                "text": text,
                "start": start,
                "end": end
            })
    
    return all_labeled


# ======================================================================================
# WHISPER ENGINE
# ======================================================================================

def process_with_whisper(file_name: str, file_content: bytes) -> Dict:
    """Process audio with OpenAI Whisper."""
    client = _get_openai_client()
    if not client:
        return {"status": "error", "engine": "whisper", "error_message": "OpenAI client not available"}
    
    try:
        audio_file = io.BytesIO(file_content)
        audio_file.name = file_name
        
        # Original transcription
        original = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
        
        # English translation
        audio_file.seek(0)
        english = client.audio.translations.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
        
        # Label speakers
        labeled_segments = _label_speakers_with_gemini(english.segments or [])
        
        return {
            "status": "success",
            "engine": "whisper",
            "original_text": _scrub_pii(original.text),
            "english_text": _scrub_pii(english.text),
            "segments": labeled_segments,
            "language": original.language,
            "duration": english.duration,
            "error_message": None
        }
    except Exception as e:
        return {"status": "error", "engine": "whisper", "error_message": str(e)}


# ======================================================================================
# GLADIA ENGINE
# ======================================================================================

def process_with_gladia(file_name: str, file_content: bytes) -> Dict:
    """Process audio with Gladia async API."""
    api_key = _get_api_key("gladia")
    if not api_key:
        return {"status": "error", "engine": "gladia", "error_message": "Gladia API key not found"}
    
    try:
        gladia_url = "https://api.gladia.io/v2"
        headers_file = {"x-gladia-key": api_key}
        headers_json = {"x-gladia-key": api_key, "Content-Type": "application/json"}
        
        # Upload
        ext = file_name.split(".")[-1].lower()
        mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4"}.get(ext, "application/octet-stream")
        files_data = {"audio": (file_name, file_content, mime)}
        up = requests.post(f"{gladia_url}/upload/", headers=headers_file, files=files_data)
        up.raise_for_status()
        audio_url = up.json().get("audio_url")
        
        # Start job
        job_payload = {
            "audio_url": audio_url,
            "diarization": True,
            "translation": True,
            "translation_config": {"target_languages": ["en"]},
            "language_config": {"languages": ["en", "fr", "ar"]}
        }
        post = requests.post(f"{gladia_url}/pre-recorded/", headers=headers_json, json=job_payload)
        post.raise_for_status()
        result_url = post.json().get("result_url")
        
        # Poll for results
        final_result = None
        for _ in range(45):
            poll = requests.get(result_url, headers=headers_file)
            poll.raise_for_status()
            j = poll.json()
            status = j.get("status")
            if status == "done":
                final_result = j
                break
            if status == "error":
                return {"status": "error", "engine": "gladia", "error_message": j.get("error")}
            time.sleep(2)
        
        if not final_result:
            return {"status": "error", "engine": "gladia", "error_message": "Polling timeout"}
        
        # Parse results
        res = final_result.get("result", {}) or {}
        transcription = res.get("transcription", {}) or {}
        translation = res.get("translation", {}) or {}
        utterances = transcription.get("utterances", []) or []
        
        language = (transcription.get("languages") or ["unknown"])[0]
        duration = transcription.get("duration", 0) or 0
        original_text = transcription.get("full_transcript", "")
        
        en_obj = (translation.get("translations") or {}).get("en") or {}
        english_text = en_obj.get("full_transcript", "Not Available")
        
        segments = []
        for u in utterances:
            segments.append({
                "start": u.get("start"),
                "end": u.get("end"),
                "text": u.get("text", ""),
                "speaker": f"SPEAKER_{u.get('speaker')}",
                "language": u.get("language")
            })
        
        return {
            "status": "success",
            "engine": "gladia",
            "language": language,
            "duration": duration,
            "original_text": _scrub_pii(original_text),
            "english_text": _scrub_pii(english_text),
            "segments": segments,
            "error_message": None
        }
    except Exception as e:
        return {"status": "error", "engine": "gladia", "error_message": str(e)}


# ======================================================================================
# DEEPGRAM ENGINE
# ======================================================================================

def process_with_deepgram(file_name: str, file_content: bytes) -> Dict:
    """Process audio with Deepgram Nova-2 Phonecall."""
    api_key = _get_api_key("deepgram")
    if not api_key:
        return {"status": "error", "engine": "deepgram", "error_message": "Deepgram API key not found"}
    
    try:
        url = "https://api.deepgram.com/v1/listen"
        params = {
            "model": "nova-2-phonecall",
            "smart_format": "false",
            "punctuate": "true",
            "diarize": "true",
            "utterances": "true",
            "detect_language": "true"
        }
        
        ext = file_name.split(".")[-1].lower()
        mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4"}.get(ext, "application/octet-stream")
        headers = {"Authorization": f"Token {api_key}", "Content-Type": mime}
        
        resp = requests.post(url, params=params, headers=headers, data=file_content)
        resp.raise_for_status()
        j = resp.json()
        
        results = j.get("results", {})
        channels = results.get("channels", []) or []
        ch0 = channels[0] if channels else {}
        alt0 = (ch0.get("alternatives") or [{}])[0]
        
        clean_transcript = alt0.get("transcript", "")
        utterances = results.get("utterances") or []
        
        language = alt0.get("language") or "unknown"
        duration = (j.get("metadata") or {}).get("duration", 0)
        
        segments = []
        for u in utterances:
            segments.append({
                "start": u.get("start"),
                "end": u.get("end"),
                "text": u.get("transcript", ""),
                "speaker": f"SPEAKER_{u.get('speaker', '1')}"
            })
        
        return {
            "status": "success",
            "engine": "deepgram",
            "language": language,
            "duration": duration,
            "original_text": _scrub_pii(clean_transcript),
            "english_text": "Not Available",
            "segments": segments,
            "error_message": None
        }
    except Exception as e:
        return {"status": "error", "engine": "deepgram", "error_message": str(e)}


# ======================================================================================
# ASSEMBLYAI ENGINE
# ======================================================================================

def process_with_assemblyai(file_name: str, file_content: bytes) -> Dict:
    """Process audio with AssemblyAI."""
    api_key = _get_api_key("assemblyai")
    if not api_key:
        return {"status": "error", "engine": "assemblyai", "error_message": "AssemblyAI API key not found"}
    
    try:
        base = "https://api.assemblyai.com/v2"
        headers_auth = {"authorization": api_key}
        headers_json = {"authorization": api_key, "content-type": "application/json"}
        
        # Upload
        up = requests.post(f"{base}/upload", headers=headers_auth, data=file_content)
        up.raise_for_status()
        upload_url = up.json().get("upload_url")
        
        # Start job
        job_payload = {
            "audio_url": upload_url,
            "speaker_labels": True,
            "language_detection": True
        }
        start = requests.post(f"{base}/transcript", headers=headers_json, json=job_payload)
        start.raise_for_status()
        tid = start.json().get("id")
        result_url = f"{base}/transcript/{tid}"
        
        # Poll
        final = None
        for _ in range(60):
            poll = requests.get(result_url, headers=headers_auth)
            poll.raise_for_status()
            j = poll.json()
            status = j.get("status")
            if status == "completed":
                final = j
                break
            if status == "error":
                return {"status": "error", "engine": "assemblyai", "error_message": j.get("error")}
            time.sleep(3)
        
        if not final:
            return {"status": "error", "engine": "assemblyai", "error_message": "Polling timeout"}
        
        language = final.get("language_code", "unknown")
        duration = final.get("audio_duration", 0)
        original_text = final.get("text", "")
        
        segments = []
        for u in (final.get("utterances") or []):
            segments.append({
                "start": u.get("start"),
                "end": u.get("end"),
                "text": u.get("text", ""),
                "speaker": f"SPEAKER_{u.get('speaker')}"
            })
        
        return {
            "status": "success",
            "engine": "assemblyai",
            "language": language,
            "duration": duration,
            "original_text": _scrub_pii(original_text),
            "english_text": "Not Available",
            "segments": segments,
            "error_message": None
        }
    except Exception as e:
        return {"status": "error", "engine": "assemblyai", "error_message": str(e)}


# ======================================================================================
# UNIFIED INTERFACE
# ======================================================================================

ENGINE_MAP = {
    "Whisper": process_with_whisper,
    "Gladia": process_with_gladia,
    "Deepgram": process_with_deepgram,
    "AssemblyAI": process_with_assemblyai
}


@st.cache_data(show_spinner=False)
def _process_audio_cached(file_hash: str, file_name: str, file_content: bytes, engine: str) -> Dict:
    """Cached audio processing."""
    validation = _validate_audio_file(file_name, file_content)
    if not validation["valid"]:
        return {
            "status": "error",
            "error": "Validation failed",
            "validation_errors": validation["errors"]
        }
    
    processor = ENGINE_MAP.get(engine)
    if not processor:
        return {"status": "error", "error": f"Unknown engine: {engine}"}
    
    return processor(file_name, file_content)


def process_audio(file_name: str, file_content: bytes, engine: str = "Whisper") -> Dict:
    """Main processing function with caching."""
    file_hash = _calculate_file_hash(file_content)
    return _process_audio_cached(file_hash, file_name, file_content, engine)


def get_available_engines() -> List[str]:
    """Return list of available engines based on API keys."""
    engines = []
    if st.secrets.get("OPENAI_API_KEY"):
        engines.append("Whisper")
    if st.secrets.get("GLADIA_API_KEY"):
        engines.append("Gladia")
    if st.secrets.get("DEEPGRAM_API_KEY"):
        engines.append("Deepgram")
    if st.secrets.get("ASSEMBLYAI_API_KEY"):
        engines.append("AssemblyAI")
    return engines
