import os
import io
import json
import hashlib
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import time
import re

# ======================================================================================
# CONSTANTS AND CONFIGURATION
# ======================================================================================

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB - Whisper's typical limit
SUPPORTED_FORMATS = ['mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm']

WHISPER_MODEL = "whisper-1"
GEMINI_MODEL = "gemini-1.5-pro"

GEMINI_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# ======================================================================================
# LAZY CLIENT INITIALIZATION
# ======================================================================================

@lru_cache(maxsize=1)
def _get_openai_client() -> Optional[OpenAI]:
    """Lazy initialization of OpenAI client."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("OpenAI API key not found. Audio transcription will not work.")
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"OpenAI client initialization failed: {e}")
        return None

@lru_cache(maxsize=1)
def _get_gemini_model():
    """Lazy initialization of Gemini model."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.warning("Gemini API key not found. Speaker labeling will not work.")
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config={"response_mime_type": "application/json"}
        )
    except Exception as e:
        st.error(f"Gemini client initialization failed: {e}")
        return None

# ======================================================================================
# UTILITY FUNCTIONS
# ======================================================================================

def _calculate_file_hash(file_content: bytes) -> str:
    """Calculate hash of file content for caching."""
    return hashlib.md5(file_content).hexdigest()

def _scrub_pii(text: str) -> str:
    """Basic PII scrubbing for production safety."""
    if not isinstance(text, str):
        return text
    # Phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', text)
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL_REDACTED]', text)
    # Credit card patterns (basic)
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD_REDACTED]', text)
    # Emirates ID pattern
    text = re.sub(r'\b784-\d{4}-\d{7}-\d{1}\b', '[EMIRATES_ID_REDACTED]', text)
    return text

def _validate_audio_file(file_name: str, file_content: bytes) -> Dict[str, any]:
    """Validate audio file before processing."""
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "file_info": {}
    }

    # Size checks
    file_size = len(file_content)
    validation_result["file_info"]["size_mb"] = round(file_size / (1024 * 1024), 2)

    if file_size > MAX_FILE_SIZE:
        validation_result["valid"] = False
        validation_result["errors"].append(
            f"File size ({validation_result['file_info']['size_mb']}MB) exceeds limit ({MAX_FILE_SIZE // (1024*1024)}MB)"
        )

    if file_size < 1024:  # Less than 1KB
        validation_result["valid"] = False
        validation_result["errors"].append("File appears to be too small to contain audio data")

    # Extension checks
    file_extension = file_name.lower().split('.')[-1] if '.' in file_name else ''
    validation_result["file_info"]["format"] = file_extension

    if file_extension not in SUPPORTED_FORMATS:
        validation_result["valid"] = False
        validation_result["errors"].append(
            f"Unsupported file format: {file_extension}. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    if file_size > 10 * 1024 * 1024:  # > 10MB
        validation_result["warnings"].append("Large file - processing may take several minutes")

    return validation_result

def _json_guard(text_response: str) -> dict:
    """Safely parse JSON response with better error handling (handles code fences)."""
    if not text_response or not text_response.strip():
        return {"error": "Empty response from AI model"}
    t = text_response.strip().strip('`')
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start = t.find('{'); end = t.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(t[start:end+1])
            except json.JSONDecodeError:
                pass
    return {
        "error": "Failed to parse AI response as JSON",
        "raw_response": t[:500] + ("..." if len(t) > 500 else "")
    }

# ======================================================================================
# CORE PROCESSING FUNCTIONS
# ======================================================================================

def _transcribe_audio_original(file_name: str, file_content: bytes) -> Dict:
    """Transcribe audio in original language."""
    oai_client = _get_openai_client()
    if not oai_client:
        return {"error": "OpenAI client not available"}

    for attempt in range(MAX_RETRIES):
        try:
            audio_file = io.BytesIO(file_content)
            audio_file.name = file_name

            response = oai_client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                response_format="verbose_json"
            )
            return {
                "text": getattr(response, "text", ""),
                "segments": getattr(response, "segments", []),
                "language": getattr(response, "language", None) or getattr(response, "detected_language", "unknown"),
                "duration": getattr(response, "duration", 0),
            }
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return {"error": f"Original transcription failed after {MAX_RETRIES} attempts: {e}"}
            time.sleep(2 ** attempt)

    return {"error": "Original transcription failed - maximum retries exceeded"}

def _translate_to_english(file_name: str, file_content: bytes) -> Dict:
    """Translate audio to English."""
    oai_client = _get_openai_client()
    if not oai_client:
        return {"error": "OpenAI client not available"}

    for attempt in range(MAX_RETRIES):
        try:
            audio_file = io.BytesIO(file_content)
            audio_file.name = file_name

            response = oai_client.audio.translations.create(
                model=WHISPER_MODEL,
                file=audio_file,
                response_format="verbose_json"
            )
            return {
                "text": getattr(response, "text", ""),
                "segments": getattr(response, "segments", []),
                "language": "en",
                "duration": getattr(response, "duration", 0),
            }
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return {"error": f"English translation failed after {MAX_RETRIES} attempts: {e}"}
            time.sleep(2 ** attempt)

    return {"error": "English translation failed - maximum retries exceeded"}

def _label_speakers_batch(segments: List, batch_size: int = 20) -> List[Dict]:
    """Label speakers with batching support, using stable local IDs."""
    model = _get_gemini_model()
    if not model:
        # Fallback: simple alternating pattern
        labeled_fallback = []
        for i, seg in enumerate(segments):
            text = seg["text"] if isinstance(seg, dict) and "text" in seg else getattr(seg, "text", str(seg))
            start = seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0)
            end = seg.get("end", 0) if isinstance(seg, dict) else getattr(seg, "end", 0)
            labeled_fallback.append({
                "id": i,
                "speaker": "AGENT" if i % 2 == 0 else "CUSTOMER",
                "text": text, "start": start, "end": end
            })
        return labeled_fallback

    all_labeled = []
    n = len(segments)
    for batch_start in range(0, n, batch_size):
        batch = segments[batch_start: batch_start + batch_size]

        # Build prompt batch with stable local_id we control
        prompt_segments = []
        id_map = {}  # local_id -> global index
        for j, seg in enumerate(batch):
            local_id = batch_start + j
            id_map[local_id] = batch_start + j
            text = seg["text"] if isinstance(seg, dict) and "text" in seg else getattr(seg, "text", str(seg))
            start = seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0)
            end = seg.get("end", 0) if isinstance(seg, dict) else getattr(seg, "end", 0)
            prompt_segments.append({"id": local_id, "text": text, "start": start, "end": end})

        prompt = f"""
You are a conversation analyst specializing in call center interactions. Identify speakers for each segment.

Rules:
1) Two speakers: AGENT (company rep) and CUSTOMER.
2) First speaker is usually AGENT.
3) Phrases like "Thank you for calling", "How can I help", "My name is" ⇒ AGENT.
4) Phrases like "I have a problem", "I need help", "Can you..." ⇒ CUSTOMER.

Segments:
{json.dumps(prompt_segments, indent=2)}

Return ONLY JSON:
{{"labels":[{{"id": <segment_id>, "speaker": "AGENT|CUSTOMER"}}]}}
""".strip()

        label_map: Dict[int, str] = {}
        for attempt in range(2):
            try:
                resp = model.generate_content(prompt, request_options={"timeout": GEMINI_TIMEOUT})
                data = _json_guard(getattr(resp, "text", "") or "")
                if "error" not in data:
                    for item in data.get("labels", []):
                        if "id" in item and "speaker" in item:
                            try:
                                label_map[int(item["id"])] = item["speaker"]
                            except Exception:
                                continue
                    break
            except Exception:
                if attempt == 1:
                    break
                time.sleep(1)

        # Apply labels; fallback = alternating
        for local_id, global_idx in id_map.items():
            seg = segments[global_idx]
            text = seg["text"] if isinstance(seg, dict) and "text" in seg else getattr(seg, "text", str(seg))
            start = seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0)
            end = seg.get("end", 0) if isinstance(seg, dict) else getattr(seg, "end", 0)
            speaker = label_map.get(local_id, "AGENT" if global_idx % 2 == 0 else "CUSTOMER")

            all_labeled.append({
                "id": local_id,
                "speaker": speaker,
                "text": text,
                "start": start,
                "end": end,
                "seek": seg.get("seek", 0) if isinstance(seg, dict) else getattr(seg, "seek", 0),
                "tokens": seg.get("tokens", []) if isinstance(seg, dict) else getattr(seg, "tokens", []),
                "temperature": seg.get("temperature", 0.0) if isinstance(seg, dict) else getattr(seg, "temperature", 0.0),
                "avg_logprob": seg.get("avg_logprob", 0.0) if isinstance(seg, dict) else getattr(seg, "avg_logprob", 0.0),
                "compression_ratio": seg.get("compression_ratio", 0.0) if isinstance(seg, dict) else getattr(seg, "compression_ratio", 0.0),
                "no_speech_prob": seg.get("no_speech_prob", 0.0) if isinstance(seg, dict) else getattr(seg, "no_speech_prob", 0.0),
            })

    return all_labeled

# ======================================================================================
# CACHED PROCESSING IMPLEMENTATION (NO RECURSION)
# ======================================================================================

@st.cache_data(show_spinner=False)
def _process_audio_cached(file_hash: str, file_name: str, file_content: bytes) -> Dict:
    """Cached implementation that does the actual heavy lifting."""
    # Step 1: Validate file
    validation = _validate_audio_file(file_name, file_content)
    if not validation["valid"]:
        return {
            "status": "error",
            "error": "File validation failed",
            "validation_errors": validation["errors"],
            "file_info": validation.get("file_info", {})
        }

    result = {
        "status": "processing",
        "file_hash": file_hash,
        "file_info": validation["file_info"],
        "validation_warnings": validation.get("warnings", []),
    }

    try:
        # Step 2: Transcribe in original language
        original_transcript = _transcribe_audio_original(file_name, file_content)
        if "error" in original_transcript:
            result["status"] = "error"
            result["error"] = f"Original transcription failed: {original_transcript['error']}"
            return result

        # Step 3: Translate to English
        english_transcript = _translate_to_english(file_name, file_content)
        if "error" in english_transcript:
            result["status"] = "error"
            result["error"] = f"English translation failed: {english_transcript['error']}"
            return result

        # Step 4: Label speakers (on English segments)
        labeled_segments = _label_speakers_batch(english_transcript.get("segments", []))

        # Step 5: PII scrubbing
        scrubbed_original = _scrub_pii(original_transcript.get("text", ""))
        scrubbed_english = _scrub_pii(english_transcript.get("text", ""))
        for seg in labeled_segments:
            seg["text"] = _scrub_pii(seg.get("text", ""))

        # Step 6: Compile final result
        result.update({
            "status": "success",
            "original_transcript": scrubbed_original,
            "english_transcript": scrubbed_english,
            "segments": labeled_segments,
            "duration": english_transcript.get("duration", 0),
            "detected_language": original_transcript.get("language", "unknown"),
            "processing_metadata": {
                "segments_count": len(labeled_segments),
                "speakers_detected": len(set(s.get("speaker") for s in labeled_segments)) if labeled_segments else 0,
                "file_hash": file_hash,
                "pii_scrubbed": True
            }
        })
        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Unexpected error during processing: {e}"
        return result

# ======================================================================================
# PUBLIC API FUNCTIONS
# ======================================================================================

def process_audio(file_name: str, file_content: bytes) -> Dict:
    """Main audio processing function - calls cached implementation."""
    file_hash = _calculate_file_hash(file_content)
    return _process_audio_cached(file_hash, file_name, file_content)

def process_audio_with_progress(file_name: str, file_content: bytes) -> Dict:
    """
    Process audio with Streamlit progress indicators.
    Note: Progress with caching is approximate since cache hits return fast.
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    file_hash = _calculate_file_hash(file_content)

    try:
        status_text.text("Checking cache...")
        progress_bar.progress(10)

        result = _process_audio_cached(file_hash, file_name, file_content)

        if result.get("status") == "success":
            status_text.text("Processing audio...")
            progress_bar.progress(50)
            time.sleep(0.4)
            status_text.text("Finalizing results...")
            progress_bar.progress(90)
            time.sleep(0.3)
            status_text.text("Complete!")
            progress_bar.progress(100)
            time.sleep(0.3)
        else:
            status_text.text("Processing failed")
            progress_bar.progress(100)

        return result
    finally:
        progress_bar.empty()
        status_text.empty()

def validate_file_before_upload(file_name: str, file_content: bytes) -> Dict:
    """Validate file without processing - useful for upload validation."""
    return _validate_audio_file(file_name, file_content)

def get_processing_capabilities() -> Dict:
    """Return current processing capabilities based on available API keys."""
    return {
        "transcription_available": _get_openai_client() is not None,
        "speaker_labeling_available": _get_gemini_model() is not None,
        "supported_formats": SUPPORTED_FORMATS,
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "features": {
            "original_language_transcription": _get_openai_client() is not None,
            "english_translation": _get_openai_client() is not None,
            "speaker_identification": _get_gemini_model() is not None,
            "batched_speaker_labeling": _get_gemini_model() is not None,
            "pii_scrubbing": True,
            "caching": True,
            "progress_tracking": True,
            "retry_logic": True,
            "timeout_handling": True
        }
    }

def clear_audio_cache():
    """Clear the audio processing cache - useful for testing."""
    _process_audio_cached.clear()
    st.success("Audio processing cache cleared")
