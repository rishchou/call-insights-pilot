# audio_processing.py
import io
import json
import hashlib
import time
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI
import google.generativeai as genai

# SDKs for the new engines
# from gladia import GladiaClient
import assemblyai as aai
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
def _guess_lang_from_text(txt: str) -> str:
    if not txt:
        return "unknown"
    # Arabic script range
    if re.search(r'[\u0600-\u06FF]', txt):
        return "ar"
    # quick French diacritics hint (very naive)
    if re.search(r'[àâçéèêëîïôûùüÿœæ]', txt, flags=re.IGNORECASE):
        return "fr"
    if re.search(r'[A-Za-z]', txt):
        return "en"
    return "unknown"

def _normalize_speakers(segments: List[Dict[str, Any]]):
    """Map engine-specific speaker ids to SPEAKER_0/1/2… consistently."""
    mapping, next_id = {}, 0
    out = []
    for s in segments:
        raw = str(s.get("speaker", "UNK"))
        if raw not in mapping:
            mapping[raw] = f"SPEAKER_{next_id}"
            next_id += 1
        out.append({**s, "speaker": mapping[raw]})
    return out, mapping

def summarize_diarization(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple turns/talk-time/interruptions/silence metrics."""
    if not segments:
        return {"turns": 0, "talk_time": {}, "talk_ratio": {}, "interruptions": 0, "avg_silence": 0.0}

    total_by_spk, turns, interruptions = {}, 0, 0
    last_spk, prev_end = None, 0.0
    silences = []

    for s in segments:
        spk = s.get("speaker") or "SPEAKER_0"
        dur = max(0.0, float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
        total_by_spk[spk] = total_by_spk.get(spk, 0.0) + dur

        # turn change
        if spk != last_spk:
            if last_spk is not None:
                # overlap -> interruption
                if float(s.get("start", 0.0)) < prev_end - 0.1:
                    interruptions += 1
                # gap -> silence
                if float(s.get("start", 0.0)) > prev_end:
                    silences.append(float(s.get("start", 0.0)) - prev_end)
            turns += 1
            last_spk = spk

        prev_end = max(prev_end, float(s.get("end", 0.0)))

    total = sum(total_by_spk.values()) or 1.0
    talk_ratio = {spk: secs / total for spk, secs in total_by_spk.items()}
    avg_silence = sum(silences) / len(silences) if silences else 0.0
    return {"turns": turns, "talk_time": total_by_spk, "talk_ratio": talk_ratio,
            "interruptions": interruptions, "avg_silence": avg_silence}
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
def transcribe_assemblyai(audio_bytes: bytes) -> Dict[str, Any]:
    """
    AssemblyAI transcription with auto language detection and diarization.
    (Summarization is OFF here; you can add it later in config if needed.)
    """
    try:
        api_key = st.secrets.get("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise ValueError("AssemblyAI API key not found in st.secrets['ASSEMBLYAI_API_KEY'].")

        # Use the alias you imported: `import assemblyai as aai`
        aai.settings.api_key = api_key
        transcriber = aai.Transcriber()
        cfg = aai.TranscriptionConfig(
            speaker_labels=True,       # diarization -> transcript.utterances
            language_detection=True    # auto language detection
        )

        transcript = transcriber.transcribe(io.BytesIO(audio_bytes), cfg)

        # If AssemblyAI signals failure
        if getattr(transcript, "status", None) == aai.TranscriptStatus.error:
            raise RuntimeError(getattr(transcript, "error", "Transcription failed."))

        text = getattr(transcript, "text", "") or ""

        # diarized utterances
        segments: List[Dict[str, Any]] = []
        for utt in getattr(transcript, "utterances", []) or []:
            segments.append({
                "start": float((utt.start or 0) / 1000.0),
                "end":   float((utt.end or 0) / 1000.0),
                "speaker": utt.speaker or "SPEAKER_0",
                "text":  utt.text or ""
            })

        # --- robust language extraction ---
        detected_language = (
            getattr(transcript, "language", None)
            or getattr(transcript, "language_code", None)
        )

        raw = getattr(transcript, "json_response", None) or getattr(transcript, "_response", None)
        if isinstance(raw, dict):
            detected_language = detected_language or raw.get("language_code") or raw.get("language")

        detected_language = detected_language or _guess_lang_from_text(text)

        duration = float(getattr(transcript, "audio_duration", 0.0) or 0.0)

        # normalize speakers + add basic diarization metrics
        segments, speaker_map = _normalize_speakers(segments)
        metrics = summarize_diarization(segments)

        return {
            "provider": "assemblyai",
            "model": "default",
            "language": detected_language,
            "original_text": text,
            "segments": segments,
            "duration": duration,
            "intelligence": None,       # keep analysis separate
            "summary": None,            # add later if you enable summarization
            "diarization_supported": True,
            "diarization_metrics": metrics,
            "speaker_map": speaker_map,
            "error_message": None
        }

    except Exception as e:
        return {
            "provider": "assemblyai",
            "model": "default",
            "language": "unknown",
            "original_text": "",
            "segments": [],
            "duration": 0.0,
            "intelligence": None,
            "summary": None,
            "diarization_supported": True,
            "diarization_metrics": {"turns":0,"talk_time":{},"talk_ratio":{},"interruptions":0,"avg_silence":0.0},
            "speaker_map": {},
            "error_message": str(e)
        }
# ======================================================================================
# ENGINE 4: Deepgram (All-in-One)
# ======================================================================================
def transcribe_deepgram(audio_bytes: bytes, *, enable_intelligence: bool = False) -> Dict[str, Any]:
    """
    Deepgram nova-3 transcription with auto language detection and diarization.
    If enable_intelligence=True, also returns summary/topics/intents/sentiment when available.
    """
    try:
        api_key = st.secrets.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("Deepgram API key not found in st.secrets['DEEPGRAM_API_KEY'].")

        dg = DeepgramClient(api_key)

        # If you didn't import FileSource, drop the annotation: payload = {"buffer": audio_bytes}
        payload: FileSource = {"buffer": audio_bytes}

        opts = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
            punctuate=True,
            utterances=True,       # segmentation
            diarize=True,          # speaker diarization
            detect_language=True   # auto language detection
        )
        # Optional hint if your calls code-switch a lot:
        # opts.language = "multi"

        if enable_intelligence:
            opts.summarize = "v2"
            opts.intents = True
            opts.topics = True
            opts.sentiment = True

        # v("1") path per DG Intelligence guide
        resp = dg.listen.prerecorded.v("1").transcribe_file(payload, opts)
        result = resp.to_dict()

        results = result.get("results", {}) or {}
        channels = results.get("channels", []) or [{}]
        ch0 = channels[0]
        alts = ch0.get("alternatives", []) or [{}]
        alt0 = alts[0]

        original_text = alt0.get("transcript", "") or ""

        # Robust language detection with fallbacks + final script guess
        detected_language = (
            ch0.get("detected_language")
            or alt0.get("language")
            or result.get("metadata", {}).get("detected_language")
            or _guess_lang_from_text(original_text)
        )

        duration = float(result.get("metadata", {}).get("duration", 0.0) or 0.0)

        # Diarized utterances
        segments: List[Dict[str, Any]] = []
        for utt in results.get("utterances", []) or []:
            segments.append({
                "start": float(utt.get("start", 0.0)),
                "end":   float(utt.get("end", 0.0)),
                # keep raw speaker id; we'll normalize to SPEAKER_n below
                "speaker": utt.get("speaker", 0),
                "text":   utt.get("transcript", "") or ""
            })

        # Normalize speakers + add diarization metrics
        segments, speaker_map = _normalize_speakers(segments)
        metrics = summarize_diarization(segments)

        intelligence = None
        if enable_intelligence:
            intelligence = {
                "summary": results.get("summary"),
                "topics": results.get("topics"),
                "intents": results.get("intents"),
                "sentiment": results.get("sentiment"),
            }

        return {
            "provider": "deepgram",
            "model": "nova-3",
            "language": detected_language,
            "original_text": original_text,
            "segments": segments,
            "duration": duration,
            "intelligence": intelligence,   # None if not requested
            "summary": None,
            "diarization_supported": True,
            "diarization_metrics": metrics,
            "speaker_map": speaker_map,
            "error_message": None
        }

    except Exception as e:
        return {
            "provider": "deepgram",
            "model": "nova-3",
            "language": "unknown",
            "original_text": "",
            "segments": [],
            "duration": 0.0,
            "intelligence": None,
            "summary": None,
            "diarization_supported": True,
            "diarization_metrics": {"turns":0,"talk_time":{},"talk_ratio":{},"interruptions":0,"avg_silence":0.0},
            "speaker_map": {},
            "error_message": str(e)
        }
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
# -------------------- helpers --------------------

def _guess_mime(fname: str) -> str:
    f = fname.lower()
    if f.endswith(".wav"):  return "audio/wav"
    if f.endswith(".mp3"):  return "audio/mpeg"
    if f.endswith(".m4a"):  return "audio/mp4"
    if f.endswith(".ogg"):  return "audio/ogg"
    if f.endswith(".webm"): return "audio/webm"
    return "application/octet-stream"

# -------------------- main router for UI --------------------

def process_audio(file_name: str, file_content: bytes, *, engine: str) -> Dict[str, Any]:
    """
    Router used by main_app.py. Calls the new transcribe_* funcs and returns
    a dict that includes `status` and `english_text` so the UI works unchanged.
    """
    try:
        if engine == "deepgram":
            out = transcribe_deepgram(file_content, enable_intelligence=True)  # flip to False if you don’t want DG Intelligence now
        elif engine == "assemblyai":
            out = transcribe_assemblyai(file_content)
        elif engine == "whisper_gemini":
            # We’ll just run Whisper here (no Gemini processing in this step)
            mime = _guess_mime(file_name)
            out = transcribe_openai_whisper(file_content, filename=file_name, mime=mime)
        elif engine == "gladia":
            return {
                "provider": "gladia",
                "model": "",
                "language": "unknown",
                "original_text": "",
                "segments": [],
                "duration": 0.0,
                "intelligence": None,
                "summary": None,
                "diarization_supported": False,
                "engine": "gladia",
                "status": "error",
                "error_message": "Gladia is not wired yet in this build."
            }
        else:
            return {
                "provider": engine,
                "model": "",
                "language": "unknown",
                "original_text": "",
                "segments": [],
                "duration": 0.0,
                "intelligence": None,
                "summary": None,
                "diarization_supported": False,
                "engine": engine,
                "status": "error",
                "error_message": f"Unknown engine '{engine}'."
            }

        # Normalize for your UI (main_app.py expects these)
        out["engine"] = engine
        out["status"] = "error" if out.get("error_message") else "success"
        # Until you add a translator step, show original text as snippet fallback
        out.setdefault("english_text", out.get("original_text", ""))

        return out

    except Exception as e:
        return {
            "provider": engine,
            "model": "",
            "language": "unknown",
            "original_text": "",
            "segments": [],
            "duration": 0.0,
            "intelligence": None,
            "summary": None,
            "diarization_supported": engine in ("deepgram", "assemblyai"),
            "engine": engine,
            "status": "error",
            "error_message": str(e)
        }
