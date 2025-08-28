# ai_engine.py — A/B anonymized dual-model engine for call QA (Streamlit-friendly, UI-agnostic)

import json
import time
import statistics
import random
import uuid
import re
import logging
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from openai import OpenAI
import google.generativeai as genai


# ======================================================================================
# API CLIENT SETUP & CONFIGURATION
# ======================================================================================

def validate_api_keys() -> Tuple[bool, bool]:
    """Validate API keys are available from st.secrets."""
    openai_key = st.secrets.get("OPENAI_API_KEY")
    gemini_key = st.secrets.get("GEMINI_API_KEY")
    return bool(openai_key), bool(gemini_key)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Cached OpenAI client."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not configured")
    return OpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def get_gemini_client():
    """Cached Gemini model object (GenerativeModel)."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not configured")
    genai.configure(api_key=api_key)
    # Force JSON MIME type to reduce parsing errors
    return genai.GenerativeModel(
        "gemini-1.5-pro",
        generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
    )


# ======================================================================================
# PARAMETER CONFIGURATIONS
# ======================================================================================

ANALYSIS_PARAMETERS = {
    "Quick Scan": [
        {"name": "greeting", "weight": 15, "anchors": "90-100: Professional greeting with company name, agent name, and polite inquiry. 70-89: Good greeting with most elements present. 50-69: Basic greeting missing some elements. 30-49: Poor greeting, unprofessional. 0-29: No proper greeting."},
        {"name": "empathy", "weight": 20, "anchors": "90-100: Shows genuine understanding and concern for customer's situation. 70-89: Demonstrates good empathy with appropriate responses. 50-69: Shows some empathy but could be more genuine. 30-49: Limited empathy shown. 0-29: No empathy demonstrated."},
        {"name": "resolution", "weight": 30, "anchors": "90-100: Completely resolves customer issue with clear solution. 70-89: Good resolution with minor gaps. 50-69: Partial resolution provided. 30-49: Minimal resolution attempts. 0-29: No resolution provided."},
    ],
    "Standard Analysis": [
        {"name": "greeting", "weight": 10, "anchors": "90-100: Professional greeting with company name, agent name, and polite inquiry. 70-89: Good greeting with most elements present. 50-69: Basic greeting missing some elements. 30-49: Poor greeting, unprofessional. 0-29: No proper greeting."},
        {"name": "empathy", "weight": 15, "anchors": "90-100: Shows genuine understanding and concern for customer's situation. 70-89: Demonstrates good empathy with appropriate responses. 50-69: Shows some empathy but could be more genuine. 30-49: Limited empathy shown. 0-29: No empathy demonstrated."},
        {"name": "resolution", "weight": 25, "anchors": "90-100: Completely resolves customer issue with clear solution. 70-89: Good resolution with minor gaps. 50-69: Partial resolution provided. 30-49: Minimal resolution attempts. 0-29: No resolution provided."},
        {"name": "compliance", "weight": 20, "anchors": "90-100: Follows all compliance requirements including mandatory statements. 70-89: Good compliance with minor gaps. 50-69: Some compliance issues noted. 30-49: Multiple compliance violations. 0-29: Major compliance failures."},
        {"name": "professionalism", "weight": 15, "anchors": "90-100: Maintains professional tone throughout, uses appropriate language. 70-89: Generally professional with minor lapses. 50-69: Somewhat professional but inconsistent. 30-49: Limited professionalism. 0-29: Unprofessional behavior."},
        {"name": "communication_clarity", "weight": 15, "anchors": "90-100: Clear, concise communication that's easy to understand. 70-89: Generally clear with minor confusion. 50-69: Somewhat clear but could be improved. 30-49: Often unclear or confusing. 0-29: Very poor communication."},
    ],
    "Deep Dive": [
        {"name": "greeting", "weight": 8, "anchors": "90-100: Professional greeting with company name, agent name, and polite inquiry. 70-89: Good greeting with most elements present. 50-69: Basic greeting missing some elements. 30-49: Poor greeting, unprofessional. 0-29: No proper greeting."},
        {"name": "empathy", "weight": 12, "anchors": "90-100: Shows genuine understanding and concern for customer's situation. 70-89: Demonstrates good empathy with appropriate responses. 50-69: Shows some empathy but could be more genuine. 30-49: Limited empathy shown. 0-29: No empathy demonstrated."},
        {"name": "resolution", "weight": 20, "anchors": "90-100: Completely resolves customer issue with clear solution. 70-89: Good resolution with minor gaps. 50-69: Partial resolution provided. 30-49: Minimal resolution attempts. 0-29: No resolution provided."},
        {"name": "compliance", "weight": 15, "anchors": "90-100: Follows all compliance requirements including mandatory statements at proper times during call. Agent must verify customer identity before sharing financial information, provide required disclosures, and use exact compliance language. 70-89: Good compliance with minor gaps. 50-69: Some compliance issues noted. 30-49: Multiple compliance violations. 0-29: Major compliance failures."},
        {"name": "professionalism", "weight": 12, "anchors": "90-100: Maintains professional tone throughout, uses appropriate language. 70-89: Generally professional with minor lapses. 50-69: Somewhat professional but inconsistent. 30-49: Limited professionalism. 0-29: Unprofessional behavior."},
        {"name": "active_listening", "weight": 10, "anchors": "90-100: Demonstrates excellent active listening with appropriate responses and clarifying questions. 70-89: Good listening skills shown. 50-69: Some listening but misses cues. 30-49: Poor listening skills. 0-29: No evidence of active listening."},
        {"name": "product_knowledge", "weight": 13, "anchors": "90-100: Expert knowledge of products/services with accurate information. 70-89: Good product knowledge with minor gaps. 50-69: Basic knowledge but some inaccuracies. 30-49: Limited product knowledge. 0-29: Poor or incorrect product knowledge."},
        {"name": "call_control", "weight": 10, "anchors": "90-100: Maintains excellent control of call flow, manages time effectively. 70-89: Good call control with minor issues. 50-69: Some call control but could be better. 30-49: Poor call control, loses direction. 0-29: No call control, chaotic flow."},
    ],
}

# Custom Parameters Storage (can be moved to DB later)
CUSTOM_PARAMETERS = {
    "Sales Outbound": [
        {"name": "emi_offer", "weight": 15, "custom": True, "anchors": "90-100: Proactively offered 0% EMI when applicable and customer showed interest in higher-priced options. 70-89: Offered EMI when customer asked or showed price sensitivity. 50-69: Mentioned EMI options briefly during price discussion. 30-49: Failed to offer EMI when clear opportunity existed. 0-29: No EMI discussion despite clear opportunity."},
        {"name": "urgency_creation", "weight": 20, "custom": True, "anchors": "90-100: Genuine urgency with limited-time offer/availability. 70-89: Some urgency with promotions. 50-69: Mild attempts. 30-49: Weak urgency. 0-29: No urgency created."},
        {"name": "objection_handling", "weight": 25, "custom": True, "anchors": "90-100: Expertly handled objections with relevant solutions. 70-89: Good handling. 50-69: Basic handling. 30-49: Poor handling. 0-29: Failed to address objections."},
    ],
    "Banking Support": [
        {"name": "identity_verification", "weight": 25, "custom": True, "anchors": "90-100: Verified identity with multiple factors before sharing details. 70-89: Good verification. 50-69: Basic verification. 30-49: Insufficient verification. 0-29: No verification before sharing sensitive info."},
        {"name": "data_protection", "weight": 20, "custom": True, "anchors": "90-100: Excellent data protection; no full numbers spoken; secure channels. 70-89: Good protection. 50-69: Adequate but some verbal sharing. 30-49: Concerns. 0-29: Poor practices."},
        {"name": "regulatory_compliance", "weight": 15, "custom": True, "anchors": "90-100: Perfect disclosures, consent captured. 70-89: Good compliance. 50-69: Most requirements met. 30-49: Some issues. 0-29: Multiple violations."},
    ],
    "Technical Support": [
        {"name": "troubleshooting_methodology", "weight": 30, "custom": True, "anchors": "90-100: Systematic approach; info gathered; step-by-step tests. 70-89: Good approach. 50-69: Basic, skipped steps. 30-49: Poor approach. 0-29: No methodology."},
        {"name": "technical_accuracy", "weight": 25, "custom": True, "anchors": "90-100: Completely accurate guidance. 70-89: Mostly accurate. 50-69: Some questionable advice. 30-49: Several inaccuracies. 0-29: Incorrect/harmful info."},
    ],
}


# ======================================================================================
# HELPERS: SANITIZATION & JSON PARSING
# ======================================================================================

def sanitize_transcript(transcript: str, max_length: int = 50000) -> str:
    """Sanitize basic PII and truncate very long transcripts."""
    if not transcript:
        return ""
    # Basic PII patterns (expand as needed for your markets)
    transcript = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD_NUMBER_REDACTED]', transcript)  # card
    transcript = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[SSN_REDACTED]', transcript)  # SSN
    transcript = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL_REDACTED]', transcript)
    transcript = re.sub(r'\b(\+?\d[\d\s-]{8,})\b', '[PHONE_REDACTED]', transcript)  # phone-ish
    transcript = re.sub(r'\b\d{4,6}\b', '[OTP_REDACTED]', transcript)  # OTP length
    transcript = re.sub(r'\b\d{3}\b', '[CVV_REDACTED]', transcript)  # CVV length
    if len(transcript) > max_length:
        transcript = transcript[:max_length] + "\n...[TRANSCRIPT_TRUNCATED]"
    return transcript


def _json_guard(text_response: str) -> dict:
    """Safely parse JSON from raw model text (strip junk, code fences, etc.)."""
    if not text_response or not isinstance(text_response, str):
        return {"error": "Empty or invalid response", "raw_response": text_response}
    t = text_response.strip().strip('`')
    try:
        # Quick path
        if t.startswith("{") and t.endswith("}"):
            return json.loads(t)
        # Search for JSON object boundaries
        start = t.find('{'); end = t.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(t[start:end+1])
    except Exception as e:
        return {"error": f"JSON parsing failed: {e}", "raw_response": text_response[:1000]}
    return {"error": "No valid JSON object found in response", "raw_response": text_response[:1000]}


# ======================================================================================
# MODEL CALLS
# ======================================================================================

def call_ai_engine_single(prompt: str, model_name: str, max_retries: int = 2) -> dict:
    """Call a single model ('gpt' or 'gemini') with retries and timing."""
    start_time = time.time()
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            if model_name == "gpt":
                client = get_openai_client()
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    timeout=30,
                )
                result = _json_guard(resp.choices[0].message.content)

            elif model_name == "gemini":
                model = get_gemini_client()
                resp = model.generate_content(prompt, request_options={"timeout": 30})
                result = _json_guard(getattr(resp, "text", "") or "")

            else:
                result = {"error": f"Unknown model '{model_name}'"}

            result["_performance"] = {
                "model": model_name,
                "response_time": round(time.time() - start_time, 2),
                "attempts": attempt,
                "success": "error" not in result,
            }
            return result

        except Exception as e:
            last_err = str(e)
            time.sleep(0.5)  # backoff

    return {
        "error": f"{model_name} failed after {max_retries} attempts: {last_err}",
        "_performance": {"model": model_name, "failed": True, "attempts": max_retries, "error": last_err},
    }


def _stash_model_comparison(detailed: dict):
    """Keep a capped history of model comparisons in session state."""
    buf = st.session_state.setdefault("_model_comparisons", [])
    buf.append({"ts": time.time(), "results": detailed})
    if len(buf) > 50:
        buf.pop(0)


def _create_ab_mapping(openai_ok: bool, gemini_ok: bool) -> Tuple[str, Dict[str, str]]:
    """
    Create a randomized A/B mapping for this run.
    Returns (run_id, {"A": "gpt"/"gemini", "B": "gpt"/"gemini"?}).
    """
    run_id = str(uuid.uuid4())[:8]
    mapping = {}
    if openai_ok and gemini_ok:
        pair = ["gpt", "gemini"]
        random.shuffle(pair)
        mapping = {"A": pair[0], "B": pair[1]}
    elif openai_ok:
        mapping = {"A": "gpt"}
    elif gemini_ok:
        mapping = {"A": "gemini"}
    else:
        mapping = {}
    # Persist mapping for admin lookup
    st.session_state.setdefault("_ab_runs", {})[run_id] = mapping
    return run_id, mapping


def _call_pair_with_mapping(prompt: str, mapping: Dict[str, str], max_retries: int = 2) -> Tuple[Dict[str, dict], dict]:
    """
    Call the models specified in mapping in parallel.
    Returns (results_by_label, perf_summary).
    """
    results: Dict[str, dict] = {}
    perf = {"models_used": [], "total_time": None}

    start = time.time()
    tasks = {}
    with ThreadPoolExecutor(max_workers=max(1, len(mapping))) as pool:
        for label, model_name in mapping.items():
            tasks[label] = pool.submit(call_ai_engine_single, prompt, model_name, max_retries)

        for label, fut in tasks.items():
            results[label] = fut.result()
            model_name = mapping[label]
            if model_name not in perf["models_used"]:
                perf["models_used"].append(model_name)

    perf["total_time"] = round(time.time() - start, 2)
    return results, perf


# ======================================================================================
# PUBLIC API: SINGLE PROMPT (A/B)
# ======================================================================================

def call_ai_engine(prompt: str, max_retries: int = 2, admin_view: bool = False) -> dict:
    """
    Run a single prompt through anonymized A/B:
      - Users get: {"results": [{"label":"A","result":...}, {"label":"B","result":...}]}
      - Admin gets (if admin_view=True): + {"admin": {"run_id":..., "mapping": {...}, "performance_summary": {...}}}
    If only one model is configured, returns just A.
    """
    openai_ok, gemini_ok = validate_api_keys()
    if not (openai_ok or gemini_ok):
        return {"error": "No API keys configured"}

    clean_prompt = sanitize_transcript(prompt)
    run_id, mapping = _create_ab_mapping(openai_ok, gemini_ok)
    results_by_label, perf = _call_pair_with_mapping(clean_prompt, mapping, max_retries)

    # Build user payload
    items = [{"label": label, "result": results_by_label[label]} for label in ("A", "B") if label in results_by_label]
    detailed = {m: results_by_label[lbl] for lbl, m in mapping.items()}
    detailed["performance_summary"] = {"total_time": perf["total_time"], "parallel_execution": True, "models_used": perf["models_used"]}

    _stash_model_comparison(detailed)

    payload = {"results": items}
    if admin_view:
        payload["admin"] = {"run_id": run_id, "mapping": mapping, "performance_summary": detailed["performance_summary"]}
    return payload


# ======================================================================================
# PROMPT GENERATION
# ======================================================================================

def get_triage_prompt(transcript: str) -> str:
    return f"""
You are an expert conversation analyst. Analyze this call transcript and return a JSON object with exact field names.

Required JSON structure:
{{
    "purpose": "Single sentence summary of why customer called",
    "category": "One of: Query, Complaint, Follow-up, Escalation, Sales, Support, Billing",
    "summary": "Three-sentence summary covering: call start, main discussion, resolution",
    "customer_sentiment": "Integer from 1-10 (1=very negative, 10=very positive)",
    "call_complexity": "Integer from 1-5 (1=simple routine, 5=very complex multi-issue)",
    "estimated_duration": "Integer estimated minutes based on transcript length and complexity"
}}

Transcript:
---
{transcript}
---

Return ONLY the JSON object with no additional text.
""".strip()


def get_business_outcome_prompt(transcript: str) -> str:
    return f"""
You are a business analyst specializing in call center outcomes. Analyze this transcript and return a JSON object.

Required JSON structure:
{{
    "business_outcome": "One of: Sale_Completed, Customer_Retained, Issue_Resolved_First_Call, Escalation_Required, Follow-up_Promised, Customer_Churn_Risk, No_Resolution, Information_Provided, Appointment_Scheduled",
    "outcome_confidence": "Integer from 1-10 indicating confidence in classification",
    "justification": "Brief explanation for the outcome classification with specific evidence",
    "compliance_adherence": "Boolean - true if agent followed standard compliance procedures",
    "compliance_details": "Specific compliance elements observed or missed",
    "risk_identified": "Boolean indicating if any legal or reputational risks were identified",
    "risk_details": "If risks identified, provide specific details and exact quotes",
    "next_steps": "Recommended next actions based on this call outcome",
    "customer_satisfaction_indicator": "Integer from 1-10 based on customer responses and call resolution"
}}

Compliance Requirements to Check:
- Agent must greet professionally with company name
- Must verify customer identity before sharing account details
- Must provide required disclosures for financial products
- Must end with professional closing statement

Transcript:
---
{transcript}
---

Return ONLY the JSON object with no additional text.
""".strip()


def get_parameter_scoring_prompt(transcript: str, parameter_name: str, anchors: str) -> str:
    return f"""
You are a meticulous QA analyst with expertise in call center quality assessment.

Score the parameter '{parameter_name}' on a scale of 0-100 using the provided behavioral anchors.

Behavioral Anchors for {parameter_name}:
{anchors}

Required JSON structure:
{{
    "score": "Integer from 0-100 based on behavioral anchors",
    "confidence": "Integer from 1-10 indicating confidence in this score",
    "justification": "Detailed reasoning referencing specific behavioral anchors and transcript evidence",
    "primary_evidence": "Best supporting quote from transcript (max 150 words)",
    "context_before": "Dialogue context before the evidence (max 100 words)",
    "context_after": "Dialogue context after the evidence (max 100 words)",
    "coaching_opportunity": "Specific, actionable coaching recommendation for improvement",
    "improvement_impact": "How addressing this would impact customer experience",
    "severity": "If score < 70, rate issue severity from 1-5 (1=minor, 5=critical)"
}}

Instructions:
- Be precise and reference specific anchor ranges
- Use exact quotes from transcript for evidence
- Provide actionable coaching, not generic advice
- If parameter doesn't apply to this call type, score as N/A with explanation

Transcript:
---
{transcript}
---

Return ONLY the JSON object with no additional text.
""".strip()


# ======================================================================================
# UTILS
# ======================================================================================

def _num(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def get_combined_parameters(depth: str, custom_rubric: Optional[str] = None) -> List[dict]:
    """Merge standard parameters with optional custom rubric and normalize weights."""
    standard_params = list(ANALYSIS_PARAMETERS.get(depth, ANALYSIS_PARAMETERS["Standard Analysis"]))
    custom_params = list(CUSTOM_PARAMETERS.get(custom_rubric, [])) if custom_rubric else []
    combined = standard_params + custom_params

    total_weight = sum(p.get("weight", 0) for p in combined) or 0
    if total_weight > 100:
        for p in combined:
            p["weight"] = round((p.get("weight", 0) / total_weight) * 100, 1)
    return combined


def calculate_overall_metrics(parameter_scores: dict) -> dict:
    """Calculate overall weighted metrics from parameter scores."""
    if not parameter_scores:
        return {"error": "No parameter scores provided"}

    total_weighted_score = 0.0
    total_weight = 0.0
    valid_scores = 0
    low_scores = []
    coaching_opportunities = []
    custom_parameter_performance = {}

    for param_name, param_data in parameter_scores.items():
        if isinstance(param_data, dict) and 'error' not in param_data and 'score' in param_data:
            score = _num(param_data.get('score', 0))
            weight = _num(param_data.get('weight', 10))
            is_custom = bool(param_data.get('custom', False))

            total_weighted_score += (score * weight)
            total_weight += weight
            valid_scores += 1

            if is_custom:
                custom_parameter_performance[param_name] = {
                    'score': score, 'weight': weight, 'confidence': _num(param_data.get('confidence', 5))
                }

            if score < 70:
                low_scores.append({
                    'parameter': param_name,
                    'score': score,
                    'coaching': param_data.get('coaching_opportunity', 'Review performance'),
                    'is_custom': is_custom
                })

            if param_data.get('coaching_opportunity'):
                coaching_opportunities.append({
                    'parameter': param_name,
                    'coaching': param_data['coaching_opportunity'],
                    'is_custom': is_custom
                })

    if total_weight <= 0:
        return {"error": "No valid weighted scores found"}

    overall_score = round(total_weighted_score / total_weight, 2)

    if overall_score >= 90:
        quality_bucket = "Excellent"
    elif overall_score >= 80:
        quality_bucket = "Good"
    elif overall_score >= 70:
        quality_bucket = "Average"
    elif overall_score >= 60:
        quality_bucket = "Below Average"
    else:
        quality_bucket = "Poor"

    score_distribution = {
        "excellent": sum(1 for _, d in parameter_scores.items() if _num(d.get('score', 0)) >= 90),
        "good":      sum(1 for _, d in parameter_scores.items() if 80 <= _num(d.get('score', 0)) < 90),
        "average":   sum(1 for _, d in parameter_scores.items() if 70 <= _num(d.get('score', 0)) < 80),
        "weak":      sum(1 for _, d in parameter_scores.items() if 60 <= _num(d.get('score', 0)) < 70),
        "poor":      sum(1 for _, d in parameter_scores.items() if _num(d.get('score', 0)) < 60),
    }

    # Top 3 coaching items (fallback to lowest scores)
    low_scores_sorted = sorted(
        low_scores, key=lambda x: (_num(x['score']), -_num(parameter_scores[x['parameter']].get('weight', 0)))
    )
    top_coaching = coaching_opportunities[:3] if coaching_opportunities else [
        {"parameter": x["parameter"], "coaching": x.get("coaching", "Improve performance"), "is_custom": x["is_custom"]}
        for x in low_scores_sorted[:3]
    ]

    return {
        "overall_score": overall_score,
        "quality_bucket": quality_bucket,
        "total_parameters_scored": valid_scores,
        "parameters_needing_attention": len(low_scores),
        "low_scoring_parameters": low_scores,
        "coaching_opportunities": top_coaching,
        "custom_parameter_performance": custom_parameter_performance,
        "score_distribution": score_distribution,
    }


# ======================================================================================
# MULTI-STAGE ANALYSIS (A/B preserved across all stages)
# ======================================================================================

def run_initial_triage_ab(transcript: str, mapping: Dict[str, str], max_retries: int = 2, admin_view: bool = False) -> dict:
    prompt = get_triage_prompt(transcript)
    results_by_label, perf = _call_pair_with_mapping(prompt, mapping, max_retries)
    payload = {"results": [{"label": lbl, "result": results_by_label[lbl]} for lbl in ("A", "B") if lbl in results_by_label]}
    if admin_view:
        payload["admin"] = {"performance_summary": {"total_time": perf["total_time"], "models_used": perf["models_used"]}}
    return payload


def run_business_outcome_ab(transcript: str, mapping: Dict[str, str], max_retries: int = 2, admin_view: bool = False) -> dict:
    prompt = get_business_outcome_prompt(transcript)
    results_by_label, perf = _call_pair_with_mapping(prompt, mapping, max_retries)
    payload = {"results": [{"label": lbl, "result": results_by_label[lbl]} for lbl in ("A", "B") if lbl in results_by_label]}
    if admin_view:
        payload["admin"] = {"performance_summary": {"total_time": perf["total_time"], "models_used": perf["models_used"]}}
    return payload


def run_parameter_scoring_ab(transcript: str, parameters: List[dict], mapping: Dict[str, str], max_retries: int = 2, admin_view: bool = False) -> dict:
    """
    Score each parameter for A and B with the same mapping.
    Returns:
      {
        "A": {param_name: result_with_weight_custom, ...},
        "B": {param_name: result_with_weight_custom, ...},
        "admin": {...?}
      }
    """
    scores = {"A": {}, "B": {}}
    perf_totals = {"A": 0.0, "B": 0.0, "calls": 0}

    for p in parameters:
        prompt = get_parameter_scoring_prompt(transcript, p["name"], p["anchors"])
        results_by_label, perf = _call_pair_with_mapping(prompt, mapping, max_retries)
        perf_totals["calls"] += 1
        for lbl in results_by_label:
            r = results_by_label[lbl]
            if "error" not in r:
                r["weight"] = p.get("weight", 10)
                r["custom"] = p.get("custom", False)
            scores[lbl][p["name"]] = r
            if r.get("_performance", {}).get("response_time"):
                perf_totals[lbl] += r["_performance"]["response_time"]

    payload = {"A": scores.get("A", {}), "B": scores.get("B", {})}
    if admin_view:
        payload["admin"] = {
            "perf_estimates": {
                "A_total_secs": round(perf_totals["A"], 2),
                "B_total_secs": round(perf_totals["B"], 2),
                "param_calls": perf_totals["calls"],
            }
        }
    return payload


def run_comprehensive_analysis(transcript: str, depth: str = "Standard Analysis", custom_rubric: Optional[str] = None, max_retries: int = 2, admin_view: bool = False) -> dict:
    """
    Orchestrated A/B analysis with a single, consistent A/B mapping:
      - triage
      - business outcome
      - parameter scoring
      - overall metrics per variant
    Returns a dict that the UI can render without needing Streamlit calls here.
    """
    if not transcript or not transcript.strip():
        return {"error": "Empty transcript provided"}

    openai_ok, gemini_ok = validate_api_keys()
    if not (openai_ok or gemini_ok):
        return {"error": "No API keys configured"}

    cleaned = sanitize_transcript(transcript)
    run_id, mapping = _create_ab_mapping(openai_ok, gemini_ok)

    # Stage 1: triage
    triage = run_initial_triage_ab(cleaned, mapping, max_retries, admin_view)

    # Stage 2: business outcomes
    outcome = run_business_outcome_ab(cleaned, mapping, max_retries, admin_view)

    # Stage 3: parameter scoring + overall
    params = get_combined_parameters(depth, custom_rubric)
    param_scores = run_parameter_scoring_ab(cleaned, params, mapping, max_retries, admin_view)

    overall = {}
    if "A" in param_scores and param_scores["A"]:
        overall["A"] = calculate_overall_metrics(param_scores["A"])
    if "B" in param_scores and param_scores["B"]:
        overall["B"] = calculate_overall_metrics(param_scores["B"])

    payload = {
        "run_id": run_id,
        "mapping_present": bool(mapping),  # not the mapping itself (kept admin-only)
        "stages": [
            {"name": "triage", "results": triage["results"]},
            {"name": "business_outcome", "results": outcome["results"]},
            {"name": "parameter_scores", "A": param_scores.get("A", {}), "B": param_scores.get("B", {})},
        ],
        "overall": overall,
    }

    if admin_view:
        payload["admin"] = {
            "mapping": mapping,
            "triage_admin": triage.get("admin"),
            "outcome_admin": outcome.get("admin"),
            "params_admin": param_scores.get("admin"),
        }

    # Optionally stash a compact detailed log:
    _stash_model_comparison({
        "run_id": run_id,
        "mapping": mapping,
        "depth": depth,
        "custom_rubric": custom_rubric,
        "notes": "comprehensive_ab",
    })
    return payload
