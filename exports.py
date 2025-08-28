# exports.py
import re, json
import pandas as pd
from datetime import datetime

# ---------- helpers ----------
def _truncate(s: str, max_len: int = 8000) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= max_len else s[:max_len] + " [TRUNCATED]"

def _extract_agent_name(transcript_data: dict) -> tuple[str, str]:
    segs = transcript_data.get("segments", []) or []
    for seg in segs[:10]:
        if str(seg.get("speaker","")).upper() == "AGENT":
            text = seg.get("text","") or ""
            m = re.search(r"(my name is|this is)\s+([A-Za-z][A-Za-z\-\s']{1,40})", text, flags=re.I)
            if m: return (m.group(2).strip(), text.strip())
            m2 = re.search(r"(you'?re|you are)\s+speaking with\s+([A-Za-z][A-Za-z\-\s']{1,40})", text, flags=re.I)
            if m2: return (m2.group(2).strip(), text.strip())
    return ("Unknown", "")

def _build_executive_summary(file_name: str, triage: dict, outcome: dict, overall: dict) -> str:
    purpose = triage.get("purpose") or "N/A"
    category = triage.get("category") or "N/A"
    bo = outcome.get("business_outcome") or "N/A"
    boc = outcome.get("outcome_confidence")
    bucket = overall.get("quality_bucket") or "N/A"
    score = overall.get("overall_score")
    parts = []
    parts.append(f"Call '{file_name}': {category} — {purpose}.")
    parts.append(f"Outcome: {bo}" + (f" (confidence {boc}/10)." if isinstance(boc, (int, float)) else "."))
    if isinstance(score, (int, float)):
        parts.append(f"Overall QA: {bucket} ({score:.1f}/100).")
    if outcome.get("compliance_adherence") is False:
        parts.append("Compliance issues observed.")
    if outcome.get("risk_identified"):
        parts.append("Risk flagged.")
    return " ".join(parts)

def _build_coaching_summary(overall: dict, outcome: dict) -> str:
    bullets = []
    lows = overall.get("low_scoring_parameters") or []
    for item in sorted(lows, key=lambda x: x.get("score", 100))[:3]:
        pname = item.get("parameter", "Parameter")
        coaching = item.get("coaching", "Improve performance against rubric.")
        bullets.append(f"{pname}: {coaching}")
    if outcome.get("compliance_adherence") is False:
        bullets.append(f"Compliance: {outcome.get('compliance_details','Non-adherence')}")
    if outcome.get("risk_identified"):
        bullets.append(f"Risk: {outcome.get('risk_details','Potential risk')}")
    return " • ".join(bullets) if bullets else ""

# ---------- main: build per-variant DataFrame ----------
def build_variant_dataframe(file_name: str, ab_result: dict, transcript_data: dict, variant: str) -> pd.DataFrame:
    ts = datetime.now().isoformat(timespec="seconds")
    run_id = ab_result.get("run_id", f"run-{ts}")
    stages = {s.get("name"): s for s in ab_result.get("stages", [])}
    overall_all = ab_result.get("overall", {}) or {}
    overall = overall_all.get(variant, {}) or {}
    triage = {x.get("label"): x.get("result", {}) for x in stages.get("triage", {}).get("results", [])}.get(variant, {})
    outcome = {x.get("label"): x.get("result", {}) for x in stages.get("business_outcome", {}).get("results", [])}.get(variant, {})
    params = (stages.get("parameter_scores", {}) or {}).get(variant, {}) or {}

    agent_name, greeting_line = _extract_agent_name(transcript_data)
    exec_summary = _truncate(_build_executive_summary(file_name, triage, outcome, overall), 2000)
    coach_summary = _truncate(_build_coaching_summary(overall, outcome), 2000)

    base = {
        "AuditTimestamp": ts,
        "RunID": run_id,
        "FileName": file_name,
        "Variant": variant,

        "DetectedLanguage": transcript_data.get("detected_language", "unknown"),
        "DurationSec": transcript_data.get("duration", 0),
        "SpeakersDetected": len(set(s.get("speaker") for s in transcript_data.get("segments", []) or [])),
        "SegmentsCount": len(transcript_data.get("segments", []) or []),

        "AgentName": agent_name,
        "AgentGreetingLine": _truncate(greeting_line, 400),

        "Category": triage.get("category", "N/A"),
        "CallPurpose": triage.get("purpose", "N/A"),

        "BusinessOutcome": outcome.get("business_outcome", "N/A"),
        "OutcomeConfidence": outcome.get("outcome_confidence", "N/A"),
        "ComplianceAdherence": outcome.get("compliance_adherence", "N/A"),
        "ComplianceDetails": _truncate(outcome.get("compliance_details", ""), 2000),
        "RiskIdentified": outcome.get("risk_identified", False),
        "RiskDetails": _truncate(outcome.get("risk_details", ""), 2000),

        "OverallScore": overall.get("overall_score", None),
        "QualityBucket": overall.get("quality_bucket", "N/A"),
        "ParametersNeedingAttention": overall.get("parameters_needing_attention", 0),

        "AIExecutiveSummary": exec_summary,
        "AICoachingSummary": coach_summary,

        "EnglishTranscript": _truncate(transcript_data.get("english_transcript", ""), 15000),
        "OriginalTranscript": _truncate(transcript_data.get("original_transcript", ""), 15000),

        "First5SegmentsJSON": json.dumps([
            {"id": s.get("id"), "speaker": s.get("speaker"), "start": s.get("start"),
             "end": s.get("end"), "text": s.get("text")}
            for s in (transcript_data.get("segments", []) or [])[:5]
        ], ensure_ascii=False),
    }

    rows = []
    if params:
        for pname, details in params.items():
            if not isinstance(details, dict) or "score" not in details or "error" in details:
                continue
            rows.append({
                **base,
                "Parameter": pname,
                "Score": details.get("score"),
                "Confidence": details.get("confidence"),
                "Justification": _truncate(details.get("justification", ""), 5000),
                "PrimaryEvidence": _truncate(details.get("primary_evidence", ""), 3000),
                "CoachingOpportunity": _truncate(details.get("coaching_opportunity", ""), 2000),
                "Weight": details.get("weight"),
                "CustomFlag": details.get("custom", False),
            })
    else:
        rows.append({**base, "Parameter": None, "Score": None, "Confidence": None,
                     "Justification": None, "PrimaryEvidence": None,
                     "CoachingOpportunity": None, "Weight": None, "CustomFlag": None})
    return pd.DataFrame(rows)
