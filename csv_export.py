# csv_export.py - CSV Export functionality for QA analysis results
"""
This module provides CSV export functionality similar to the notebook implementation,
with detailed parameter rows for each analysis.
"""

import pandas as pd
import os
from typing import Dict, List, Optional


def _safe(val):
    """Return empty string if value is None."""
    return "" if val is None else val


def _maybe_trunc(s, limit):
    """Truncate string to limit if needed."""
    if limit and isinstance(s, str) and len(s) > limit:
        return s[:limit] + " ...[TRUNCATED]"
    return s


def build_stt_context(stt_result: dict, file_name: str, rubric="Standard Analysis") -> dict:
    """Build STT context from transcription result."""
    # Build labeled transcript if not present
    if "labeled_transcript" in stt_result and stt_result.get("labeled_transcript"):
        labeled_txt = stt_result["labeled_transcript"]
    else:
        segs = stt_result.get("segments") or []
        labeled_txt = "\n".join(
            f"{seg.get('speaker','SPEAKER')} : {seg.get('text','')}" 
            for seg in segs
        )

    return {
        "FileName": file_name,
        "Engine": stt_result.get("engine", ""),
        "Language": stt_result.get("language", "unknown"),
        "DurationSeconds": stt_result.get("duration", 0),
        "OriginalTranscript": stt_result.get("original_text", ""),
        "EnglishTranscript": stt_result.get("english_text", "Not Available"),
        "LabeledTranscript": labeled_txt,
        "Rubric": rubric,
    }


def _param_rows_with_context(stt_ctx: dict, analysis: dict, truncate_len: int = 8000) -> List[dict]:
    """Build parameter rows with context from analysis results."""
    rows = []
    param_scores = analysis.get("parameter_scores") or {}

    for pname, pdata in (param_scores or {}).items():
        err = ""
        if isinstance(pdata, dict) and "error" in pdata:
            err = str(pdata.get("error"))

        severity = ""
        try:
            sc = float(pdata.get("score", 0))
            if sc < 70:
                severity = (pdata.get("severity") or "")
        except Exception:
            pass

        rows.append({
            # STT context
            "FileName": _safe(stt_ctx.get("FileName")),
            "Engine": _safe(stt_ctx.get("Engine")),
            "Language": _safe(stt_ctx.get("Language")),
            "DurationSeconds": _safe(stt_ctx.get("DurationSeconds")),
            "OriginalTranscript": _maybe_trunc(_safe(stt_ctx.get("OriginalTranscript")), truncate_len),
            "EnglishTranscript": _maybe_trunc(_safe(stt_ctx.get("EnglishTranscript")), truncate_len),
            "LabeledTranscript": _maybe_trunc(_safe(stt_ctx.get("LabeledTranscript")), truncate_len),

            # QA context
            "Rubric": _safe(stt_ctx.get("Rubric", "Standard Analysis")),
            "ModelName": "gemini",  # Always gemini now
            "Error": err,

            # Parameter details
            "Parameter": pname,
            "Score": _safe(pdata.get("score")),
            "Confidence": _safe(pdata.get("confidence")),
            "Justification": _maybe_trunc(_safe(pdata.get("justification")), truncate_len),
            "PrimaryEvidence": _maybe_trunc(_safe(pdata.get("primary_evidence")), truncate_len),
            "ContextBefore": _maybe_trunc(_safe(pdata.get("context_before")), truncate_len),
            "ContextAfter": _maybe_trunc(_safe(pdata.get("context_after")), truncate_len),
            "CoachingOpportunity": _maybe_trunc(_safe(pdata.get("coaching_opportunity")), truncate_len),
            "ImprovementImpact": _maybe_trunc(_safe(pdata.get("improvement_impact")), truncate_len),
            "Severity": severity,
            "Weight": _safe(pdata.get("weight")),
            "CustomFlag": bool(pdata.get("custom", False)),
        })
    return rows


def save_detailed_params_csv(stt_ctx: dict, analysis: dict, out_csv_path: str, truncate_len: int = 8000):
    """Save detailed parameter rows to CSV."""
    rows = _param_rows_with_context(stt_ctx, analysis, truncate_len=truncate_len)
    df = pd.DataFrame(rows)

    if os.path.exists(out_csv_path):
        df.to_csv(out_csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
        print(f"✅ Appended {len(rows)} rows → {out_csv_path}")
    else:
        df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ Created {out_csv_path} with {len(rows)} rows")


def export_analysis_to_csv(
    file_name: str,
    stt_result: dict,
    analysis: dict,
    rubric: str = "Standard Analysis",
    truncate_len: int = 8000
) -> pd.DataFrame:
    """
    Export analysis to CSV format.
    Returns a DataFrame that can be downloaded or saved.
    """
    stt_ctx = build_stt_context(stt_result, file_name, rubric=rubric)
    rows = _param_rows_with_context(stt_ctx, analysis, truncate_len=truncate_len)
    return pd.DataFrame(rows)


def export_multiple_analyses_to_csv(
    analyses: List[Dict],
    out_csv_path: str,
    truncate_len: int = 8000
):
    """
    Export multiple analyses to a single CSV file.
    
    analyses: List of dicts with keys: file_name, stt_result, analysis, rubric
    """
    all_rows = []
    
    for item in analyses:
        stt_ctx = build_stt_context(
            item["stt_result"], 
            item["file_name"], 
            rubric=item.get("rubric", "Standard Analysis")
        )
        rows = _param_rows_with_context(stt_ctx, item["analysis"], truncate_len=truncate_len)
        all_rows.extend(rows)
    
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Exported {len(all_rows)} rows to {out_csv_path}")
    
    return df


def create_summary_df(analyses: List[Dict]) -> pd.DataFrame:
    """
    Create a summary DataFrame from multiple analyses.
    
    analyses: List of dicts with keys: file_name, stt_result, analysis
    """
    summary_rows = []
    
    for item in analyses:
        file_name = item["file_name"]
        stt = item["stt_result"]
        analysis = item["analysis"]
        
        # Extract key metrics
        overall = analysis.get("overall", {})
        triage = analysis.get("triage", {})
        outcome = analysis.get("business_outcome", {})
        
        summary_rows.append({
            "FileName": file_name,
            "Engine": stt.get("engine", ""),
            "Language": stt.get("language", "unknown"),
            "Duration": stt.get("duration", 0),
            "CallPurpose": triage.get("purpose", "N/A"),
            "Category": triage.get("category", "N/A"),
            "CustomerSentiment": triage.get("customer_sentiment", "N/A"),
            "BusinessOutcome": outcome.get("business_outcome", "N/A"),
            "ComplianceAdherence": outcome.get("compliance_adherence", "N/A"),
            "RiskIdentified": outcome.get("risk_identified", False),
            "OverallScore": overall.get("overall_score", "N/A"),
            "QualityBucket": overall.get("quality_bucket", "N/A"),
            "ParametersScored": overall.get("total_parameters_scored", 0),
            "ParametersNeedingAttention": overall.get("parameters_needing_attention", 0)
        })
    
    return pd.DataFrame(summary_rows)
