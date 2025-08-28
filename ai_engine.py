# ======================================================================================
# PROMPT GENERATION FUNCTIONS
# ======================================================================================

def _get_triage_prompt(transcript: str) -> str:
    """Generate triage analysis prompt"""
    return f"""
    You are a conversation analyst. Analyze the provided transcript and return a JSON object.
    The JSON object must contain:
    1. "purpose": A one-sentence summary of why the customer is calling.
    2. "category": Classify the call into one of: 'Query', 'Complaint', 'Follow-up', 'Escalation', 'Sales', 'Support', 'Billing'.
    3. "summary": A three-sentence summary of the entire call from start to finish.
    4. "customer_sentiment": Rate customer sentiment from 1-10 (1=very negative, 10=very positive).
    5. "call_complexity": Rate call complexity from 1-5 (1=simple, 5=very complex).
    6. "estimated_duration": Estimate call duration in minutes based on content.

    Transcript:
    ---
    {transcript}
    ---
    Return ONLY the JSON object.
    """

def _get_business_outcome_prompt(transcript: str) -> str:
    """Generate business outcome analysis prompt"""
    compliance_statement = "Thank you for calling [Company]. Have a great day."
    
    return f"""
    You are a business analyst. Analyze the provided transcript and return a JSON object.
    The JSON object must contain:
    1. "business_outcome": Classify the final outcome as one of: 'Sale_Completed', 'Customer_Retained', 'Issue_Resolved_First_Call', 'Escalation_Required', 'Follow-up_Promised', 'Customer_Churn_Risk', 'No_Resolution', 'Information_Provided', 'Appointment_Scheduled'.
    2. "outcome_confidence": Rate confidence in the outcome classification from 1-10.
    3. "justification": Brief explanation for the business outcome classification.
    4. "compliance_adherence": Boolean indicating if agent followed compliance requirements.
    5. "risk_identified": Boolean indicating if any legal or reputational risks were identified.
    6. "risk_details": If risks identified, provide specific details and quotes.
    7. "next_steps": What should happen next based on this call.
    8. "customer_satisfaction_indicator": Rate likely customer satisfaction from 1-10 based on call flow.

    Mandatory compliance check: '{compliance_statement}'
    
    Transcript:
    ---
    {transcript}
    ---
    Return ONLY the JSON object.
    """

def _get_parameter_scoring_prompt(transcript: str, parameter_name: str, anchors: str) -> str:
    """Generate parameter scoring prompt"""
    return f"""
    You are a meticulous QA Analyst. Score the parameter '{parameter_name}' on a scale of 0-100.
    
    Behavioral Anchors:
    {anchors}

    Return a JSON object with:
    - "score": Numeric score (0-100)
    - "confidence": Confidence in score (1-10)
    - "justification": Detailed reasoning referencing behavioral anchors
    - "primary_evidence": Best supporting quote from transcript
    - "context_before": 30 seconds of dialogue before the evidence
    - "context_after": 30 seconds of dialogue after the evidence
    - "coaching_opportunity": Specific, actionable coaching recommendation
    - "improvement_impact": How fixing this would impact customer experience
    - "severity": If score is low, rate severity of the issue (1-5)

    Transcript:
    ---
    {transcript}
    ---
    Return ONLY the JSON object.
    """

# ======================================================================================
# LEGACY COMPATIBILITY FUNCTIONS (for existing code)
# ======================================================================================

@st.cache_data(show_spinner="Running initial triage...")
def run_initial_triage(transcript: str) -> dict:
    """
    Enhanced triage analysis - now uses dual models automatically.
    """
    prompt = _get_triage_prompt(transcript)
    return call_ai_engine(prompt)

@st.cache_data(show_spinner="Analyzing business outcome...")
def run_business_outcome_analysis(transcript: str) -> dict:
    """
    Enhanced business outcome analysis - now uses dual models automatically.
    """
    prompt = _get_business_outcome_prompt(transcript)
    return call_ai_engine(prompt)

@st.cache_data(show_spinner="Scoring parameter...")
def score_single_parameter(transcript: str, parameter_name: str, anchors: str) -> dict:
    """
    Enhanced parameter scoring - now uses dual models automatically.
    """
    prompt = _get_parameter_scoring_prompt(transcript, parameter_name, anchors)
    return call_ai_engine(prompt)

# ======================================================================================
# CUSTOM PARAMETER MANAGEMENT
# ======================================================================================

def add_custom_parameter(rubric_name: str, parameter: dict):
    """Add a custom parameter to a rubric"""
    if rubric_name not in CUSTOM_PARAMETERS:
        CUSTOM_PARAMETERS[rubric_name] = []
    
    CUSTOM_PARAMETERS[rubric_name].append(parameter)

def get_available_custom_rubrics():
    """Get list of available custom rubrics"""
    return list(CUSTOM_PARAMETERS.keys())

def create_custom_rubric(rubric_name: str, industry: str, parameters: list):
    """Create a new custom rubric"""
    CUSTOM_PARAMETERS[rubric_name] = []
    for param in parameters:
        param['custom'] = True
        CUSTOM_PARAMETERS[rubric_name].append(param)

# ======================================================================================
# ENHANCED UTILITY FUNCTIONS
# ======================================================================================

def calculate_overall_metrics(parameter_scores: dict) -> dict:
    """
    Enhanced calculate overall weighted metrics from parameter scores.
    """
    if not parameter_scores:
        return {"error": "No parameter scores provided"}
    
    total_weighted_score = 0
    total_weight = 0
    valid_scores = 0
    low_scores = []
    coaching_opportunities = []
    custom_parameter_performance = {}
    
    for param_name, param_data in parameter_scores.items():
        if 'error' not in param_data and 'score' in param_data:
            score = param_data.get('score', 0)
            weight = param_data.get('weight', 10)  # Default weight
            is_custom = param_data.get('custom_parameter', False)
            
            total_weighted_score += (score * weight)
            total_weight += weight
            valid_scores += 1
            
            # Track custom parameter performance
            if is_custom:
                custom_parameter_performance[param_name] = {
                    'score': score,
                    'weight': weight,
                    'confidence': param_data.get('confidence', 5)
                }
            
            # Collect low scores for attention
            if score < 70:
                low_scores.append({
                    'parameter': param_name,
                    'score': score,
                    'coaching': param_data.get('coaching_opportunity', 'No coaching provided'),
                    'is_custom': is_custom
                })
            
            # Collect coaching opportunities
            if param_data.get('coaching_opportunity'):
                coaching_opportunities.append({
                    'parameter': param_name,
                    'coaching': param_data['coaching_opportunity'],
                    'is_custom': is_custom
                })
    
    if total_weight == 0:
        return {"error": "No valid weighted scores found"}
    
    overall_score = round(total_weighted_score / total_weight, 2)
    
    # Determine quality bucket
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
    
    return {
        "overall_score": overall_score,
        "quality_bucket": quality_bucket,
        "total_parameters_scored": valid_scores,
        "parameters_needing_attention": len(low_scores),
        "low_scoring_parameters": low_scores,
        "coaching_opportunities": coaching_opportunities[:3],  # Top 3
        "custom_parameter_performance": custom_parameter_performance,
        "score_distribution": {
            "excellent": sum(1 for _, data in parameter_scores.items() 
                           if data.get('score', 0) >= 90),
            "good": sum(1 for _, data in parameter_scores.items() 
                       if 80 <= data.get('score', 0) < 90),
            "average": sum(1 for _, data in parameter_scores.items() 
                          if 70 <= data.get('score', 0) < 80),
            "below_average": sum(1 for _, data in parameter_scores.items() 
                                if 60 <= data.get('score', 0) < 70),
            "poor": sum(1 for _, data in parameter_scores.items() 
                       if data.get('score', 0) < 60)
        }
    }

def get_available_analysis_depths():
    """Return available analysis depth options."""
    return list(ANALYSIS_PARAMETERS.keys())

def get_parameters_for_depth(depth: str):
    """Get parameter configuration for specified analysis depth."""
    return ANALYSIS_PARAMETERS.get(depth, ANALYSIS_PARAMETERS["Standard Analysis"])

# ======================================================================================
# ADMIN/DEVELOPER FUNCTIONS
# ======================================================================================

def run_model_comparison_analysis(transcript: str, depth: str = "Standard Analysis", 
                                custom_rubric: str = None) -> dict:
    """
    Special function for admin to get detailed model comparison.
    This is your secret function to evaluate model performance!
    """
    return run_comprehensive_analysis(
        transcript=transcript,
        depth=depth,
        custom_rubric=custom_rubric,
        admin_mode=True
    )

def get_model_performance_stats(comparison_results: dict) -> dict:
    """
    Analyze model performance from comparison results.
    """
    stats = {
        "gpt_wins": 0,
        "gemini_wins": 0,
        "ties": 0,
        "average_variance": 0,
        "reliability_breakdown": {},
        "performance_summary": {}
    }
    
    variances = []
    
    # Analyze each comparison in the results
    for stage, data in comparison_results.get('_model_comparison_data', {}).items():
        if 'gpt' in data and 'gemini' in data:
            gpt_data = data['gpt']
            gemini_data = data['gemini']
            
            # Compare response completeness and quality
            gpt_valid = 'error' not in gpt_data
            gemini_valid = 'error' not in gemini_data
            
            if gpt_valid and not gemini_valid:
                stats["gpt_wins"] += 1
            elif gemini_valid and not gpt_valid:
                stats["gemini_wins"] += 1
            elif gpt_valid and gemini_valid:
                stats["ties"] += 1
                
                # Calculate variance if scores exist
                if 'score' in gpt_data and 'score' in gemini_data:
                    variance = abs(gpt_data['score'] - gemini_data['score'])
                    variances.append(variance)
    
    if variances:
        stats["average_variance"] = round(sum(variances) / len(variances), 2)
    
    return statsimport json
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import time

# ======================================================================================
# API CLIENT SETUP & CONFIGURATION
# ======================================================================================

# Load API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# Define the available AI models for selection
AVAILABLE_MODELS = {
    "GPT-4o (OpenAI)": "gpt-4o",
    "Gemini 1.5 Pro (Google)": "gemini-1.5-pro"
}

# Custom Parameters Storage (will be moved to database later)
CUSTOM_PARAMETERS = {
    "Sales Outbound": [
        {"name": "emi_offer", "weight": 15, "anchors": "90-100: Proactively offered 0% EMI when applicable. 70-89: Offered EMI when asked. 50-69: Mentioned EMI options briefly. 30-49: Failed to offer EMI when opportunity existed. 0-29: No EMI discussion despite clear opportunity."},
        {"name": "urgency_creation", "weight": 20, "anchors": "90-100: Effectively created urgency with limited-time offers. 70-89: Some urgency created. 50-69: Mild urgency attempts. 30-49: Weak urgency creation. 0-29: No urgency created."},
    ],
    "Banking Support": [
        {"name": "identity_verification", "weight": 25, "anchors": "90-100: Properly verified customer identity before sharing financial details. 70-89: Good verification with minor gaps. 50-69: Basic verification done. 30-49: Insufficient verification. 0-29: No identity verification before sharing sensitive info."},
        {"name": "data_protection", "weight": 20, "anchors": "90-100: Excellent data protection practices throughout call. 70-89: Good data protection with minor issues. 50-69: Adequate protection. 30-49: Some data protection concerns. 0-29: Poor data protection practices."},
    ]
}

# Define analysis parameters based on depth
ANALYSIS_PARAMETERS = {
    "Quick Scan": [
        {"name": "greeting", "weight": 15, "anchors": "90-100: Professional greeting with company name, agent name, and polite inquiry. 70-89: Good greeting with most elements present. 50-69: Basic greeting missing some elements. 30-49: Poor greeting, unprofessional. 0-29: No proper greeting."},
        {"name": "empathy", "weight": 20, "anchors": "90-100: Shows genuine understanding and concern for customer's situation. 70-89: Demonstrates good empathy with appropriate responses. 50-69: Shows some empathy but could be more genuine. 30-49: Limited empathy shown. 0-29: No empathy demonstrated."},
        {"name": "resolution", "weight": 30, "anchors": "90-100: Completely resolves customer issue with clear solution. 70-89: Good resolution with minor gaps. 50-69: Partial resolution provided. 30-49: Minimal resolution attempts. 0-29: No resolution provided."}
    ],
    "Standard Analysis": [
        {"name": "greeting", "weight": 10, "anchors": "90-100: Professional greeting with company name, agent name, and polite inquiry. 70-89: Good greeting with most elements present. 50-69: Basic greeting missing some elements. 30-49: Poor greeting, unprofessional. 0-29: No proper greeting."},
        {"name": "empathy", "weight": 15, "anchors": "90-100: Shows genuine understanding and concern for customer's situation. 70-89: Demonstrates good empathy with appropriate responses. 50-69: Shows some empathy but could be more genuine. 30-49: Limited empathy shown. 0-29: No empathy demonstrated."},
        {"name": "resolution", "weight": 25, "anchors": "90-100: Completely resolves customer issue with clear solution. 70-89: Good resolution with minor gaps. 50-69: Partial resolution provided. 30-49: Minimal resolution attempts. 0-29: No resolution provided."},
        {"name": "compliance", "weight": 20, "anchors": "90-100: Follows all compliance requirements including mandatory statements. 70-89: Good compliance with minor gaps. 50-69: Some compliance issues noted. 30-49: Multiple compliance violations. 0-29: Major compliance failures."},
        {"name": "professionalism", "weight": 15, "anchors": "90-100: Maintains professional tone throughout, uses appropriate language. 70-89: Generally professional with minor lapses. 50-69: Somewhat professional but inconsistent. 30-49: Limited professionalism. 0-29: Unprofessional behavior."},
        {"name": "communication_clarity", "weight": 15, "anchors": "90-100: Clear, concise communication that's easy to understand. 70-89: Generally clear with minor confusion. 50-69: Somewhat clear but could be improved. 30-49: Often unclear or confusing. 0-29: Very poor communication."}
    ],
    "Deep Dive": [
        {"name": "greeting", "weight": 8, "anchors": "90-100: Professional greeting with company name, agent name, and polite inquiry. 70-89: Good greeting with most elements present. 50-69: Basic greeting missing some elements. 30-49: Poor greeting, unprofessional. 0-29: No proper greeting."},
        {"name": "empathy", "weight": 12, "anchors": "90-100: Shows genuine understanding and concern for customer's situation. 70-89: Demonstrates good empathy with appropriate responses. 50-69: Shows some empathy but could be more genuine. 30-49: Limited empathy shown. 0-29: No empathy demonstrated."},
        {"name": "resolution", "weight": 20, "anchors": "90-100: Completely resolves customer issue with clear solution. 70-89: Good resolution with minor gaps. 50-69: Partial resolution provided. 30-49: Minimal resolution attempts. 0-29: No resolution provided."},
        {"name": "compliance", "weight": 15, "anchors": "90-100: Follows all compliance requirements including mandatory statements. 70-89: Good compliance with minor gaps. 50-69: Some compliance issues noted. 30-49: Multiple compliance violations. 0-29: Major compliance failures."},
        {"name": "professionalism", "weight": 12, "anchors": "90-100: Maintains professional tone throughout, uses appropriate language. 70-89: Generally professional with minor lapses. 50-69: Somewhat professional but inconsistent. 30-49: Limited professionalism. 0-29: Unprofessional behavior."},
        {"name": "active_listening", "weight": 10, "anchors": "90-100: Demonstrates excellent active listening with appropriate responses and clarifying questions. 70-89: Good listening skills shown. 50-69: Some listening but misses cues. 30-49: Poor listening skills. 0-29: No evidence of active listening."},
        {"name": "product_knowledge", "weight": 13, "anchors": "90-100: Expert knowledge of products/services with accurate information. 70-89: Good product knowledge with minor gaps. 50-69: Basic knowledge but some inaccuracies. 30-49: Limited product knowledge. 0-29: Poor or incorrect product knowledge."},
        {"name": "call_control", "weight": 10, "anchors": "90-100: Maintains excellent control of call flow, manages time effectively. 70-89: Good call control with minor issues. 50-69: Some call control but could be better. 30-49: Poor call control, loses direction. 0-29: No call control, chaotic flow."}
    ]
}

# ======================================================================================
# STEALTH DUAL MODEL SYSTEM
# ======================================================================================

class ModelConsensus:
    """Handles dual model analysis and consensus building"""
    
    @staticmethod
    def calculate_consensus_score(gpt_score: float, gemini_score: float) -> dict:
        """Calculate consensus score with confidence metrics"""
        scores = [gpt_score, gemini_score]
        consensus = round(statistics.mean(scores), 1)
        variance = abs(gpt_score - gemini_score)
        
        # Confidence based on agreement between models
        if variance <= 5:
            confidence = 10  # High confidence - models agree closely
            reliability = "Very High"
        elif variance <= 10:
            confidence = 8   # Good confidence
            reliability = "High"
        elif variance <= 20:
            confidence = 6   # Medium confidence
            reliability = "Medium"
        else:
            confidence = 4   # Low confidence - significant disagreement
            reliability = "Low - Review Required"
        
        return {
            "consensus_score": consensus,
            "confidence": confidence,
            "reliability": reliability,
            "variance": variance,
            "gpt_score": gpt_score,
            "gemini_score": gemini_score
        }
    
    @staticmethod
    def combine_text_responses(gpt_response: str, gemini_response: str) -> str:
        """Intelligently combine text responses from both models"""
        # For now, prefer GPT's response but could implement more sophisticated merging
        if len(gpt_response) > len(gemini_response):
            return gpt_response
        return gemini_response
    
    @staticmethod
    def analyze_model_performance(gpt_result: dict, gemini_result: dict) -> dict:
        """Analyze which model performed better"""
        performance_metrics = {
            "gpt_reliability": 0,
            "gemini_reliability": 0,
            "preferred_model": "GPT",
            "reasoning": []
        }
        
        # Check response completeness
        gpt_complete = 'error' not in gpt_result
        gemini_complete = 'error' not in gemini_result
        
        if gpt_complete and not gemini_complete:
            performance_metrics["preferred_model"] = "GPT"
            performance_metrics["reasoning"].append("GPT provided complete response")
        elif gemini_complete and not gpt_complete:
            performance_metrics["preferred_model"] = "Gemini"
            performance_metrics["reasoning"].append("Gemini provided complete response")
        
        return performance_metrics
