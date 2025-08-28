import json
import streamlit as st
from openai import OpenAI
import google.generativeai as genai

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
# HELPER FUNCTIONS
# ======================================================================================

def _json_guard(text_response: str) -> dict:
    """
    Safely parses a JSON string from an AI response, even if it's embedded in markdown.
    """
    try:
        # Find the start and end of the JSON object
        start = text_response.find('{')
        end = text_response.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text_response[start:end+1])
    except (json.JSONDecodeError, IndexError):
        # Return a structured error if parsing fails
        return {"error": "Failed to parse AI response as JSON.", "raw_response": text_response}
    return {"error": "No valid JSON object found in the response.", "raw_response": text_response}

def call_ai_engine(prompt: str, selected_model: str, max_retries: int = 3) -> dict:
    """
    Enhanced AI engine call with retry logic and better error handling.
    """
    for attempt in range(max_retries):
        try:
            if "Gemini" in selected_model:
                if not GEMINI_API_KEY:
                    return {"error": "Gemini API key is not configured."}
                
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel(
                    AVAILABLE_MODELS[selected_model],
                    generation_config={"temperature": 0.2, "response_mime_type": "application/json"}
                )
                response = model.generate_content(prompt)
                return _json_guard(response.text)
                
            elif "GPT" in selected_model:
                if not OPENAI_API_KEY:
                    return {"error": "OpenAI API key is not configured."}
                
                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=AVAILABLE_MODELS[selected_model],
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                return _json_guard(response.choices[0].message.content)
            else:
                return {"error": "Invalid model selected."}
                
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                st.error(f"AI engine failed after {max_retries} attempts: {str(e)}")
                return {"error": f"Failed after {max_retries} attempts: {str(e)}"}
            else:
                st.warning(f"Attempt {attempt + 1} failed, retrying... ({str(e)})")
                continue

# ======================================================================================
# MAIN ORCHESTRATOR FUNCTION
# ======================================================================================

def run_comprehensive_analysis(transcript: str, selected_model: str, depth: str = "Standard Analysis") -> dict:
    """
    Main orchestrator function that runs all analysis stages based on selected depth.
    Returns comprehensive analysis results.
    """
    if not transcript.strip():
        return {"error": "Empty transcript provided"}
    
    results = {}
    
    try:
        # Stage 1: Initial Triage
        st.write("🔍 **Stage 1:** Running initial triage analysis...")
        results['triage'] = run_initial_triage(transcript, selected_model)
        
        if 'error' in results['triage']:
            st.error(f"Triage analysis failed: {results['triage']['error']}")
            return results
        
        # Stage 2: Business Outcome Analysis  
        st.write("💼 **Stage 2:** Analyzing business outcomes...")
        results['business_outcome'] = run_business_outcome_analysis(transcript, selected_model)
        
        if 'error' in results['business_outcome']:
            st.error(f"Business outcome analysis failed: {results['business_outcome']['error']}")
        
        # Stage 3: Parameter Scoring (based on selected depth)
        parameters = ANALYSIS_PARAMETERS.get(depth, ANALYSIS_PARAMETERS["Standard Analysis"])
        st.write(f"📊 **Stage 3:** Scoring {len(parameters)} parameters ({depth})...")
        
        results['parameter_scores'] = {}
        progress_bar = st.progress(0)
        
        for i, param in enumerate(parameters):
            with st.spinner(f"Analyzing {param['name']}..."):
                score_result = score_single_parameter(
                    transcript, 
                    param['name'], 
                    param['anchors'], 
                    selected_model
                )
                
                # Add weight to the result
                if 'error' not in score_result:
                    score_result['weight'] = param['weight']
                
                results['parameter_scores'][param['name']] = score_result
                progress_bar.progress((i + 1) / len(parameters))
        
        # Calculate overall weighted score
        results['overall_metrics'] = calculate_overall_metrics(results['parameter_scores'])
        
        st.success("✅ Comprehensive analysis completed!")
        return results
        
    except Exception as e:
        st.error(f"Comprehensive analysis failed: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}"}

# ======================================================================================
# CORE ANALYSIS FUNCTIONS
# ======================================================================================

@st.cache_data(show_spinner="Running initial triage...")
def run_initial_triage(transcript: str, selected_model: str) -> dict:
    """
    Enhanced triage analysis with additional insights.
    """
    prompt = f"""
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
    return call_ai_engine(prompt, selected_model)

@st.cache_data(show_spinner="Analyzing business outcome...")
def run_business_outcome_analysis(transcript: str, selected_model: str) -> dict:
    """
    Enhanced business outcome analysis with more detailed classifications.
    """
    compliance_statement = "Thank you for calling [Company]. Have a great day."
    
    prompt = f"""
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
    return call_ai_engine(prompt, selected_model)

@st.cache_data(show_spinner="Scoring parameter: {parameter_name}...")
def score_single_parameter(transcript: str, parameter_name: str, anchors: str, selected_model: str) -> dict:
    """
    Enhanced parameter scoring with more detailed evidence extraction.
    """
    prompt = f"""
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
    return call_ai_engine(prompt, selected_model)

# ======================================================================================
# UTILITY FUNCTIONS
# ======================================================================================

def calculate_overall_metrics(parameter_scores: dict) -> dict:
    """
    Calculate overall weighted metrics from parameter scores.
    """
    if not parameter_scores:
        return {"error": "No parameter scores provided"}
    
    total_weighted_score = 0
    total_weight = 0
    valid_scores = 0
    low_scores = []
    coaching_opportunities = []
    
    for param_name, param_data in parameter_scores.items():
        if 'error' not in param_data and 'score' in param_data:
            score = param_data.get('score', 0)
            weight = param_data.get('weight', 10)  # Default weight
            
            total_weighted_score += (score * weight)
            total_weight += weight
            valid_scores += 1
            
            # Collect low scores for attention
            if score < 70:
                low_scores.append({
                    'parameter': param_name,
                    'score': score,
                    'coaching': param_data.get('coaching_opportunity', 'No coaching provided')
                })
            
            # Collect coaching opportunities
            if param_data.get('coaching_opportunity'):
                coaching_opportunities.append({
                    'parameter': param_name,
                    'coaching': param_data['coaching_opportunity']
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
