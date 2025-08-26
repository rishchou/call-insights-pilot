import json
import streamlit as st
from openai import OpenAI
import google.generativeai as genai

# --- API Client Setup ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

AVAILABLE_MODELS = {
    "GPT-4o (OpenAI)": "gpt-4o",
    "Gemini 1.5 Pro (Google)": "gemini-1.5-pro"
}

# --- Helper Functions ---
def _json_guard(text_response: str) -> dict:
    """Ensures the AI response is a valid JSON object."""
    try:
        start = text_response.find('{')
        end = text_response.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text_response[start:end+1])
    except (json.JSONDecodeError, IndexError):
        return {"error": "Failed to parse AI response as JSON.", "raw_response": text_response}
    return {"error": "No valid JSON object found in the response.", "raw_response": text_response}

def call_ai_engine(prompt: str, selected_model: str) -> dict:
    """Calls the selected AI engine and returns a structured dictionary."""
    if "Gemini" in selected_model:
        if not GEMINI_API_KEY: return {"error": "Gemini API key not configured."}
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            AVAILABLE_MODELS[selected_model],
            generation_config={"temperature": 0.2, "response_mime_type": "application/json"}
        )
        response = model.generate_content(prompt)
        return _json_guard(response.text)
        
    elif "GPT" in selected_model:
        if not OPENAI_API_KEY: return {"error": "OpenAI API key not configured."}
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

# --- Core Analysis Functions ---

def run_initial_triage(transcript: str, selected_model: str) -> dict:
    """
    Performs the first pass analysis to get summary, category, and call phases.
    """
    prompt = f"""
    You are a conversation analyst. Your task is to analyze the provided transcript and return a JSON object.
    The JSON object must contain:
    1. "purpose": A one-sentence summary of why the customer is calling.
    2. "category": Classify the call into one of: 'Query', 'Complaint', 'Follow-up', 'Escalation', 'Sales'.
    3. "summary": A three-sentence summary of the entire call from start to finish.
    4. "call_phases": Identify the start and end timestamps for each phase: 'opening', 'verification', 'problem_identification', 'resolution', 'closing'.

    Transcript:
    ---
    {transcript}
    ---
    Return ONLY the JSON object.
    """
    return call_ai_engine(prompt, selected_model)

def run_business_outcome_analysis(transcript: str, selected_model: str) -> dict:
    """
    Determines the final business outcome and checks for compliance/risk.
    """
    # In a real app, the compliance statement would be a configurable variable
    compliance_statement = "Thank you for calling [Company]. Have a great day."
    
    prompt = f"""
    You are a business analyst. Analyze the provided transcript and return a JSON object.
    The JSON object must contain:
    1. "business_outcome": Classify the final outcome as one of: 'Sale_Completed', 'Customer_Retained', 'Issue_Resolved_First_Call', 'Escalation_Required', 'Follow-up_Promised', 'Customer_Churn_Risk', 'No_Resolution'. Provide a brief justification.
    2. "compliance_adherence": A boolean (true/false) indicating if the agent recited the mandatory statement: '{compliance_statement}'.
    3. "risk_identified": A boolean (true/false) indicating if any legal or reputational risks were mentioned. If true, provide the quote.

    Transcript:
    ---
    {transcript}
    ---
    Return ONLY the JSON object.
    """
    return call_ai_engine(prompt, selected_model)

# The function for scoring a single parameter would go here.
# It would be called in a loop from the main app file.
def score_single_parameter(transcript: str, parameter_name: str, anchors: str, selected_model: str) -> dict:
    """
    Scores a single, specific parameter based on a transcript and behavioral anchors.
    """
    prompt = f"""
    You are a meticulous QA Analyst. Your only task is to score the parameter '{parameter_name}' on a scale of 0-100 based on the provided transcript.
    Use the following behavioral anchors to determine your score:
    {anchors}

    After determining the score, you MUST provide a detailed `evidence_sequence` in JSON format. The sequence must include:
    - "score": The numeric score from 0-100.
    - "justification": Your reasoning for the score, referencing the behavioral anchors.
    - "primary_evidence": The single best quote from the transcript that justifies your score.
    - "context_before": The dialogue from the 30 seconds immediately preceding the primary evidence.
    - "context_after": The dialogue from the 30 seconds immediately following the primary evidence.
    - "coaching_opportunity": A specific, actionable recommendation for the agent based on this interaction.

    Transcript:
    ---
    {transcript}
    ---
    Return ONLY the JSON object with the evidence sequence.
    """
    return call_ai_engine(prompt, selected_model)
