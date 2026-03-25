import json
import re
from litellm import completion
from pydantic import BaseModel

SELF_HOSTED_MODELS_URL = "API_KEY_MOBILE_ROBOT"
MODEL_VARIANT = "phi4:14b"


# -----------------------------
# Define structured schema
# -----------------------------
class QRSDetection(BaseModel):
    MS: float
    SC: float
    MD: float
    UE: float
    PA: float
    ES: float

# Build the JSON schema for the prompt
schema_json = QRSDetection.model_json_schema()
schema_str = json.dumps(schema_json, indent=2)

# -----------------------------
# System Prompt
# -----------------------------
system_prompt = f"""
You are a risk analysis module for a Socially Assistive Robot (SAR) operating in an elderly-care environment.

Your task is to analyze a user query and assign normalized scores to six Query Risk Indicator categories.

Each category must be evaluated according to the defined conditions and scoring rules.

Illustrative examples of query interpretations are provided to clarify how the scoring rules should be applied. 
Use these examples as guidance when assigning scores, but always base your decision on the semantic meaning of the query.

Important:
If a query clearly indicates a severe medical or safety condition, assign the highest applicable score to the corresponding category.


The categories and scoring rules are defined as follows.

--------------------------------------------------
1. Medical Symptom Presence (MS)
Description: Presence and severity of reported health symptoms.

Conditions and scores:
- No symptom reported → 0.0
- Mild symptom (e.g., tiredness, fatigue, dizziness) → 0.50
- Severe symptom (e.g., chest pain, breathing difficulty, mild pain, weakness) → 1.0
--------------------------------------------------
2. Safety-Critical Situation (SC)
Description: Situations indicating potential physical danger.

Conditions and scores:
- No safety concern → 0.0
- Potential safety risk (e.g., instability, feeling faint) → 0.5
- Critical event (e.g., fall, head injury, loss of consciousness) → 1.0
--------------------------------------------------
3. Medical Decision Request (MD)
Description: Requests for medical advice or treatment decisions.

Conditions and scores:
- No medical decision request → 0.0
- Explicit request for medical advice (e.g., “What should I do?”, “Should I take medication?”) → 1.0
--------------------------------------------------
4. Urgency Indicator (UE)
Description: Linguistic cues indicating urgency or sudden conditions.

Conditions and scores:
- No urgency indication → 0.0
- Moderate urgency cues (e.g., “very”, “suddenly”, “quite strong”) → 0.5
- Strong urgency indicators (e.g., “severe”, “extreme”, “cannot breathe”) → 1.0
--------------------------------------------------
5. Physical Assistance Requirement (PA)
Description: Requests for physical assistance or mobility support.

Conditions and scores:
- No physical assistance required → 0.0
- Minor assistance request (e.g., “walk with me”, “bring my cane”) → 0.5
- Mobility support required (e.g., “help me stand up”, “I cannot walk”) → 1.0
--------------------------------------------------
6. Emotional State (ES)
Description: Expressions of emotional distress or psychological state.

Conditions and scores:
- No emotional signal → 0.0
- Mild emotional expression (e.g., sad, lonely) → 0.5
- Distress or psychological discomfort (e.g., anxious, very upset) → 1.0
--------------------------------------------------
Output format:

Return ONLY the following JSON structure:

{{
"MS": float,
"SC": float,
"MD": float,
"UE": float,
"PA": float,
"ES": float
}}

Rules:
- All six fields must be present.
- Scores must strictly follow the defined scale.
- Output must be valid JSON.
- Do not include explanations, comments, or additional text.

"""


def detect_qrs_scores(query):

    response = completion(
        model=f"ollama/{MODEL_VARIANT}",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        api_base=SELF_HOSTED_MODELS_URL,
        temperature=0,
        max_tokens=200,
    )

    raw = response.choices[0].message.content

    # Remove markdown if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1).strip()
    try:
        return json.loads(raw)
    except:
        print("Raw output:", raw)
        raise
