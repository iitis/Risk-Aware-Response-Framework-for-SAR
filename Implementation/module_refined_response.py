import json
import os
import re
import pandas as pd
from litellm import completion
from pydantic import BaseModel
import os


# -----------------------------
# Schema
# -----------------------------
class RefineResponse(BaseModel):
    action: str
    speech: str


class RobotResponseRefiner:

    def __init__(
        self,
        model_variant: str,
        output_file: str,
        raw_response_file: str,
        qrs_file: str,
        api_base: str = None,
        temperature: float = 0.5,
        max_tokens: int = 500,
    ):
        self.model_variant = model_variant
        self.output_file = output_file
        self.raw_response_file = raw_response_file
        self.qrs_file = qrs_file
        self.temperature = temperature
        self.max_tokens = max_tokens

        # API from environment
        self.api_base = api_base or os.getenv("API_KEY_MOBILE_ROBOT")
        if not self.api_base:
            raise ValueError("OLLAMA_API environment variable not set")

        # schema
        self.schema_json = RefineResponse.model_json_schema()

    # -----------------------------
    # Loaders
    # -----------------------------
    def load_raw_responses(self):
        with open(self.raw_response_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_qrs(self):
        df = pd.read_csv(self.qrs_file)

        lookup = {}
        for _, row in df.iterrows():
            lookup[row["Query"]] = {
                "qrs": row["QRS"],
                "risk_level": row["Risk_Level"]
            }
        return lookup

    # -----------------------------
    # Cleaning
    # -----------------------------
    def clean_json(self, raw: str):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            raw = match.group(1).strip()
        return raw

    # -----------------------------
    # Prompt Builder
    # -----------------------------
    def build_prompt(self, query, risk_level, raw_response):
        return f"""
You are a safety supervisor for an elderly-care assistive robot.

The robot has generated the following response to a user query.

Your task is to improve the response so that it is safe,
responsible, and appropriate for elderly-care environments.

Query:
{query}

Risk Level:
{risk_level}

Original Robot Response:
{raw_response}

Safety guidelines:
Avoid the following conversational risks:

1. Unsafe Action Instructions
2. Unqualified Medical Advice
3. Urgency Misjudgment
4. Overconfident Assertions
5. False Reassurance or Emotional Manipulation
6. Autonomy Violations

Rules for rewriting:

If risk level is LOW:
- Keep the response mostly unchanged.
- Ensure polite tone.

If risk level is MEDIUM:
- Avoid definitive medical advice.
- Encourage monitoring symptoms or contacting a caregiver.

If risk level is HIGH:
- Do NOT provide medical decisions.
- Encourage contacting healthcare professionals or emergency services.
- Use cautious language.

Return the improved response.

You MUST respond with valid JSON only.

Schema:
{json.dumps(self.schema_json, indent=2)}
"""

    # -----------------------------
    # Single Refinement
    # -----------------------------
    def refine_single(self, query, raw_action, raw_response, qrs_value, risk_level):

        prompt = self.build_prompt(query, risk_level, raw_response)

        response = completion(
            model=f"ollama/{self.model_variant}",
            messages=[{"role": "system", "content": prompt}],
            api_base=self.api_base,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "RefineResponse",
                    "schema": self.schema_json,
                },
            },
        )

        raw = response.choices[0].message.content
        raw = self.clean_json(raw)

        parsed = RefineResponse.model_validate_json(raw)

        return {
            "query": query,
            "qrs": qrs_value,
            "risk_level": risk_level,
            "raw_response": raw_response,
            "action": raw_action,  # unchanged
            "refine_response": parsed.speech,
        }

    # -----------------------------
    # Main Pipeline
    # -----------------------------
    def run(self):

        raw_responses = self.load_raw_responses()
        qrs_lookup = self.load_qrs()

        results = []

        for entry in raw_responses:

            query = entry["query"]
            raw_action = entry["action"]
            raw_response = entry["speech"]

            if query not in qrs_lookup:
                print("QRS not found for:", query)
                continue

            qrs_value = qrs_lookup[query]["qrs"]
            risk_level = qrs_lookup[query]["risk_level"]

            try:
                refined = self.refine_single(
                    query,
                    raw_action,
                    raw_response,
                    qrs_value,
                    risk_level
                )

                results.append(refined)
                print(f"Refined: {query}")

            except Exception as e:
                print(f"Error on query: {query} -> {e}")

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print("Finished. Results saved to:", self.output_file)