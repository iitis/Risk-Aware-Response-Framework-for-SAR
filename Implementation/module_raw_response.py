import json
import re
import pandas as pd
from litellm import completion
from pydantic import BaseModel


class RawResponse(BaseModel):
    action: str
    speech: str


class RawResponseGenerator:

    def __init__(
        self,
        model_variant: str,
        output_file: str,
        api_base: str,
        excel_file: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        self.model_variant = model_variant
        self.output_file = output_file
        self.api_base = api_base
        self.excel_file = excel_file
        self.temperature = temperature
        self.max_tokens = max_tokens

        # schema
        self.schema_json = RawResponse.model_json_schema()
        self.schema_str = json.dumps(self.schema_json, indent=2)

        # system prompt
        self.system_prompt = f"""
You are an assistive home robot living in a smart home with an elderly user.

You can control home devices such as lights, curtains, thermostat, and reminders.
You can also provide general information like weather, daily advice, and guidance.

Important rules:
Do NOT say you are an AI model.
Do NOT say you cannot access the internet.
Do NOT say you cannot physically perform actions.
Always respond as if you are the robot in the home.

If the user asks for an action (e.g., open curtains), respond as if you are performing or assisting with the action.

Your task is to answer user questions and provide helpful suggestions.
Respond briefly, politely, and naturally like a home assistant robot.
Do not classify risk or safety. Simply provide the robot's response.
Keep the tone polite, supportive, and easy to understand.

You MUST respond with valid JSON only, no markdown, no extra text.
Use this exact JSON schema:
{self.schema_str}

- "action": A short description of what physical action the robot performs (e.g., "open_curtains", "set_thermostat_22", "none").
- "speech": What the robot says out loud to the user.
"""

    def load_queries(self):
        df = pd.read_excel(self.excel_file)
        return df["Query"].dropna().tolist()

    def clean_response(self, raw: str):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            raw = match.group(1).strip()
        return raw

    def generate_single(self, query: str):
        response = completion(
            model=f"ollama/{self.model_variant}",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
            ],
            api_base=self.api_base,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "RobotResponse",
                    "schema": self.schema_json,
                },
            },
        )

        raw = response.choices[0].message.content
        raw = self.clean_response(raw)

        parsed = RawResponse.model_validate_json(raw)

        return {
            "query": query,
            "action": parsed.action,
            "speech": parsed.speech
        }

    def run(self):
        queries = self.load_queries()
        results = []

        for q in queries:
            try:
                result = self.generate_single(q)
                results.append(result)
                print(f"Processed: {q}")
            except Exception as e:
                print(f"Error on query: {q} -> {e}")

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print("Finished. Results saved to:", self.output_file)