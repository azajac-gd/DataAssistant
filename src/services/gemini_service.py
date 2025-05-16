from google import genai
from typing import List, Dict
import os
from dotenv import load_dotenv
import json

load_dotenv()


def generate_data_with_gemini(tables: List[Dict], prompt: str, temperature: float) -> str:
    client = genai.Client(
        vertexai=os.getenv("USE_VERTEXAI", "False") == "True",
        project=os.getenv("PROJECT_ID"),
        location=os.getenv("LOCATION")
    )

    tables_desc = "\n".join([
        f"Table '{table['table_name']}': " +
        ", ".join([f"{col['name']} ({col['type']})" for col in table['columns']])
        for table in tables
    ])

    full_prompt = f"""
You are a data generator. Given the following table definitions, generate realistic and consistent sample data.
Return the data as a JSON array in this format:
[
  {{
    "table_name": "<table_name>",
    "rows": [
      {{ "col1": "value1", "col2": "value2", ... }},
      ...
    ]
  }},
  ...
]
Rules:
- Each table must contain 2 sample rows unless specified otherwise.
- The data should be realistic and consistent with the table definitions.
- Do not create None or empty values in the data unless specified otherwise.

Table definitions:
{tables_desc}

Additional context: {prompt}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=full_prompt,
        config={
            "temperature": temperature
        }
    )

    return response.text


def generate_data_from_prompt(prompt: str, temperature: float) -> str:
    client = genai.Client(
        vertexai=os.getenv("USE_VERTEXAI", "False") == "True",
        project=os.getenv("PROJECT_ID"),
        location=os.getenv("LOCATION")
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={"temperature": temperature}
    )
    return response.text


def build_edit_prompt(current_data: List[Dict], edit_history: List[str], new_instruction: str) -> str:
    json_data = json.dumps(current_data, indent=2)
    edit_steps = "\n".join([f"{i+1}. {edit}" for i, edit in enumerate(edit_history)])
    next_step = f"{len(edit_history) + 1}. {new_instruction}"

    return f"""
You are a data editor. Below is the current data in JSON format:

{json_data}

Previously applied modifications:
{edit_steps}

Apply this additional modification:
{next_step}

Rules:
 - Modify only what is necessary.
 - Return only the updated JSON in the same structure.
 - Previous modifications should be preserved unless additional changes are specified to them.
"""


response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "table_name": {"type": "string"},
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True             
                }
            }
        },
        "required": ["table_name", "rows"]
    }
}