from google import genai
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()


def generate_data_with_gemini(tables: Dict[str, List[str]], prompt: str, temperature: float) -> List[Dict]:
    client = genai.Client(
        vertexai=os.getenv("USE_VERTEXAI", "False") == "True",
        project=os.getenv("PROJECT_ID"),
        location=os.getenv("LOCATION")
    )
    
    tables_desc = "\n".join([
        f"Table '{table_name}': " + ", ".join([f"{col} (string)" for col in columns])
        for table_name, columns in tables.items()
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
            "temperature": temperature,
            #"response_mime_type": "application/json",
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