from google import genai
from typing import List, Dict
import os
from dotenv import load_dotenv
import json
from google.genai import types
import logging


load_dotenv()


def generate_data_with_gemini(tables: List[Dict], prompt: str, temperature: float) -> str:
    client = genai.Client(
        vertexai=os.getenv("USE_VERTEXAI", "False") == "True",
        project=os.getenv("PROJECT_ID"),
        location=os.getenv("LOCATION")
    )

    validate_data_declaration = {
        "name": "validate_data",
        "description": "Check if the input data is valid and return issues if any",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Raw input data to be validated"
                }
            },
            "required": ["data"],
        },
    }

    tools = types.Tool(function_declarations=[validate_data_declaration])
    config = types.GenerateContentConfig(tools=[tools], temperature=temperature)

    previously_generated_data = {}
    all_generated_data = []
    generated_tables = []

    for table in tables:
        table_name = table["table_name"]
        columns = ", ".join([f"{col['name']} ({col['type']})" for col in table["columns"]])

        # Kontekst do zapewnienia spójności (np. z ID innych tabel)
        context_json = json.dumps(previously_generated_data, indent=2) if previously_generated_data else "None"

        full_prompt = f"""
You are a data generator. Your task is to generate realistic and consistent data for the following table:
Table '{table_name}': {columns}

Use the following previously generated data as reference to maintain consistency (foreign keys, IDs, names, etc.):
{context_json}

Rules:
- Generate 15 rows of data for the table unless specified otherwise.
- Ensure the data is realistic and consistent with the table structure.
- Ensure all foreign key references are valid and consistent with the reference data.
- Return ONLY valid JSON in this format:
{{
  "table_name": "{table_name}",
  "rows": [
    {{ "col1": "value1", "col2": "value2", ... }},
    ...
  ]
}}

Additional context from user: {prompt}
        """
        contents = [types.Content(role="user", parts=[types.Part(text=full_prompt)])]

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=config
        )
        result = response.candidates[0].content.parts[0].text.strip()

        try:
            if result.startswith("```json"):
                result = result[len("```json"):].strip()
            if result.endswith("```"):
                result = result[:-3].strip()
            data_json = json.loads(result)
            all_generated_data.append(data_json)
            previously_generated_data[table_name] = data_json["rows"]
        except Exception as e:
            logging.warning(f"Could not parse table {table_name}: {e}")
            continue

    final_json= json.dumps(all_generated_data, indent=2)
        

    validation_prompt = f"""
You are a data validator. Check if the following JSON data is realistic, internally consistent, and valid given the table structure.

Respond with:
- "OK" if the data is valid.
- Otherwise, list the problems briefly.

Data:
{final_json}
"""

    validation_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[types.Part(text=validation_prompt)])]
    )

    result_text = validation_response.candidates[0].content.parts[0].text.strip()
    logging.info(f"Validation result: {result_text}")
    print(f"Validation result: {result_text}")
    if result_text == "OK":
        return final_json
    else:
        return final_json



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
You are a step-by-step reasoning data editor (ReACT agent). You receive the entire database as input.

Your task is to:
1. Understand the new modification instruction.
2. Think step-by-step about which tables and rows are affected.
3. Apply the necessary changes only to those relevant tables.
4. Return ONLY the updated tables as valid JSON. Do not include any unchanged tables.

Current database (multiple tables, JSON format):
{json_data}

Previously applied modifications:
{edit_steps if edit_steps else "None"}

New modification to apply:
{next_step}

Rules:
- Modify only what is necessary.
- Return ONLY the updated JSON structure for the tables that were changed.
- DO NOT explain what you are doing.
- DO NOT include markdown formatting like ```json.
- DO NOT include any other text — ONLY pure JSON array output.
- Analyze the instruction carefully and reason which tables are involved.
- Maintain correct relationships (e.g. foreign keys, references).
- Keep the table structure identical to the input.
"""

