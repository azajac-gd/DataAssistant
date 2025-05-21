from google import genai
from typing import List, Dict
import os
from dotenv import load_dotenv
import json
from google.genai import types
import logging


load_dotenv()


client = genai.Client(
    vertexai=os.getenv("USE_VERTEXAI", "False") == "True",
    project=os.getenv("PROJECT_ID"),
    location=os.getenv("LOCATION")
)

def validate_data(data: str) -> str:
    validation_prompt = f"""
    You are a data validator. 
    1. Check if the following data is realistic, consistent and valid.
    2. Check if the data is in valid JSON format.
    3. Check if the data is consistent with the table structure and relationships.

    Output:
    - If valid, return "OK".
    - Otherwise, return only a bullet list of problems found (short descriptions).

    Data:
    {data}
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[types.Part(text=validation_prompt)])]
    )
    return response.candidates[0].content.parts[0].text.strip()


def generate_data_with_gemini(tables: List[Dict], prompt: str, temperature: float) -> str:
    logging.info("Generating data with Gemini...")

    validate_data_declaration = {
        "name": "validate_data",
        "description": "Validate the data. Check if the input data is valid and return issues if any",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Input JSON data to be validated"
                }
            },
            "required": ["data"],
        },
    }

    tools = types.Tool(function_declarations=[validate_data_declaration])
    config = types.GenerateContentConfig(tools=[tools], temperature=temperature)

    previously_generated_data = {}
    all_generated_data = []

    for table in tables:
        table_name = table["table_name"]
        columns = ", ".join([f"{col['name']} ({col['type']})" for col in table["columns"]])

        context_json = json.dumps(previously_generated_data, indent=2) if previously_generated_data else "None"

        full_prompt = f"""
You are a data generator. Your task is to generate realistic and consistent data for the following table:
Table '{table_name}': {columns}

Use the following previously generated data as reference to maintain consistency (foreign keys, IDs, names, etc.):
{context_json}

Rules:
- Generate 7 rows of data for the table unless specified otherwise.
- Ensure the data is realistic and consistent with the table structure.
- Ensure all foreign key references are valid and consistent with the reference data.
- Return ONLY valid JSON in this format!!:
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

    final_json = json.dumps(all_generated_data, indent=2)
        

    tools = types.Tool(function_declarations=[validate_data_declaration])
    config = types.GenerateContentConfig(tools=[tools])

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text="Validate the following JSON data."),
                types.Part(text=final_json)
            ]
        )
    ]


    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=config,
        contents=contents
    )
    candidate = response.candidates[0]

    if not candidate.content.parts:
        logging.warning("Model response has no parts - returning original data.")
        return final_json

    part = candidate.content.parts[0]

    if not hasattr(part, "function_call") or part.function_call is None:
        logging.info("Model did not call any function – returning original data.")
        return final_json

    # 4. Jeśli model wywołał funkcję
    tool_call = part.function_call

    if tool_call.name == "validate_data":
        try:
            result = validate_data(**tool_call.args)
            logging.info(f"Validation result: {result}")
        except Exception as e:
            logging.error(f"Error during validation: {e}")
            return final_json

        if result == "OK":
            return final_json
        else:
            logging.warning(f"Validation issues: {result}")
            return final_json
    else:
        logging.info(f"Unexpected function call: {tool_call.name} – returning original data.")
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

