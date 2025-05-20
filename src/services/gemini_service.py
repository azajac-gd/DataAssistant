from google import genai
from typing import List, Dict
import os
from dotenv import load_dotenv
import json
from google.genai import types
import logging


load_dotenv()

# function calling is not working correctly 

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
- Each table must contain 5 sample rows unless specified otherwise.
- The data should be realistic and consistent with the table definitions.
- Do not create None or empty values in the data unless specified otherwise.
- Return ONLY valid JSON.
- Do NOT add explanations, comments, or surrounding text.
- Ensure the JSON is well-formed and parsable.
- After generating the JSON sample data, validate it using the "validate_data" function.

Table definitions:
{tables_desc}
Additional context: {prompt}
"""

    contents = [
        types.Content(
            role="user", parts=[types.Part(text=full_prompt)]
        )
    ]

    response = client.models.generate_content(
        model="gemini-2.0-flash",  
        contents=contents,
        config=config
    )


    candidate = response.candidates[0]
    function_call = candidate.content.parts[0].function_call

    if function_call:
        logging.info(f"Function to call: {function_call.name}")
        logging.info(f"Arguments: {function_call.args}")

        if function_call.name == "validate_data":
            data_to_validate = function_call.args.get("data", "")

            validation_prompt = f"""
You are a data validator. Check if the following JSON data is realistic, internally consistent, and valid given the table structure.

Respond with:
- "OK" if the data is valid.
- Otherwise, list the problems briefly.

Data:
{data_to_validate}
"""

            validation_response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[types.Content(role="user", parts=[types.Part(text=validation_prompt)])]
            )

            result_text = validation_response.candidates[0].content.parts[0].text.strip()
            logging.info(f"Validation result: {result_text}")
            return result_text
    else:
        logging.info("No function call found in the response.")
        return candidate.content.parts[0].text.strip()




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

