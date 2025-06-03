from typing import List, Dict
import json
from services.gemini_client import client
from langfuse.decorators import observe

@observe()
def generate_data_with_gemini(ddl_schema: str, prompt: str, temperature: float) -> str:
    full_prompt = f"""
You are a data generator. Given the following ddl schema, generate realistic and consistent sample data.
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
- Make sure that primary keys and foreign keys are consistent and relations between tables are correct!

DDL schema:
{ddl_schema}

Additional context: {prompt}
"""


    response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=full_prompt,
            config={"temperature": temperature}
        )

    output = response.text.strip()

    return output


@observe()
def validate_generated_data(ddl_schema: str, generated_data: str):
    prompt = f"""
You are a data validator. 
Rules:
- Each table must contain 15 sample rows unless specified otherwise.
- The data should be realistic and consistent with the table definitions.
- Check if primary keys and foreign keys are consistent and relations between tables are correct!

Return:
- 'OK' if validated data are correct
- Description what is wrong and what has to be changed

DDL Schema:
{ddl_schema}

Data to validate:
{generated_data}

"""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "temperature": 0.0
        }
    )

    return response.text

@observe()
def generate_data_from_prompt(prompt: str, temperature: float) -> str:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={"temperature": temperature}
    )
    return response.text

@observe()
def build_edit_prompt(current_data: List[Dict], edit_history: List[str], new_instruction: str, ddl_schema: str) -> str:
    json_data = json.dumps(current_data, indent=2)
    edit_steps = "\n".join([f"{i+1}. {edit}" for i, edit in enumerate(edit_history)])
    next_step = f"{len(edit_history) + 1}. {new_instruction}"

    return f"""
    You are a step-by-step reasoning data editor (ReACT agent). You receive the entire database as input.

    Your task is to:
    1. Understand the new modification instruction.
    2. Think step-by-step about which tables and rows are affected.
    3. Apply the necessary changes only to those relevant tables.
    4. Data changes shoudl be correct with ddl schema
    5. Return ONLY the updated tables as valid JSON. Do not include any unchanged tables.

    Current database (multiple tables, JSON format):
    {json_data}

    DDL schema:
    {ddl_schema}

    Previously applied modifications:
    {edit_steps if edit_steps else "None"}

    New modification to apply:
    {next_step}

    Rules:
    - Modify only what is necessary.
    - Data must be correct with ddl schema
    - Return ONLY the updated JSON structure for the tables that were changed.
    - DO NOT explain what you are doing.
    - DO NOT include markdown formatting like ```json.
    - DO NOT include any other text — ONLY pure JSON array output.
    - Analyze the instruction carefully and reason which tables are involved.
    - Maintain correct relationships (e.g. foreign keys, references).
    - Keep the table structure identical to the input.
    """