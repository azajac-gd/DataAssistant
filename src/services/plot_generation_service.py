from services.gemini_client import client
from services.sql_generation_service import sql_generation
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import uuid
import os
import traceback
from google.genai import types
from langfuse.decorators import observe,langfuse_context 
from dotenv import load_dotenv
load_dotenv()


def plot_generator(user_query: str, ddl_schema: str, messages:str) -> dict:
    error = 'first run'
    while error:
        sql_query, df = sql_generation(ddl_schema, user_query, messages)
        logging.info(f"Dataframe: {df}")
        plot_request = generate_code_for_plot(user_query, ddl_schema, df, error, messages)
        logging.info(f"Code {plot_request}")
        plot_path, error = execute_plot(plot_request, df)
        logging.info(plot_path)
    return plot_path


plot_generation_declaration = {
        "name": "plot_generator",
        "description": "Generate data visualization and plots based on user request or add changes to the previous generated plot",
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "The user's natural language question about the data."
                }
            },
            "required": ["user_query"]
        }
}

@observe(as_type="generation")
def generate_code_for_plot(user_query: str, ddl_schema: str, df: str, error: str, messages: str) -> str:
    if error != 'first run':
        prompt = f"""
        You previously generated an invalid plot code with the following error: {error}
        Please correct it based on the schema: {ddl_schema}
        And user question: {user_query}

        Guidelines:
        - Output only valid Python code for generating a plot.
        - The result must be directly executable in Python.

        Your output should be a single Python code block, nothing else.
        """
    else:
        prompt = f"""
        You are a professional code generator. Your task is to generate valid Python code using Seaborn library for creating a plot based on the given schema and the user's natural language request.

        Schema:
        {ddl_schema}

        Dataframe:
        {df}

        User request:
        \"{user_query}\"
        """
    gemini_messages = []
    for message in messages:
        if "role" in message and "content" in message:
            gemini_messages.append(
                types.Content(role=message["role"], parts=[types.Part(text=message["content"])])
            )
    gemini_messages.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    model = "gemini-2.0-flash"
    response = client.models.generate_content(
        model=model,
        contents=gemini_messages,
        config={
            "temperature": 0.0
        }
    )
    langfuse_context.update_current_observation(
    input=input,
    model=model,
    usage_details={
          "input": response.usage_metadata.prompt_token_count,
          "output": response.usage_metadata.candidates_token_count,
          "total": response.usage_metadata.total_token_count
      })
    raw = response.text.strip()

    if raw.startswith("```python"):
        raw = raw.removeprefix("```python").strip()
    if raw.endswith("```"):
        raw = raw.removesuffix("```").strip()

    return raw

def execute_plot(plot_code: str, df: pd.DataFrame) -> tuple[str | None, str | None]:
    local_vars = {"df": df, "sns": sns, "plt": plt, "pd": pd}

    try:
        plt.clf()
        exec(plot_code, {}, local_vars)

        os.makedirs("plots", exist_ok=True)
        plot_path = f"plots/{uuid.uuid4().hex}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        return plot_path, None

    except Exception as e:
        logging.error(f"Error: {e}")
        error_msg = traceback.format_exc()
        return None, error_msg