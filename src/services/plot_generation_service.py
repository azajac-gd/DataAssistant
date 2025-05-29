from services.gemini_client import client
import logging


def plot_generator(user_query: str, ddl_schema: str) -> dict:
    error = 'first run'
    while error:
        plot_request = generate_code_for_plot(user_query, ddl_schema, error)
        plot, error = execute_plot(plot_request)
    return plot


plot_generation_declaration = {
        "name": "plot_generator",
        "description": "Generate and run a data visualization based on user request",
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

def generate_code_for_plot(user_query: str, ddl_schema: str, error: str) -> str:
    if error != 'first run':
        prompt = f"""
        You previously generated an invalid plot code with the following error: {error}
        Please correct it based on the schema: {ddl_schema}
        And user question: {user_query}

        Guidelines:
        - Output only valid Python code for generating a plot.
        - Do NOT include markdown formatting like ```python or ``` at any time.
        - Do NOT include any explanations, comments, or extra text.
        - The result must be directly executable in Python.

        Your output should be a single Python code block, nothing else.
        """
    else:
        prompt = f"""
        You are a professional data visualization generator. Your task is to generate valid Python code for creating a plot based on the given schema and the user's natural language request.

        Schema:
        {ddl_schema}

        User request:
        \"{user_query}\"

        Guidelines:
        - Output only valid Python code for generating a plot.
        - Do NOT include markdown formatting like ```python or ``` at any time.
        - Do NOT include any explanations, comments, or extra text.
        - The result must be directly executable in Python.

        Your output should be a single Python code block, nothing else.
        """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "temperature": 0.0
        }
    )
    raw = response.text.strip()

    if raw.startswith("```python"):
        raw = raw.removeprefix("```python").strip()
    if raw.endswith("```"):
        raw = raw.removesuffix("```").strip()

    return raw

def execute_plot(plot_code: str) -> tuple[str, str]:
    try:
        local_scope = {}
        exec(plot_code, {"plt": plt, "sns": sns}, local_scope)
        
        if 'plt' in local_scope and hasattr(local_scope['plt'], 'show'):
            plt = local_scope['plt']
            plt.show()
            return "Plot generated successfully", None
        else:
            return "No plot generated", "No plot object found in the code."
    except Exception as e:
        logging.error(f"Error executing plot code: {e}")
        return None, str(e)