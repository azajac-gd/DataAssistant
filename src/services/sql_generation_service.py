from services.postgres_service import execute_sql
from services.gemini_client import client
from google.genai import types
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv
load_dotenv()

@observe(as_type="generation")
def generate_sql(ddl_schema: str, input_query: str, error: str, messages: str) -> str:
    if error != 'first run' and error is not None:
        prompt = f"""
        You previously generated an invalid SQL query with the following error: {error}
        Please correct it based on the schema: {ddl_schema}
        And user question: {input_query}

        Guidelines:
        - Output only a valid SQL query.
        - Do NOT include markdown formatting like ```sql or ``` at any time.
        - Do NOT include any explanations, comments, or extra text.
        - The result must be directly executable in PostgreSQL.

        Your output should be a single SQL query, nothing else.
        """
    else:
        prompt = f"""
        You are a professional SQL generator. Your task is to generate a valid, executable PostgreSQL query based strictly on the given schema and the user's natural language request.

        Schema:
        {ddl_schema}

        User request:
        \"{input_query}\"

        Guidelines:
        - Output only a valid SQL query.
        - Do NOT include markdown formatting like ```sql or ``` at any time.
        - Do NOT include any explanations, comments, or extra text.
        - The result must be directly executable in PostgreSQL.

        Your output should be a single SQL query, nothing else.
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

    if raw.startswith("```sql"):
        raw = raw.removeprefix("```sql").strip()
    if raw.endswith("```"):
        raw = raw.removesuffix("```").strip()

    return raw

def sql_generation(ddl_schema: str, user_query: str, messages: str) -> dict:
    error = 'first run'
    while error:
        sql_query = generate_sql(ddl_schema, user_query, error, messages) 
        result_df, error = execute_sql(sql_query)
    return sql_query, result_df

sql_generation_declaration={
        "name": "sql_generation",
        "description": "Generate and run SQL query from user input",
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