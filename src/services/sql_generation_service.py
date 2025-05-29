from services.postgres_service import execute_sql
from services.gemini_client import client


def generate_sql(ddl_schema: str, input_query: str, error: str) -> str:
    if error != 'first run':
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

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "temperature": 0.0
        }
    )
    raw = response.text.strip()

    if raw.startswith("```sql"):
        raw = raw.removeprefix("```sql").strip()
    if raw.endswith("```"):
        raw = raw.removesuffix("```").strip()

    return raw


def sql_generation(ddl_schema: str, user_query: str) -> dict:
    error = 'first run'
    while error:
        sql_query = generate_sql(ddl_schema, user_query, error) 
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