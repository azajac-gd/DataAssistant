from services.gemini_client import client
from services.sql_generation_service import sql_generation, sql_generation_declaration
from services.plot_generation_service import plot_generator, plot_generation_declaration
from google.genai import types

from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv
load_dotenv()

@observe(as_type="generation")
def chat_response(ddl_schema: str, user_query: str, messages: str):
    tools = types.Tool(function_declarations=[sql_generation_declaration, plot_generation_declaration])
    config = types.GenerateContentConfig(
        temperature=0.0,
        system_instruction="""You are a data assistant that uses tools. 
        Based on the user's question, you can either:
        1. Generate and run SQL query from user input.
        2. Generate and run a data visualization based on user request.
        
        You must choose one of these tools to answer the user's question.
        Do not respond with conversational text.""",
        tools=[tools],
        tool_config= {"function_calling_config": {"mode": "any"}})
    
    contents = [
        types.Content(
            role="user", parts=[types.Part(text=f"User question: {user_query}")]
        )
    ]
    model = "gemini-2.0-flash"
    response = client.models.generate_content(
        model=model, config=config, contents=contents
    )
    
    langfuse_context.update_current_observation(
    input=input,
    model=model,
    usage_details={
          "input": response.usage_metadata.prompt_token_count,
          "output": response.usage_metadata.candidates_token_count,
          "total": response.usage_metadata.total_token_count
      })

    tool_call = response.candidates[0].content.parts[0].function_call
    if tool_call is None:
        return "Model did not choose any tool."

    name = tool_call.name
    args = tool_call.args

    if name == "sql_generation":
        return sql_generation(ddl_schema, user_query, messages)
    elif name == "plot_generator":
        return plot_generator(user_query, ddl_schema, messages)

