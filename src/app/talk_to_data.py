import streamlit as st
from services.gemini_service import generate_sql
from services.postgres_service import execute_sql
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging


def show_talk_to_data():
    if "ddl_schema" not in st.session_state:
        st.warning("Please upload a DDL schema first in the data generation section")
        return
    
    ddl_schema = st.session_state.ddl_schema
    chat_container(ddl_schema)


def chat_container(ddl_schema: str):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            elif message["role"] == "assistant":
                sql_code, df_html = message["content"]
                st.code(sql_code, language="sql")
                logging.info(f"DataFrame: {df_html}")
                st.markdown(df_html, unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question about your data?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            sql_query, df = sql_generation(prompt, ddl_schema)

            with st.chat_message("assistant"):
                st.code(sql_query, language="sql")

                if df.empty:
                    st.warning("Query returned no results.")
                else:
                    st.dataframe(df)
            st.session_state.messages.append({
                "role": "assistant",
                "content": [sql_query, df.to_html(index=False, escape=False)]
            })

        except Exception as e:
            st.error(f"Error: {e}")



def sql_generation(user_query:str, ddl_schema:str) -> str:
    sql_request = generate_sql(ddl_schema, user_query)
    df = execute_sql(sql_request)
    return sql_request, df
