import streamlit as st
import logging
from services.chat_service import chat_response
import base64
import pickle


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
                if "sql" in message:
                    st.code(message["sql"], language="sql")

                if "df" in message:
                    try:
                        df = pickle.loads(base64.b64decode(message["df"]))
                        st.dataframe(df, use_container_width=False)
                    except Exception as e:
                        st.warning(f"Could not load previous DataFrame: {e}")

                if "plot_image" in message:
                    st.image(message["plot_image"])


    if prompt := st.chat_input("Ask a question about your data?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            response = chat_response(ddl_schema, prompt, st.session_state.messages)

            with st.chat_message("assistant"):
                assistant_msg = {"role": "assistant"}

                if isinstance(response, tuple) and len(response) == 2:
                    sql_query, df = response

                    st.code(sql_query, language="sql")
                    assistant_msg["sql"] = sql_query

                    if not df.empty:
                        st.dataframe(df, use_container_width=False)
                        df = base64.b64encode(pickle.dumps(df)).decode("utf-8")
                        assistant_msg["df"] = df

                    else:
                        st.warning("Query returned no results.")

                else:
                    image, plot_code, error = response
                    st.image(image)
                    assistant_msg["plot_image"] = image
                    assistant_msg["plot_code"] = plot_code

                st.session_state.messages.append(assistant_msg)

        except Exception as e:
            st.error(f"Error: {e}")

