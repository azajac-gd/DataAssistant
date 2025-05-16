import streamlit as st
import json
import pandas as pd
from typing import List, Dict

from services.ddl_service import parse_ddl
from services.gemini_service import generate_data_with_gemini, generate_data_from_prompt, build_edit_prompt


def show_data_generation():
    with st.container(border=True):
        prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

        ddl_file = st.file_uploader("Upload your DDL schema", type=["sql", "ddl", "txt"])
        ddl_content = ddl_file.read().decode("utf-8") if ddl_file else None

        st.markdown("**Advanced Parameters**")
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
        with col2:
            max_tokens = st.number_input("Max Tokens", min_value=1, max_value=4096, value=1000)

        generate_button = st.button("Generate")

    if generate_button:
        if ddl_content is None:
            st.error("Please upload a DDL file!")
            return

        with st.spinner("Generating data..."):
            tables = parse_ddl(ddl_content)
            generated_data = generate_data_with_gemini(tables, prompt, temperature)
            parsed_data = parse_json_block(generated_data)

        if parsed_data:
            st.session_state['generated_data'] = parsed_data
            st.session_state['edit_prompts'] = []
        else:
            st.error("Failed to parse generated data!")

    with st.container(border=True):
        if 'generated_data' in st.session_state:
            show_tables(st.session_state['generated_data'])

            col1, col2 = st.columns([5, 1])
            with col1:
                edit_prompt = st.text_input("edit_prompt", placeholder="Enter quick edit instructions...", label_visibility="collapsed")
            with col2:
                if st.button("Submit", use_container_width=True) and edit_prompt.strip():
                    with st.spinner("Applying edit..."):
                        process_edit_prompt(edit_prompt, temperature)



def show_tables(data: List[Dict]):
    col1, col2 = st.columns(2)
    with col1:
        st.write("Data Preview")
    with col2:
        table_names = [table['table_name'] for table in data]
        selected_table = st.selectbox("Select table", table_names, label_visibility="collapsed")

    for table_data in data:
        if table_data['table_name'] == selected_table:
            df = pd.DataFrame(table_data['rows'])
            st.dataframe(df, use_container_width=True, hide_index=True)
            break


def process_edit_prompt(edit_prompt: str, temperature: float):
    previous_data = st.session_state['generated_data']
    previous_prompts = st.session_state.get('edit_prompts', [])

    full_prompt = build_edit_prompt(previous_data, previous_prompts, edit_prompt)
    updated_data_str = generate_data_from_prompt(full_prompt, temperature)
    parsed_data = parse_json_block(updated_data_str)

    if parsed_data:
        st.session_state['generated_data'] = parsed_data
        st.session_state['edit_prompts'].append(edit_prompt)
    else:
        st.error("Failed to parse updated data.")


def parse_json_block(raw: str):
    try:
        if raw.startswith("```json"):
            raw = raw[len("```json"):].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        return json.loads(raw)
    except Exception as e:
        st.error(f"JSON parsing error: {e}")
        return None
