import streamlit as st
import json
import pandas as pd
from typing import List, Dict
import io
import zipfile
import logging

from services.data_generation_service import generate_data_with_gemini, build_edit_prompt, generate_data_from_prompt, validate_generated_data
from services.postgres_service import execute_ddl_and_save_data
from services.validation_service import validate_prompt, extract_affected_tables
from langfuse.decorators import observe
from dotenv import load_dotenv
load_dotenv()


def show_data_generation():
    with st.container(border=True):
        prompt = st.text_area("Prompt", placeholder="Enter your prompt here...")

        ddl_file = st.file_uploader("Upload your DDL schema", type=["sql", "ddl", "txt"])
        if ddl_file:
            ddl_content = ddl_file.read().decode("utf-8")
            st.session_state.ddl_schema = ddl_content  

        st.markdown("**Advanced Parameters**")
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
        with col2:
            max_tokens = st.number_input("Max Tokens", min_value=1, max_value=4096, value=1000)

        generate_button = st.button("Generate")

    if generate_button:
        validation = validate_prompt(prompt)
        if validation != "OK":
            st.error(f"Prompt rejected!")
        else:
            if ddl_content is None:
                st.error("Please upload a DDL file!")
                return

            with st.spinner("Generating data..."):
                logging.info("Generating data...")
                generated_data = generate_data_with_gemini(ddl_content, prompt, temperature)
                checked_data = validate_generated_data(ddl_content, generated_data)
                logging.info(f"Validation result: {checked_data}")
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
                    validation = validate_prompt(edit_prompt)
                    if validation != "OK":
                        st.error(f"Prompt rejected!")
                    else:
                        with st.spinner("Applying edit..."):
                            process_edit_prompt(edit_prompt, temperature, ddl_content)
            download_all_tables()
            if st.button("Save to PostgreSQL"):
                try:
                    execute_ddl_and_save_data(ddl_content, st.session_state['generated_data'])

                except Exception as e:
                    st.error(f"Error saving to PostgreSQL: {e}")

def download_all_tables():
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for table in st.session_state["generated_data"]:
            table_name = table["table_name"]
            df = pd.DataFrame(table["rows"])
            csv_io = io.StringIO()
            df.to_csv(csv_io, index=False)
            zip_file.writestr(f"{table_name}.csv", csv_io.getvalue())

    zip_buffer.seek(0)

    st.download_button(
        label="Download all tables",
        data=zip_buffer,
        file_name="generated_data.zip",
        mime="application/zip"
    )

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

def process_edit_prompt(edit_prompt: str, temperature: float, ddl_schema: str):
    full_data = st.session_state['generated_data']
    edit_history = st.session_state.get('edit_prompts', [])

    all_table_names = [table["table_name"] for table in full_data]
    affected_tables = extract_affected_tables(edit_prompt, all_table_names)
    filtered_data = [table for table in full_data if table["table_name"] in affected_tables]
    full_prompt = build_edit_prompt(filtered_data, edit_history, edit_prompt, ddl_schema)

    updated_data_str = generate_data_from_prompt(full_prompt, temperature)

    parsed_data = parse_json_block(updated_data_str)

    if parsed_data:
        updated_table_names = {table['table_name'] for table in parsed_data}
        for i, table in enumerate(full_data):
            if table['table_name'] in updated_table_names:
                full_data[i] = next(t for t in parsed_data if t['table_name'] == table['table_name'])

        st.session_state['generated_data'] = full_data
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




