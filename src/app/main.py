import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import streamlit as st
from data_generation import show_data_generation
from talk_to_data import show_talk_to_data
from langfuse.decorators import observe
from dotenv import load_dotenv
import atexit
load_dotenv()

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Data Assistant")
    selection = st.sidebar.radio(" ", ["Data Generation", "Talk to Your Data"])

    if selection == "Data Generation":
        show_data_generation()
    elif selection == "Talk to Your Data":
        show_talk_to_data()

def cleanup_plots():
    folder = "plots"
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

if __name__ == "__main__":
    atexit.register(cleanup_plots)
    main()

