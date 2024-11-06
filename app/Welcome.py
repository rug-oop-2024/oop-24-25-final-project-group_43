
import streamlit as st
import sys
import os

# Ensure the autoop module is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from autoop.core.ml.artifact import Artifact

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.sidebar.success("Select a page above.")

# Specify UTF-8 encoding when opening the file
with open("README.md", "r", encoding="utf-8") as f:
    readme_content = f.read()

st.markdown(readme_content)

# Run the app with:
# cd path/to/Welcome.py
# streamlit run Welcome.py