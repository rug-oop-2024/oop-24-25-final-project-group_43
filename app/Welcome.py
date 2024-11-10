"""A simple Streamlit app to display a README.md file."""
import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.sidebar.success("Select a page above.")

# Specify UTF-8 encoding when opening the file
with open("README.md", "r", encoding="utf-8") as f:
    readme_content = f.read()

st.markdown(readme_content)
