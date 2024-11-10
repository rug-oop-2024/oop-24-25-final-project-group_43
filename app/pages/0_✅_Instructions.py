"""Instructions page shown when the app is launched."""
import streamlit as st

st.set_page_config(
    page_title="Instructions",
    page_icon="👋",
)

st.markdown(open("INSTRUCTIONS.md").read())
