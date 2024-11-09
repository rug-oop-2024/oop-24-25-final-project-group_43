from autoop.core.ml.artifact import Artifact

import streamlit as st
import sys
import os

# Ensure the autoop module is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../../')))

st.set_page_config(
    page_title="Instructions",
    page_icon="👋",
)

st.markdown(open("INSTRUCTIONS.md").read())
