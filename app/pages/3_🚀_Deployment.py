"""
Module for deployment page functionality in the app.

This module contains the functions and classes for managing
the deployment aspects of the app using the AutoMLSystem.
"""
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact

import streamlit as st

st.set_page_config(page_title="Deployment", page_icon="🚀")


def write_helper_text(text: str) -> None:
    """
    Display the given text as a styled paragraph in the Streamlit app.

    Args:
        text (str): The text to be displayed in the Streamlit app.
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# 🚀 Deployment")
write_helper_text("In this section, load a pipeline")

automl = AutoMLSystem.get_instance()

pipelines = automl.registry.list(type="pipeline")


def select_pipeline(pipelines: list) -> Artifact:
    """
    Allow the user to select a pipeline from a list of pipelines.

    Args:
        pipelines (list): A list of pipeline objects.

    Returns:
        Artifact: The selected pipeline object.
    """
    selected_pipeline_name = st.selectbox("Select a pipeline",
                                          [pipeline.name for pipeline in
                                           pipelines])
    selected_pipeline = next(pipeline for pipeline in pipelines
                             if pipeline.name == selected_pipeline_name)
    return selected_pipeline


def load_pipeline(selected_pipeline: Artifact) -> Pipeline:
    """
    Load and display the selected machine learning pipeline.

    Args:
        selected_pipeline (Artifact): The selected pipeline artifact
        to be loaded.

    Returns:
        Pipeline: The loaded machine learning pipeline.
    """
    st.write(f"Selected pipeline: {selected_pipeline.name}")
    pipeline = selected_pipeline.to_pipeline()
    st.write(pipeline)
    st.write(selected_pipeline)


selected_pipeline = select_pipeline(pipelines)
pipeline = load_pipeline(selected_pipeline)
