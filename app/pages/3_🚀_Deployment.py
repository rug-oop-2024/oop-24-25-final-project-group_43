from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact

import streamlit as st

st.set_page_config(page_title="Deployment", page_icon="🚀")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# 🚀 Deployment")
write_helper_text("In this section, load a pipeline")

automl = AutoMLSystem.get_instance()

pipelines = automl.registry.list(type="pipeline")


def select_pipeline(pipelines: list) -> Artifact:
    selected_pipeline_name = st.selectbox("Select a pipeline",
                                          [pipeline.name for pipeline in
                                           pipelines])
    selected_pipeline = next(pipeline for pipeline in pipelines
                             if pipeline.name == selected_pipeline_name)
    return selected_pipeline


def load_pipeline(selected_pipeline: Artifact) -> Pipeline:
    st.write(f"Selected pipeline: {selected_pipeline.name}")
    pipeline = selected_pipeline.to_pipeline()
    st.write(pipeline)
    st.write(selected_pipeline)


selected_pipeline = select_pipeline(pipelines)
pipeline = load_pipeline(selected_pipeline)
