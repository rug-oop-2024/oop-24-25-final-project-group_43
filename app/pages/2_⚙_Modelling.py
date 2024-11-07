import streamlit as st
import pandas as pd
import sys
import os
from autoop.functional.feature import detect_feature_types

# Ensure the autoop module is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")



# ST/modelling/datasets/list
# ST/modelling/datasets/features


dataset_names = [dataset.name for dataset in datasets]
selected_dataset_name = st.selectbox("Select a dataset", dataset_names)

selected_dataset = next((dataset for dataset in datasets if dataset.name == selected_dataset_name), None)

if selected_dataset:
    st.write(f"Selected dataset: {selected_dataset.name}")
      
    # Detect features
    # features = detect_feature_types(selected_dataset)
        # -> AttributeError: 'bytes' object has no attribute 'columns'

    # feature_names = [feature.name for feature in features]
    
    # # Generate selection menus for input features and target feature
    # input_features = st.multiselect("Select input features", feature_names)
    # target_feature = st.selectbox("Select target feature", feature_names)
    
    # if input_features and target_feature:
    #     # Determine task type based on target feature
    #     target_feature_type = next((feature for feature in features if feature.name == target_feature), None).dtype
        
    #     if target_feature_type in ['int', 'float']:
    #         task_type = "regression"
    #     else:
    #         task_type = "classification"
        
    #     st.write(f"Detected task type: {task_type}")
else:
    st.write("No dataset selected.")

