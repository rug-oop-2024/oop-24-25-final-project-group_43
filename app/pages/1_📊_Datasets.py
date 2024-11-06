import streamlit as st
import pandas as pd
import sys
import os

# Ensure the autoop module is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

st.write("# 📊 Datasets")
# your code here
st.write("Upload your dataset to get started.")
file = st.file_uploader("Upload a CSV file", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    st.write(df)
    dataset = Dataset.from_dataframe(df, name=file.name[:-4], asset_path=file.name)
    automl.registry.register(dataset)
    
    st.success("Dataset uploaded successfully.")