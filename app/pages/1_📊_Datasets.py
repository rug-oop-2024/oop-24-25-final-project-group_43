from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

import streamlit as st
import pandas as pd

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

st.write("# 📊 Datasets")
# your code here

st.subheader("Upload your dataset to get started.")
file = st.file_uploader("Upload a CSV file", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    st.write(df)
    dataset = Dataset.from_dataframe(df, name=file.name[:-4],
                                     asset_path=f'datasets/{file.name}')
    automl.registry.register(dataset)

    st.success("Dataset uploaded successfully.")
