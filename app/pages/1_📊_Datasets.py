from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

import streamlit as st
import pandas as pd


def select_dataset(datasets: list) -> Dataset:
    selected_dataset_name = st.selectbox("Select a dataset",
                                         [dataset.name for dataset in
                                          datasets])
    selected_dataset = next(dataset for dataset in datasets
                            if dataset.name == selected_dataset_name)
    return selected_dataset


def upload_dataset() -> None:
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        dataset = Dataset.from_dataframe(df, name=file.name[:-4],
                                         asset_path=f'datasets/{file.name}')
        automl.registry.register(dataset)
        st.success("Dataset uploaded successfully.")
        st.rerun()


automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")


st.write("# 📊 Datasets")


if datasets:
    st.write("## Available datasets")
    selected_dataset = select_dataset(datasets)

    if st.button("Preview Dataset"):
        st.write(pd.read_csv(f'assets/objects/{selected_dataset.asset_path}'))

    if st.button("Delete Dataset"):
        automl.registry.delete(selected_dataset.id)
        st.rerun()
else:
    st.write("## No datasets available.")
    st.subheader("Upload your dataset to get started.")
    upload_dataset()
