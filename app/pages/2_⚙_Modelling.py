import streamlit as st
import pandas as pd
import sys
import os
from typing import List

# Ensure the autoop module is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from autoop.core.ml.metric import get_metric
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model import get_model
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning "\
                  "pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

##################################################################################
# Styled code with the use of functions
##################################################################################

def select_dataset(datasets: list) -> Dataset:
    selected_dataset_name = st.selectbox("Select a dataset",
                                      [dataset.name for dataset in datasets])
    selected_dataset = next(dataset for dataset in datasets 
                            if dataset.name == selected_dataset_name)
    return selected_dataset


def load_dataset(selected_dataset: Dataset) -> Dataset:
    st.write(f"Selected dataset: {selected_dataset.name}")

    # Load the dataset (couldn't figure out how to load the dataset from the registry, 
    # since it is a list of artifacts)
    file_path = os.path.join('assets/objects/', selected_dataset.asset_path)
    df = pd.read_csv(file_path)
    st.write(df)
    datasett = Dataset.from_dataframe(df, name=selected_dataset.name, 
                                      asset_path=selected_dataset.asset_path)
    return datasett


def load_features(datasett: Dataset) -> list:
    features = detect_feature_types(datasett)
    return features

def determine_task_type(input_features: List[str], 
                        target_feature: str|int|float, features: List[str]) -> str:
    if input_features and target_feature:
    # Determine task type based on target feature
        target_feature_type = next((feature.type for feature in features if 
                                    feature.name == target_feature), None)
        
        if target_feature_type == 'numerical':
            task_type = "regression"
        else:
            task_type = "classification"
        
        st.write(f"Detected task type: {task_type}")
        return task_type
    
def select_model(task_type: str) -> str:
    st.subheader("Model Selection")
    if task_type == "classification":
        model_type = st.selectbox("Select a "\
            "classification model", ["KNN", "Logistic Regression", "Random Forest"])
    elif task_type == "regression":
        model_type = st.selectbox("Select a "\
            "regression model", ["Lasso", "Multiple Linear Regression",
                                    "Polynomial Regression"])
    return model_type

def split_data() -> float:
    st.subheader("Splitting Data")
    split_ratio = st.slider("Select the training set split ratio",
                            0.1, 0.9, 0.5, 0.05)
    return split_ratio

def select_metrics(task_type: str) -> List[str]:
    st.subheader("Metrics Selection")
    if task_type == "classification":
        metrics = st.multiselect("Select metrics for classification",
                                ["Accuracy", "Recall", "Precision"])
    elif task_type == "regression":
        metrics = st.multiselect("Select metrics for regression",
                                ["Mean Squared Error",
                                    "Root Mean Squared Error",
                                    "Mean Absolute Error"])
    return metrics

def display_summary(selected_dataset_name: str, input_features: List[str],
                     target_feature: str|int|float, model_type: str, 
                     split_ratio: float, metrics: List[str]) -> None:
    if st.button("Show Summary"):
        st.write(f"Selected dataset: {selected_dataset_name}")
        st.write(f"Selected input features: {input_features}")
        st.write(f"Selected target feature: {target_feature}")
        st.write(f"Selected model: {model_type}")
        st.write(f"Training set split ratio: {split_ratio}")
        st.write(f"Selected metrics: {metrics}")

def train_model(datasett: Dataset, features: List[str],  input_features: List[str],
                 target_feature: str|int|float, model_type: str, split_ratio: float,
                 metrics: List[str]) -> tuple[Pipeline, dict]:
    if st.button("Train Model"):
        st.write("Training model...")
        input_features = [next(feature for feature in features 
                                if feature.name == feature_name) 
                                for feature_name in input_features]
        target_feature = next(feature for feature in features 
                                if feature.name == target_feature)
        metrics = [get_metric(metric) for metric in metrics]
        pipeline = Pipeline(metrics=metrics, dataset=datasett,
                            model=get_model(model_type),
                                input_features=input_features,
                                target_feature=target_feature, 
                                split=split_ratio)
        result = pipeline.execute()
        st.write("Model trained successfully.")
        return pipeline, result

def save_pipeline(pipeline: Pipeline) -> None:
    if st.button("Save Pipeline"):
        pipeline_name = st.text_input("Enter pipeline name")
        pipeline_version = st.text_input("Enter pipeline version")
        if st.button("Save Pipeline"):
            st.write("Saving pipeline...")
            for artifact in pipeline.artifacts:
                artifact.save_pipeline_artifact(pipeline_name, pipeline_version)
            st.write("Pipeline saved successfully.")

def print_result(result: dict) -> None:

    metrics = result["metrics"]
    metrics_train = result["metrics_train"]

    metric_values = [metric[1].item() for metric in metrics]
    metric_names = [str(metric[0]).split('.')[4].split(' ')[0] for metric in metrics]

    metric_train_values = [metric[1].item() for metric in metrics_train]
    metric_train_names = [str(metric[0]).split('.')[4].split(' ')[0] for metric in metrics_train]
    

    st.subheader("Metrics:")
    for names, values in zip(metric_names, metric_values):
        st.write(f"{names}: {values}")

    st.subheader("Training Metrics:")
    for names, values in zip(metric_train_names, metric_train_values):
        st.write(f"{names}: {values}")


##############################################################################
# "main" 
##############################################################################


st.subheader("Dataset Selection")
selected_dataset = select_dataset(datasets)
datasett = load_dataset(selected_dataset)
features = load_features(datasett)

feature_names = [feature.name for feature in features]
st.subheader("Feature Selection")
input_features = st.multiselect("Select input features", feature_names)
target_feature = st.selectbox("Select target feature", feature_names)

if input_features and target_feature:
        task_type = determine_task_type(input_features, target_feature, features)
        model_type = select_model(task_type)
        split_ratio = split_data() 
        metrics = select_metrics(task_type)
        display_summary(selected_dataset.name, input_features, target_feature,
                         model_type, split_ratio, metrics)
        (pipeline, result) = train_model(datasett, features, input_features, target_feature,
                              model_type, split_ratio, metrics)
        save_pipeline(pipeline)
        if result:
            print_result(result)
        