from autoop.core.ml.metric import get_metric
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model import get_model
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.dataset import Artifact

import streamlit as st
import pandas as pd
import os
from typing import List

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """
    Writes the given text as HTML with a specific style to Streamlit.

    Args:
        text (str): The text to be written in the Streamlit app.
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning "
                  "pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")


def select_dataset(datasets: list) -> Dataset:
    """
    Prompts the user to select a dataset from a list of datasets
        using a selectbox.

    Args:
        datasets (list): A list of Dataset objects to choose from.

    Returns:
        Dataset: The selected Dataset object.

    Notes:
        - If no dataset is selected, an informational message is displayed.
    """
    selected_dataset_name = st.selectbox("Select a dataset",
                                         [dataset.name for dataset in
                                          datasets])
    if selected_dataset_name is not None:
        selected_dataset = next(dataset for dataset in datasets
                                if dataset.name == selected_dataset_name)
        return selected_dataset
    else:
        st.info("Please select a dataset to proceed.")


def load_dataset(selected_dataset: Dataset) -> Dataset:
    """
    Loads a dataset from a CSV file and returns it as a Dataset object.

    Args:
        selected_dataset (Dataset): The selected dataset object
        containing the name and asset path.

    Returns:
        Dataset: A new Dataset object created from the loaded CSV file.

    """
    st.write(f"Selected dataset: {selected_dataset.name}")
    file_path = os.path.join('assets/objects/', selected_dataset.asset_path)
    df = pd.read_csv(file_path)
    st.write(df)
    datasett = Dataset.from_dataframe(df, name=selected_dataset.name,
                                      asset_path=selected_dataset.asset_path)
    return datasett


def load_features(datasett: Dataset) -> list:
    """
    Load and detect feature types from the given dataset.

    Args:
        datasett (Dataset): The dataset from which to
            detect and load features.

    Returns:
        list: A list of detected features.
    """
    features = detect_feature_types(datasett)
    return features


def determine_task_type(input_features: List[str],
                        target_feature: str | int | float,
                        features: List[str]) -> str:
    """
    Determines the type of machine learning task based on the
        input features and target feature.

    Args:
        input_features (List[str]): A list of input feature names.
        target_feature (str | int | float): The target feature name.
        features (List[str]): A list of feature objects, where each
            object has 'name' and 'type' attributes.

    Returns:
        str: The type of task, either "regression" or "classification".
    """
    if input_features and target_feature:
        target_feature_type = next((feature.type for feature in features if
                                    feature.name == target_feature), None)

        if target_feature_type == 'numerical':
            task_type = "regression"
        else:
            task_type = "classification"

        st.info(f"Detected task type: {task_type}")
        return task_type


def select_model(task_type: str) -> str:
    """
    Displays a model selection dropdown based on the task type
        and returns the selected model.

    Parameters:
    task_type (str): The type of task for which the model is being selected.
                     It can be either "classification" or "regression".

    Returns:
    str: The name of the selected model.
    """
    st.subheader("Model Selection")
    if task_type == "classification":
        model = st.selectbox("Select a "
                             "classification model",
                             ["KNN",
                              "Logistic Regression",
                              "Random Forest"],
                             index=0,
                             placeholder='None')
    elif task_type == "regression":
        model = st.selectbox("Select a "
                             "regression model",
                             ["Lasso",
                              "Multiple Linear Regression",
                              "Polynomial Regression"],
                             index=0,
                             placeholder='None')
    return model


def split_data() -> float:
    """
    Displays a subheader and a slider widget in a Streamlit
        app to select the training set split ratio.

    Returns:
        float: The selected split ratio for the training set.
    """
    st.subheader("Splitting Data")
    split_ratio = st.slider("Select the training set split ratio",
                            0.1, 0.9, 0.5, 0.05)
    return split_ratio


def select_metrics(task_type: str) -> List[str]:
    """
    Display a Streamlit multiselect widget based on the task type.

    Parameters:
    task_type (str): The type of machine learning task.
    It can be either "classification" or "regression".

    Returns:
    List[str]: A list of selected metrics as strings.
    """
    st.subheader("Metrics Selection")
    if task_type == "classification":
        metrics = st.multiselect("Select metrics for classification",
                                 ["Accuracy", "Recall", "Precision"],
                                 default=None, placeholder='')
    elif task_type == "regression":
        metrics = st.multiselect("Select metrics for regression",
                                 ["Mean Squared Error",
                                  "Root Mean Squared Error",
                                  "Mean Absolute Error"],
                                 default=None, placeholder='')
    return metrics


def show_summary(selected_dataset_name: str, input_features: List[str],
                 target_feature: str | int | float, model_type: str,
                 split_ratio: float, metrics: List[str]) -> None:
    """
    Displays a summary of the selected dataset and model configuration.

    Args:
        selected_dataset_name (str): The name of the selected dataset.
        input_features (List[str]): A list of input feature names.
        target_feature (str | int | float): The target feature name or value.
        model_type (str): The type of model selected.
        split_ratio (float): The ratio for splitting the dataset into training
        and testing sets.
        metrics (List[str]): A list of metrics to evaluate the model.

    Returns:
        None
    """
    st.write(f"Selected dataset: {selected_dataset_name}")
    st.write(f"Selected input features: {input_features}")
    st.write(f"Selected target feature: {target_feature}")
    st.write(f"Selected model: {model_type}")
    st.write(f"Training set split ratio: {split_ratio}")
    st.write(f"Selected metrics: {metrics}")


def get_pipeline(datasett: Dataset, features: List[str],
                 input_features: List[str],
                 target_feature: str | int | float, model: str,
                 split_ratio: float,
                 metrics: List[str]) -> Pipeline:
    """
    Trains a machine learning model using the provided dataset and parameters,
        and returns the resulting pipeline.
    Args:
        datasett (Dataset): The dataset to be used for training the model.
        features (List[str]): A list of feature names available
                                in the dataset.
        input_features (List[str]): A list of feature names
                                to be used as input features for the model.
        target_feature (str | int | float): The name of the target feature
                                to be predicted by the model.
        model (str): The name of the model to be used for training.
        split_ratio (float): The ratio to split the
            dataset into training and testing sets.
        metrics (List[str]): A list of metric names
            to evaluate the model's performance.
    Returns:
        Pipeline: The trained machine learning pipeline.
    """
    st.write("Training model...")
    input_features = [
        next(feature for feature in features if
             feature.name == feature_name)
        for feature_name in input_features
    ]
    target_feature = next(
        feature for feature in features if feature.name == target_feature
    )
    metrics = [get_metric(metric) for metric in metrics]
    pipeline = Pipeline(
        metrics=metrics,
        dataset=datasett,
        model=get_model(model),
        input_features=input_features,
        target_feature=target_feature,
        split=split_ratio
    )
    st.write("Model trained successfully.")
    return pipeline


def save_pipeline(pipeline: Pipeline) -> None:
    """
    Prompts the user to enter a pipeline name and version,
        and saves the given pipeline
    as an artifact with the specified name and version.

    Args:
        pipeline (Pipeline): The pipeline to be saved.

    Returns:
        None
    """
    pipeline_name = st.text_input("Enter pipeline name")
    pipeline_version = st.text_input("Enter pipeline version")
    if st.button("Save Pipeline"):
        st.write("Saving pipeline...")
        pipeline_artifact = Artifact.from_pipeline(
            cls=Artifact,
            type="pipeline",
            name=pipeline_name,
            asset_path=f'pipelines/{pipeline_name}_{pipeline_version}.pkl',
            version=pipeline_version,
            tags=None,
            data=pipeline,
            metadata=None
        )
        automl.registry.register(pipeline_artifact)
        st.write("Pipeline saved successfully.")


def print_result(result: dict) -> None:
    """
    Prints the metrics, training metrics, and predictions from the
        result dictionary.

    Args:
        result (dict): A dictionary containing the following keys:
            - "metrics": A list of tuples where each tuple contains a
                metric name and its value.
            - "metrics_train": A list of tuples where each tuple
                contains a training metric name and its value.
            - "predictions": The predictions to be printed.

    Returns:
        None
    """

    metrics = result["metrics"]
    metrics_train = result["metrics_train"]
    predictions = result["predictions"]

    metric_values = [metric[1].item() for metric in metrics]
    metric_names = [str(metric[0]).split('.')[4].split(' ')[0]
                    for metric in metrics]

    metric_train_values = [metric[1].item() for metric in metrics_train]
    metric_train_names = [str(metric[0]).split('.')[4].split(' ')[0] for
                          metric in metrics_train]

    st.subheader("Metrics:")
    for names, values in zip(metric_names, metric_values):
        st.write(f"{names}: {values}")

    st.subheader("Training Metrics:")
    for names, values in zip(metric_train_names, metric_train_values):
        st.write(f"{names}: {values}")

    st.subheader("Predictions:")
    st.write(predictions)


# Main code
st.subheader("Dataset Selection")

if datasets:
    selected_dataset = select_dataset(datasets)
    st.markdown("---")
    if selected_dataset:
        datasett = load_dataset(selected_dataset)
        features = load_features(datasett)
        feature_names = [feature.name for feature in features]
        st.markdown("---")
        if feature_names:
            st.subheader("Feature Selection")
            input_features = st.multiselect("Select input features",
                                            feature_names,
                                            default=None,
                                            placeholder='')
            target_feature = st.selectbox("Select target feature",
                                          feature_names,
                                          index=None,
                                          placeholder='')
            if input_features and target_feature:
                task_type = determine_task_type(input_features,
                                                target_feature,
                                                features)
                model = select_model(task_type)
                split_ratio = split_data()
                metrics = select_metrics(task_type)
                if st.checkbox("Show Summary"):
                    show_summary(selected_dataset.name,
                                 input_features,
                                 target_feature,
                                 model,
                                 split_ratio,
                                 metrics)
                if split_ratio and metrics and model:
                    pipeline = get_pipeline(datasett,
                                            features,
                                            input_features,
                                            target_feature,
                                            model,
                                            split_ratio,
                                            metrics)
                    if pipeline:
                        if st.checkbox("Show Results"):
                            print_result(pipeline.execute())

                        if st.checkbox("Save Pipeline"):
                            save_pipeline(pipeline)

else:
    st.warning("There are no Datasets available, "
               "please upload a Dataset in the Datasets page")
