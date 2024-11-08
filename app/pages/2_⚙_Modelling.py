import streamlit as st
import pandas as pd
import sys
import os

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
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")


# Select a dataset from uploaded datasets
st.subheader("Dataset Selection")
#dataset_names = [dataset.name for dataset in datasets]
selected_dataset_name = st.selectbox("Select a dataset", [dataset.name for dataset in datasets])
selected_dataset = next(dataset for dataset in datasets if dataset.name == selected_dataset_name)

"""
if st.button('Refresh Datasets'):
    delete_datasets('path/to/datasets')
    st.experimental_rerun()
"""

if selected_dataset:
    st.write(f"Selected dataset: {selected_dataset.name}")


    # Load the dataset (couldn't figure out how to load the dataset from the registry, since it is a list of artifacts)
    file_path = os.path.join('assets/objects/', selected_dataset.asset_path)
    df = pd.read_csv(file_path)
    st.write(df)
    datasett = Dataset.from_dataframe(df, name=selected_dataset.name, asset_path=selected_dataset.asset_path)
    features = detect_feature_types(datasett)

    feature_names = [feature.name for feature in features]
    
    st.subheader("Feature Selection")

    # Generate selection menus for input features and target feature
    input_features = st.multiselect("Select input features", feature_names)
    target_feature = st.selectbox("Select target feature", feature_names)
    
    if input_features and target_feature:
    # Determine task type based on target feature
        target_feature_type = next((feature.type for feature in features if feature.name == target_feature), None)
        
        if target_feature_type == 'numerical':
            task_type = "regression"
        else:
            task_type = "classification"
        
        st.write(f"Detected task type: {task_type}")


        # Select model type based on task type
        st.subheader("Model Selection")
        if task_type == "classification":
            model_type = st.selectbox("Select a classification model", ["KNN", "Logistic Regression", "Random Forest"])
        elif task_type == "regression":
            model_type = st.selectbox("Select a regression model", ["Lasso", "Multiple Linear Regression", "Polynomial Regression"])  


        # Splitting Data
        st.subheader("Splitting Data")
        split_ratio = st.slider("Select the training set split ratio", 0.1, 0.9, 0.5, 0.05)


        # Select metrics based on task type
        st.subheader("Metrics Selection")
        if task_type == "classification":
            metrics = st.multiselect("Select metrics for classification", ["Accuracy", "Recall", "Precision"])
        elif task_type == "regression":
            metrics = st.multiselect("Select metrics for regression", ["Mean Squared Error", "Root Mean Squared Error", "Mean Absolute Error"])


        # Display summary of selected options
        if st.button("Show Summary"):
            st.write(f"Selected dataset: {selected_dataset_name}")
            st.write(f"Selected input features: {input_features}")
            st.write(f"Selected target feature: {target_feature}")
            st.write(f"Selected model: {model_type}")
            st.write(f"Training set split ratio: {split_ratio}")
            st.write(f"Selected metrics: {metrics}")


        # Train the model
        if st.button("Train Model"):
            st.write("Training model...")
            input_features = [next(feature for feature in features if feature.name == feature_name) for feature_name in input_features]
            target_feature = next(feature for feature in features if feature.name == target_feature)
            metrics = [get_metric(metric) for metric in metrics]
            pipeline = Pipeline(metrics=metrics, dataset=datasett, model=get_model(model_type), input_features=input_features, target_feature=target_feature, split=split_ratio)
            result = pipeline.execute()
            st.write(result)


            # Errors: 
            # KNN:     
                #    return np.bincount(k_nearest_labels).argmax()
                # TypeError: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'

            # Logistic Regression:
                # y = column_or_1d(y, warn=True)
                # raise ValueError(
                # ValueError: y should be a 1d array, got an array of shape (75, 3) instead.



            # Train the model
            # Evaluate the model
            # Display evaluation results
            st.write("Model trained successfully.")


                # def __init__(self, 
                #  metrics: List[Metric],
                #  dataset: Dataset, 
                #  model: Model,
                #  input_features: List[Feature],
                #  target_feature: Feature,
                #  split=0.8,
                #  ):

else:
    st.write("No dataset selected.")





