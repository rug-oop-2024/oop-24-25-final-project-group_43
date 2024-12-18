from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():
    """
    Pipeline class for managing the machine learning workflow.

    This class encapsulates the entire machine learning pipeline, including
    preprocessing, data splitting, model training, and evaluation. It supports
    both classification and regression tasks and allows for the registration
    and retrieval of artifacts generated during the pipeline execution.

        _dataset (Dataset): The dataset to be used in the pipeline.
        _model (Model): The machine learning model to be trained
        and evaluated.
        _input_features (List[Feature]): A list of features to be
        used as input for the model.
        _target_feature (Feature): The feature to be predicted by
        the model.
        _metrics (List[Metric]): A list of metrics to evaluate the model.
        _artifacts (dict): A dictionary to store artifacts generated
        during the pipeline execution.
        _split (float): The proportion of the dataset to be used for training.
        _output_vector (list): The preprocessed target data.
        _input_vectors (list of lists): The preprocessed input data.
        _train_x (list of lists): The training subset of the input data.
        _test_x (list of lists): The testing subset of the input data.
        _train_y (list): The training subset of the output data.
        _test_y (list): The testing subset of the output data.
        _metrics_results (list): The evaluation results of the model
        on the test data.
        _predictions (list): The predictions made by the model on
        the test data.
        _metrics_results_train (list): The evaluation results of the
        model on the training data.

    Methods:
        __init__(metrics, dataset, model, input_features,
            target_feature, split=0.8):
            Initializes the pipeline with the given dataset, model,
            input features, target feature, and metrics.
        __str__():
        model():
        artifacts():
            Returns the artifacts generated during the pipeline execution
            to be saved.
        _register_artifact(name, artifact):
        _preprocess_features():
        _split_data():
            Splits the input and output data into training and testing
            sets based on the specified split ratio.
        _compact_vectors(vectors):
        _train():
        _evaluate():
            Evaluates the model using the test data and computes the
            specified metrics.
        _evaluate_train():
        execute():
            Executes the machine learning pipeline by performing preprocessing,
            data splitting, training, and evaluation.
    """
    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 ) -> None:
        """
        Initializes the pipeline with the given dataset, model,
        input features, target feature, and metrics.

        Args:
            metrics (List[Metric]): A list of metrics to evaluate the model.
            dataset (Dataset): The dataset to be used in the pipeline.
            model (Model): The machine learning model to be trained and
                evaluated.
            input_features (List[Feature]): A list of features to be used as
            input for the model.
            target_feature (Feature): The feature to be predicted by the model.
            split (float, optional): The proportion of the dataset to be used
            for training.
                                        Defaults to 0.8.

        Raises:
            ValueError: If the target feature is categorical and the
                model type is not classification.
            ValueError: If the target feature is continuous and the model
                type is not regression.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical" and \
                model.type != "classification":
            raise ValueError("Model type must be classification "
                             "for categorical target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError("Model type must be regression for"
                             " continuous target feature")

    def __str__(self) -> str:
        """
        Returns a string representation of the Pipeline object.

        The string includes the type of the model, a list of input features,
        the target feature, the data split, and a list of metrics.

        Returns:
            str: A formatted string representation of the Pipeline object.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Returns the machine learning model instance.

        Returns:
            Model: The machine learning model instance.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Used to get the artifacts generated during the
        pipeline execution to be saved.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(
            name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """
        Registers an artifact with a given name.

        Args:
            name (str): The name to register the artifact under.
            artifact: The artifact to be registered.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocesses the features of the dataset.

        This method preprocesses both the target feature and
        input features of the dataset.
        It registers artifacts for each feature and stores
        the preprocessed data.

        Steps:
        1. Preprocess the target feature and register its artifact.
        2. Preprocess the input features and register their artifacts.
        3. Store the preprocessed target data in `_output_vector`.
        4. Store the preprocessed input data in `_input_vectors`.

        Returns:
            None
        """
        (target_feature_name, target_data, artifact) = \
            preprocess_features([self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector, sort by feature name
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data,
                                         artifact) in input_results]

    def _split_data(self) -> None:
        """
        Splits the input and output data into training and testing
        sets based on the specified split ratio.

        This method divides the input vectors and output vector
        into training and testing subsets.
        The split ratio is determined by the `self._split`
        attribute.

        Attributes:
            self._input_vectors (list of lists): The input data to be split.
            self._output_vector (list): The output data to be split.
            self._split (float): The ratio to split the data into
                training and testing sets.

        Sets:
            self._train_x (list of lists): The training subset of
                the input data.
            self._test_x (list of lists): The testing subset of
                the input data.
            self._train_y (list): The training subset of the output data.
            self._test_y (list): The testing subset of the output data.
        """
        # Split the data into training and testing sets
        split = self._split
        self._train_x = [vector[:int(split * len(vector))] for vector
                         in self._input_vectors]
        self._test_x = [vector[int(split * len(vector)):] for vector
                        in self._input_vectors]
        self._train_y = self._output_vector[:int(
            split * len(self._output_vector))]
        self._test_y = self._output_vector[int(
            split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Concatenates a list of numpy arrays along the second axis (axis=1).

        Args:
            vectors (List[np.array]): A list of numpy arrays
            to be concatenated.

        Returns:
            np.array: A single numpy array resulting from the
            concatenation of the input arrays.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Trains the model using the training data.

        This method compacts the feature vectors from the training
        data and fits the model using the compacted feature vectors
        and the corresponding target values.

        Returns:
            None
        """
        X = self._compact_vectors(self._train_x)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        Evaluates the model using the test data and computes
        the specified metrics.

        This method performs the following steps:
        1. Compacts the test feature vectors.
        2. Uses the model to predict the target values based on the
            compacted test features.
        3. Evaluates the predictions using each metric in the metrics list.
        4. Stores the results of each metric evaluation in the
            metrics_results list.
        5. Stores the predictions in the predictions attribute.

        Returns:
            None
        """
        X = self._compact_vectors(self._test_x)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def _evaluate_train(self) -> None:
        """
        Evaluates the model on the training data using the specified metrics.

        This method compacts the training feature vectors, makes
        predictions using the model,
        and evaluates these predictions against the true labels using each
        metric in the metrics list. The results are stored in the
        _metrics_results_train attribute.

        Returns:
            None
        """
        X = self._compact_vectors(self._train_x)
        Y = self._train_y
        self._metrics_results_train = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results_train.append((metric, result))

    def execute(self) -> dict:
        """
        Executes the machine learning pipeline by performing preprocessing,
        data splitting, training, and evaluation.

        Returns:
            dict: A dictionary containing the following keys:
                - "metrics": The evaluation metrics of the model on the
                    test data.
                - "predictions": The predictions made by the model on
                    the test data.
                - "metrics_train": The evaluation metrics of the model
                    on the training data.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        self._evaluate_train()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
            "metrics_train": self._metrics_results_train
        }
