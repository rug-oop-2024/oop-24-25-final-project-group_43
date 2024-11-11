"""	Module for defining evaluation metrics for machine learning models.	"""
from abc import ABC, abstractmethod
import numpy as np

METRICS = [
    "mean_squared_error",  # regression
    "root_mean_squared_error",  # regression
    "mean_absolute_error",  # regression
    "accuracy",  # classification
    "recall",  # classification
    "precision",  # classification
]


def get_metric(name: str) -> 'Metric':
    """
    Retrieve a metric object based on the provided metric name.

    Args:
        name (str): The name of the metric to retrieve. The metrics are:
                    - "Mean Squared Error"
                    - "Root Mean Squared Error"
                    - "Mean Absolute Error"
                    - "Accuracy"
                    - "Recall"
                    - "Precision"

    Returns:
        Metric: An instance of the requested metric.

    Raises:
        ValueError: If the provided metric name is not supported.
    """
    match name:
        case "Mean Squared Error":
            return MeanSquaredError()
        case "Root Mean Squared Error":
            return RootMeanSquaredError()
        case "Mean Absolute Error":
            return MeanAbsoluteError()
        case "Accuracy":
            return Accuracy()
        case "Recall":
            return Recall()
        case "Precision":
            return Precision()
        case _:
            raise ValueError(f"Metric {name} not found.")


class Metric(ABC):
    """Base class for all metrics."""

    def __call__(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """
        Evaluate the model's prediction against the ground truth.

        Args:
            ground_truth (np.ndarray): The actual values.
            prediction (np.ndarray): The predicted values by the model.

        Returns:
            float: The evaluation result as a float.
        """
        self.evaluate(ground_truth, prediction)

    @abstractmethod
    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """
        Evaluate the prediction against the ground truth.

        Parameters:
        ground_truth (np.ndarray): The actual values.
        prediction (np.ndarray): The predicted values.

        Returns:
        float: The evaluation score.
        """
        return self(ground_truth, prediction)


class MeanSquaredError(Metric):
    """Mean Squared Error metric."""

    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """
        Evaluate the mean squared error.

        Parameters:
        ground_truth (np.ndarray): The actual values.
        prediction (np.ndarray): The predicted values.

        Returns:
        float: The mean squared error between the ground truth
        and prediction.
        """
        return np.mean((ground_truth - prediction) ** 2)


class RootMeanSquaredError(Metric):
    """Root Mean Squared Error metric."""

    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """
        Evaluate the root mean squared error.

        Parameters:
        ground_truth (np.ndarray): The actual values.
        prediction (np.ndarray): The predicted values.

        Returns:
        float: The root mean squared error between the ground truth
        and prediction.
        """
        return np.sqrt(np.mean((ground_truth - prediction) ** 2))


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric."""

    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """
        Evaluate the mean absolute error.

        Parameters:
        ground_truth (np.ndarray): The actual values.
        prediction (np.ndarray): The predicted values.

        Returns:
        float: The mean absolute error between the ground
        truth and prediction.
        """
        return np.mean(np.abs(ground_truth - prediction))


class Accuracy(Metric):
    """Accuracy metric."""

    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """
        Evaluate the accuracy of the predictions.

        Parameters:
        ground_truth (np.ndarray): The true labels.
        prediction (np.ndarray): The predicted labels.

        Returns:
        float: The accuracy of the predictions as a float.
        """
        return np.mean(ground_truth == prediction)


class Recall(Metric):
    """Recall metric."""

    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """
        Evaluate the performance of a prediction against the ground truth.

        This method calculates the recall metric, which is the ratio
        of true positives to the sum of true positives and false negatives.

        Args:
            ground_truth (np.ndarray): The ground truth binary labels.
            prediction (np.ndarray): The predicted binary labels.

        Returns:
            float: The recall score.
        """
        tp = np.sum((ground_truth == 1) & (prediction == 1))
        fn = np.sum((ground_truth == 1) & (prediction == 0))
        return tp / (tp + fn)


class Precision(Metric):
    """Precision metric."""

    def evaluate(self, ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """
        Evaluate the precision of the predictions.

        Parameters:
        ground_truth (np.ndarray): Array containing the ground
            truth binary labels.
        prediction (np.ndarray): Array containing the predicted
            binary labels.

        Returns:
        float: The precision of the predictions, calculated as
                the number of true positives divided by the sum
                of true positives and false positives.
        """
        tp = np.sum((ground_truth == 1) & (prediction == 1))
        fp = np.sum((ground_truth == 0) & (prediction == 1))
        return tp / (tp + fp)
