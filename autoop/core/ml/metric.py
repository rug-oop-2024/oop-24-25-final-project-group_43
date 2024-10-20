from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error", #regression
    "root_mean_squared_error", #regression
    "mean_absolute_error", #regression
    "accuracy", #classification
    "recall", #classification
    "precision", #classification
] # add the names (in strings) of the metrics you implement

def get_metric(name: str) -> Any:
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    match name:
        case "mean_squared_error":
            return MeanSquaredError()
        case "root_mean_squared_error":
            return RootMeanSquaredError()
        case "mean_absolute_error":
            return MeanAbsoluteError()
        case "accuracy":
            return Accuracy()
        case "recall":
            return Recall()
        case "precision":
            return Precision()
        case _:
            raise ValueError(f"Metric {name} not found.")

class Metric(ABC):
    """Base class for all metrics."""
    # remember: metrics take ground truth and prediction as input and return a real number

    @abstractmethod
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        raise NotImplementedError("To be implemented.")

# add here concrete implementations of the Metric class

class MeanSquaredError(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        return np.mean((ground_truth - prediction) ** 2)
    
class RootMeanSquaredError(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        return np.sqrt(np.mean((ground_truth - prediction) ** 2))
    
class MeanAbsoluteError(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        return np.mean(np.abs(ground_truth - prediction))

class Accuracy(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        return np.mean(ground_truth == prediction)
    
class Recall(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        tp = np.sum((ground_truth == 1) & (prediction == 1))
        fn = np.sum((ground_truth == 1) & (prediction == 0))
        return tp / (tp + fn)
    
class Precision(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        tp = np.sum((ground_truth == 1) & (prediction == 1))
        fp = np.sum((ground_truth == 0) & (prediction == 1))
        return tp / (tp + fp)