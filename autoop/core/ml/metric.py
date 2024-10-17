from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
] # add the names (in strings) of the metrics you implement

def get_metric(name: str):
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    raise NotImplementedError("To be implemented.")

class Metric(...):
    """Base class for all metrics.
    """
    # remember: metrics take ground truth and prediction as input and return a real number
    # Code by Jasmijn implemented below
    def __init__(self, name: str) -> None:
        if name not in METRICS:
            raise ValueError(f"Invalid metric name: {name}")
        self.name = name

    def __call__(self, ground_truth, prediction) -> float:
        raise NotImplementedError("To be implemented.")

# what you can do now:
# accuracy_metric = Metric("accuracy")
# accuracy = accuracy_metric(ground_truth, prediction)

# add here concrete implementations of the Metric class
    