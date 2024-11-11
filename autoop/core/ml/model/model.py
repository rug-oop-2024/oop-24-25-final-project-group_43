from abc import ABC, abstractmethod
import numpy as np
from typing import Literal


class Model(ABC):
    """
    Abstract base class for machine learning models.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the model with given arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Attributes:
            parameters (dict): A dictionary to store model parameters.
            type: The type of the model.
        """
        self.parameters: dict = {}
        self.type = type

    @abstractmethod
    def fit(self, observations: Literal[np.ndarray],  # type: ignore
            ground_truth: Literal[np.ndarray]) -> None:  # type: ignore
        """
        Fits the model to the provided observations and ground truth data.

        Parameters:
        observations (np.ndarray): The input data to fit the model.
        ground_truth (np.ndarray): The actual output values
            corresponding to the input data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, observations:
                Literal[np.ndarray]) -> np.ndarray:  # type: ignore
        """
        Predict the output for given observations.

        Parameters:
        observations (np.ndarray): A numpy array containing the input
            data for prediction.

        Returns:
        np.ndarray: A numpy array containing the predicted outputs.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        Retrieve the parameters of the model.

        Returns:
            dict: A dictionary containing the model parameters.
        """
        pass
