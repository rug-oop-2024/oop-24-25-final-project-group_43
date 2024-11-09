import numpy as np
from autoop.core.ml.model.model import Model


class MultipleLinearRegression(Model):
    """
    Implements a multiple linear regression model.

    It uses the normal equation to calculate the coefficients.
    """

    def __init__(self) -> None:
        """Initialize the model with an empty dictionary for the coefficients.
        """
        super().__init__()
        self._coefficients = None
        self.type = "regression"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the data by calculating the coefficients.

        :param np.ndarray observations: A 2D numpy array where each row
            represents an observation and each column represents a feature.
        :param np.ndarray ground_truth: A 1D numpy array containing
            the ground truth values for the observations.
        return: None
        """
        # Add a column of ones for the intercept term
        observations = np.concatenate(
            (np.ones((observations.shape[0], 1)), observations), axis=1
        )
        xtx = np.dot(observations.T, observations)
        xtx_inv = np.linalg.inv(xtx)
        xty = np.dot(observations.T, ground_truth)
        self._coefficients = np.dot(xtx_inv, xty)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        :param np.ndarray observations: A 2D numpy array where each row
            represents an observation and each column represents a feature.
        return: np.ndarray predictions: A 1D numpy array containing the
            predicted values
        """
        # Add a column of ones for the intercept term
        observations = np.concatenate(
            (np.ones((observations.shape[0], 1)), observations), axis=1
        )
        # Make predictions
        return np.dot(observations, self._coefficients)

    def get_params(self) -> dict[str, np.ndarray]:
        """
        Return the parameters of the model.

        return: dict[str, np.ndarray] parameters: A dictionary containing the
            parameters of the model.
        """
        parameters = {
            "intercept": self._coefficients[0],
            "weight_coefficient": self._coefficients[1],
            "length_coefficient": self._coefficients[2],
        }
        return parameters
