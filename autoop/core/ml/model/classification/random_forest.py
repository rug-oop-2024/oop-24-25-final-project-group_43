import numpy as np


from autoop.core.ml.model.model import Model
from sklearn.ensemble import RandomForestClassifier

class RandomForest(Model):
    """
    Implements a Random Forest model for classification.

    It uses an ensemble of decision trees to make predictions.
    """

    def __init__(self, n_estimators: int = 100) -> None:
        """
        Initialize the Random Forest model with the number of estimators.

        :param int n_estimators: The number of trees in the forest.
        """
        super().__init__()
        self._model = RandomForestClassifier(n_estimators=n_estimators)
        self.type = "classification"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the Random Forest model.

        :param np.ndarray observations: A 2D numpy array where each row represents an
            observation and each column represents a feature.
        :param np.ndarray ground_truth: A 1D numpy array containing the ground truth
            values for the observations.
        return: None
        """
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained Random Forest model.

        :param np.ndarray observations: A 2D numpy array where each row represents an
            observation and each column represents a feature.
        return: np.ndarray predictions: A 1D numpy array containing the predicted values
        """
        return self._model.predict(observations)

    def get_params(self) -> dict:
        """
        Provide access to the parameters of the model.

        return: The parameters dictionary
        """
        return self._model.get_params()