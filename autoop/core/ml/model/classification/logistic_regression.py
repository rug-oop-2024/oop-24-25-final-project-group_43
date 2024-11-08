import numpy as np

from autoop.core.ml.model import Model
from sklearn.linear_model import LogisticRegression

class LogisticRegressionWrapper(Model):
    """
    It used the logistic regression model for training and predictions.
    """

    def __init__(self, penalty: str = 'l2', C: float = 1.0) -> None:
        """
        Initialize the Logistic Regression model.

        param: penalty: The penalty to be used in the Logistic Regression model.
        param: C: The regularization parameter for the Logistic Regression model.
        """
        super().__init__()
        self._model = LogisticRegression(penalty=penalty, C=C)
        self.type = "classification"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Trains the Logistic Regression model.

        param: observations: A ndarray of the observations to train the model on.
        param: ground_truth: A ndarray of the target values for the provided
            observations.
        return: None
        """
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "penalty": self._model.penalty,
            "C": self._model.C,
            "intercept_": self._model.intercept_,
            "coef_": self._model.coef_,
            "n_iter_": self._model.n_iter_,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data using the fitted Logistic Regression model.

        param: observations: A ndarray of the observations to make predictions on.
        return: A ndarray of the predicted values for the provided observations.
        """
        return self._model.predict(observations)

    def get_params(self) -> dict:
        """
        Provide access to the parameters of the model.

        param: None
        return: The parameters dictionary
        """
        return self._parameters
