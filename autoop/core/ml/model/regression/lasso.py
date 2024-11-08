import numpy as np

from autoop.core.ml.model.model import Model
from sklearn.linear_model import Lasso


class LassoWrapper(Model):
    """
    It provides an interface for training and making predictions using the Lasso model.
    """

    def __init__(self, alpha: int = 1) -> None:
        """
        Initialize the Lasso model with a regularization parameter.

        param: alpha: The regularization parameter for the Lasso model.
        """
        super().__init__()
        self._model = Lasso()
        self.type = "regression"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Trains the Lasso model.

        param: observations: A ndarray of the observations to train the model on.
        param: ground_truth: A ndarray of the target values for the provided
            observations.
        return: None
        """
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "alpha": self._model.alpha,
            "coef_": self._model.coef_,
            "intercept_": self._model.intercept_,
            "tol": self._model.tol,
            "max_iter": self._model.max_iter,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data using the fitted Lasso model.

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
