import numpy as np

from autoop.core.ml.model.model import Model


class KNN(Model):
    """
    Implements a K-Nearest Neigbors (KNN) model.

    It predicts the category of a new instance
        based on its proximity to the training dataset.
    """

    def __init__(self, k_value: int = 3) -> None:
        """
        Add more parameters to the class.

        :param int k_value: The number of neighbors to consider when making a
            prediction.
        return: None.
        """
        super().__init__()
        self.type = "classification"
        self.k_value = k_value
        self._observations = None
        self._ground_truth = None
        self._parameters = {}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Trains the model by storing the training data.

        :param np.ndarray observations: A ndarray that contains
            the feature data for the training model.
        :param np.ndarray ground_truth: A ndarray that contains
            the corresponding categories for each observation
            in the training data.
        return: None.
        """
        self._observations = observations
        self._ground_truth = ground_truth

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions based on new observations.

        :param np.ndarray observations: A ndarray that contains the
            feature data
        return: A ndarray that contains predicted labels for the
            provided observations.
        """
        array = [[self._predict_single(x)] for x in observations]
        array = np.array(array)
        return array

    def _predict_single(self, observation: np.ndarray) -> np.int64:
        """
        Predicts the label for a single observation.

        :param np.ndarray observation: A ndarray that contains the
            feature data
        return: The function returns an integer representing
            the predicted label for the given observation.
        """
        x = self._observations
        y = self._ground_truth
        distances = np.linalg.norm(x - observation, axis=1)
        k_indices = np.argpartition(distances, self.k_value)[: self.k_value]
        k_nearest_labels = np.take(y, k_indices).astype(np.int64)
        return np.bincount(k_nearest_labels).argmax()

    def get_params(self) -> dict:
        """
        Provide access to the parameters of the model.

        param: None
        return: The parameters dictionary
        """
        return self._parameters
