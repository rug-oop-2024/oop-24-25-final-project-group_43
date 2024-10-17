import numpy as np

class KNN():
    """
    Implements a K-Nearest Neigbors (KNN) model.

    It predicts the category of a new instance
        based on its proximity to the training dataset.
    """

    def __init__(self, k_value: int) -> None:
        """
        Add more parameters to the class.

        :param int k_value: The number of neighbors to consider when making a
            prediction.
        return: None.
        """
        super().__init__()
        self.k_value = k_value
        self._observations = None
        self._ground_truth = None
        self._parameters = {}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Trains the model by storing the training data.

        :param np.ndarray observations: A ndarray that contains the feature data
            for the training model.
        :param np.ndarray ground_truth: A ndarray that contains the corresponfing
            categories for each observation in the training data.
        return: None.
        """
        self._observations = observations
        self._ground_truth = ground_truth

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions based on new observations.

        :param np.ndarray observations: A ndarray that contains the feature data
        return: A ndarray that contains predicted labels for the provided observations.
        """
        return [self._predict_single(x) for x in observations]

    def _predict_single(self, observation: np.ndarray) -> int:
        """
        Predicts the label for a single observation.

        :param np.ndarray observation: A ndarray that contains the feature data
        return: The function returns an integer representing the predicted label for
            the given observation.
        """
        # step 1: calculate the distance between the observation and all the
        # points in the training set
        distances = np.linalg.norm(self._observations - observation, axis=1)
        # step 2: sort the distances
        k_indices = np.argpartition(distances, self.k_value)[: self.k_value]
        # step 3 take the k first elements
        k_nearest_labels = np.take(self._ground_truth, k_indices)
        # step 4: take the majority vote and return the label
        return np.bincount(k_nearest_labels).argmax()
