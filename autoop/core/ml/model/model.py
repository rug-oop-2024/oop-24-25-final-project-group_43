
from abc import ABC,abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal


# Changes:
# Replaced Artifact with ABC as parent class
# added self.parameters and self.type
# commented out def to_artifact()

class Model(ABC):
    def __init__(self, *args, **kwargs):
        self.parameters: dict = {}
        self.type = type


    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass

    # Need to implement this method
    # @abstractmethod
    # def to_artifact(self):
    #     pass

    
