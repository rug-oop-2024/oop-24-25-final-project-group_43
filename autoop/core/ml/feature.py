
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature(BaseModel):
    def __init__(self, name: str, type: Literal["numerical", "categorical"]) -> None:
        self.name = name
        self.type = type

    def __str__(self):
        return f"{self.name}: {self.type}"