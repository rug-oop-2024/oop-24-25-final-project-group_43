
from typing import List
import pandas as pd
import json
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []
    _data = dataset.read()
    for col in _data.columns:
        if _data[col].dtype == "object":
            features.append(Feature(name=col, type="categorical"))
        elif _data[col].dtype == "int64" or _data[col].dtype == "float64":
            features.append(Feature(name=col, type="numerical"))
    return features
