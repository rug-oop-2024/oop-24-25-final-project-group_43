
from typing import List
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
    first_row = dataset.data.iloc[0]
    for col in first_row:
        if isinstance(col, str):
            features.append(Feature(name=col, type="categorical"))
        elif isinstance(col, (int, float)):
            features.append(Feature(name=col, type="numerical"))
    return features
