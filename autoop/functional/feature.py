
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
    decoded_data = dataset.data.decode('utf-8')  # Adjust encoding if necessary
    #print("Decoded data:", decoded_data)
    
    # Parse JSON from the decoded data
    if isinstance(decoded_data, str):
        data_dict = json.loads(decoded_data)
        print("Parsed JSON data:", data_dict)

    # Create DataFrame from parsed data
    df = pd.DataFrame(data_dict)

    first_row = df.iloc[0]
    print(f"This is the first row!!! \n {first_row}")
    for col in first_row:
        if isinstance(col, str):
            features.append(Feature(name=col, type="categorical"))
        elif isinstance(col, (int, float)):
            features.append(Feature(name=col, type="numerical"))
    return features

# i don't think categorical and numerical can be determined by being instances of strings and integers/floats
#       this is from test_features.py:

        # numerical_columns = [
        #     "age",
        #     "education-num",
        #     "capital-gain",
        #     "capital-loss",
        #     "hours-per-week",
        # ]
        # categorical_columns = [
        #     "workclass",
        #     "education",
        #     "marital-status",
        #     "occupation",
        #     "relationship",
        #     "race",
        #     "sex",
        #     "native-country",
        # ]