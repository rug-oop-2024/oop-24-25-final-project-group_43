"""
This package contains the implementation of multiple linear regression
"""
from autoop.core.ml.model.regression.multiple_linear_regression import \
    MultipleLinearRegression

# To make sure that flake8 doesn't complain about the unused import
MLR_DICT = {
    "multiple_linear_regression": MultipleLinearRegression,
}
