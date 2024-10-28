
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.regression.polynominal import PolynomialRegression
from autoop.core.ml.model.regression.lasso import LassoWrapper
from autoop.core.ml.model.classification.knn import KNN
from autoop.core.ml.model.classification.logistic_regression import LogisticRegressionWrapper
from autoop.core.ml.model.classification.random_forest import RandomForest

REGRESSION_MODELS = [
    "Lasso",
    "Multiple Linear Regression",
    "Polynomial"
] # add your models as str here

CLASSIFICATION_MODELS = [
    "KNN",
    "Logistic Regression",
    "Random forest"
] # add your models as str here

def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    raise NotImplementedError("To be implemented.")