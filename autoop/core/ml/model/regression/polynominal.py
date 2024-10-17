import numpy as np

from sklearn.preprocessing import PolynomialFeatures

from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegressor

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=self.degree)
        self.mlr = MultipleLinearRegressor()  # Using MLR for the regression part

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Transform the input features to polynomial features
        X_poly = self.poly_features.fit_transform(X)
        
        # Fit the MLR model on the transformed features
        self.mlr.fit(X_poly, y)

    def predict(self, X: np.ndarray):
        # Transform the input features to polynomial features before predicting
        X_poly = self.poly_features.transform(X)
        
        # Use the MLR model to predict
        return self.mlr.predict(X_poly)

    def get_params(self):
        # Get the coefficients and intercept from the underlying MLR model
        return self.mlr.get_parameters()