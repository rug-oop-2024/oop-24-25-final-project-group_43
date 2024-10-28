import numpy as np

from sklearn.preprocessing import PolynomialFeatures

from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression

class PolynomialRegression:
    def __init__(self, degree=2):
        """
        Initializes the Polynomial Regression model with a specified degree.

        Parameters:
            degree (int): The degree of the polynomial features. Default is 2.

        Attributes:
            degree (int): The degree of the polynomial features.
            poly_features (PolynomialFeatures): An instance of PolynomialFeatures
              to generate polynomial features.
            mlr (MultipleLinearRegressor): An instance of MultipleLinearRegressor
              for performing the regression.
        """
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=self.degree)
        self.mlr = MultipleLinearRegression()  # Using MLR for the regression part

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the polynomial regression model to the provided data.
        Parameters:
        X (np.ndarray): The input features, a 2D array where each row represents
          a sample and each column represents a feature.
        y (np.ndarray): The target values, a 1D array where each element corresponds
          to the target value of a sample.
        Returns:
        None
        """
        # Transform the input features to polynomial features
        X_poly = self.poly_features.fit_transform(X)
        
        # Fit the MLR model on the transformed features
        self.mlr.fit(X_poly, y)

    def predict(self, X: np.ndarray):
        """
        Predicts the target values for the given input features using the
          polynomial regression model.
        Parameters:
        X (np.ndarray): The input features for which predictions are to be made.
        Returns:
        np.ndarray: The predicted target values.
        """
        # Transform the input features to polynomial features before predicting
        X_poly = self.poly_features.transform(X)
        
        # Use the MLR model to predict
        return self.mlr.predict(X_poly)

    def get_params(self):
        """
        Retrieve the parameters of the polynomial regression model.

        Returns:
            dict: A dictionary containing the coefficients and intercept
              of the underlying multiple linear regression (MLR) model.
        """
        # Get the coefficients and intercept from the underlying MLR model
        return self.mlr.get_params()