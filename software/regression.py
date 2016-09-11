"""Bayesian linear regression."""

from numbers import Number

import numpy as np


class LinearRegression:
    """Bayesian linear regression with full predictive distribution."""
    
    def __init__(self, basis, α=1., β=1.):
        """Set up basis and hyperparameters of a linear regression model.
        
        basis: callable that transforms a predictor vector into the basis of
            the regression.
        α: regularization parameter (lower → less regulation).
        β: inverse of target error variance (lower → higher errors). Beta is
            currently fixed
        """
        assert callable(basis)
        self.basis = basis
        self.α = α
        self.β = β
    
    def _design_matrix(self, predictors):
        return np.apply_along_axis(self.basis, 1, predictors)
    
    def fit(self, predictors, targets):
        """Fit the model to a set of training data.
        
        Rows of predictors and targets belong together. A scalar target must
        be given as a column vector.
        """
        Φ = self._design_matrix(predictors)
        self.coef_covi = self.β * (Φ.T @ Φ) + self.α * np.eye(Φ.shape[1], dtype=Φ.dtype)
        lhs = self.coef_covi / self.β
        rhs = Φ.T @ targets
        self.coef_mean = np.linalg.solve(lhs, rhs)
    
    def predict(self, predictors, samples=None):
        """Predict values using the fitted model.
        
        If samples is std the pointwise standard deviation of the predictive
            distribution is returned as a second output. If samples is an
            integer, additional targets are returned, each predicted by a
            random sample from the distribution of coefficients.
        """
        if isinstance(predictors, Number):
            predictors = np.array([[predictors]])
        elif len(predictors.shape) == 1:
            predictors = predictors.reshape(1,-1)
        Φ = self._design_matrix(predictors)
        mean = Φ @ self.coef_mean
        if samples is None:
            return mean
        elif samples == "std":
            var_term = np.sum((Φ @ np.linalg.inv(self.coef_covi)) * Φ,
                    axis=1, keepdims=True)
            std = np.sqrt(1./self.β + var_term)
            return mean, std
        elif isinstance(samples, int):
            cov = np.linalg.inv(self.coef_covi)
            cs = np.zeros([samples] + list(self.coef_mean.shape), dtype=float)
            for i in range(self.coef_mean.shape[1]):
                cs[:,:,i] = np.random.multivariate_normal(self.coef_mean[:,i], cov, size=samples)
            return mean, [Φ @ c for c in cs]
        else:
            raise ValueError("Invalid value for argument sample.")

