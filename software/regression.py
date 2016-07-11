"""Bayesian linear regression."""

from numbers import Number

import numpy as np


class LinearRegression:
    """Bayesian linear regression with full predictive distribution."""
    
    def __init__(self, basis, α=1., β=1.):
        """
        
        Lower α → less regulation
        Lower β → higher errors associated with target
        """
        assert callable(basis)
        self.basis = basis
        self.α = α
        self.β = β
    
    def _design_matrix(self, predictors):
        return np.apply_along_axis(self.basis, 1, predictors)
    
    def fit(self, predictors, targets):
        """
        
        targets must be column vector.
        """
        Φ = self._design_matrix(predictors)
        self.coef_covi = self.β * (Φ.T @ Φ) + self.α * np.eye(Φ.shape[1], dtype=Φ.dtype)
        lhs = self.coef_covi / self.β
        rhs = Φ.T @ targets
        self.coef_mean = np.linalg.solve(lhs, rhs)
    
    def predict(self, predictors, error_estimate=True):
        """"""
        if isinstance(predictors, Number):
            predictors = np.array([[predictors]])
        elif len(predictors.shape) == 1:
            predictors = predictors.reshape(1,-1)
        Φ = self._design_matrix(predictors)
        mean = Φ @ self.coef_mean
        if error_estimate is None:
            return mean
        var_term = np.sum((Φ @ np.linalg.inv(self.coef_covi)) * Φ, axis=1, keepdims=True)
        std = np.sqrt(1./self.β + var_term)
        return mean, std