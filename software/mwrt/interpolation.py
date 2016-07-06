"""Interpolation utilities."""

import numpy as np


class LinearInterpolation:
    """Matrix-based linear interpolation of 1-dimensional vectors.
    
    More costly than scipy.interpolate.interp1d, but the matrix also is the
    Jacobian and important for the differentiation of the numerical model. The
    interpolation matrix is dense, care has to be taken of memory requirements.
    """

    def __init__(self, source, target):
        assert strictly_monotonic(source)
        assert strictly_monotonic(target)
        assert np.min(target) >= np.min(source)
        assert np.max(target) <= np.max(source)
        self.source = source
        self.target = target
        self._generate_matrix()

    def _generate_matrix(self):
        distances = self.target[:,None] - self.source[None,:]
        # Find neighbouring points to each target point in source and turn
        # indices into appropriate slices
        dim0 = np.arange(distances.shape[0]), 
        upper_idx = dim0, np.nanargmax(
                np.where(distances <= 0, distances, np.nan), axis=1)
        lower_idx = dim0, np.nanargmin(
                np.where(distances >= 0, distances, np.nan), axis=1)
        # Prepare interpolation weights
        upper_distances = distances[upper_idx]
        lower_distances = distances[lower_idx]
        differences = upper_distances - lower_distances
        differences[differences==0] = 1. # Avoid division by 0
        # Fill matrix with weights
        self.matrix = np.zeros_like(distances, dtype=float)
        self.matrix[upper_idx] = 1 - np.abs(upper_distances / differences)
        self.matrix[lower_idx] = 1 - np.abs(lower_distances / differences)
        # Test basic interpolation property
        assert np.isclose(1., np.sum(self.matrix, axis=1)).all()

    def __call__(self, value):
        return self.matrix @ value

    def __matmul__(self, other):
        return self.matrix @ other

    def __rmatmul__(self, other):
        return other @ self.matrix

    def __repr__(self):
        return "LinearInterpolation(source={}, target={})".format(
                repr(self.source), repr(self.target))


def strictly_monotonic(x):
    difference = x[:-1] - x[1:]
    return np.all(difference > 0) or np.all(difference < 0)

