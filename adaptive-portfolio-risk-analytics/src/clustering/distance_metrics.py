"""Distance and correlation utilities for hierarchical clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform


class DistanceMetrics:
    """Distance transformations for portfolio clustering."""

    @staticmethod
    def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        """Compute asset correlation matrix from returns."""
        if returns.empty:
            raise ValueError("returns must not be empty")
        return returns.corr()

    @staticmethod
    def correlation_distance(correlation: np.ndarray) -> np.ndarray:
        """Convert correlation matrix into metric distance matrix."""
        if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
            raise ValueError("correlation must be a square matrix")

        clipped = np.clip(correlation, -1.0, 1.0)
        distance = np.sqrt(0.5 * (1.0 - clipped))
        np.fill_diagonal(distance, 0.0)
        return distance

    @staticmethod
    def to_condensed(distance_matrix: np.ndarray) -> np.ndarray:
        """Convert square distance matrix to condensed form for scipy linkage."""
        if distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        symmetric = (distance_matrix + distance_matrix.T) / 2.0
        np.fill_diagonal(symmetric, 0.0)
        return squareform(symmetric, checks=False)
