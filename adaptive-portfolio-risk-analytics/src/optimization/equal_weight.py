"""Equal-weight allocator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseAllocator


class EqualWeightAllocator(BaseAllocator):
    """Allocate equally across all assets."""

    def __init__(self):
        self._weights: np.ndarray | None = None

    def fit(
        self,
        returns: pd.DataFrame,
        cov_matrix: np.ndarray | None = None,
    ) -> "EqualWeightAllocator":
        _ = cov_matrix
        if returns.empty:
            raise ValueError("returns must not be empty")
        n_assets = returns.shape[1]
        self._weights = np.ones(n_assets) / n_assets
        return self

    def get_weights(self) -> np.ndarray:
        if self._weights is None:
            raise ValueError("allocator not fitted")
        return self._weights
