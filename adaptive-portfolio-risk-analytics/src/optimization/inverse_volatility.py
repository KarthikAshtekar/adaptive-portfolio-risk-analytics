"""Inverse-volatility allocator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseAllocator


class InverseVolatilityAllocator(BaseAllocator):
    """Allocate inversely proportional to asset volatility."""

    def __init__(self):
        self._weights: np.ndarray | None = None

    def fit(
        self,
        returns: pd.DataFrame,
        cov_matrix: np.ndarray | None = None,
    ) -> "InverseVolatilityAllocator":
        _ = cov_matrix
        if returns.empty:
            raise ValueError("returns must not be empty")

        clean = returns.dropna(how="any")
        if clean.empty:
            raise ValueError("returns has no valid rows after dropping NaNs")

        vol = clean.std().values
        inv_vol = 1.0 / np.clip(vol, 1e-12, None)
        self._weights = inv_vol / inv_vol.sum()
        return self

    def get_weights(self) -> np.ndarray:
        if self._weights is None:
            raise ValueError("allocator not fitted")
        return self._weights
