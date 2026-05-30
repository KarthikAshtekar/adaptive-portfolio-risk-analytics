"""HRP allocator adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.clustering.hrp import HierarchicalRiskParity

from .base import BaseAllocator


class HRPAllocator(BaseAllocator):
    """Allocate using Hierarchical Risk Parity."""

    def __init__(self, linkage_method: str = "single"):
        self.linkage_method = linkage_method
        self._weights: np.ndarray | None = None

    def fit(self, returns: pd.DataFrame, cov_matrix: np.ndarray | None = None) -> "HRPAllocator":
        _ = cov_matrix
        hrp = HierarchicalRiskParity(linkage_method=self.linkage_method)
        hrp.fit(returns)
        self._weights = hrp.get_weights()
        return self

    def get_weights(self) -> np.ndarray:
        if self._weights is None:
            raise ValueError("allocator not fitted")
        return self._weights
