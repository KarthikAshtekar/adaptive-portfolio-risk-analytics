"""Base interfaces for portfolio allocators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseAllocator(ABC):
    """Abstract allocator interface for Phase 1 and extensions."""

    @abstractmethod
    def fit(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame | np.ndarray | None = None,
    ) -> "BaseAllocator":
        """Fit allocator on returns and optional covariance."""

    @abstractmethod
    def get_weights(self) -> pd.Series | np.ndarray:
        """Get computed portfolio weights."""

    def optimize(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame | np.ndarray | None = None,
    ) -> pd.Series | np.ndarray:
        """Backward-compatible optimization method."""
        return self.fit(returns, cov_matrix).get_weights()
