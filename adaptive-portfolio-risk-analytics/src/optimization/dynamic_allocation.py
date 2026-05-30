"""Future dynamic allocation extension point (Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseAllocator


class DynamicAllocationAllocator(BaseAllocator):
    """Phase 3 extension point."""

    def fit(
        self,
        returns: pd.DataFrame,
        cov_matrix: np.ndarray | None = None,
    ) -> "DynamicAllocationAllocator":
        _ = (returns, cov_matrix)
        raise NotImplementedError("Dynamic allocation is reserved for Phase 3.")

    def get_weights(self) -> np.ndarray:
        raise NotImplementedError("Dynamic allocation is reserved for Phase 3.")
