"""Extension points for future covariance estimators (Phase 2/3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sample_covariance import BaseCovarianceEstimator


@dataclass
class LedoitWolfEstimator(BaseCovarianceEstimator):
    """Phase 2 extension point (not implemented)."""

    shrinkage: float | None = None
    target: str = "identity"

    def fit(self, returns: pd.DataFrame) -> "LedoitWolfEstimator":
        _ = returns
        raise NotImplementedError("Ledoit-Wolf is reserved for Phase 2 and is not implemented.")

    def get_covariance(self) -> np.ndarray:
        raise NotImplementedError("Ledoit-Wolf is reserved for Phase 2 and is not implemented.")


@dataclass
class GerberCovarianceEstimator(BaseCovarianceEstimator):
    """Phase 3 extension point (not implemented)."""

    correlation_type: str = "RS"

    def fit(self, returns: pd.DataFrame) -> "GerberCovarianceEstimator":
        _ = returns
        raise NotImplementedError(
            "Gerber covariance is reserved for Phase 3 and is not implemented."
        )

    def get_covariance(self) -> np.ndarray:
        raise NotImplementedError(
            "Gerber covariance is reserved for Phase 3 and is not implemented."
        )
