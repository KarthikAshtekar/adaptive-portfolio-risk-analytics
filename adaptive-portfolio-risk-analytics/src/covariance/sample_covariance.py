"""Base and Phase 1 covariance estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


class BaseCovarianceEstimator(ABC):
    """Abstract covariance estimator interface."""

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> "BaseCovarianceEstimator":
        """Fit estimator on returns."""

    @abstractmethod
    def get_covariance(self) -> np.ndarray:
        """Return fitted covariance matrix."""

    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """Convenience method for one-shot estimation."""
        return self.fit(returns).get_covariance()


@dataclass
class SampleCovarianceEstimator(BaseCovarianceEstimator):
    """Phase 1 sample covariance estimator."""

    ddof: int = 1
    _covariance: np.ndarray | None = None

    def fit(self, returns: pd.DataFrame) -> "SampleCovarianceEstimator":
        if returns.empty:
            raise ValueError("returns must not be empty")

        clean = returns.dropna(how="any")
        if clean.empty:
            raise ValueError("returns has no valid rows after dropping NaNs")

        self._covariance = np.cov(clean.values, rowvar=False, ddof=self.ddof)
        return self

    def get_covariance(self) -> np.ndarray:
        if self._covariance is None:
            raise ValueError("estimator not fitted")
        return self._covariance


@dataclass
class RollingCovarianceEstimator(BaseCovarianceEstimator):
    """Rolling-window covariance estimator."""

    window: int = 252
    method: str = "standard"
    _covariance: np.ndarray | None = None

    def fit(self, returns: pd.DataFrame) -> "RollingCovarianceEstimator":
        if returns.empty:
            raise ValueError("returns must not be empty")
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if len(returns) < self.window:
            raise ValueError("returns length must be >= window")

        recent = returns.iloc[-self.window:].dropna(how="any")
        if recent.empty:
            raise ValueError("no valid rows in rolling window")

        if self.method == "standard":
            self._covariance = recent.cov().values
        elif self.method == "exponential_weighted":
            ewm_cov = recent.ewm(span=self.window, adjust=False).cov()
            latest_cov = ewm_cov.xs(recent.index[-1], level=0)
            self._covariance = latest_cov.values
        else:
            raise ValueError(f"unsupported rolling covariance method: {self.method}")

        return self

    def get_covariance(self) -> np.ndarray:
        if self._covariance is None:
            raise ValueError("estimator not fitted")
        return self._covariance

    def estimate_series(self, returns: pd.DataFrame) -> dict[pd.Timestamp, np.ndarray]:
        """Estimate covariance across all rolling windows."""
        if len(returns) < self.window:
            return {}

        cov_series: dict[pd.Timestamp, np.ndarray] = {}
        for i in range(self.window, len(returns) + 1):
            window_slice = returns.iloc[i - self.window: i].dropna(how="any")
            if window_slice.empty:
                continue
            cov_series[returns.index[i - 1]] = window_slice.cov().values
        return cov_series
