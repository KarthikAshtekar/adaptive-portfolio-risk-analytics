"""Mean-variance (Markowitz) allocator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import BaseAllocator


class MeanVarianceAllocator(BaseAllocator):
    """Max-Sharpe long-only allocator."""

    def __init__(
        self,
        target_return: float | None = None,
        risk_free_rate: float = 0.02,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ):
        self.target_return = target_return
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self._weights: np.ndarray | None = None

    def fit(
        self,
        returns: pd.DataFrame,
        cov_matrix: np.ndarray | None = None,
    ) -> "MeanVarianceAllocator":
        if returns.empty:
            raise ValueError("returns must not be empty")

        clean = returns.dropna(how="any")
        if clean.empty:
            raise ValueError("returns has no valid rows after dropping NaNs")

        mu = clean.mean().values * 252.0
        cov = cov_matrix if cov_matrix is not None else clean.cov().values

        n_assets = clean.shape[1]
        cov = np.asarray(cov, dtype=float)
        cov = 0.5 * (cov + cov.T)
        cov += np.eye(n_assets) * 1e-8

        rf = self.risk_free_rate

        def objective(weights: np.ndarray) -> float:
            port_ret = float(weights @ mu)
            port_var = float(weights.T @ cov @ weights)
            port_vol = np.sqrt(max(port_var, 1e-12))
            sharpe = (port_ret - rf) / port_vol
            return -sharpe

        constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
        if self.target_return is not None:
            constraints.append(
                {"type": "ineq", "fun": lambda w: float((w @ mu) - self.target_return)}
            )

        bounds = [(self.min_weight, self.max_weight)] * n_assets
        x0 = np.ones(n_assets) / n_assets

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9, "disp": False},
        )

        if not result.success:
            self._weights = x0
            return self

        weights = np.clip(result.x, self.min_weight, self.max_weight)
        weights = weights / weights.sum()
        self._weights = weights
        return self

    def get_weights(self) -> np.ndarray:
        if self._weights is None:
            raise ValueError("allocator not fitted")
        return self._weights
