"""Hierarchical Risk Parity (HRP) portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

from .distance_metrics import DistanceMetrics


class HierarchicalRiskParity:
    """HRP portfolio optimizer using recursive bisection."""

    def __init__(self, linkage_method: str = "single"):
        self.linkage_method = linkage_method
        self.linkage_matrix: np.ndarray | None = None
        self.asset_order: list[str] | None = None
        self.weights: pd.Series | None = None

    def fit(self, returns: pd.DataFrame) -> "HierarchicalRiskParity":
        if returns.empty:
            raise ValueError("returns must not be empty")

        returns = returns.dropna(how="any")
        if returns.empty:
            raise ValueError("returns has no valid rows after dropping NaNs")

        cov = returns.cov()
        corr = returns.corr()
        dist = DistanceMetrics.correlation_distance(corr.values)
        condensed = DistanceMetrics.to_condensed(dist)

        self.linkage_matrix = linkage(condensed, method=self.linkage_method)
        order_idx = leaves_list(self.linkage_matrix).tolist()
        self.asset_order = [returns.columns[i] for i in order_idx]

        ordered_cov = cov.loc[self.asset_order, self.asset_order]
        self.weights = self._recursive_bisection(ordered_cov, self.asset_order)

        # Return in original column order.
        self.weights = self.weights.reindex(returns.columns).fillna(0.0)
        self.weights = self.weights / self.weights.sum()
        return self

    def _cluster_variance(self, cov: pd.DataFrame, cluster: list[str]) -> float:
        sub_cov = cov.loc[cluster, cluster]
        diag = np.diag(sub_cov.values)
        inv_diag = 1.0 / np.clip(diag, 1e-12, None)
        ivp = inv_diag / inv_diag.sum()
        return float(ivp.T @ sub_cov.values @ ivp)

    def _recursive_bisection(self, cov: pd.DataFrame, ordered_assets: list[str]) -> pd.Series:
        weights = pd.Series(1.0, index=ordered_assets)
        clusters: list[list[str]] = [ordered_assets]

        while clusters:
            cluster = clusters.pop(0)
            if len(cluster) <= 1:
                continue

            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]

            left_var = self._cluster_variance(cov, left)
            right_var = self._cluster_variance(cov, right)

            alpha = 1.0 - left_var / (left_var + right_var)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha

            if len(left) > 1:
                clusters.append(left)
            if len(right) > 1:
                clusters.append(right)

        return weights / weights.sum()

    def get_weights(self) -> np.ndarray:
        if self.weights is None:
            raise ValueError("model not fitted")
        return self.weights.values


class ConstrainedHRP(HierarchicalRiskParity):
    """Phase 2 extension point for constrained HRP."""

    def __init__(
        self,
        linkage_method: str = "single",
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ):
        super().__init__(linkage_method=linkage_method)
        self.min_weight = min_weight
        self.max_weight = max_weight
