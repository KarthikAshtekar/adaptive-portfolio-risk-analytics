"""HRP allocator adapter and stage 5 utility functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

from src.clustering.distance_metrics import DistanceMetrics

from .base import BaseAllocator


def get_quasi_diagonal_order(linkage_matrix: np.ndarray) -> list[int]:
    """Return leaf ordering for a quasi-diagonalized linkage tree."""
    if linkage_matrix.ndim != 2 or linkage_matrix.shape[1] != 4:
        raise ValueError("linkage_matrix must be a valid scipy linkage matrix")

    n = linkage_matrix.shape[0] + 1

    def traverse(node: int) -> list[int]:
        if node < n:
            return [node]
        left = int(linkage_matrix[node - n, 0])
        right = int(linkage_matrix[node - n, 1])
        return traverse(left) + traverse(right)

    root = 2 * n - 2
    return traverse(root)


def compute_cluster_variance(
    covariance_matrix: pd.DataFrame,
    cluster_assets: list[str],
) -> float:
    """Compute the variance of a cluster using inverse-variance weights."""
    if covariance_matrix.empty:
        raise ValueError("covariance_matrix must not be empty")
    if len(cluster_assets) == 0:
        raise ValueError("cluster_assets must not be empty")

    sub_cov = covariance_matrix.loc[cluster_assets, cluster_assets]
    diag = np.diag(sub_cov.values)
    inv_diag = 1.0 / np.clip(diag, 1e-12, None)
    ivp_weights = inv_diag / inv_diag.sum()
    variance = float(ivp_weights.T @ sub_cov.values @ ivp_weights)

    if variance <= 0.0:
        raise ValueError("cluster variance must be positive")

    return variance


def recursive_bisection(
    covariance_matrix: pd.DataFrame,
    ordered_assets: list[str],
) -> pd.Series:
    """Recursively allocate portfolio weights through cluster bisection."""
    if covariance_matrix.empty:
        raise ValueError("covariance_matrix must not be empty")
    if len(ordered_assets) == 0:
        raise ValueError("ordered_assets must not be empty")

    weights = pd.Series(1.0, index=ordered_assets, dtype=float)
    clusters = [ordered_assets.copy()]

    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue

        split = len(cluster) // 2
        left = cluster[:split]
        right = cluster[split:]

        left_variance = compute_cluster_variance(covariance_matrix, left)
        right_variance = compute_cluster_variance(covariance_matrix, right)
        allocation = right_variance / (left_variance + right_variance)

        weights[left] *= allocation
        weights[right] *= 1.0 - allocation

        if len(left) > 1:
            clusters.append(left)
        if len(right) > 1:
            clusters.append(right)

    weights = weights.clip(lower=0.0)
    return weights / weights.sum()


def allocate_hrp_weights(
    covariance_matrix_df: pd.DataFrame,
    linkage_matrix: np.ndarray,
) -> pd.Series:
    """Allocate portfolio weights from a covariance matrix and linkage tree."""
    if not isinstance(covariance_matrix_df, pd.DataFrame):
        raise TypeError("covariance_matrix_df must be a pandas DataFrame")
    if covariance_matrix_df.shape[0] != covariance_matrix_df.shape[1]:
        raise ValueError("covariance_matrix_df must be square")
    if not covariance_matrix_df.index.equals(covariance_matrix_df.columns):
        raise ValueError("covariance_matrix_df must have identical labels")

    ordered_indices = get_quasi_diagonal_order(linkage_matrix)
    ordered_assets = [covariance_matrix_df.index[i] for i in ordered_indices]
    hrp_weights = recursive_bisection(covariance_matrix_df, ordered_assets)

    return hrp_weights.reindex(covariance_matrix_df.index).fillna(0.0).astype(float)


class HRPAllocator(BaseAllocator):
    """Allocate using Hierarchical Risk Parity."""

    def __init__(self, linkage_method: str = "single"):
        self.linkage_method = linkage_method
        self._weights: np.ndarray | None = None

    def fit(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame | np.ndarray | None = None,
        linkage_matrix: np.ndarray | None = None,
    ) -> "HRPAllocator":
        if returns.empty:
            raise ValueError("returns must not be empty")

        clean_returns = returns.dropna(how="any")
        if clean_returns.empty:
            raise ValueError("returns has no valid rows after dropping NaNs")

        if cov_matrix is None:
            covariance_df = clean_returns.cov()
        elif isinstance(cov_matrix, np.ndarray):
            covariance_df = pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
        else:
            covariance_df = cov_matrix

        if linkage_matrix is None:
            corr = clean_returns.corr()
            distance = DistanceMetrics.correlation_distance(corr.values)
            condensed = DistanceMetrics.to_condensed(distance)
            linkage_matrix = linkage(condensed, method=self.linkage_method)

        weights = allocate_hrp_weights(covariance_df, linkage_matrix).values
        self._weights = weights
        return self

    def get_weights(self) -> np.ndarray:
        if self._weights is None:
            raise ValueError("allocator not fitted")
        return self._weights
