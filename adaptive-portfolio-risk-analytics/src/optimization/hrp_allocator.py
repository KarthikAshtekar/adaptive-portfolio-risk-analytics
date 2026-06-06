"""HRP allocator adapter and stage 5 utility functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

from src.clustering.distance_metrics import DistanceMetrics
from src.covariance import CovarianceFactory

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


def covariance_to_correlation(covariance_matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix into a labeled correlation matrix."""
    if not isinstance(covariance_matrix, pd.DataFrame):
        raise TypeError("covariance_matrix must be a pandas DataFrame")
    if covariance_matrix.empty:
        raise ValueError("covariance_matrix must not be empty")
    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("covariance_matrix must be square")
    if not covariance_matrix.index.equals(covariance_matrix.columns):
        raise ValueError("covariance_matrix must have identical row and column labels")

    vol = np.sqrt(np.clip(np.diag(covariance_matrix.values), 1e-12, None))
    scale = np.outer(vol, vol)
    correlation = covariance_matrix.values / scale
    correlation = np.clip(correlation, -1.0, 1.0)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)

    return pd.DataFrame(
        correlation,
        index=covariance_matrix.index,
        columns=covariance_matrix.columns,
    )


class HRPAllocator(BaseAllocator):
    """Allocate using Hierarchical Risk Parity."""

    def __init__(
        self,
        linkage_method: str = "single",
        covariance_method: str = "sample",
        covariance_kwargs: dict | None = None,
    ):
        self.linkage_method = linkage_method
        self.covariance_method = covariance_method
        self.covariance_kwargs = dict(covariance_kwargs or {})
        self.covariance_matrix_: pd.DataFrame | None = None
        self.correlation_matrix_: pd.DataFrame | None = None
        self._weights: pd.Series | None = None

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
            covariance_df = CovarianceFactory.compute(
                clean_returns,
                method=self.covariance_method,
                **self.covariance_kwargs,
            )
        elif isinstance(cov_matrix, np.ndarray):
            covariance_df = pd.DataFrame(
                cov_matrix,
                index=clean_returns.columns,
                columns=clean_returns.columns,
            )
        else:
            covariance_df = cov_matrix.loc[clean_returns.columns, clean_returns.columns]

        if linkage_matrix is None:
            correlation_df = covariance_to_correlation(covariance_df)
            distance = DistanceMetrics.correlation_distance(correlation_df.values)
            condensed = DistanceMetrics.to_condensed(distance)
            linkage_matrix = linkage(condensed, method=self.linkage_method)
        else:
            correlation_df = covariance_to_correlation(covariance_df)

        weights = allocate_hrp_weights(covariance_df, linkage_matrix).reindex(clean_returns.columns)
        weights.name = "weight"
        self.covariance_matrix_ = covariance_df
        self.correlation_matrix_ = correlation_df
        self._weights = weights
        return self

    def get_weights(self) -> pd.Series:
        if self._weights is None:
            raise ValueError("allocator not fitted")
        return self._weights.copy()
