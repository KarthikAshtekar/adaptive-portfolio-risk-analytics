"""Hierarchical Equal Risk Contribution allocator and comparison helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.covariance import CovarianceFactory
from src.covariance.distance import compute_distance_matrix
from src.optimization.base import BaseAllocator

from .hierarchical import compute_linkage_matrix


def _prepare_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean returns before optimization."""
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame")
    if returns_df.empty:
        raise ValueError("returns_df must not be empty")

    clean_returns = returns_df.dropna(how="any")
    if clean_returns.empty:
        raise ValueError("returns_df has no valid rows after dropping NaNs")
    if clean_returns.shape[1] < 2:
        raise ValueError("returns_df must contain at least two assets")

    return clean_returns


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


def _build_cluster_tree(
    linkage_matrix: np.ndarray,
    asset_names: list[str],
) -> tuple[dict[int, tuple[int, int]], dict[int, list[str]], int]:
    """Build a node -> children / node -> assets representation of the linkage tree."""
    if linkage_matrix.ndim != 2 or linkage_matrix.shape[1] != 4:
        raise ValueError("linkage_matrix must be a valid scipy linkage matrix")

    n_assets = len(asset_names)
    children: dict[int, tuple[int, int]] = {}
    members: dict[int, list[str]] = {idx: [asset] for idx, asset in enumerate(asset_names)}

    for row_idx, row in enumerate(linkage_matrix):
        node_id = n_assets + row_idx
        left = int(row[0])
        right = int(row[1])
        children[node_id] = (left, right)
        members[node_id] = members[left] + members[right]

    root = n_assets + linkage_matrix.shape[0] - 1
    return children, members, root


def _inverse_volatility_weights(sub_covariance: pd.DataFrame) -> pd.Series:
    """Proxy equal-risk weights inside a cluster using inverse volatility."""
    volatility = np.sqrt(np.clip(np.diag(sub_covariance.values), 1e-12, None))
    inv_volatility = 1.0 / volatility
    weights = inv_volatility / inv_volatility.sum()
    return pd.Series(weights, index=sub_covariance.index, dtype=float)


def compute_cluster_risk(
    covariance_matrix: pd.DataFrame,
    cluster_assets: list[str],
) -> float:
    """Compute cluster volatility from a covariance submatrix.

    HERC uses a local equal-risk proxy inside each cluster before capital is
    passed further down the hierarchy. Here that proxy is inverse-volatility
    weighting inside the cluster, followed by cluster volatility calculation.
    """
    if not isinstance(covariance_matrix, pd.DataFrame):
        raise TypeError("covariance_matrix must be a pandas DataFrame")
    if covariance_matrix.empty:
        raise ValueError("covariance_matrix must not be empty")
    if len(cluster_assets) == 0:
        raise ValueError("cluster_assets must not be empty")

    sub_covariance = covariance_matrix.loc[cluster_assets, cluster_assets]
    cluster_weights = _inverse_volatility_weights(sub_covariance)
    cluster_variance = float(
        cluster_weights.values.T @ sub_covariance.values @ cluster_weights.values
    )

    return float(np.sqrt(max(cluster_variance, 1e-12)))


def validate_weights(weights: pd.Series, atol: float = 1e-8) -> pd.Series:
    """Validate and normalize a weight vector."""
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series")
    if weights.empty:
        raise ValueError("weights must not be empty")
    if weights.isna().any():
        raise ValueError("weights must not contain NaN values")
    if not np.isfinite(weights.values).all():
        raise ValueError("weights must be finite")
    if (weights < -atol).any():
        raise ValueError("weights must be non-negative")

    clean_weights = weights.clip(lower=0.0)
    total_weight = float(clean_weights.sum())
    if total_weight <= 0.0:
        raise ValueError("weights must sum to a positive value")

    normalized_weights = clean_weights / total_weight
    if not np.isclose(float(normalized_weights.sum()), 1.0, atol=atol):
        raise ValueError("weights must sum to 1 after normalization")

    return normalized_weights.astype(float)


def allocate_herc_weights(
    covariance_matrix_df: pd.DataFrame,
    linkage_matrix: np.ndarray,
) -> pd.Series:
    """Allocate portfolio weights by equalizing risk down the linkage tree.

    This differs from the existing HRP adapter in two ways:
    1. The recursion follows the explicit linkage tree splits instead of
       quasi-diagonal midpoint bisection.
    2. Cluster risk is measured as cluster volatility using local
       inverse-volatility weights, which is closer to a HERC-style risk budget
       than HRP's ordered recursive variance split.
    """
    if not isinstance(covariance_matrix_df, pd.DataFrame):
        raise TypeError("covariance_matrix_df must be a pandas DataFrame")
    if covariance_matrix_df.shape[0] != covariance_matrix_df.shape[1]:
        raise ValueError("covariance_matrix_df must be square")
    if not covariance_matrix_df.index.equals(covariance_matrix_df.columns):
        raise ValueError("covariance_matrix_df must have identical row and column labels")

    asset_names = list(covariance_matrix_df.index)
    children, members, root = _build_cluster_tree(linkage_matrix, asset_names)
    weights = pd.Series(0.0, index=asset_names, dtype=float)
    n_assets = len(asset_names)

    def recurse(node_id: int, cluster_weight: float) -> None:
        if node_id < n_assets:
            weights.loc[asset_names[node_id]] = cluster_weight
            return

        left_id, right_id = children[node_id]
        left_assets = members[left_id]
        right_assets = members[right_id]

        left_risk = compute_cluster_risk(covariance_matrix_df, left_assets)
        right_risk = compute_cluster_risk(covariance_matrix_df, right_assets)
        total_risk = left_risk + right_risk
        if total_risk <= 0.0:
            raise ValueError("cluster risks must be positive")

        # Equal branch risk means capital is inversely proportional to branch risk.
        left_weight = cluster_weight * (right_risk / total_risk)
        right_weight = cluster_weight * (left_risk / total_risk)

        recurse(left_id, left_weight)
        recurse(right_id, right_weight)

    recurse(root, 1.0)
    return validate_weights(weights)


def compare_hrp_herc_weights(
    returns_df: pd.DataFrame,
    covariance_method: str = "sample",
    linkage_method: str = "single",
    **covariance_kwargs: Any,
) -> pd.DataFrame:
    """Compare HRP and HERC weights using the same covariance estimate."""
    from src.optimization.hrp_allocator import allocate_hrp_weights

    clean_returns = _prepare_returns(returns_df)
    covariance_matrix = CovarianceFactory.compute(
        clean_returns,
        method=covariance_method,
        **covariance_kwargs,
    )
    correlation_matrix = covariance_to_correlation(covariance_matrix)
    distance_matrix = compute_distance_matrix(correlation_matrix)
    linkage_matrix = compute_linkage_matrix(distance_matrix, method=linkage_method)

    hrp_weights = allocate_hrp_weights(covariance_matrix, linkage_matrix)
    herc_weights = allocate_herc_weights(covariance_matrix, linkage_matrix)

    comparison_df = pd.DataFrame(
        {
            "Asset": clean_returns.columns,
            "HRP Weight": hrp_weights.reindex(clean_returns.columns).values,
            "HERC Weight": herc_weights.reindex(clean_returns.columns).values,
        }
    )
    comparison_df["Difference"] = (
        comparison_df["HERC Weight"] - comparison_df["HRP Weight"]
    )
    return comparison_df


class HERCAllocator(BaseAllocator):
    """Allocate using Hierarchical Equal Risk Contribution."""

    def __init__(
        self,
        covariance_method: str = "sample",
        linkage_method: str = "single",
        covariance_kwargs: dict[str, Any] | None = None,
    ):
        self.covariance_method = covariance_method
        self.linkage_method = linkage_method
        self.covariance_kwargs = dict(covariance_kwargs or {})
        self.covariance_matrix_: pd.DataFrame | None = None
        self.correlation_matrix_: pd.DataFrame | None = None
        self.linkage_matrix_: np.ndarray | None = None
        self.weights_: pd.Series | None = None

    @staticmethod
    def validate_weights(weights: pd.Series, atol: float = 1e-8) -> pd.Series:
        """Public validation helper for tests and notebook analysis."""
        return validate_weights(weights, atol=atol)

    @staticmethod
    def compute_cluster_risk(
        covariance_matrix: pd.DataFrame,
        cluster_assets: list[str],
    ) -> float:
        """Public cluster-risk helper for tests and notebook analysis."""
        return compute_cluster_risk(covariance_matrix, cluster_assets)

    def fit(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame | np.ndarray | None = None,
        linkage_matrix: np.ndarray | None = None,
    ) -> "HERCAllocator":
        clean_returns = _prepare_returns(returns)

        if cov_matrix is None:
            covariance_matrix = CovarianceFactory.compute(
                clean_returns,
                method=self.covariance_method,
                **self.covariance_kwargs,
            )
        elif isinstance(cov_matrix, np.ndarray):
            covariance_matrix = pd.DataFrame(
                cov_matrix,
                index=clean_returns.columns,
                columns=clean_returns.columns,
            )
        else:
            covariance_matrix = cov_matrix.loc[clean_returns.columns, clean_returns.columns]

        if linkage_matrix is None:
            correlation_matrix = covariance_to_correlation(covariance_matrix)
            distance_matrix = compute_distance_matrix(correlation_matrix)
            linkage_matrix = compute_linkage_matrix(distance_matrix, method=self.linkage_method)
        else:
            correlation_matrix = covariance_to_correlation(covariance_matrix)

        weights = allocate_herc_weights(covariance_matrix, linkage_matrix)

        self.covariance_matrix_ = covariance_matrix
        self.correlation_matrix_ = correlation_matrix
        self.linkage_matrix_ = linkage_matrix
        self.weights_ = weights.reindex(clean_returns.columns).astype(float)
        self.weights_.name = "weight"
        return self

    def get_weights(self) -> pd.Series:
        if self.weights_ is None:
            raise ValueError("allocator not fitted")
        return self.weights_.copy()
