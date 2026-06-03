"""Hierarchical clustering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


SUPPORTED_LINKAGE_METHODS = {"ward", "single", "complete", "average"}


def compute_linkage_matrix(
    distance_matrix_df: pd.DataFrame,
    method: str = "ward",
) -> np.ndarray:
    """Compute a hierarchical linkage matrix from a distance matrix."""
    if method not in SUPPORTED_LINKAGE_METHODS:
        raise ValueError(
            f"Unsupported linkage method '{method}'. "
            f"Supported methods: {', '.join(sorted(SUPPORTED_LINKAGE_METHODS))}."
        )

    if not isinstance(distance_matrix_df, pd.DataFrame):
        raise TypeError("distance_matrix_df must be a pandas DataFrame")

    if distance_matrix_df.shape[0] != distance_matrix_df.shape[1]:
        raise ValueError("distance_matrix_df must be square")

    if not distance_matrix_df.index.equals(distance_matrix_df.columns):
        raise ValueError("distance_matrix_df must have identical row and column labels")

    distance_matrix = distance_matrix_df.values.astype(float)
    if np.isnan(distance_matrix).any():
        raise ValueError("distance_matrix_df must not contain NaN values")

    np.fill_diagonal(distance_matrix, 0.0)
    condensed = squareform(distance_matrix, checks=True)
    return linkage(condensed, method=method)


def assign_clusters(linkage_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """Assign cluster labels from a linkage matrix."""
    if not isinstance(linkage_matrix, np.ndarray):
        raise TypeError("linkage_matrix must be a numpy array")
    if linkage_matrix.ndim != 2 or linkage_matrix.shape[1] != 4:
        raise ValueError("linkage_matrix must have shape (n-1, 4)")
    if n_clusters < 1 or n_clusters > linkage_matrix.shape[0] + 1:
        raise ValueError("n_clusters must be between 1 and the number of observations")

    return fcluster(linkage_matrix, n_clusters, criterion="maxclust")


def get_cluster_members(
    assets: list[str], cluster_labels: np.ndarray
) -> dict[int, list[str]]:
    """Map assets to cluster memberships."""
    if len(assets) != len(cluster_labels):
        raise ValueError("assets and cluster_labels must have the same length")
    if cluster_labels.ndim != 1:
        raise ValueError("cluster_labels must be a one-dimensional array")

    members: dict[int, list[str]] = {}
    for asset, label in zip(assets, cluster_labels.tolist()):
        members.setdefault(int(label), []).append(asset)

    return members
