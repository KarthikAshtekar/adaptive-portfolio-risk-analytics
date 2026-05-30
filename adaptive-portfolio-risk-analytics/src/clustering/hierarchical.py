"""Hierarchical clustering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from .distance_metrics import DistanceMetrics


class HierarchicalClusterer:
    """Run hierarchical clustering on asset return correlations."""

    def __init__(self, linkage_method: str = "single"):
        self.linkage_method = linkage_method
        self.linkage_matrix: np.ndarray | None = None

    def fit(self, returns: pd.DataFrame) -> "HierarchicalClusterer":
        corr = DistanceMetrics.correlation_matrix(returns)
        dist = DistanceMetrics.correlation_distance(corr.values)
        condensed = DistanceMetrics.to_condensed(dist)
        self.linkage_matrix = linkage(condensed, method=self.linkage_method)
        return self

    def get_clusters(self, n_clusters: int) -> np.ndarray:
        if self.linkage_matrix is None:
            raise ValueError("clusterer not fitted")
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        return fcluster(self.linkage_matrix, n_clusters, criterion="maxclust")
