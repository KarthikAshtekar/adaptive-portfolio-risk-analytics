"""
Hierarchical Risk Parity (HRP) portfolio construction.

References
----------
- López de Prado, M. (2016). "Building Diversified Portfolios that Outperform"
- Raffinot, T. (2018). "Hierarchical Clustering Based Asset Allocation"
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity portfolio optimizer.

    Constructs portfolios by:
    1. Clustering assets based on correlation
    2. Recursively allocating risk through the hierarchy
    3. Optimizing weights within each cluster
    """

    def __init__(self, linkage_method: str = "ward"):
        """
        Initialize HRP optimizer.

        Parameters
        ----------
        linkage_method : str
            Hierarchical linkage method

        TODO: Add divergence-based distance metrics
        """
        self.linkage_method = linkage_method
        self.linkage_matrix = None
        self.weights = None

    def fit(self, returns: pd.DataFrame) -> "HierarchicalRiskParity":
        """
        Fit HRP model.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns

        Returns
        -------
        HierarchicalRiskParity
            Fitted model

        TODO: Implement full HRP algorithm
        TODO: Add variance-minimization refinement
        """
        # Calculate correlation matrix
        corr = returns.corr()

        # Calculate distance matrix
        dist = 1 - corr
        dist_condensed = squareform(dist)

        # Perform hierarchical clustering
        self.linkage_matrix = linkage(dist_condensed, method=self.linkage_method)

        # Get optimal ordering
        order = leaves_list(self.linkage_matrix)

        # Calculate weights using HRP algorithm
        self.weights = self._calculate_hrp_weights(
            cov=returns.cov(),
            order=order,
        )

        return self

    def _calculate_hrp_weights(
        self, cov: pd.DataFrame, order: np.ndarray
    ) -> np.ndarray:
        """
        Calculate HRP weights recursively.

        Parameters
        ----------
        cov : pd.DataFrame
            Covariance matrix
        order : np.ndarray
            Asset ordering from dendrogram

        Returns
        -------
        np.ndarray
            Portfolio weights

        TODO: Implement recursive weight calculation
        TODO: Implement inverse volatility refinement
        """
        # TODO: Implement full recursive algorithm
        n_assets = len(order)
        return np.ones(n_assets) / n_assets

    def get_weights(self) -> np.ndarray:
        """
        Get HRP portfolio weights.

        Returns
        -------
        np.ndarray
            Portfolio weights
        """
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.weights


class ConstrainedHRP(HierarchicalRiskParity):
    """
    HRP with position constraints.

    TODO: Implement min/max weight constraints
    TODO: Implement sector constraints
    """

    def __init__(
        self,
        linkage_method: str = "ward",
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ):
        """
        Initialize constrained HRP.

        Parameters
        ----------
        linkage_method : str
            Hierarchical linkage method
        min_weight : float
            Minimum asset weight
        max_weight : float
            Maximum asset weight

        TODO: Add sector-level constraints
        """
        super().__init__(linkage_method)
        self.min_weight = min_weight
        self.max_weight = max_weight
