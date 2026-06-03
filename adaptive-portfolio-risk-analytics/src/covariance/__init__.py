"""
Stage 3 (Covariance / Correlation / Distance) public API.

Notebooks and dashboards should import these business-logic functions from:
`src.covariance`
"""

from .covariance import (
    compute_covariance_matrix,
    compute_correlation_matrix,
    validate_covariance_matrix,
    validate_correlation_matrix,
    rank_correlations,
    compute_average_correlation,
)

from .distance import compute_distance_matrix

__all__ = [
    "compute_covariance_matrix",
    "compute_correlation_matrix",
    "validate_covariance_matrix",
    "validate_correlation_matrix",
    "rank_correlations",
    "compute_average_correlation",
    "compute_distance_matrix",
]