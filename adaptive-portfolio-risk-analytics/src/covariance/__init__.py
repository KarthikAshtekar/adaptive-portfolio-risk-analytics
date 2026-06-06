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
from .sample_covariance import (
    BaseCovarianceEstimator,
    SampleCovarianceEstimator,
    RollingCovarianceEstimator,
)
from .ledoit_wolf import compute_ledoit_wolf_covariance
from .ewma_covariance import (
    compute_ewma_covariance,
    compute_ewma_ledoit_wolf_covariance,
)
from .covariance_factory import (
    CovarianceFactory,
    SUPPORTED_COVARIANCE_METHODS,
    extract_covariance_metadata,
    validate_estimated_covariance_matrix,
    assert_valid_covariance_matrix,
)

__all__ = [
    "compute_covariance_matrix",
    "compute_correlation_matrix",
    "validate_covariance_matrix",
    "validate_correlation_matrix",
    "rank_correlations",
    "compute_average_correlation",
    "compute_distance_matrix",
    "BaseCovarianceEstimator",
    "SampleCovarianceEstimator",
    "RollingCovarianceEstimator",
    "compute_ledoit_wolf_covariance",
    "compute_ewma_covariance",
    "compute_ewma_ledoit_wolf_covariance",
    "CovarianceFactory",
    "SUPPORTED_COVARIANCE_METHODS",
    "extract_covariance_metadata",
    "validate_estimated_covariance_matrix",
    "assert_valid_covariance_matrix",
]
