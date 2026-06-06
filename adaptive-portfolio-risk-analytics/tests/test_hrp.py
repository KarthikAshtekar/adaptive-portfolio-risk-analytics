"""Unit tests for Phase 1 HRP and clustering."""

import numpy as np
import pandas as pd
import pytest

from src.clustering import DistanceMetrics, compute_linkage_matrix
from src.clustering.hrp import HierarchicalRiskParity
from src.optimization import HRPAllocator
from src.optimization.hrp_allocator import (
    allocate_hrp_weights,
    compute_cluster_variance,
    get_quasi_diagonal_order,
    recursive_bisection,
)


def _sample_returns() -> pd.DataFrame:
    np.random.seed(7)
    dates = pd.date_range(start="2020-01-01", periods=260, freq="B")
    data = np.random.randn(260, 5) * 0.015
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D", "E"])


def test_correlation_distance_matrix_properties() -> None:
    returns = _sample_returns()
    corr = returns.corr().values
    dist = DistanceMetrics.correlation_distance(corr)

    assert dist.shape == corr.shape
    assert np.allclose(np.diag(dist), 0.0)
    assert np.allclose(dist, dist.T, atol=1e-12)


def test_compute_linkage_matrix_produces_valid_tree() -> None:
    returns = _sample_returns()
    distance = DistanceMetrics.correlation_distance(returns.corr().values)
    distance_df = pd.DataFrame(distance, index=returns.columns, columns=returns.columns)
    linkage_matrix = compute_linkage_matrix(distance_df, method="single")

    assert linkage_matrix is not None
    assert linkage_matrix.shape[0] == returns.shape[1] - 1


def test_hrp_weights_sum_to_one() -> None:
    returns = _sample_returns()
    hrp = HierarchicalRiskParity(linkage_method="single").fit(returns)
    weights = hrp.get_weights()

    assert weights.shape == (returns.shape[1],)
    assert np.isclose(weights.sum(), 1.0)


def test_hrp_weights_non_negative() -> None:
    returns = _sample_returns()
    weights = HierarchicalRiskParity().fit(returns).get_weights()

    assert np.all(weights >= 0.0)


def test_hrp_different_from_equal_weight_when_covariance_varies() -> None:
    np.random.seed(21)
    dates = pd.date_range(start="2020-01-01", periods=260, freq="B")
    low_vol = np.random.randn(260) * 0.005
    high_vol = np.random.randn(260) * 0.03
    mid_vol = np.random.randn(260) * 0.015
    returns = pd.DataFrame({"L": low_vol, "M": mid_vol, "H": high_vol}, index=dates)

    weights = HierarchicalRiskParity().fit(returns).get_weights()
    equal = np.array([1 / 3, 1 / 3, 1 / 3])

    assert not np.allclose(weights, equal)


def test_get_quasi_diagonal_order_returns_all_assets() -> None:
    returns = _sample_returns()
    covariance_df = returns.cov()
    linkage_matrix = DistanceMetrics.to_condensed(DistanceMetrics.correlation_distance(returns.corr().values))
    linkage_matrix = np.asarray(linkage_matrix)
    # Use hierarchical linkage from scipy if needed for correct shape
    from scipy.cluster.hierarchy import linkage as scipy_linkage

    linkage_matrix = scipy_linkage(linkage_matrix, method="single")
    asset_order = get_quasi_diagonal_order(linkage_matrix)

    assert set(asset_order) == set(range(returns.shape[1]))
    assert len(asset_order) == returns.shape[1]


def test_compute_cluster_variance_positive() -> None:
    returns = _sample_returns()
    covariance_df = returns.cov()
    cluster_assets = ["A", "B", "C"]
    variance = compute_cluster_variance(covariance_df, cluster_assets)

    assert variance > 0.0


def test_recursive_bisection_completes_successfully() -> None:
    returns = _sample_returns()
    covariance_df = returns.cov()
    linkage_matrix = DistanceMetrics.to_condensed(DistanceMetrics.correlation_distance(returns.corr().values))
    from scipy.cluster.hierarchy import linkage as scipy_linkage

    linkage_matrix = scipy_linkage(linkage_matrix, method="single")
    order = [returns.columns[i] for i in get_quasi_diagonal_order(linkage_matrix)]
    weights = recursive_bisection(covariance_df, order)

    assert weights.shape[0] == returns.shape[1]
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)


def test_allocate_hrp_weights_reproducible() -> None:
    returns = _sample_returns()
    covariance_df = returns.cov()
    condensed = DistanceMetrics.to_condensed(DistanceMetrics.correlation_distance(returns.corr().values))
    from scipy.cluster.hierarchy import linkage as scipy_linkage

    linkage_matrix = scipy_linkage(condensed, method="single")
    first = allocate_hrp_weights(covariance_df, linkage_matrix)
    second = allocate_hrp_weights(covariance_df, linkage_matrix)

    assert np.allclose(first.values, second.values)
    assert set(first.index) == set(returns.columns)


@pytest.mark.parametrize(
    ("covariance_method", "covariance_kwargs"),
    [
        ("sample", {}),
        ("ledoit_wolf", {}),
        ("ewma", {"span": 126}),
        ("ewma_ledoit_wolf", {"span": 126}),
    ],
)
def test_hrp_allocator_supports_covariance_methods(
    covariance_method: str,
    covariance_kwargs: dict,
) -> None:
    returns = _sample_returns()
    weights = HRPAllocator(
        covariance_method=covariance_method,
        covariance_kwargs=covariance_kwargs,
    ).optimize(returns)

    assert isinstance(weights, pd.Series)
    assert weights.index.tolist() == returns.columns.tolist()
    assert weights.name == "weight"
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)
    assert np.isfinite(weights.values).all()
