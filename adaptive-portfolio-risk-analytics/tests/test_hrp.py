"""Unit tests for Phase 1 HRP and clustering."""

import numpy as np
import pandas as pd

from src.clustering import DistanceMetrics, HierarchicalClusterer
from src.clustering.hrp import HierarchicalRiskParity


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


def test_hierarchical_clusterer_produces_linkage() -> None:
    returns = _sample_returns()
    clusterer = HierarchicalClusterer(linkage_method="single").fit(returns)

    assert clusterer.linkage_matrix is not None
    assert clusterer.linkage_matrix.shape[0] == returns.shape[1] - 1


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
