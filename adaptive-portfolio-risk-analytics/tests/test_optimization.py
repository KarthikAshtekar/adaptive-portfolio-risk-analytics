"""Unit tests for Phase 1 portfolio allocators."""

import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    EqualWeightAllocator,
    HRPAllocator,
    InverseVolatilityAllocator,
    MeanVarianceAllocator,
)


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    np.random.seed(123)
    dates = pd.date_range(start="2020-01-01", periods=320, freq="B")
    data = np.random.randn(320, 5) * np.array([0.01, 0.015, 0.02, 0.012, 0.018])
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D", "E"])


def test_equal_weight_allocator(sample_returns: pd.DataFrame) -> None:
    weights = EqualWeightAllocator().optimize(sample_returns)
    expected = np.ones(sample_returns.shape[1]) / sample_returns.shape[1]

    assert np.allclose(weights, expected)


def test_inverse_volatility_allocator_properties(sample_returns: pd.DataFrame) -> None:
    weights = InverseVolatilityAllocator().optimize(sample_returns)

    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)

    vol = sample_returns.std().values
    inv_vol = 1.0 / vol
    expected = inv_vol / inv_vol.sum()
    assert np.allclose(weights, expected)


def test_mean_variance_allocator_valid_weights(sample_returns: pd.DataFrame) -> None:
    weights = MeanVarianceAllocator().optimize(sample_returns)

    assert weights.shape == (sample_returns.shape[1],)
    assert np.isclose(weights.sum(), 1.0, atol=1e-8)
    assert np.all(weights >= -1e-8)


def test_mean_variance_handles_covariance_input(sample_returns: pd.DataFrame) -> None:
    cov = sample_returns.cov().values
    weights = MeanVarianceAllocator().optimize(sample_returns, cov_matrix=cov)

    assert np.isclose(weights.sum(), 1.0, atol=1e-8)


def test_hrp_allocator_valid_weights(sample_returns: pd.DataFrame) -> None:
    weights = HRPAllocator().optimize(sample_returns)

    assert weights.shape == (sample_returns.shape[1],)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)


def test_allocators_raise_on_empty_input() -> None:
    empty = pd.DataFrame()
    with pytest.raises(ValueError):
        EqualWeightAllocator().optimize(empty)
    with pytest.raises(ValueError):
        InverseVolatilityAllocator().optimize(empty)
    with pytest.raises(ValueError):
        MeanVarianceAllocator().optimize(empty)
