"""Unit tests for Phase 1 covariance estimators."""

import numpy as np
import pandas as pd
import pytest

from src.covariance import RollingCovarianceEstimator, SampleCovarianceEstimator


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=300, freq="B")
    data = np.random.randn(300, 6) * 0.01
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(6)])


def test_sample_covariance_is_symmetric(sample_returns: pd.DataFrame) -> None:
    estimator = SampleCovarianceEstimator().fit(sample_returns)
    cov = estimator.get_covariance()

    assert cov.shape == (sample_returns.shape[1], sample_returns.shape[1])
    assert np.allclose(cov, cov.T, atol=1e-12)


def test_sample_covariance_positive_semidefinite(sample_returns: pd.DataFrame) -> None:
    cov = SampleCovarianceEstimator().estimate(sample_returns)
    eigenvalues = np.linalg.eigvalsh(cov)

    assert np.all(eigenvalues >= -1e-10)


def test_rolling_covariance_uses_window(sample_returns: pd.DataFrame) -> None:
    window = 63
    estimator = RollingCovarianceEstimator(window=window, method="standard")
    cov = estimator.estimate(sample_returns)

    expected = sample_returns.iloc[-window:].cov().values
    assert np.allclose(cov, expected)


def test_rolling_covariance_series_length(sample_returns: pd.DataFrame) -> None:
    window = 50
    estimator = RollingCovarianceEstimator(window=window)
    series = estimator.estimate_series(sample_returns)

    assert len(series) == len(sample_returns) - window + 1


def test_rolling_covariance_invalid_window(sample_returns: pd.DataFrame) -> None:
    estimator = RollingCovarianceEstimator(window=1000)
    with pytest.raises(ValueError):
        estimator.fit(sample_returns)
