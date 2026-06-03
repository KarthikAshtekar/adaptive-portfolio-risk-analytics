"""Tests for the covariance/correlation utilities.

These tests exercise the public API exposed by `src.covariance` and are
deterministic so they can run in CI and on developer machines.
"""

import numpy as np
import pandas as pd
import pytest

from src.covariance import (
    compute_covariance_matrix,
    compute_correlation_matrix,
    compute_distance_matrix,
    rank_correlations,
    compute_average_correlation,
    validate_covariance_matrix,
    validate_correlation_matrix,
)


@pytest.fixture
def simple_returns() -> pd.DataFrame:
    """Small deterministic returns DataFrame.

    Asset_A and Asset_B are identical (perfect correlation), Asset_C is the
    reverse series (negative correlation).
    """
    return pd.DataFrame(
        {
            "Asset_A": [0.01, 0.02, 0.03, 0.04, 0.05],
            "Asset_B": [0.01, 0.02, 0.03, 0.04, 0.05],
            "Asset_C": [0.05, 0.04, 0.03, 0.02, 0.01],
        }
    )


def test_api_exposes_expected_symbols():
    # smoke-test that the module API contains the expected callables
    assert callable(compute_covariance_matrix)
    assert callable(compute_correlation_matrix)
    assert callable(compute_distance_matrix)
    assert callable(rank_correlations)
    assert callable(compute_average_correlation)
    assert callable(validate_covariance_matrix)
    assert callable(validate_correlation_matrix)


def test_covariance_and_correlation_properties(simple_returns):
    cov = compute_covariance_matrix(simple_returns)
    corr = compute_correlation_matrix(simple_returns)

    # shapes
    assert cov.shape == (3, 3)
    assert corr.shape == (3, 3)

    # symmetry
    assert np.allclose(cov.values, cov.values.T)
    assert np.allclose(corr.values, corr.values.T)

    # diagonal checks
    assert np.all(np.diag(corr.values) == 1.0)
    assert np.all(np.diag(cov.values) >= 0.0)


def test_validate_matrix_helpers(simple_returns):
    cov = compute_covariance_matrix(simple_returns)
    corr = compute_correlation_matrix(simple_returns)

    assert validate_covariance_matrix(cov) is True
    assert validate_correlation_matrix(corr) is True


def test_distance_matrix_properties(simple_returns):
    corr = compute_correlation_matrix(simple_returns)
    dist = compute_distance_matrix(corr)

    # diagonal zeros
    assert np.allclose(np.diag(dist.values), 0.0)

    # bounds [0, 1]
    assert (dist.values >= 0.0).all()
    assert (dist.values <= 1.0).all()


def test_rank_correlations_and_average():
    correlation = pd.DataFrame(
        [
            [1.0, 0.9, 0.2],
            [0.9, 1.0, -0.3],
            [0.2, -0.3, 1.0],
        ],
        columns=["A", "B", "C"],
        index=["A", "B", "C"],
    )

    rankings = rank_correlations(correlation)

    # Expect the top pair to be (A, B) with correlation 0.9
    top = rankings.iloc[0]
    assert {top["Asset A"], top["Asset B"]} == {"A", "B"}
    assert pytest.approx(top["Correlation"], rel=1e-9) == 0.9

    # Expect the bottom correlation to be -0.3
    bottom = rankings.iloc[-1]
    assert pytest.approx(bottom["Correlation"], rel=1e-9) == -0.3

    # Average correlation (unique off-diagonals): (0.9 + 0.2 - 0.3) / 3
    avg = compute_average_correlation(correlation)
    expected = (0.9 + 0.2 - 0.3) / 3
    assert pytest.approx(avg, rel=1e-12) == expected
