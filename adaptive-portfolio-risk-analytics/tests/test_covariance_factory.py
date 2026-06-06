"""Tests for the covariance factory research API."""

import numpy as np
import pandas as pd
import pytest

from src.covariance import (
    CovarianceFactory,
    compute_covariance_matrix,
    extract_covariance_metadata,
    validate_estimated_covariance_matrix,
    assert_valid_covariance_matrix,
)


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("sample", {}),
        ("ledoit_wolf", {}),
        ("ewma", {"span": 126}),
        ("ewma_ledoit_wolf", {"span": 126}),
    ],
)
def test_covariance_factory_routes_supported_methods(sample_returns, method, kwargs):
    covariance_matrix = CovarianceFactory.compute(sample_returns, method=method, **kwargs)

    assert isinstance(covariance_matrix, pd.DataFrame)
    assert covariance_matrix.shape == (sample_returns.shape[1], sample_returns.shape[1])
    assert list(covariance_matrix.index) == list(sample_returns.columns)
    assert list(covariance_matrix.columns) == list(sample_returns.columns)
    assert validate_estimated_covariance_matrix(covariance_matrix) is True


def test_covariance_factory_sample_matches_existing_api(sample_returns):
    expected = compute_covariance_matrix(sample_returns)
    actual = CovarianceFactory.compute(sample_returns, method="sample")

    pd.testing.assert_frame_equal(actual, expected)
    assert extract_covariance_metadata(actual) == {"method": "sample"}


def test_covariance_factory_rejects_unknown_method(sample_returns):
    with pytest.raises(ValueError, match="unsupported covariance method"):
        CovarianceFactory.compute(sample_returns, method="unknown_method")


def test_covariance_validation_helpers_reject_invalid_matrix():
    invalid_covariance = pd.DataFrame(
        [[0.01, np.nan], [0.00, 0.02]],
        index=["Asset_A", "Asset_B"],
        columns=["Asset_A", "Asset_B"],
    )

    assert validate_estimated_covariance_matrix(invalid_covariance) is False

    with pytest.raises(ValueError, match="invalid covariance matrix"):
        assert_valid_covariance_matrix(invalid_covariance, method="broken")
