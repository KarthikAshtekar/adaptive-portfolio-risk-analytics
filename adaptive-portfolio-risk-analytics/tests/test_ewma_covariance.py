"""Tests for EWMA covariance estimation."""

import numpy as np
import pandas as pd
import pytest

from src.covariance import (
    compute_ewma_covariance,
    compute_ewma_ledoit_wolf_covariance,
    extract_covariance_metadata,
    validate_estimated_covariance_matrix,
)


def test_ewma_covariance_returns_valid_matrix(sample_returns):
    covariance_matrix = compute_ewma_covariance(sample_returns, span=126)

    assert isinstance(covariance_matrix, pd.DataFrame)
    assert covariance_matrix.shape == (sample_returns.shape[1], sample_returns.shape[1])
    assert list(covariance_matrix.index) == list(sample_returns.columns)
    assert list(covariance_matrix.columns) == list(sample_returns.columns)
    assert validate_estimated_covariance_matrix(covariance_matrix) is True
    assert not covariance_matrix.isna().any().any()


def test_ewma_covariance_span_changes_result(sample_returns):
    short_span_covariance = compute_ewma_covariance(sample_returns, span=63)
    long_span_covariance = compute_ewma_covariance(sample_returns, span=252)

    assert not np.allclose(short_span_covariance.values, long_span_covariance.values)


def test_ewma_ledoit_wolf_covariance_returns_valid_matrix(sample_returns):
    covariance_matrix = compute_ewma_ledoit_wolf_covariance(sample_returns, span=126)
    metadata = extract_covariance_metadata(covariance_matrix)

    assert validate_estimated_covariance_matrix(covariance_matrix) is True
    assert metadata["method"] == "ewma_ledoit_wolf"
    assert metadata["span"] == 126
    assert 0.0 <= metadata["shrinkage"] <= 1.0


def test_ewma_metadata_includes_span(sample_returns):
    covariance_matrix = compute_ewma_covariance(sample_returns, span=84)
    metadata = extract_covariance_metadata(covariance_matrix)

    assert metadata == {"method": "ewma", "span": 84}


def test_ewma_rejects_invalid_span(sample_returns):
    with pytest.raises(ValueError, match="span must be at least 2"):
        compute_ewma_covariance(sample_returns, span=1)
