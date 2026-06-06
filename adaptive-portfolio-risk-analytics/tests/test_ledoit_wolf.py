"""Tests for Ledoit-Wolf covariance estimation."""

import numpy as np
import pandas as pd

from src.covariance import (
    compute_ledoit_wolf_covariance,
    extract_covariance_metadata,
    validate_estimated_covariance_matrix,
)


def test_ledoit_wolf_covariance_preserves_labels(sample_returns):
    covariance_matrix = compute_ledoit_wolf_covariance(sample_returns)

    assert isinstance(covariance_matrix, pd.DataFrame)
    assert list(covariance_matrix.index) == list(sample_returns.columns)
    assert list(covariance_matrix.columns) == list(sample_returns.columns)
    assert covariance_matrix.shape == (sample_returns.shape[1], sample_returns.shape[1])


def test_ledoit_wolf_covariance_has_expected_properties(sample_returns):
    covariance_matrix = compute_ledoit_wolf_covariance(sample_returns)

    assert validate_estimated_covariance_matrix(covariance_matrix) is True
    assert np.allclose(covariance_matrix.values, covariance_matrix.values.T)
    assert (np.diag(covariance_matrix.values) > 0.0).all()
    assert not covariance_matrix.isna().any().any()


def test_ledoit_wolf_covariance_exposes_shrinkage_metadata(sample_returns):
    covariance_matrix = compute_ledoit_wolf_covariance(sample_returns)
    metadata = extract_covariance_metadata(covariance_matrix)

    assert metadata["method"] == "ledoit_wolf"
    assert 0.0 <= metadata["shrinkage"] <= 1.0


def test_ledoit_wolf_drops_nan_rows_and_keeps_asset_names(sample_returns):
    returns_with_nan = sample_returns.copy()
    returns_with_nan.iloc[0, 0] = np.nan

    covariance_matrix = compute_ledoit_wolf_covariance(returns_with_nan)

    assert validate_estimated_covariance_matrix(covariance_matrix) is True
    assert list(covariance_matrix.columns) == list(sample_returns.columns)
