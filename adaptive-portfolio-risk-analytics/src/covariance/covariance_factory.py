"""Factory and validation helpers for covariance research estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .covariance import compute_covariance_matrix
from .ewma_covariance import (
    compute_ewma_covariance,
    compute_ewma_ledoit_wolf_covariance,
)
from .ledoit_wolf import compute_ledoit_wolf_covariance

SUPPORTED_COVARIANCE_METHODS = (
    "sample",
    "ledoit_wolf",
    "ewma",
    "ewma_ledoit_wolf",
)


def extract_covariance_metadata(covariance_matrix: pd.DataFrame) -> dict[str, Any]:
    """Return estimator metadata attached to a covariance DataFrame."""
    metadata = covariance_matrix.attrs.get("covariance_metadata", {})
    return dict(metadata)


def validate_estimated_covariance_matrix(
    covariance_matrix: pd.DataFrame, atol: float = 1e-10
) -> bool:
    """Validate core covariance matrix properties for research estimators."""
    if not isinstance(covariance_matrix, pd.DataFrame):
        return False
    if covariance_matrix.empty:
        return False
    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        return False

    values = covariance_matrix.values
    if np.isnan(values).any():
        return False
    if not np.allclose(values, values.T, atol=atol):
        return False
    if not (np.diag(values) > 0.0).all():
        return False

    return True


def assert_valid_covariance_matrix(covariance_matrix: pd.DataFrame, *, method: str) -> pd.DataFrame:
    """Raise a descriptive error if an estimated covariance matrix is invalid."""
    if not validate_estimated_covariance_matrix(covariance_matrix):
        raise ValueError(f"{method} produced an invalid covariance matrix")
    return covariance_matrix


def _prepare_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean returns before factory routing."""
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame")
    if returns_df.empty:
        raise ValueError("returns_df must not be empty")

    clean_returns = returns_df.dropna(how="any")
    if clean_returns.empty:
        raise ValueError("returns_df has no valid rows after dropping NaNs")
    if clean_returns.shape[1] < 2:
        raise ValueError("returns_df must contain at least two assets")

    return clean_returns


def _attach_metadata(covariance_matrix: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """Attach estimator metadata while preserving DataFrame output."""
    covariance_matrix.attrs["covariance_metadata"] = metadata
    return covariance_matrix


def _compute_sample_covariance(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute sample covariance with aligned labels and metadata."""
    clean_returns = _prepare_returns(returns_df)
    covariance_matrix = compute_covariance_matrix(clean_returns)
    covariance_matrix = covariance_matrix.loc[clean_returns.columns, clean_returns.columns]
    return _attach_metadata(covariance_matrix, {"method": "sample"})


class CovarianceFactory:
    """Single entry point for covariance estimator research."""

    @staticmethod
    def compute(returns_df: pd.DataFrame, method: str = "sample", **kwargs: Any) -> pd.DataFrame:
        """Compute a covariance matrix using the requested estimation method."""
        normalized_method = method.lower()

        if normalized_method == "sample":
            covariance_matrix = _compute_sample_covariance(returns_df)
        elif normalized_method == "ledoit_wolf":
            covariance_matrix = compute_ledoit_wolf_covariance(returns_df)
        elif normalized_method == "ewma":
            covariance_matrix = compute_ewma_covariance(returns_df, **kwargs)
        elif normalized_method == "ewma_ledoit_wolf":
            covariance_matrix = compute_ewma_ledoit_wolf_covariance(returns_df, **kwargs)
        else:
            supported = ", ".join(SUPPORTED_COVARIANCE_METHODS)
            raise ValueError(
                f"unsupported covariance method '{method}'. Supported methods: {supported}"
            )

        return assert_valid_covariance_matrix(covariance_matrix, method=normalized_method)
