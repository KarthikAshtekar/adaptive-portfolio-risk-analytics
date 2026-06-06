"""Exponentially weighted covariance estimation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _prepare_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean returns before covariance estimation."""
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


def _validate_span(span: int) -> int:
    """Ensure the EWMA span is valid."""
    if not isinstance(span, int):
        raise TypeError("span must be an integer")
    if span < 2:
        raise ValueError("span must be at least 2")
    return span


def _attach_metadata(covariance_matrix: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """Attach estimator metadata while preserving DataFrame output."""
    covariance_matrix.attrs["covariance_metadata"] = metadata
    return covariance_matrix


def _compute_ewma_weights(length: int, span: int) -> np.ndarray:
    """Compute normalized EWMA weights for the full return history."""
    alpha = 2.0 / (span + 1.0)
    exponents = np.arange(length - 1, -1, -1, dtype=float)
    weights = alpha * np.power(1.0 - alpha, exponents)
    weights /= weights.sum()
    return weights


def compute_ewma_covariance(returns_df: pd.DataFrame, span: int = 252) -> pd.DataFrame:
    """Estimate the latest EWMA covariance matrix."""
    clean_returns = _prepare_returns(returns_df)
    span = _validate_span(span)

    ewm_covariance = clean_returns.ewm(span=span, adjust=True).cov()
    latest_covariance = ewm_covariance.xs(clean_returns.index[-1], level=0)
    covariance_matrix = (latest_covariance + latest_covariance.T) / 2.0
    covariance_matrix = covariance_matrix.loc[clean_returns.columns, clean_returns.columns]

    metadata = {
        "method": "ewma",
        "span": span,
    }
    return _attach_metadata(covariance_matrix, metadata)


def compute_ewma_ledoit_wolf_covariance(returns_df: pd.DataFrame, span: int = 252) -> pd.DataFrame:
    """Estimate covariance from EWMA-weighted returns with Ledoit-Wolf shrinkage."""
    clean_returns = _prepare_returns(returns_df)
    span = _validate_span(span)

    weights = _compute_ewma_weights(len(clean_returns), span)
    weighted_mean = np.average(clean_returns.values, axis=0, weights=weights)
    centered_returns = clean_returns.values - weighted_mean
    weighted_returns = centered_returns * np.sqrt(len(clean_returns) * weights)[:, None]

    estimator = LedoitWolf(assume_centered=True)
    estimator.fit(weighted_returns)

    covariance_matrix = pd.DataFrame(
        estimator.covariance_,
        index=clean_returns.columns,
        columns=clean_returns.columns,
    )
    metadata = {
        "method": "ewma_ledoit_wolf",
        "span": span,
        "shrinkage": float(estimator.shrinkage_),
    }
    return _attach_metadata(covariance_matrix, metadata)
