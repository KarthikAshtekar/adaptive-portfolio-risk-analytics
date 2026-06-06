"""Ledoit-Wolf covariance estimation utilities."""

from __future__ import annotations

from typing import Any

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


def _attach_metadata(covariance_matrix: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """Attach estimator metadata while preserving DataFrame output."""
    covariance_matrix.attrs["covariance_metadata"] = metadata
    return covariance_matrix


def compute_ledoit_wolf_covariance(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate covariance using Ledoit-Wolf shrinkage."""
    clean_returns = _prepare_returns(returns_df)
    estimator = LedoitWolf()
    estimator.fit(clean_returns.values)

    covariance_matrix = pd.DataFrame(
        estimator.covariance_,
        index=clean_returns.columns,
        columns=clean_returns.columns,
    )
    metadata = {
        "method": "ledoit_wolf",
        "shrinkage": float(estimator.shrinkage_),
    }
    return _attach_metadata(covariance_matrix, metadata)
