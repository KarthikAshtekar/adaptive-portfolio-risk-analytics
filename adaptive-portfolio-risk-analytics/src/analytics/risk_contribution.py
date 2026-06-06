"""Risk contribution analytics for portfolio attribution research."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_covariance_matrix(covariance_matrix: pd.DataFrame) -> pd.DataFrame:
    """Validate covariance matrix structure."""
    if not isinstance(covariance_matrix, pd.DataFrame):
        raise TypeError("covariance_matrix must be a pandas DataFrame")
    if covariance_matrix.empty:
        raise ValueError("covariance_matrix must not be empty")
    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("covariance_matrix must be square")
    if not covariance_matrix.index.equals(covariance_matrix.columns):
        raise ValueError("covariance_matrix row and column labels must match")
    if covariance_matrix.isna().any().any():
        raise ValueError("covariance_matrix must not contain NaN values")

    return covariance_matrix.astype(float)


def _normalize_weights(
    weights: pd.Series | np.ndarray,
    covariance_matrix: pd.DataFrame,
) -> pd.Series:
    """Validate and align weights against covariance labels."""
    covariance_matrix = _validate_covariance_matrix(covariance_matrix)

    if isinstance(weights, pd.Series):
        weight_series = weights.astype(float).copy()
        if weight_series.empty:
            raise ValueError("weights must not be empty")
        if weight_series.isna().any():
            raise ValueError("weights must not contain NaN values")
        if not np.isfinite(weight_series.values).all():
            raise ValueError("weights must be finite")
        if set(weight_series.index) != set(covariance_matrix.columns):
            raise ValueError("weights labels must match covariance matrix labels")
        weight_series = weight_series.reindex(covariance_matrix.columns)
    elif isinstance(weights, np.ndarray):
        if weights.ndim != 1:
            raise ValueError("weights ndarray must be one-dimensional")
        if len(weights) != covariance_matrix.shape[0]:
            raise ValueError("weights length must match covariance matrix dimension")
        if np.isnan(weights).any():
            raise ValueError("weights must not contain NaN values")
        if not np.isfinite(weights).all():
            raise ValueError("weights must be finite")
        weight_series = pd.Series(weights.astype(float), index=covariance_matrix.columns, name="weight")
    else:
        raise TypeError("weights must be a pandas Series or numpy ndarray")

    if (weight_series < -1e-8).any():
        raise ValueError("weights must be non-negative")
    if not np.isclose(float(weight_series.sum()), 1.0, atol=1e-6):
        raise ValueError("weights must sum approximately to 1")

    weight_series = weight_series.clip(lower=0.0)
    weight_series = weight_series / float(weight_series.sum())
    weight_series.name = "weight"
    return weight_series


def portfolio_volatility(
    weights: pd.Series | np.ndarray,
    covariance_matrix: pd.DataFrame,
) -> float:
    """Compute portfolio volatility from weights and covariance."""
    covariance_matrix = _validate_covariance_matrix(covariance_matrix)
    weight_series = _normalize_weights(weights, covariance_matrix)
    variance = float(weight_series.values.T @ covariance_matrix.values @ weight_series.values)
    return float(np.sqrt(max(variance, 0.0)))


def marginal_risk_contribution(
    weights: pd.Series | np.ndarray,
    covariance_matrix: pd.DataFrame,
) -> pd.Series:
    """Compute asset-level marginal risk contribution."""
    covariance_matrix = _validate_covariance_matrix(covariance_matrix)
    weight_series = _normalize_weights(weights, covariance_matrix)
    portfolio_vol = portfolio_volatility(weight_series, covariance_matrix)
    if portfolio_vol <= 0.0:
        raise ValueError("portfolio volatility must be positive")

    marginal = covariance_matrix.values @ weight_series.values / portfolio_vol
    return pd.Series(
        marginal,
        index=weight_series.index,
        name="Marginal Risk Contribution",
        dtype=float,
    )


def total_risk_contribution(
    weights: pd.Series | np.ndarray,
    covariance_matrix: pd.DataFrame,
) -> pd.Series:
    """Compute asset-level total risk contribution."""
    covariance_matrix = _validate_covariance_matrix(covariance_matrix)
    weight_series = _normalize_weights(weights, covariance_matrix)
    mrc = marginal_risk_contribution(weight_series, covariance_matrix)
    trc = weight_series * mrc
    trc.name = "Total Risk Contribution"
    return trc.astype(float)


def percentage_risk_contribution(
    weights: pd.Series | np.ndarray,
    covariance_matrix: pd.DataFrame,
) -> pd.Series:
    """Compute asset-level percentage risk contribution."""
    covariance_matrix = _validate_covariance_matrix(covariance_matrix)
    trc = total_risk_contribution(weights, covariance_matrix)
    portfolio_vol = portfolio_volatility(weights, covariance_matrix)
    if portfolio_vol <= 0.0:
        raise ValueError("portfolio volatility must be positive")

    prc = trc / portfolio_vol
    prc.name = "Percentage Risk Contribution"
    if not np.isclose(float(prc.sum()), 1.0, atol=1e-6):
        raise ValueError("percentage risk contributions must sum approximately to 1")
    return prc.astype(float)


def risk_contribution_table(
    weights: pd.Series | np.ndarray,
    covariance_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Return a risk contribution table sorted by percentage risk contribution."""
    covariance_matrix = _validate_covariance_matrix(covariance_matrix)
    weight_series = _normalize_weights(weights, covariance_matrix)
    mrc = marginal_risk_contribution(weight_series, covariance_matrix)
    trc = total_risk_contribution(weight_series, covariance_matrix)
    prc = percentage_risk_contribution(weight_series, covariance_matrix)

    table = pd.DataFrame(
        {
            "Asset": weight_series.index,
            "Weight": weight_series.values,
            "Marginal Risk Contribution": mrc.reindex(weight_series.index).values,
            "Total Risk Contribution": trc.reindex(weight_series.index).values,
            "Percentage Risk Contribution": prc.reindex(weight_series.index).values,
        }
    )
    return table.sort_values(
        by="Percentage Risk Contribution",
        ascending=False,
        ignore_index=True,
    )


def compare_risk_contributions(
    hrp_weights: pd.Series | np.ndarray,
    herc_weights: pd.Series | np.ndarray,
    covariance_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Compare HRP and HERC weights and percentage risk contributions."""
    covariance_matrix = _validate_covariance_matrix(covariance_matrix)
    hrp_series = _normalize_weights(hrp_weights, covariance_matrix)
    herc_series = _normalize_weights(herc_weights, covariance_matrix)

    hrp_prc = percentage_risk_contribution(hrp_series, covariance_matrix)
    herc_prc = percentage_risk_contribution(herc_series, covariance_matrix)

    comparison_df = pd.DataFrame(
        {
            "Asset": covariance_matrix.columns,
            "HRP Weight": hrp_series.reindex(covariance_matrix.columns).values,
            "HERC Weight": herc_series.reindex(covariance_matrix.columns).values,
            "HRP % Risk Contribution": hrp_prc.reindex(covariance_matrix.columns).values,
            "HERC % Risk Contribution": herc_prc.reindex(covariance_matrix.columns).values,
        }
    )
    comparison_df["Risk Contribution Difference"] = (
        comparison_df["HERC % Risk Contribution"] - comparison_df["HRP % Risk Contribution"]
    )
    return comparison_df
