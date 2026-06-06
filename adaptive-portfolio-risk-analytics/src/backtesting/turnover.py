"""Turnover analytics for realistic portfolio backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _align_weights(
    current_weights: pd.Series | np.ndarray,
    target_weights: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align weight vectors before turnover calculation."""
    if isinstance(current_weights, pd.Series) and isinstance(target_weights, pd.Series):
        if set(current_weights.index) != set(target_weights.index):
            raise ValueError("current_weights and target_weights labels must match")
        current_aligned = current_weights.reindex(target_weights.index).astype(float)
        target_aligned = target_weights.astype(float)
        return current_aligned.values, target_aligned.values

    current_array = np.asarray(current_weights, dtype=float)
    target_array = np.asarray(target_weights, dtype=float)
    if current_array.ndim != 1 or target_array.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if len(current_array) != len(target_array):
        raise ValueError("current_weights and target_weights must have the same length")
    return current_array, target_array


def calculate_turnover(
    current_weights: pd.Series | np.ndarray,
    target_weights: pd.Series | np.ndarray,
) -> float:
    """Calculate portfolio turnover as half the L1 weight change."""
    current_array, target_array = _align_weights(current_weights, target_weights)
    return float(0.5 * np.abs(target_array - current_array).sum())


def calculate_turnover_series(weights_history_df: pd.DataFrame) -> pd.Series:
    """Calculate turnover for each rebalance date from a weights history DataFrame."""
    if not isinstance(weights_history_df, pd.DataFrame):
        raise TypeError("weights_history_df must be a pandas DataFrame")
    if weights_history_df.empty:
        return pd.Series(dtype=float, name="turnover")

    turnovers = [0.0]
    for idx in range(1, len(weights_history_df)):
        turnover = calculate_turnover(
            weights_history_df.iloc[idx - 1],
            weights_history_df.iloc[idx],
        )
        turnovers.append(turnover)

    return pd.Series(turnovers, index=weights_history_df.index, name="turnover", dtype=float)


def summarize_turnover(turnover_series: pd.Series) -> dict[str, float | int]:
    """Summarize turnover history."""
    if not isinstance(turnover_series, pd.Series):
        raise TypeError("turnover_series must be a pandas Series")
    if turnover_series.empty:
        return {
            "total_turnover": 0.0,
            "average_turnover": 0.0,
            "max_turnover": 0.0,
            "num_rebalances": 0,
        }

    return {
        "total_turnover": float(turnover_series.sum()),
        "average_turnover": float(turnover_series.mean()),
        "max_turnover": float(turnover_series.max()),
        "num_rebalances": int(len(turnover_series)),
    }
