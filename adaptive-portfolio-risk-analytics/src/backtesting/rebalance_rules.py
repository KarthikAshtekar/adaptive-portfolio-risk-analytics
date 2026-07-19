"""Rebalance rule helpers for realistic backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_rebalance_frequency(freq: str) -> str:
    """Normalize rebalance frequencies for pandas offset aliases."""
    if not isinstance(freq, str):
        raise TypeError("freq must be a string")

    normalized = freq.upper()
    if normalized == "M":
        return "ME"
    return normalized


def _period_frequency(freq: str) -> str:
    normalized = normalize_rebalance_frequency(freq)
    if normalized == "ME":
        return "M"
    return normalized


def should_rebalance_calendar(
    current_date,
    previous_rebalance_date,
    frequency: str,
) -> bool:
    """Determine whether a calendar rebalance should occur."""
    if previous_rebalance_date is None:
        return True

    normalized_frequency = normalize_rebalance_frequency(frequency)
    current_timestamp = pd.Timestamp(current_date)
    previous_timestamp = pd.Timestamp(previous_rebalance_date)

    if normalized_frequency == "D":
        return current_timestamp > previous_timestamp

    period_frequency = _period_frequency(normalized_frequency)
    return current_timestamp.to_period(period_frequency) != previous_timestamp.to_period(
        period_frequency
    )


def _align_weight_vectors(
    current_weights: pd.Series | np.ndarray,
    target_weights: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(current_weights, pd.Series) and isinstance(target_weights, pd.Series):
        if set(current_weights.index) != set(target_weights.index):
            raise ValueError("current_weights and target_weights labels must match")
        target_aligned = target_weights.reindex(current_weights.index).astype(float)
        current_aligned = current_weights.astype(float)
        return current_aligned.values, target_aligned.values

    current_array = np.asarray(current_weights, dtype=float)
    target_array = np.asarray(target_weights, dtype=float)
    if current_array.ndim != 1 or target_array.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if len(current_array) != len(target_array):
        raise ValueError("current_weights and target_weights must have the same length")
    return current_array, target_array


def should_rebalance_threshold(
    current_weights: pd.Series | np.ndarray,
    target_weights: pd.Series | np.ndarray,
    threshold: float = 0.05,
) -> bool:
    """Rebalance when the maximum absolute weight drift exceeds the threshold."""
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")

    current_array, target_array = _align_weight_vectors(current_weights, target_weights)
    max_drift = float(np.abs(current_array - target_array).max())
    return max_drift >= threshold
