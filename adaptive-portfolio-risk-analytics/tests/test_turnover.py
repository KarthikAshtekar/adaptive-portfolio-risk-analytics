"""Tests for turnover analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtesting import calculate_turnover, calculate_turnover_series, summarize_turnover


def test_turnover_is_zero_when_weights_unchanged() -> None:
    weights = np.array([0.25, 0.25, 0.25, 0.25])

    turnover = calculate_turnover(weights, weights.copy())

    assert turnover == 0.0


def test_turnover_is_positive_when_weights_differ() -> None:
    current = np.array([0.25, 0.25, 0.25, 0.25])
    target = np.array([0.40, 0.20, 0.20, 0.20])

    turnover = calculate_turnover(current, target)

    assert turnover > 0.0


def test_turnover_aligns_series_labels_correctly() -> None:
    current = pd.Series({"A": 0.60, "B": 0.40})
    target = pd.Series({"B": 0.20, "A": 0.80})

    turnover = calculate_turnover(current, target)

    assert np.isclose(turnover, 0.20, atol=1e-8)


def test_turnover_summary_has_expected_keys() -> None:
    turnover_series = pd.Series(
        [0.0, 0.15, 0.05], index=pd.date_range("2020-01-31", periods=3, freq="ME")
    )

    summary = summarize_turnover(turnover_series)

    assert set(summary.keys()) == {
        "total_turnover",
        "average_turnover",
        "max_turnover",
        "num_rebalances",
    }


def test_calculate_turnover_series_returns_series() -> None:
    weights_history = pd.DataFrame(
        [
            [0.50, 0.50],
            [0.60, 0.40],
            [0.55, 0.45],
        ],
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
        columns=["A", "B"],
    )

    turnover_series = calculate_turnover_series(weights_history)

    assert isinstance(turnover_series, pd.Series)
    assert turnover_series.index.equals(weights_history.index)
