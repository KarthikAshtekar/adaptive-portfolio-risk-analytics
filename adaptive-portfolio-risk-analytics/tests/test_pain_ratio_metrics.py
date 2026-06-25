"""Tests for Pain Index and Pain Ratio metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import PerformanceAnalytics, compute_drawdown_series, compute_pain_index
from src.analytics.performance_metrics import compute_pain_ratio


def test_pain_index_computed_correctly_on_toy_wealth_path() -> None:
    values = pd.Series([100.0, 120.0, 90.0, 110.0])
    drawdown = compute_drawdown_series(values)

    assert drawdown.tolist() == pytest.approx([0.0, 0.0, -0.25, -1.0 / 12.0])


def test_pain_ratio_computed_correctly_on_toy_returns() -> None:
    returns = pd.Series([0.10, -0.10, 0.05])
    expected_pain = (0.0 + 0.10 + (1.0 - 1.0395 / 1.10)) / 3.0
    expected_return = PerformanceAnalytics.annualized_return(returns, periods_per_year=3)

    assert compute_pain_index(returns) == pytest.approx(expected_pain)
    assert compute_pain_ratio(returns, periods_per_year=3) == pytest.approx(
        expected_return / expected_pain
    )


def test_pain_ratio_returns_nan_when_pain_index_is_zero() -> None:
    returns = pd.Series([0.01, 0.02, 0.03])

    assert compute_pain_index(returns) == pytest.approx(0.0)
    assert np.isnan(compute_pain_ratio(returns, periods_per_year=3))


def test_persistent_drawdown_has_positive_pain_index() -> None:
    returns = pd.Series([-0.10, 0.0, 0.0])

    assert compute_pain_index(returns) == pytest.approx(0.10)
    assert compute_pain_ratio(returns, periods_per_year=3) < 0.0


def test_negative_series_and_calmar_behavior_are_distinct() -> None:
    returns = pd.Series([-0.02, -0.01, -0.03, 0.01])

    assert compute_pain_index(returns) > 0.0
    assert np.isfinite(compute_pain_ratio(returns, periods_per_year=4))
    assert PerformanceAnalytics.calmar_ratio(returns, periods_per_year=4) < 0.0


def test_summary_table_includes_pain_metrics_without_removing_calmar() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, -0.01])
    summary = PerformanceAnalytics.summary_table(returns, risk_free_rate=0.0)

    assert "calmar" in summary
    assert "pain_index" in summary
    assert "pain_ratio" in summary
