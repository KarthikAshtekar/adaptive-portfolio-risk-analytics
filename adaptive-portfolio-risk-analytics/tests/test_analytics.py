"""Unit tests for Phase 1 analytics metrics."""

import numpy as np
import pandas as pd

from src.analytics import PerformanceAnalytics, RiskAnalytics


def _series() -> pd.Series:
    np.random.seed(9)
    data = np.random.randn(252) * 0.01 + 0.0004
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    return pd.Series(data, index=idx)


def test_risk_metrics_basic_properties() -> None:
    returns = _series()

    vol = RiskAnalytics.volatility(returns)
    max_dd = RiskAnalytics.maximum_drawdown(returns)
    var95 = RiskAnalytics.value_at_risk(returns)
    cvar95 = RiskAnalytics.conditional_value_at_risk(returns)

    assert vol > 0
    assert max_dd <= 0
    assert cvar95 <= var95


def test_performance_metrics_basic_properties() -> None:
    returns = _series()

    cum = PerformanceAnalytics.cumulative_return(returns)
    cagr = PerformanceAnalytics.annualized_return(returns)
    sharpe = PerformanceAnalytics.sharpe_ratio(returns)
    sortino = PerformanceAnalytics.sortino_ratio(returns)

    assert isinstance(cum, float)
    assert isinstance(cagr, float)
    assert np.isfinite(sharpe)
    assert np.isfinite(sortino) or np.isinf(sortino)


def test_summary_table_contains_required_keys() -> None:
    returns = _series()
    summary = PerformanceAnalytics.summary_table(returns)

    for key in ["cumulative_return", "cagr", "sharpe", "sortino", "volatility", "max_drawdown"]:
        assert key in summary
