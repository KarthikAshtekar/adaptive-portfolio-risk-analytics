"""Unit tests for Phase 1 analytics metrics."""

import numpy as np
import pandas as pd

from src.analytics import PerformanceAnalytics, RiskAnalytics


def _series() -> pd.Series:
    np.random.seed(9)
    data = np.random.randn(252) * 0.01 + 0.0004
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    return pd.Series(data, index=idx)


def _positive_series() -> pd.Series:
    """Generate positive return series for testing metrics that require positive values."""
    np.random.seed(42)
    data = np.random.randn(252) * 0.005 + 0.0008
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


def test_annualized_return_calculation() -> None:
    """Test annualized return calculation."""
    returns = _positive_series()
    annual = PerformanceAnalytics.annualized_return(returns)
    assert isinstance(annual, float)
    assert annual > -1.0


def test_cagr_equals_annualized_return() -> None:
    """Test that CAGR alias works correctly."""
    returns = _positive_series()
    cagr = PerformanceAnalytics.cagr(returns)
    annual = PerformanceAnalytics.annualized_return(returns)
    assert np.isclose(cagr, annual)


def test_annualized_volatility() -> None:
    """Test annualized volatility calculation."""
    returns = _series()
    vol = PerformanceAnalytics.annualized_volatility(returns)
    risk_vol = RiskAnalytics.volatility(returns)
    assert np.isclose(vol, risk_vol)
    assert vol > 0


def test_sharpe_ratio_positive_returns() -> None:
    """Test Sharpe ratio with positive returns."""
    returns = _positive_series()
    sharpe = PerformanceAnalytics.sharpe_ratio(returns)
    assert isinstance(sharpe, float)
    assert np.isfinite(sharpe)


def test_sortino_ratio_calculation() -> None:
    """Test Sortino ratio calculation."""
    returns = _series()
    sortino = PerformanceAnalytics.sortino_ratio(returns)
    assert isinstance(sortino, float)
    assert np.isfinite(sortino) or np.isinf(sortino)


def test_calmar_ratio() -> None:
    """Test Calmar ratio calculation."""
    returns = _positive_series()
    calmar = PerformanceAnalytics.calmar_ratio(returns)
    assert isinstance(calmar, float)
    assert calmar > 0


def test_maximum_drawdown_properties() -> None:
    """Test maximum drawdown calculation."""
    returns = _series()
    max_dd = RiskAnalytics.maximum_drawdown(returns)
    assert max_dd <= 0
    assert max_dd >= -1.0


def test_max_drawdown_alias() -> None:
    """Test max_drawdown alias."""
    returns = _series()
    max_dd1 = RiskAnalytics.max_drawdown(returns)
    max_dd2 = RiskAnalytics.maximum_drawdown(returns)
    assert np.isclose(max_dd1, max_dd2)


def test_drawdown_series_properties() -> None:
    """Test drawdown series calculation."""
    returns = _series()
    dd_series = RiskAnalytics.drawdown_series(returns)
    assert len(dd_series) == len(returns)
    assert (dd_series <= 0).all()


def test_rolling_volatility() -> None:
    """Test rolling volatility calculation."""
    returns = _series()
    rolling_vol = RiskAnalytics.rolling_volatility(returns, window=30)
    assert len(rolling_vol) == len(returns)
    assert (rolling_vol[~rolling_vol.isna()] >= 0).all()


def test_rolling_sharpe() -> None:
    """Test rolling Sharpe ratio calculation."""
    returns = _positive_series()
    rolling_sharpe = RiskAnalytics.rolling_sharpe(returns, window=30)
    assert len(rolling_sharpe) == len(returns)
    assert np.isfinite(rolling_sharpe).any()


def test_downside_deviation() -> None:
    """Test downside deviation calculation."""
    returns = _series()
    dd = RiskAnalytics.downside_deviation(returns)
    assert dd >= 0


def test_empty_series_handling() -> None:
    """Test handling of empty series."""
    empty = pd.Series(dtype=float)

    assert RiskAnalytics.volatility(empty) == 0.0
    assert PerformanceAnalytics.cumulative_return(empty) == 0.0
    assert PerformanceAnalytics.annualized_return(empty) == 0.0
    assert PerformanceAnalytics.sharpe_ratio(empty) == 0.0


def test_metrics_reproducibility() -> None:
    """Test that metrics are reproducible."""
    returns1 = _positive_series()
    returns2 = _positive_series()

    sharpe1 = PerformanceAnalytics.sharpe_ratio(returns1)
    sharpe2 = PerformanceAnalytics.sharpe_ratio(returns2)

    assert np.isclose(sharpe1, sharpe2)
