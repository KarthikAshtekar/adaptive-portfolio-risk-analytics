"""Unit tests for Phase 1 rolling backtester."""

import numpy as np
import pandas as pd

from src.backtesting import RollingBacktester, compare_strategies, generate_rebalance_dates
from src.optimization import (
    EqualWeightAllocator,
    InverseVolatilityAllocator,
    HRPAllocator,
)


def _returns() -> pd.DataFrame:
    np.random.seed(11)
    dates = pd.date_range(start="2019-01-01", periods=400, freq="B")
    data = np.random.randn(400, 4) * 0.01
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])


def _prices() -> pd.DataFrame:
    """Generate sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=504, freq="B")
    prices = pd.DataFrame(
        np.random.uniform(90, 110, size=(len(dates), 4))
        * (1 + np.random.randn(len(dates), 4) * 0.01).cumprod(axis=0),
        index=dates,
        columns=["Asset A", "Asset B", "Asset C", "Asset D"],
    )
    return prices


def test_rolling_backtester_generates_outputs() -> None:
    returns = _returns()
    bt = RollingBacktester(
        allocator=EqualWeightAllocator(),
        train_window=126,
        rebalance_frequency="M",
        initial_capital=1_000_000,
    )
    result = bt.run(returns)

    assert "portfolio_returns" in result
    assert "portfolio_values" in result
    assert "weights_history" in result
    assert "performance_metrics" in result
    assert not result["portfolio_returns"].empty
    assert not result["portfolio_values"].empty
    assert result["portfolio_values"].iloc[-1] > 0


def test_rolling_backtester_weights_sum_to_one() -> None:
    returns = _returns()
    result = RollingBacktester(allocator=EqualWeightAllocator(), train_window=126).run(returns)

    weights_history = result["weights_history"]
    row_sums = weights_history.sum(axis=1)
    assert np.allclose(row_sums.values, np.ones(len(row_sums)))


def test_rolling_backtester_drawdown_non_positive() -> None:
    returns = _returns()
    result = RollingBacktester(allocator=EqualWeightAllocator(), train_window=126).run(returns)

    drawdown = result["drawdown"]
    assert (drawdown <= 1e-12).all()


def test_generate_rebalance_dates_monthly() -> None:
    """Test monthly rebalance date generation."""
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-12-31")
    dates = generate_rebalance_dates(start, end, frequency="M")

    assert len(dates) >= 11
    assert dates[0] == start or dates[0].month == 1
    assert dates[-1] <= end


def test_compare_strategies_multiple() -> None:
    """Test comparison of multiple strategies."""
    prices = _prices()

    strategies = {
        "Equal Weight": EqualWeightAllocator(),
        "Inverse Volatility": InverseVolatilityAllocator(),
        "HRP": HRPAllocator(),
    }

    comparison = compare_strategies(
        prices,
        strategies,
        lookback_window=252,
        rebalance_frequency="M",
    )

    assert "strategy_returns_df" in comparison
    assert "performance_summary_df" in comparison
    assert comparison["strategy_returns_df"].shape[1] == 3
    assert comparison["performance_summary_df"].shape[0] == 3


def test_performance_summary_has_metrics() -> None:
    """Test that performance summary includes required metrics."""
    prices = _prices()

    strategies = {
        "Equal Weight": EqualWeightAllocator(),
    }

    comparison = compare_strategies(prices, strategies)
    metrics_df = comparison["performance_summary_df"]

    required_metrics = [
        "cumulative_return",
        "cagr",
        "sharpe",
        "volatility",
        "max_drawdown",
    ]

    for metric in required_metrics:
        assert metric in metrics_df.columns


def test_hrp_backtest_produces_output() -> None:
    """Test rolling backtester with HRP strategy."""
    prices = _prices()
    returns = prices.pct_change().dropna()

    allocator = HRPAllocator(linkage_method="single")
    backtester = RollingBacktester(
        allocator=allocator,
        train_window=252,
        rebalance_frequency="M",
    )

    results = backtester.run(returns)

    assert results is not None
    assert len(results["portfolio_returns"]) > 0
    assert "weights_history" in results


def test_backtest_reproducible() -> None:
    """Test that backtest results are reproducible."""
    prices = _prices()
    returns = prices.pct_change().dropna()

    allocator1 = EqualWeightAllocator()
    backtester1 = RollingBacktester(allocator=allocator1, train_window=252)
    results1 = backtester1.run(returns.copy())

    allocator2 = EqualWeightAllocator()
    backtester2 = RollingBacktester(allocator=allocator2, train_window=252)
    results2 = backtester2.run(returns.copy())

    assert np.allclose(
        results1["portfolio_returns"].values,
        results2["portfolio_returns"].values,
        rtol=1e-10,
    )
