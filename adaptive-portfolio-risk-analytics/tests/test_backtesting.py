"""Unit tests for Phase 1 rolling backtester."""

import warnings

import numpy as np
import pandas as pd

from src.backtesting import (
    RollingBacktester,
    compare_strategies,
    generate_rebalance_dates,
    normalize_rebalance_frequency,
    TransactionCostModel,
)
from src.optimization import (
    BaseAllocator,
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


def _constant_returns_no_drift() -> pd.DataFrame:
    dates = pd.date_range(start="2021-01-01", periods=120, freq="B")
    data = np.full((len(dates), 4), 0.001)
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])


def _drifty_returns() -> pd.DataFrame:
    dates = pd.date_range(start="2021-01-01", periods=140, freq="B")
    base = np.linspace(-0.006, 0.006, len(dates))
    data = np.column_stack(
        [
            0.002 + 1.5 * base,
            0.001 - 1.0 * base,
            -0.001 + 0.5 * base,
            0.0005 - 0.75 * base,
        ]
    )
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])


class CountingAllocator(BaseAllocator):
    def __init__(self) -> None:
        self.optimize_calls = 0
        self._weights: np.ndarray | None = None

    def fit(self, returns: pd.DataFrame) -> "CountingAllocator":
        self._weights = self.optimize(returns)
        return self

    def optimize(self, returns: pd.DataFrame) -> np.ndarray:
        self.optimize_calls += 1
        self._weights = np.ones(returns.shape[1], dtype=float) / returns.shape[1]
        return self._weights

    def get_weights(self) -> np.ndarray:
        if self._weights is None:
            raise ValueError("weights are not available before fit/optimize")
        return self._weights


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
    assert "rebalance_log" in result
    assert "turnover_summary" in result
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dates = generate_rebalance_dates(start, end, frequency="M")

    assert len(dates) >= 11
    assert dates[0] == start or dates[0].month == 1
    assert dates[-1] <= end
    assert not any("'M' is deprecated" in str(w.message) for w in caught)


def test_generate_rebalance_dates_me_matches_m() -> None:
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-12-31")

    dates_m = generate_rebalance_dates(start, end, frequency="M")
    dates_me = generate_rebalance_dates(start, end, frequency="ME")

    assert dates_m == dates_me


def test_normalize_rebalance_frequency_maps_month_end_alias() -> None:
    assert normalize_rebalance_frequency("M") == "ME"
    assert normalize_rebalance_frequency("ME") == "ME"
    assert normalize_rebalance_frequency("W") == "W"


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


def test_calendar_mode_still_works() -> None:
    returns = _returns()
    results = RollingBacktester(
        allocator=EqualWeightAllocator(),
        train_window=60,
        rebalance_mode="calendar",
        rebalance_frequency="M",
    ).run(returns)

    assert not results["portfolio_returns"].empty
    assert "rebalance_log" in results


def test_threshold_mode_works() -> None:
    returns = _returns()
    results = RollingBacktester(
        allocator=InverseVolatilityAllocator(),
        train_window=60,
        rebalance_mode="threshold",
        threshold=0.01,
        rebalance_frequency="M",
    ).run(returns)

    assert not results["portfolio_returns"].empty
    assert "rebalance_log" in results


def test_threshold_mode_does_not_rebalance_repeatedly_without_drift() -> None:
    returns = _constant_returns_no_drift()
    results = RollingBacktester(
        allocator=EqualWeightAllocator(),
        train_window=20,
        rebalance_mode="threshold",
        threshold=0.03,
    ).run(returns)

    assert results["rebalance_summary"]["total_rebalances"] == 1
    assert results["rebalance_summary"]["threshold_rebalances"] == 0


def test_threshold_rebalance_counts_decline_as_threshold_increases() -> None:
    returns = _drifty_returns()
    counts = []

    for threshold in (0.03, 0.05, 0.10):
        results = RollingBacktester(
            allocator=EqualWeightAllocator(),
            train_window=20,
            rebalance_mode="threshold",
            threshold=threshold,
        ).run(returns)
        counts.append(results["rebalance_summary"]["total_rebalances"])

    assert counts[0] >= counts[1] >= counts[2]


def test_target_weights_update_on_target_frequency_not_daily() -> None:
    returns = _drifty_returns()
    allocator = CountingAllocator()
    results = RollingBacktester(
        allocator=allocator,
        train_window=20,
        rebalance_mode="threshold",
        threshold=0.05,
        target_update_frequency="M",
    ).run(returns)

    expected_updates = returns.index[19:-1].to_period("M").nunique()
    assert allocator.optimize_calls == expected_updates
    assert results["rebalance_summary"]["total_rebalances"] >= 1


def test_calendar_or_threshold_mode_works() -> None:
    returns = _returns()
    results = RollingBacktester(
        allocator=HRPAllocator(),
        train_window=60,
        rebalance_mode="calendar_or_threshold",
        threshold=0.01,
        rebalance_frequency="M",
    ).run(returns)

    assert not results["portfolio_returns"].empty
    assert "rebalance_log" in results


def test_calendar_mode_rebalances_on_schedule() -> None:
    returns = _drifty_returns()
    train_window = 20
    results = RollingBacktester(
        allocator=EqualWeightAllocator(),
        train_window=train_window,
        rebalance_mode="calendar",
        rebalance_frequency="M",
    ).run(returns)

    expected_rebalances = returns.index[train_window - 1 : -1].to_period("M").nunique()
    assert results["rebalance_summary"]["total_rebalances"] == expected_rebalances
    assert results["rebalance_summary"]["calendar_rebalances"] == max(
        expected_rebalances - 1,
        0,
    )


def test_calendar_or_threshold_has_at_least_as_many_rebalances_as_calendar() -> None:
    returns = _drifty_returns()
    calendar_results = RollingBacktester(
        allocator=EqualWeightAllocator(),
        train_window=20,
        rebalance_mode="calendar",
        rebalance_frequency="M",
    ).run(returns)
    hybrid_results = RollingBacktester(
        allocator=EqualWeightAllocator(),
        train_window=20,
        rebalance_mode="calendar_or_threshold",
        rebalance_frequency="M",
        threshold=0.03,
    ).run(returns)

    assert (
        hybrid_results["rebalance_summary"]["total_rebalances"]
        >= calendar_results["rebalance_summary"]["total_rebalances"]
    )


def test_rebalance_log_records_required_fields() -> None:
    returns = _drifty_returns()
    results = RollingBacktester(
        allocator=InverseVolatilityAllocator(),
        train_window=20,
        rebalance_mode="calendar_or_threshold",
        threshold=0.03,
        transaction_cost_model=TransactionCostModel(base_bps=25.0, slippage_bps=10.0),
    ).run(returns)

    rebalance_log = results["rebalance_log"]
    assert not rebalance_log.empty
    for column in (
        "rebalance_date",
        "rebalance_reason",
        "turnover",
        "transaction_cost",
        "max_weight_drift",
    ):
        assert column in rebalance_log.columns


def test_transaction_costs_reduce_portfolio_value() -> None:
    returns = _returns()
    results = RollingBacktester(
        allocator=EqualWeightAllocator(),
        train_window=60,
        rebalance_mode="calendar",
        rebalance_frequency="M",
        transaction_cost_model=TransactionCostModel(base_bps=50.0, slippage_bps=50.0),
    ).run(returns)

    assert (results["gross_portfolio_values"] >= results["portfolio_values"]).all()
    assert results["performance_metrics"]["total_transaction_cost"] >= 0.0


def test_existing_result_keys_remain_present() -> None:
    returns = _returns()
    results = RollingBacktester(
        allocator=EqualWeightAllocator(),
        train_window=60,
        rebalance_frequency="M",
    ).run(returns)

    for key in (
        "portfolio_returns",
        "portfolio_values",
        "drawdown",
        "weights_history",
        "performance_metrics",
    ):
        assert key in results

    for new_key in (
        "gross_portfolio_values",
        "rebalance_log",
        "turnover_summary",
        "rebalance_summary",
        "cost_drag_summary",
    ):
        assert new_key in results


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
