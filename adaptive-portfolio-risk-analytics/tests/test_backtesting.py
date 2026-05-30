"""Unit tests for Phase 1 rolling backtester."""

import numpy as np
import pandas as pd

from src.backtesting import RollingBacktester
from src.optimization import EqualWeightAllocator


def _returns() -> pd.DataFrame:
    np.random.seed(11)
    dates = pd.date_range(start="2019-01-01", periods=400, freq="B")
    data = np.random.randn(400, 4) * 0.01
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])


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
