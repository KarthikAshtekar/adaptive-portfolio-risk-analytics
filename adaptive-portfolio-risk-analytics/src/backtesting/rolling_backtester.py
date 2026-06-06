"""Rolling-window backtester for Phase 1 portfolio strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import PerformanceAnalytics, RiskAnalytics
from src.optimization import BaseAllocator, EqualWeightAllocator

from .base import BaseBacktester
from .transaction_costs import TransactionCostCalculator


def generate_rebalance_dates(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    frequency: str = "M",
) -> list[pd.Timestamp]:
    """Generate rebalancing dates at specified frequency (default: monthly)."""
    if start_date >= end_date:
        raise ValueError("start_date must be before end_date")
    date_range = pd.date_range(
        start=start_date,
        end=end_date,
        freq=normalize_rebalance_frequency(frequency),
    )
    return date_range.tolist()


def normalize_rebalance_frequency(frequency: str) -> str:
    """Normalize rebalance frequency for pandas date offsets."""
    if not isinstance(frequency, str):
        raise TypeError("frequency must be a string")

    normalized = frequency.upper()
    if normalized == "M":
        return "ME"
    return normalized


def rebalance_period_frequency(frequency: str) -> str:
    """Return a Period-compatible frequency for rebalance boundary checks."""
    normalized = normalize_rebalance_frequency(frequency)
    if normalized == "ME":
        return "M"
    return normalized


def compare_strategies(
    prices_df: pd.DataFrame,
    strategies: dict[str, BaseAllocator],
    lookback_window: int = 252,
    rebalance_frequency: str = "M",
) -> dict:
    """Run backtest for multiple strategies and compare results.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Asset price history.
    strategies : dict[str, BaseAllocator]
        Dictionary of strategy name -> allocator instance.
    lookback_window : int
        Training window length in trading days.
    rebalance_frequency : str
        Rebalancing frequency.

    Returns
    -------
    dict
        Combined results with strategy_returns_df and performance_summary_df.
    """
    if not strategies:
        raise ValueError("strategies must not be empty")

    all_results = {}
    for strategy_name, allocator in strategies.items():
        backtester = RollingBacktester(
            allocator=allocator,
            train_window=lookback_window,
            rebalance_frequency=rebalance_frequency,
        )
        returns_df = prices_df.pct_change().dropna(how="all")
        results = backtester.run(returns_df)
        all_results[strategy_name] = results

    strategy_returns_dict = {}
    for strategy_name, results in all_results.items():
        strategy_returns_dict[strategy_name] = results["portfolio_returns"]

    strategy_returns_df = pd.DataFrame(strategy_returns_dict).dropna(how="all")

    performance_summary = {}
    for strategy_name, results in all_results.items():
        performance_summary[strategy_name] = results["performance_metrics"]

    performance_summary_df = pd.DataFrame(performance_summary).T

    return {
        "strategy_returns_df": strategy_returns_df,
        "performance_summary_df": performance_summary_df,
        "all_results": all_results,
    }


class RollingBacktester(BaseBacktester):
    """Run rolling rebalanced backtests for a single allocator."""

    def __init__(
        self,
        allocator: BaseAllocator | None = None,
        train_window: int = 252,
        rebalance_frequency: str = "M",
        initial_capital: float = 1_000_000.0,
        transaction_cost_calculator: TransactionCostCalculator | None = None,
    ):
        self.allocator = allocator or EqualWeightAllocator()
        self.train_window = train_window
        self.rebalance_frequency = rebalance_frequency
        self.initial_capital = initial_capital
        self.transaction_cost_calculator = (
            transaction_cost_calculator or TransactionCostCalculator()
        )

    def _rebalance_flags(self, index: pd.DatetimeIndex) -> np.ndarray:
        if normalize_rebalance_frequency(self.rebalance_frequency) == "D":
            return np.ones(len(index), dtype=bool)

        periods = index.to_period(rebalance_period_frequency(self.rebalance_frequency))
        flags = np.zeros(len(index), dtype=bool)
        flags[0] = True
        for i in range(1, len(index)):
            flags[i] = periods[i] != periods[i - 1]
        return flags

    def run(self, returns: pd.DataFrame) -> dict:
        if returns.empty:
            raise ValueError("returns must not be empty")
        if self.train_window < 20:
            raise ValueError("train_window must be >= 20")

        clean = returns.dropna(how="any").copy()
        if len(clean) <= self.train_window:
            raise ValueError("not enough rows for train_window")

        if not isinstance(clean.index, pd.DatetimeIndex):
            raise ValueError("returns index must be a DatetimeIndex")

        clean = clean.sort_index()
        assets = list(clean.columns)
        rebalance_flags = self._rebalance_flags(clean.index)

        current_weights = np.ones(len(assets)) / len(assets)
        portfolio_value = float(self.initial_capital)

        returns_idx: list[pd.Timestamp] = []
        portfolio_returns: list[float] = []
        values_idx: list[pd.Timestamp] = [clean.index[self.train_window - 1]]
        portfolio_values: list[float] = [portfolio_value]
        weights_records: list[np.ndarray] = []
        weights_dates: list[pd.Timestamp] = []
        total_cost = 0.0

        for t in range(self.train_window - 1, len(clean) - 1):
            if t == self.train_window - 1 or rebalance_flags[t]:
                train_slice = clean.iloc[t - self.train_window + 1 : t + 1]
                target_weights = self.allocator.optimize(train_slice)
                target_weights = np.asarray(target_weights, dtype=float)
                target_weights = np.clip(target_weights, 0.0, None)
                target_weights = target_weights / target_weights.sum()

                cost = self.transaction_cost_calculator.calculate_rebalancing_cost(
                    current_weights,
                    target_weights,
                    portfolio_value,
                )
                total_cost += cost
                portfolio_value = max(0.0, portfolio_value - cost)

                current_weights = target_weights
                weights_records.append(current_weights.copy())
                weights_dates.append(clean.index[t])

            next_ret = float(np.dot(current_weights, clean.iloc[t + 1].values))
            portfolio_value *= 1.0 + next_ret

            returns_idx.append(clean.index[t + 1])
            portfolio_returns.append(next_ret)
            values_idx.append(clean.index[t + 1])
            portfolio_values.append(portfolio_value)

        portfolio_returns_s = pd.Series(
            portfolio_returns,
            index=returns_idx,
            name="portfolio_return",
        )
        portfolio_values_s = pd.Series(portfolio_values, index=values_idx, name="portfolio_value")

        cumulative_max = portfolio_values_s.cummax()
        drawdown = (portfolio_values_s / cumulative_max) - 1.0

        performance_metrics = {
            "cumulative_return": PerformanceAnalytics.cumulative_return(portfolio_returns_s),
            "cagr": PerformanceAnalytics.annualized_return(portfolio_returns_s),
            "sharpe": PerformanceAnalytics.sharpe_ratio(portfolio_returns_s),
            "sortino": PerformanceAnalytics.sortino_ratio(portfolio_returns_s),
            "volatility": RiskAnalytics.volatility(portfolio_returns_s),
            "max_drawdown": float(drawdown.min()),
            "final_value": float(portfolio_values_s.iloc[-1]),
            "transaction_cost": float(total_cost),
        }

        weights_history = pd.DataFrame(weights_records, index=weights_dates, columns=assets)

        return {
            "portfolio_returns": portfolio_returns_s,
            "portfolio_values": portfolio_values_s,
            "drawdown": drawdown,
            "weights_history": weights_history,
            "performance_metrics": performance_metrics,
            "trades": len(weights_records),
        }
