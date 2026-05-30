"""Rolling-window backtester for Phase 1 portfolio strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import PerformanceAnalytics, RiskAnalytics
from src.optimization import BaseAllocator, EqualWeightAllocator

from .base import BaseBacktester
from .transaction_costs import TransactionCostCalculator


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
        if self.rebalance_frequency.upper() == "D":
            return np.ones(len(index), dtype=bool)

        periods = index.to_period(self.rebalance_frequency)
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
