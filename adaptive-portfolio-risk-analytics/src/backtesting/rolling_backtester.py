"""Rolling-window backtester with realistic turnover and trading frictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import PerformanceAnalytics
from src.optimization import BaseAllocator, EqualWeightAllocator

from .backtest_diagnostics import build_rebalance_summary, compare_cost_drag
from .base import BaseBacktester
from .rebalance_rules import (
    normalize_rebalance_frequency,
    should_rebalance_calendar,
    should_rebalance_threshold,
)
from .transaction_costs import TransactionCostCalculator, TransactionCostModel
from .turnover import summarize_turnover


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


def compare_strategies(
    prices_df: pd.DataFrame,
    strategies: dict[str, BaseAllocator],
    lookback_window: int = 252,
    rebalance_frequency: str = "M",
) -> dict:
    """Run backtest for multiple strategies and compare results."""
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
    """Run realistic rolling rebalanced backtests for a single allocator."""

    def __init__(
        self,
        allocator: BaseAllocator | None = None,
        train_window: int = 252,
        rebalance_frequency: str = "M",
        target_update_frequency: str = "M",
        initial_capital: float = 1_000_000.0,
        transaction_cost_calculator: TransactionCostCalculator | None = None,
        rebalance_mode: str = "calendar",
        threshold: float = 0.05,
        transaction_cost_model: TransactionCostModel | None = None,
        track_diagnostics: bool = True,
    ):
        self.allocator = allocator or EqualWeightAllocator()
        self.train_window = train_window
        self.rebalance_frequency = rebalance_frequency
        self.target_update_frequency = target_update_frequency
        self.initial_capital = initial_capital
        self.rebalance_mode = rebalance_mode
        self.threshold = threshold
        self.track_diagnostics = track_diagnostics
        self.transaction_cost_model = transaction_cost_model
        self.transaction_cost_calculator = (
            transaction_cost_calculator or TransactionCostCalculator()
        )

    @staticmethod
    def _normalize_weights(weights: np.ndarray) -> np.ndarray:
        normalized = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        total = normalized.sum()
        if total <= 0.0:
            raise ValueError("allocator returned weights with non-positive total")
        return normalized / total

    def _compute_target_weights(self, train_slice: pd.DataFrame) -> np.ndarray:
        target_weights = self.allocator.optimize(train_slice)
        return self._normalize_weights(np.asarray(target_weights, dtype=float))

    def _estimate_transaction_cost(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        portfolio_value: float,
        portfolio_volatility: float | None,
    ) -> float:
        turnover = float(0.5 * np.abs(target_weights - current_weights).sum())
        if self.transaction_cost_model is not None:
            return self.transaction_cost_model.estimate_cost(
                turnover=turnover,
                portfolio_value=portfolio_value,
                portfolio_volatility=portfolio_volatility,
            )
        return self.transaction_cost_calculator.calculate_rebalancing_cost(
            current_weights,
            target_weights,
            portfolio_value,
            portfolio_volatility=portfolio_volatility,
        )

    def _should_rebalance(
        self,
        *,
        current_date: pd.Timestamp,
        previous_rebalance_date: pd.Timestamp | None,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
    ) -> tuple[bool, str | None, float]:
        calendar_flag = should_rebalance_calendar(
            current_date=current_date,
            previous_rebalance_date=previous_rebalance_date,
            frequency=self.rebalance_frequency,
        )
        max_weight_drift = float(np.abs(current_weights - target_weights).max())
        threshold_flag = should_rebalance_threshold(
            current_weights,
            target_weights,
            threshold=self.threshold,
        )

        if self.rebalance_mode == "calendar":
            return calendar_flag, "calendar" if calendar_flag else None, max_weight_drift
        if self.rebalance_mode == "threshold":
            return threshold_flag, "threshold" if threshold_flag else None, max_weight_drift
        if self.rebalance_mode == "calendar_or_threshold":
            if calendar_flag:
                return (
                    True,
                    "calendar" if not threshold_flag else "calendar_or_threshold",
                    max_weight_drift,
                )
            if threshold_flag:
                return True, "threshold", max_weight_drift
            return False, None, max_weight_drift

        raise ValueError(
            "rebalance_mode must be one of: 'calendar', 'threshold', 'calendar_or_threshold'"
        )

    @staticmethod
    def _post_return_weights(
        current_weights: np.ndarray,
        asset_returns: np.ndarray,
        portfolio_return: float,
    ) -> np.ndarray:
        """Update weights after returns to reflect natural drift."""
        denominator = 1.0 + portfolio_return
        if denominator <= 0.0:
            return current_weights.copy()

        drifted = current_weights * (1.0 + asset_returns) / denominator
        drifted = np.clip(drifted, 0.0, None)
        total = drifted.sum()
        if total <= 0.0:
            return current_weights.copy()
        return drifted / total

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

        current_weights = np.ones(len(assets), dtype=float) / len(assets)
        target_weights = current_weights.copy()
        gross_portfolio_value = float(self.initial_capital)
        net_portfolio_value = float(self.initial_capital)
        previous_rebalance_date: pd.Timestamp | None = None
        previous_target_update_date: pd.Timestamp | None = None

        returns_idx: list[pd.Timestamp] = []
        net_portfolio_returns: list[float] = []
        gross_portfolio_returns: list[float] = []
        values_idx: list[pd.Timestamp] = [clean.index[self.train_window - 1]]
        gross_values: list[float] = [gross_portfolio_value]
        net_values: list[float] = [net_portfolio_value]
        weights_records: list[np.ndarray] = []
        weights_dates: list[pd.Timestamp] = []
        rebalance_records: list[dict[str, float | str | pd.Timestamp]] = []
        turnover_records: list[float] = []

        for t in range(self.train_window - 1, len(clean) - 1):
            current_date = clean.index[t]
            train_slice = clean.iloc[t - self.train_window + 1 : t + 1]

            should_update_target = should_rebalance_calendar(
                current_date=current_date,
                previous_rebalance_date=previous_target_update_date,
                frequency=self.target_update_frequency,
            )
            if previous_target_update_date is None or should_update_target:
                target_weights = self._compute_target_weights(train_slice)
                previous_target_update_date = current_date

            should_rebalance, rebalance_reason, max_weight_drift = self._should_rebalance(
                current_date=current_date,
                previous_rebalance_date=previous_rebalance_date,
                current_weights=current_weights,
                target_weights=target_weights,
            )

            if previous_rebalance_date is None:
                should_rebalance = True
                rebalance_reason = "initial"

            net_value_before_cost = net_portfolio_value
            if should_rebalance:
                portfolio_volatility = float(train_slice.std().mean())
                turnover = float(0.5 * np.abs(target_weights - current_weights).sum())
                transaction_cost = self._estimate_transaction_cost(
                    current_weights=current_weights,
                    target_weights=target_weights,
                    portfolio_value=net_value_before_cost,
                    portfolio_volatility=portfolio_volatility,
                )
                net_portfolio_value = max(0.0, net_portfolio_value - transaction_cost)

                current_weights = target_weights
                previous_rebalance_date = current_date
                weights_records.append(current_weights.copy())
                weights_dates.append(current_date)
                turnover_records.append(turnover)

                if self.track_diagnostics:
                    rebalance_records.append(
                        {
                            "rebalance_date": current_date,
                            "rebalance_reason": str(rebalance_reason),
                            "turnover": turnover,
                            "transaction_cost": transaction_cost,
                            "portfolio_value_before_cost": net_value_before_cost,
                            "portfolio_value_after_cost": net_portfolio_value,
                            "max_weight_drift": max_weight_drift,
                        }
                    )

            next_asset_returns = clean.iloc[t + 1].values.astype(float)
            gross_next_return = float(np.dot(current_weights, next_asset_returns))

            gross_portfolio_value *= 1.0 + gross_next_return
            net_value_after_return = net_portfolio_value * (1.0 + gross_next_return)
            net_next_return = (
                net_value_after_return / net_value_before_cost - 1.0
                if net_value_before_cost > 0.0
                else 0.0
            )
            net_portfolio_value = net_value_after_return

            current_weights = self._post_return_weights(
                current_weights=current_weights,
                asset_returns=next_asset_returns,
                portfolio_return=gross_next_return,
            )

            returns_idx.append(clean.index[t + 1])
            net_portfolio_returns.append(net_next_return)
            gross_portfolio_returns.append(gross_next_return)
            values_idx.append(clean.index[t + 1])
            gross_values.append(gross_portfolio_value)
            net_values.append(net_portfolio_value)

        portfolio_returns_s = pd.Series(
            net_portfolio_returns,
            index=returns_idx,
            name="portfolio_return",
        )
        gross_portfolio_returns_s = pd.Series(
            gross_portfolio_returns,
            index=returns_idx,
            name="gross_portfolio_return",
        )
        gross_portfolio_values_s = pd.Series(
            gross_values,
            index=values_idx,
            name="gross_portfolio_value",
        )
        portfolio_values_s = pd.Series(
            net_values,
            index=values_idx,
            name="portfolio_value",
        )

        cumulative_max = portfolio_values_s.cummax()
        drawdown = (portfolio_values_s / cumulative_max) - 1.0

        weights_history = pd.DataFrame(weights_records, index=weights_dates, columns=assets)
        rebalance_log_df = pd.DataFrame(rebalance_records)
        if not rebalance_log_df.empty:
            rebalance_log_df["rebalance_date"] = pd.to_datetime(rebalance_log_df["rebalance_date"])

        turnover_summary = summarize_turnover(
            pd.Series(turnover_records, index=weights_dates, dtype=float, name="turnover")
            if turnover_records
            else pd.Series(dtype=float, name="turnover")
        )
        rebalance_summary = build_rebalance_summary(rebalance_log_df)
        cost_drag_summary = compare_cost_drag(
            gross_portfolio_values=gross_portfolio_values_s,
            net_portfolio_values=portfolio_values_s,
        )

        summary_metrics = PerformanceAnalytics.summary_table(portfolio_returns_s)
        performance_metrics = {
            **summary_metrics,
            "max_drawdown": float(drawdown.min()),
            "final_value": float(portfolio_values_s.iloc[-1]),
            "transaction_cost": float(rebalance_summary["total_transaction_cost"]),
            "total_transaction_cost": float(rebalance_summary["total_transaction_cost"]),
            "total_turnover": float(turnover_summary["total_turnover"]),
            "average_turnover": float(turnover_summary["average_turnover"]),
            "number_of_rebalances": int(rebalance_summary["total_rebalances"]),
        }

        return {
            "portfolio_returns": portfolio_returns_s,
            "gross_portfolio_returns": gross_portfolio_returns_s,
            "portfolio_values": portfolio_values_s,
            "gross_portfolio_values": gross_portfolio_values_s,
            "drawdown": drawdown,
            "weights_history": weights_history,
            "performance_metrics": performance_metrics,
            "trades": len(weights_records),
            "rebalance_log": rebalance_log_df,
            "turnover_summary": turnover_summary,
            "rebalance_summary": rebalance_summary,
            "cost_drag_summary": cost_drag_summary,
        }
