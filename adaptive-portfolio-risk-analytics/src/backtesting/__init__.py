"""Backtesting exports."""

from .backtest_diagnostics import build_rebalance_summary, compare_cost_drag
from .base import BaseBacktester
from .rebalance_rules import (
    normalize_rebalance_frequency,
    should_rebalance_calendar,
    should_rebalance_threshold,
)
from .rolling_backtester import (
    RollingBacktester,
    compare_strategies,
    generate_rebalance_dates,
)
from .transaction_costs import TransactionCostCalculator, TransactionCostModel
from .turnover import calculate_turnover, calculate_turnover_series, summarize_turnover
from .cpcv import CPCVBacktester

# Backward-compatible aliases.
BacktestEngine = BaseBacktester
RollingBacktest = RollingBacktester
CPCVValidator = CPCVBacktester

__all__ = [
    "BaseBacktester",
    "RollingBacktester",
    "TransactionCostCalculator",
    "TransactionCostModel",
    "CPCVBacktester",
    "generate_rebalance_dates",
    "normalize_rebalance_frequency",
    "should_rebalance_calendar",
    "should_rebalance_threshold",
    "calculate_turnover",
    "calculate_turnover_series",
    "summarize_turnover",
    "build_rebalance_summary",
    "compare_cost_drag",
    "compare_strategies",
    "BacktestEngine",
    "RollingBacktest",
    "CPCVValidator",
]
