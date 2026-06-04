"""Backtesting exports."""

from .base import BaseBacktester
from .rolling_backtester import RollingBacktester, generate_rebalance_dates, compare_strategies
from .transaction_costs import TransactionCostCalculator
from .cpcv import CPCVBacktester

# Backward-compatible aliases.
BacktestEngine = BaseBacktester
RollingBacktest = RollingBacktester
CPCVValidator = CPCVBacktester

__all__ = [
    "BaseBacktester",
    "RollingBacktester",
    "TransactionCostCalculator",
    "CPCVBacktester",
    "generate_rebalance_dates",
    "compare_strategies",
    "BacktestEngine",
    "RollingBacktest",
    "CPCVValidator",
]
