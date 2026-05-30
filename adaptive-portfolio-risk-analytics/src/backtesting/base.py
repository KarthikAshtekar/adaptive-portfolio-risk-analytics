"""Base backtesting interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseBacktester(ABC):
    """Abstract interface for backtesting engines."""

    @abstractmethod
    def run(self, returns: pd.DataFrame) -> dict:
        """Run backtest and return result dictionary."""
