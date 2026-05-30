"""Risk metric calculations for portfolio returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RiskAnalytics:
    """Calculate risk metrics for portfolio return series."""

    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
        if returns.empty:
            return 0.0
        return float(returns.quantile(1.0 - confidence_level))

    @staticmethod
    def conditional_value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
        if returns.empty:
            return 0.0
        var = RiskAnalytics.value_at_risk(returns, confidence_level)
        tail = returns[returns <= var]
        return float(tail.mean()) if not tail.empty else float(var)

    @staticmethod
    def maximum_drawdown(returns: pd.Series) -> float:
        if returns.empty:
            return 0.0
        cumulative = (1.0 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1.0
        return float(drawdown.min())

    @staticmethod
    def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        if returns.empty:
            return 0.0
        return float(returns.std(ddof=1) * np.sqrt(periods_per_year))

    @staticmethod
    def drawdown_series(returns: pd.Series) -> pd.Series:
        if returns.empty:
            return pd.Series(dtype=float)
        cumulative = (1.0 + returns).cumprod()
        running_max = cumulative.cummax()
        return cumulative / running_max - 1.0
