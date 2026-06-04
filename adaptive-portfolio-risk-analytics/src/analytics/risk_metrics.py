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

    @staticmethod
    def rolling_volatility(returns: pd.Series, window: int = 30, periods_per_year: int = 252) -> pd.Series:
        """Calculate rolling volatility."""
        if returns.empty or len(returns) < window:
            return pd.Series(dtype=float)
        return returns.rolling(window=window).std(ddof=1) * np.sqrt(periods_per_year)

    @staticmethod
    def rolling_sharpe(
        returns: pd.Series,
        window: int = 30,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        if returns.empty or len(returns) < window:
            return pd.Series(dtype=float)
        excess_returns = returns - risk_free_rate / periods_per_year
        mean_excess = excess_returns.rolling(window=window).mean()
        std_excess = excess_returns.rolling(window=window).std(ddof=1)
        return (mean_excess / std_excess) * np.sqrt(periods_per_year)

    @staticmethod
    def downside_deviation(returns: pd.Series, target_return: float = 0.0, periods_per_year: int = 252) -> float:
        """Calculate annualized downside deviation (for Sortino ratio calculation)."""
        if returns.empty:
            return 0.0
        excess = returns - target_return / periods_per_year
        downside = excess[excess < 0]
        if downside.empty:
            return 0.0
        return float(np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year))

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Alias for maximum_drawdown."""
        return RiskAnalytics.maximum_drawdown(returns)
