"""Performance metric calculations for portfolio returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .risk_metrics import RiskAnalytics


class PerformanceAnalytics:
    """Calculate return and risk-adjusted performance metrics."""

    @staticmethod
    def cumulative_return(returns: pd.Series) -> float:
        if returns.empty:
            return 0.0
        return float((1.0 + returns).prod() - 1.0)

    @staticmethod
    def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
        if returns.empty:
            return 0.0
        n_periods = len(returns)
        total_return = float((1.0 + returns).prod())
        return float(total_return ** (periods_per_year / n_periods) - 1.0)

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> float:
        if returns.empty:
            return 0.0
        excess_returns = returns - risk_free_rate / periods_per_year
        std = float(excess_returns.std(ddof=1))
        if std <= 0:
            return 0.0
        return float((excess_returns.mean() / std) * np.sqrt(periods_per_year))

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        target_return: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        if returns.empty:
            return 0.0
        excess = returns - target_return / periods_per_year
        downside = excess[excess < 0]
        if downside.empty:
            return float("inf")
        downside_dev = float(np.sqrt((downside**2).mean()))
        if downside_dev <= 0:
            return 0.0
        return float((excess.mean() / downside_dev) * np.sqrt(periods_per_year))

    @staticmethod
    def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        if returns.empty:
            return 0.0
        annual_return = PerformanceAnalytics.annualized_return(returns, periods_per_year)
        max_dd = RiskAnalytics.maximum_drawdown(returns)
        if max_dd >= 0:
            return float("inf")
        return float(annual_return / abs(max_dd))

    @staticmethod
    def summary_table(returns: pd.Series, risk_free_rate: float = 0.02) -> dict[str, float]:
        return {
            "cumulative_return": PerformanceAnalytics.cumulative_return(returns),
            "cagr": PerformanceAnalytics.annualized_return(returns),
            "sharpe": PerformanceAnalytics.sharpe_ratio(returns, risk_free_rate=risk_free_rate),
            "sortino": PerformanceAnalytics.sortino_ratio(returns),
            "volatility": RiskAnalytics.volatility(returns),
            "max_drawdown": RiskAnalytics.maximum_drawdown(returns),
            "var_95": RiskAnalytics.value_at_risk(returns),
            "cvar_95": RiskAnalytics.conditional_value_at_risk(returns),
        }
