"""Risk metric calculations for portfolio returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_numeric_series(values) -> pd.Series:
    """Return a finite numeric Series while preserving the original index."""
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.dropna().astype(float)


def _looks_like_return_series(values: pd.Series) -> bool:
    """Heuristic for accepting either returns or portfolio values.

    Portfolio value paths are strictly positive and usually start near 1.0 or a
    capital value. Return series are usually small and may include negatives.
    The public helper accepts both because older notebooks sometimes pass
    normalized wealth paths directly.
    """
    if values.empty:
        return False
    if (values <= 0.0).any():
        return True
    first = float(values.iloc[0])
    if np.isclose(first, 1.0, atol=1e-8):
        return False
    return bool(values.abs().max() < 0.5)


def compute_drawdown_series(portfolio_values_or_returns) -> pd.Series:
    """Compute drawdown from a portfolio value path or simple-return series.

    Drawdown is defined as ``PortfolioValue_t / RunningPeak_t - 1``. Empty
    inputs return an empty float Series.
    """
    values = _clean_numeric_series(portfolio_values_or_returns)
    if values.empty:
        return pd.Series(dtype=float)

    if _looks_like_return_series(values):
        wealth = (1.0 + values).cumprod()
        anchor_index = (
            values.index[0] - pd.Timedelta(nanoseconds=1)
            if isinstance(values.index, pd.DatetimeIndex)
            else -1
        )
        anchored = pd.concat([pd.Series([1.0], index=[anchor_index]), wealth])
        running_peak = anchored.cummax().iloc[1:]
    else:
        wealth = values.copy()
        running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    drawdown.name = "drawdown"
    return drawdown.astype(float)


def compute_pain_index(portfolio_returns, initial_value: float = 1.0) -> float:
    """Return the mean absolute drawdown over the return path."""
    returns = _clean_numeric_series(portfolio_returns)
    if returns.empty:
        return 0.0
    if float(initial_value) <= 0.0:
        raise ValueError("initial_value must be positive")

    values = (1.0 + returns).cumprod() * float(initial_value)
    if isinstance(values.index, pd.DatetimeIndex):
        anchor_index = values.index[0] - pd.Timedelta(nanoseconds=1)
    else:
        anchor_index = -1
    anchored = pd.concat([pd.Series([float(initial_value)], index=[anchor_index]), values])
    running_peak = anchored.cummax().iloc[1:]
    drawdown = values / running_peak - 1.0
    return float(drawdown.abs().mean())


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
        return compute_drawdown_series(returns)

    @staticmethod
    def compute_drawdown_series(portfolio_values_or_returns) -> pd.Series:
        """Alias for the module-level drawdown helper."""
        return compute_drawdown_series(portfolio_values_or_returns)

    @staticmethod
    def pain_index(returns: pd.Series, initial_value: float = 1.0) -> float:
        """Return the mean absolute drawdown over the return path."""
        return compute_pain_index(returns, initial_value=initial_value)

    @staticmethod
    def rolling_volatility(
        returns: pd.Series, window: int = 30, periods_per_year: int = 252
    ) -> pd.Series:
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
    def downside_deviation(
        returns: pd.Series, target_return: float = 0.0, periods_per_year: int = 252
    ) -> float:
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
