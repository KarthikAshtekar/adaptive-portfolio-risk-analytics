"""Time-series-safe feature engineering for explainable market regimes."""

from __future__ import annotations

from bisect import bisect_right, insort
from itertools import combinations

import numpy as np
import pandas as pd

REQUIRED_FEATURE_COLUMNS = [
    "rolling_volatility",
    "volatility_percentile",
    "rolling_drawdown",
    "trend_126d",
    "momentum_63d",
    "average_correlation",
    "correlation_percentile",
    "benchmark_return_21d",
    "benchmark_volatility_63d",
]


def _validate_window(value: int, name: str) -> int:
    value = int(value)
    if value <= 1:
        raise ValueError(f"{name} must be greater than 1")
    return value


def _coerce_returns_frame(returns, prices=None) -> pd.DataFrame:
    if returns is None:
        if prices is None:
            return pd.DataFrame()
        if isinstance(prices, pd.Series):
            frame = prices.to_frame(name=prices.name or "asset").pct_change()
        elif isinstance(prices, pd.DataFrame):
            frame = prices.pct_change()
        else:
            raise TypeError("prices must be a pandas Series or DataFrame")
    elif isinstance(returns, pd.Series):
        frame = returns.to_frame(name=returns.name or "portfolio")
    elif isinstance(returns, pd.DataFrame):
        frame = returns.copy()
    else:
        raise TypeError("returns must be a pandas Series or DataFrame")

    if frame.empty:
        return frame
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a DatetimeIndex")

    frame = frame.apply(pd.to_numeric, errors="coerce").sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def _coerce_optional_series(values, index: pd.DatetimeIndex, name: str) -> pd.Series | None:
    if values is None:
        return None
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"{name} must be a Series or single-column DataFrame")
        series = values.iloc[:, 0].copy()
    elif isinstance(values, pd.Series):
        series = values.copy()
    else:
        raise TypeError(f"{name} must be a pandas Series or single-column DataFrame")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{name} index must be a DatetimeIndex")
    series = pd.to_numeric(series, errors="coerce").sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series.reindex(index)


def _rolling_compound(returns: pd.Series, window: int) -> pd.Series:
    return (1.0 + returns).rolling(window=window, min_periods=window).apply(
        np.prod,
        raw=True,
    ) - 1.0


def _expanding_percentile(values: pd.Series) -> pd.Series:
    """Rank each value against only the observations available through that date."""
    observed: list[float] = []
    percentiles: list[float] = []
    for value in values.to_numpy(dtype=float):
        if not np.isfinite(value):
            percentiles.append(np.nan)
            continue
        insort(observed, float(value))
        percentiles.append(float(bisect_right(observed, float(value)) / len(observed)))
    return pd.Series(percentiles, index=values.index, dtype=float)


def _average_rolling_correlation(returns: pd.DataFrame, window: int) -> pd.Series:
    valid_columns = [
        column for column in returns.columns if returns[column].notna().sum() >= window
    ]
    if len(valid_columns) < 2:
        return pd.Series(np.nan, index=returns.index, dtype=float)

    pairwise = [
        returns[left].rolling(window=window, min_periods=window).corr(returns[right])
        for left, right in combinations(valid_columns, 2)
    ]
    return pd.concat(pairwise, axis=1).mean(axis=1, skipna=True)


def calculate_regime_features(
    returns,
    benchmark_returns=None,
    prices=None,
    gold_returns=None,
    silver_returns=None,
    lookback_vol: int = 63,
    lookback_trend: int = 126,
    lookback_corr: int = 63,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Calculate explainable market-state features using historical data only.

    Features
    --------
    rolling_volatility
        Annualized rolling standard deviation of benchmark returns, or equal-
        weighted asset returns when no benchmark is supplied.
    volatility_percentile
        Expanding percentile rank of rolling volatility through the current date.
    rolling_drawdown
        Current drawdown from the running peak of the benchmark/equity wealth index.
    trend_126d
        Compounded benchmark/equity return over ``lookback_trend`` observations.
    momentum_63d
        Compounded benchmark/equity return over 63 observations.
    average_correlation
        Mean rolling pairwise correlation across available assets.
    correlation_percentile
        Expanding percentile rank of average correlation through the current date.
    benchmark_return_21d
        Compounded benchmark/equity return over 21 observations.
    benchmark_volatility_63d
        Annualized benchmark/equity volatility over 63 observations.
    gold_equity_spread_63d / silver_equity_spread_63d
        Optional metal return minus benchmark/equity return over 63 observations.
    """
    lookback_vol = _validate_window(lookback_vol, "lookback_vol")
    lookback_trend = _validate_window(lookback_trend, "lookback_trend")
    lookback_corr = _validate_window(lookback_corr, "lookback_corr")
    if int(periods_per_year) <= 0:
        raise ValueError("periods_per_year must be positive")

    returns_frame = _coerce_returns_frame(returns, prices=prices)
    if returns_frame.empty:
        return pd.DataFrame(columns=REQUIRED_FEATURE_COLUMNS)

    benchmark = _coerce_optional_series(
        benchmark_returns,
        returns_frame.index,
        "benchmark_returns",
    )
    equal_weight_returns = returns_frame.mean(axis=1, skipna=True).rename("equal_weight_return")
    equity_returns = (
        benchmark.combine_first(equal_weight_returns)
        if benchmark is not None
        else equal_weight_returns
    )

    rolling_volatility = equity_returns.rolling(window=lookback_vol, min_periods=lookback_vol).std(
        ddof=1
    ) * np.sqrt(int(periods_per_year))
    rolling_volatility.name = "rolling_volatility"

    wealth = (1.0 + equity_returns.fillna(0.0)).cumprod()
    rolling_drawdown = (wealth / wealth.cummax()) - 1.0
    rolling_drawdown.name = "rolling_drawdown"

    average_correlation = _average_rolling_correlation(returns_frame, lookback_corr)
    average_correlation.name = "average_correlation"

    features = pd.DataFrame(
        {
            "rolling_volatility": rolling_volatility,
            "volatility_percentile": _expanding_percentile(rolling_volatility),
            "rolling_drawdown": rolling_drawdown,
            "trend_126d": _rolling_compound(equity_returns, lookback_trend),
            "momentum_63d": _rolling_compound(equity_returns, 63),
            "average_correlation": average_correlation,
            "correlation_percentile": _expanding_percentile(average_correlation),
            "benchmark_return_21d": _rolling_compound(equity_returns, 21),
            "benchmark_volatility_63d": (
                equity_returns.rolling(window=63, min_periods=63).std(ddof=1)
                * np.sqrt(int(periods_per_year))
            ),
        },
        index=returns_frame.index,
    )

    equity_return_63d = _rolling_compound(equity_returns, 63)
    gold = _coerce_optional_series(gold_returns, returns_frame.index, "gold_returns")
    if gold is not None:
        features["gold_equity_spread_63d"] = _rolling_compound(gold, 63) - equity_return_63d

    silver = _coerce_optional_series(
        silver_returns,
        returns_frame.index,
        "silver_returns",
    )
    if silver is not None:
        features["silver_equity_spread_63d"] = _rolling_compound(silver, 63) - equity_return_63d

    return features
