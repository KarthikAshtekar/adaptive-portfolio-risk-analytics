"""Historical VaR, expected shortfall, and exception diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_historical_var(
    returns,
    confidence_level: float = 0.95,
    holding_period_days: int = 1,
    portfolio_value: float | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Calculate historical VaR as a positive loss return."""
    clean_returns = _clean_returns(returns)
    tail_probability = _tail_probability(confidence_level)
    result = {
        "confidence_level": float(confidence_level),
        "holding_period_days": int(holding_period_days),
        "var_return": float("nan"),
        "var_amount": float("nan"),
        "tail_probability": tail_probability,
    }
    if clean_returns.empty or holding_period_days <= 0 or periods_per_year <= 0:
        return result

    var_threshold = float(clean_returns.quantile(tail_probability))
    one_day_var = max(-var_threshold, 0.0)
    holding_period_var = float(one_day_var * np.sqrt(float(holding_period_days)))
    result["var_return"] = holding_period_var
    result["var_amount"] = _amount_from_return(holding_period_var, portfolio_value)
    return result


def calculate_historical_es(
    returns,
    confidence_level: float = 0.95,
    holding_period_days: int = 1,
    portfolio_value: float | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Calculate historical expected shortfall as a positive loss return."""
    clean_returns = _clean_returns(returns)
    tail_probability = _tail_probability(confidence_level)
    result = {
        "confidence_level": float(confidence_level),
        "holding_period_days": int(holding_period_days),
        "es_return": float("nan"),
        "es_amount": float("nan"),
        "tail_probability": tail_probability,
    }
    if clean_returns.empty or holding_period_days <= 0 or periods_per_year <= 0:
        return result

    var_threshold = float(clean_returns.quantile(tail_probability))
    tail_returns = clean_returns[clean_returns <= var_threshold]
    if tail_returns.empty:
        return result

    one_day_es = max(-float(tail_returns.mean()), 0.0)
    holding_period_es = float(one_day_es * np.sqrt(float(holding_period_days)))
    result["es_return"] = holding_period_es
    result["es_amount"] = _amount_from_return(holding_period_es, portfolio_value)
    return result


def calculate_var_exceptions(
    returns,
    confidence_level: float = 0.95,
    var_threshold: float | None = None,
    rolling_window: int | None = None,
) -> dict[str, float]:
    """Count VaR exceptions using static or lagged rolling historical VaR."""
    clean_returns = _clean_returns(returns)
    tail_probability = _tail_probability(confidence_level)
    empty_result = {
        "actual_exceptions": 0,
        "expected_exceptions": float("nan"),
        "exception_ratio": float("nan"),
        "exception_rate": float("nan"),
        "expected_exception_rate": tail_probability,
        "n_observations": 0,
    }
    if clean_returns.empty:
        return empty_result

    if rolling_window is not None:
        if rolling_window < 2 or len(clean_returns) <= rolling_window:
            return empty_result
        rolling_threshold = clean_returns.rolling(rolling_window).quantile(tail_probability).shift(1)
        comparable = pd.DataFrame(
            {
                "return": clean_returns,
                "threshold": rolling_threshold,
            }
        ).dropna()
        if comparable.empty:
            return empty_result
        breaches = comparable["return"] < comparable["threshold"]
        n_observations = int(len(comparable))
    else:
        signed_threshold = (
            _signed_var_threshold(float(var_threshold))
            if var_threshold is not None
            else float(clean_returns.quantile(tail_probability))
        )
        breaches = clean_returns < signed_threshold
        n_observations = int(len(clean_returns))

    actual_exceptions = int(breaches.sum())
    expected_exceptions = float(n_observations * tail_probability)
    exception_rate = float(actual_exceptions / n_observations) if n_observations > 0 else float("nan")
    exception_ratio = (
        float(actual_exceptions / expected_exceptions)
        if expected_exceptions > 0.0
        else float("nan")
    )

    return {
        "actual_exceptions": actual_exceptions,
        "expected_exceptions": expected_exceptions,
        "exception_ratio": exception_ratio,
        "exception_rate": exception_rate,
        "expected_exception_rate": tail_probability,
        "n_observations": n_observations,
    }


def _clean_returns(returns) -> pd.Series:
    if returns is None:
        return pd.Series(dtype=float)
    if isinstance(returns, pd.Series):
        series = returns.copy()
    elif isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise ValueError("returns must be a Series or single-column DataFrame")
        series = returns.iloc[:, 0].copy()
    else:
        series = pd.Series(returns)

    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    return series.sort_index()


def _tail_probability(confidence_level: float) -> float:
    confidence = float(confidence_level)
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    return 1.0 - confidence


def _amount_from_return(loss_return: float, portfolio_value: float | None) -> float:
    if portfolio_value is None:
        return float("nan")
    value = float(portfolio_value)
    if not np.isfinite(value):
        return float("nan")
    return float(value * loss_return)


def _signed_var_threshold(var_threshold: float) -> float:
    if not np.isfinite(var_threshold):
        return float("nan")
    return -var_threshold if var_threshold >= 0.0 else var_threshold
