"""Benchmark-relative, drawdown-duration, and concentration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .performance_metrics import PerformanceAnalytics

VARIANCE_EPSILON = 1e-20


def calculate_simple_alpha(
    strategy_returns=None,
    benchmark_returns=None,
    *,
    strategy_cagr: float | None = None,
    benchmark_cagr: float | None = None,
    periods_per_year: int = 252,
) -> float:
    """Return strategy CAGR minus benchmark CAGR as a decimal."""
    if strategy_cagr is None or benchmark_cagr is None:
        aligned = _align_returns(strategy_returns, benchmark_returns)
        if aligned.empty:
            return float("nan")
        if strategy_cagr is None:
            strategy_cagr = PerformanceAnalytics.cagr(
                aligned["strategy"],
                periods_per_year=periods_per_year,
            )
        if benchmark_cagr is None:
            benchmark_cagr = PerformanceAnalytics.cagr(
                aligned["benchmark"],
                periods_per_year=periods_per_year,
            )

    strategy_value = _safe_float(strategy_cagr)
    benchmark_value = _safe_float(benchmark_cagr)
    if not np.isfinite(strategy_value) or not np.isfinite(benchmark_value):
        return float("nan")
    return float(strategy_value - benchmark_value)


def calculate_beta(strategy_returns, benchmark_returns) -> float:
    """Return beta of strategy daily returns against benchmark daily returns."""
    aligned = _align_returns(strategy_returns, benchmark_returns)
    if len(aligned) < 2:
        return float("nan")

    benchmark_variance = float(np.var(aligned["benchmark"].values, ddof=1))
    if benchmark_variance <= VARIANCE_EPSILON or not np.isfinite(benchmark_variance):
        return float("nan")

    covariance = float(np.cov(aligned["strategy"].values, aligned["benchmark"].values, ddof=1)[0, 1])
    return float(covariance / benchmark_variance)


def calculate_jensens_alpha(
    strategy_returns,
    benchmark_returns,
    *,
    annual_risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Estimate daily and annualized Jensen's alpha from a CAPM-style regression."""
    aligned = _align_returns(strategy_returns, benchmark_returns)
    empty_result = {
        "daily_jensen_alpha": float("nan"),
        "annualized_jensen_alpha": float("nan"),
    }
    if len(aligned) < 2:
        return empty_result

    daily_rf = float(annual_risk_free_rate) / float(periods_per_year)
    benchmark_excess = aligned["benchmark"].values - daily_rf
    strategy_excess = aligned["strategy"].values - daily_rf

    benchmark_variance = float(np.var(benchmark_excess, ddof=1))
    if benchmark_variance <= VARIANCE_EPSILON or not np.isfinite(benchmark_variance):
        return empty_result

    design_matrix = np.column_stack([np.ones(len(benchmark_excess)), benchmark_excess])
    try:
        alpha, _beta = np.linalg.lstsq(design_matrix, strategy_excess, rcond=None)[0]
    except np.linalg.LinAlgError:
        return empty_result

    if not np.isfinite(alpha):
        return empty_result

    daily_alpha = float(alpha)
    return {
        "daily_jensen_alpha": daily_alpha,
        "annualized_jensen_alpha": float(daily_alpha * periods_per_year),
    }


def calculate_tracking_error(
    strategy_returns,
    benchmark_returns,
    *,
    periods_per_year: int = 252,
) -> float:
    """Return annualized standard deviation of active daily returns."""
    aligned = _align_returns(strategy_returns, benchmark_returns)
    if len(aligned) < 2:
        return float("nan")

    active_returns = aligned["strategy"] - aligned["benchmark"]
    return float(active_returns.std(ddof=1) * np.sqrt(periods_per_year))


def calculate_information_ratio(
    strategy_returns,
    benchmark_returns,
    *,
    periods_per_year: int = 252,
) -> float:
    """Return annualized active return divided by annualized tracking error."""
    aligned = _align_returns(strategy_returns, benchmark_returns)
    if aligned.empty:
        return float("nan")

    tracking_error = calculate_tracking_error(
        aligned["strategy"],
        aligned["benchmark"],
        periods_per_year=periods_per_year,
    )
    if not np.isfinite(tracking_error) or tracking_error <= 1e-12:
        return float("nan")

    active_returns = aligned["strategy"] - aligned["benchmark"]
    annualized_active_return = float(active_returns.mean() * periods_per_year)
    return float(annualized_active_return / tracking_error)


def calculate_hit_ratio(strategy_returns, benchmark_returns) -> float:
    """Return the share of comparable days where strategy beats benchmark."""
    aligned = _align_returns(strategy_returns, benchmark_returns)
    if aligned.empty:
        return float("nan")
    return float((aligned["strategy"] > aligned["benchmark"]).mean())


def calculate_drawdown_durations(values) -> dict[str, float]:
    """Return max, current, and average drawdown duration in trading periods."""
    clean_values = _to_numeric_series(values, "portfolio_value").dropna()
    clean_values = clean_values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean_values.empty:
        return _empty_duration_metrics()

    running_peak = -np.inf
    current_duration = 0
    completed_durations: list[int] = []

    for value in clean_values.astype(float):
        if value >= running_peak:
            if current_duration > 0:
                completed_durations.append(current_duration)
                current_duration = 0
            running_peak = max(running_peak, float(value))
        else:
            current_duration += 1

    all_durations = completed_durations.copy()
    if current_duration > 0:
        all_durations.append(current_duration)

    if not all_durations:
        return _empty_duration_metrics()

    return {
        "max_drawdown_duration": int(max(all_durations)),
        "current_drawdown_duration": int(current_duration),
        "average_drawdown_duration": float(np.mean(all_durations)),
    }


def calculate_concentration_metrics(weights) -> dict[str, float]:
    """Return HHI/effective-N concentration metrics for weights or weight history."""
    if isinstance(weights, pd.DataFrame):
        return _calculate_concentration_history(weights)
    return _calculate_concentration_vector(weights)


def calculate_active_risk_metrics(
    strategy_returns,
    benchmark_returns,
    *,
    strategy_values=None,
    weights=None,
    strategy_cagr: float | None = None,
    benchmark_cagr: float | None = None,
    annual_risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Return a compact benchmark-relative diagnostic metric dictionary."""
    simple_alpha = calculate_simple_alpha(
        strategy_returns,
        benchmark_returns,
        strategy_cagr=strategy_cagr,
        benchmark_cagr=benchmark_cagr,
        periods_per_year=periods_per_year,
    )
    jensen = calculate_jensens_alpha(
        strategy_returns,
        benchmark_returns,
        annual_risk_free_rate=annual_risk_free_rate,
        periods_per_year=periods_per_year,
    )
    durations = calculate_drawdown_durations(
        strategy_values
        if strategy_values is not None
        else (1.0 + _to_numeric_series(strategy_returns, "strategy_returns")).cumprod()
    )
    concentration = (
        calculate_concentration_metrics(weights)
        if weights is not None
        else _empty_vector_concentration_metrics()
    )

    if "latest_hhi" in concentration:
        hhi = concentration["latest_hhi"]
        effective_n = concentration["latest_effective_n"]
        max_weight = concentration["latest_max_weight"]
        top_5_weight_sum = concentration["latest_top_5_weight_sum"]
        average_hhi = concentration["average_hhi"]
        average_effective_n = concentration["average_effective_n"]
    else:
        hhi = concentration["hhi"]
        effective_n = concentration["effective_n"]
        max_weight = concentration["max_weight"]
        top_5_weight_sum = concentration["top_5_weight_sum"]
        average_hhi = float("nan")
        average_effective_n = float("nan")

    return {
        "simple_alpha": simple_alpha,
        "jensen_alpha_daily": jensen["daily_jensen_alpha"],
        "jensen_alpha_annualized": jensen["annualized_jensen_alpha"],
        "beta": calculate_beta(strategy_returns, benchmark_returns),
        "tracking_error": calculate_tracking_error(
            strategy_returns,
            benchmark_returns,
            periods_per_year=periods_per_year,
        ),
        "information_ratio": calculate_information_ratio(
            strategy_returns,
            benchmark_returns,
            periods_per_year=periods_per_year,
        ),
        "hit_ratio": calculate_hit_ratio(strategy_returns, benchmark_returns),
        "max_drawdown_duration": durations["max_drawdown_duration"],
        "current_drawdown_duration": durations["current_drawdown_duration"],
        "average_drawdown_duration": durations["average_drawdown_duration"],
        "hhi": hhi,
        "effective_n": effective_n,
        "max_weight": max_weight,
        "top_5_weight_sum": top_5_weight_sum,
        "average_hhi": average_hhi,
        "average_effective_n": average_effective_n,
    }


def _align_returns(strategy_returns, benchmark_returns) -> pd.DataFrame:
    strategy = _to_numeric_series(strategy_returns, "strategy")
    benchmark = _to_numeric_series(benchmark_returns, "benchmark")
    aligned = pd.concat([strategy, benchmark], axis=1, join="inner")
    return aligned.dropna(how="any").astype(float)


def _to_numeric_series(values, name: str) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float, name=name)
    if isinstance(values, pd.Series):
        series = values.copy()
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"{name} must be a Series or single-column DataFrame")
        series = values.iloc[:, 0].copy()
    else:
        series = pd.Series(values)

    series = pd.to_numeric(series, errors="coerce")
    series.name = name
    return series.sort_index()


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _empty_duration_metrics() -> dict[str, float]:
    return {
        "max_drawdown_duration": 0,
        "current_drawdown_duration": 0,
        "average_drawdown_duration": 0.0,
    }


def _calculate_concentration_vector(weights) -> dict[str, float]:
    weight_series = _to_numeric_series(weights, "weight")
    weight_series = weight_series.replace([np.inf, -np.inf], np.nan).dropna()
    if weight_series.empty:
        return _empty_vector_concentration_metrics()

    total_weight = float(weight_series.sum())
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        return _empty_vector_concentration_metrics()

    normalized_weights = weight_series.astype(float) / total_weight
    hhi = float(np.sum(np.square(normalized_weights.values)))
    if hhi <= 0.0 or not np.isfinite(hhi):
        return _empty_vector_concentration_metrics()

    return {
        "hhi": hhi,
        "effective_n": float(1.0 / hhi),
        "max_weight": float(normalized_weights.max()),
        "top_5_weight_sum": float(normalized_weights.sort_values(ascending=False).head(5).sum()),
    }


def _calculate_concentration_history(weights_history: pd.DataFrame) -> dict[str, float]:
    if weights_history.empty:
        return _empty_history_concentration_metrics()

    metric_rows: list[dict[str, float]] = []
    for _, row in weights_history.dropna(how="all").iterrows():
        metrics = _calculate_concentration_vector(row)
        if np.isfinite(metrics["hhi"]):
            metric_rows.append(metrics)

    if not metric_rows:
        return _empty_history_concentration_metrics()

    latest = metric_rows[-1]
    hhi_values = [row["hhi"] for row in metric_rows]
    effective_n_values = [row["effective_n"] for row in metric_rows]

    return {
        "latest_hhi": latest["hhi"],
        "latest_effective_n": latest["effective_n"],
        "average_hhi": float(np.mean(hhi_values)),
        "average_effective_n": float(np.mean(effective_n_values)),
        "latest_max_weight": latest["max_weight"],
        "latest_top_5_weight_sum": latest["top_5_weight_sum"],
    }


def _empty_vector_concentration_metrics() -> dict[str, float]:
    return {
        "hhi": float("nan"),
        "effective_n": float("nan"),
        "max_weight": float("nan"),
        "top_5_weight_sum": float("nan"),
    }


def _empty_history_concentration_metrics() -> dict[str, float]:
    return {
        "latest_hhi": float("nan"),
        "latest_effective_n": float("nan"),
        "average_hhi": float("nan"),
        "average_effective_n": float("nan"),
        "latest_max_weight": float("nan"),
        "latest_top_5_weight_sum": float("nan"),
    }
