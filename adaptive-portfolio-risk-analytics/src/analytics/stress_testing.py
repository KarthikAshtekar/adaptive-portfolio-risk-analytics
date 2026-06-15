"""Stress-testing utilities for portfolio diagnostics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .active_risk_metrics import calculate_drawdown_durations
from .risk_metrics import RiskAnalytics
from .var_es import calculate_historical_es, calculate_historical_var


DEFAULT_STRESS_PERIODS = {
    "COVID Crash": ("2020-02-01", "2020-04-30"),
    "2022 Rate/Inflation Shock": ("2022-01-01", "2022-12-31"),
}

DEFAULT_HYPOTHETICAL_SCENARIOS = {
    "Equity Shock -10%": {"equity": -0.10, "gold": 0.00, "silver": 0.00, "defensive": 0.00},
    "Equity Shock -20%": {"equity": -0.20, "gold": 0.00, "silver": 0.00, "defensive": 0.00},
    "Equity Shock -30%": {"equity": -0.30, "gold": 0.00, "silver": 0.00, "defensive": 0.00},
    "Defensive Failure": {"equity": -0.20, "gold": -0.10, "silver": -0.15, "defensive": 0.00},
    "All Risk Assets Fall": {"equity": -0.20, "gold": -0.20, "silver": -0.20, "defensive": 0.00},
}


def calculate_historical_stress_performance(
    strategy_returns,
    strategy_values=None,
    stress_periods: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Evaluate return/risk metrics over named historical stress windows."""
    returns_s = _clean_returns(strategy_returns)
    values_s = _clean_values(strategy_values)
    periods = stress_periods or DEFAULT_STRESS_PERIODS
    rows = []

    for period_name, (start_date, end_date) in periods.items():
        period_returns = returns_s.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
        if period_returns.empty:
            rows.append(_empty_stress_row(period_name, start_date, end_date))
            continue

        period_values = values_s.loc[period_returns.index.min() : period_returns.index.max()]
        var_result = calculate_historical_var(period_returns, confidence_level=0.95)
        es_result = calculate_historical_es(period_returns, confidence_level=0.95)
        values_for_duration = (
            period_values
            if not period_values.empty
            else (1.0 + period_returns).cumprod()
        )
        durations = calculate_drawdown_durations(values_for_duration)

        rows.append(
            {
                "stress_period": period_name,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "period_return": float((1.0 + period_returns).prod() - 1.0),
                "max_drawdown": RiskAnalytics.maximum_drawdown(period_returns),
                "volatility": RiskAnalytics.volatility(period_returns),
                "var_95": var_result["var_return"],
                "es_95": es_result["es_return"],
                "max_drawdown_duration": durations["max_drawdown_duration"],
                "n_observations": int(len(period_returns)),
                "status": "ok",
            }
        )

    return pd.DataFrame(rows)


def find_worst_periods(
    returns,
    windows: tuple[int, ...] = (21, 63, 126),
) -> pd.DataFrame:
    """Find worst compounded return windows for selected trading-day lengths."""
    returns_s = _clean_returns(returns)
    rows = []
    for window in windows:
        if window <= 0 or len(returns_s) < window:
            rows.append(
                {
                    "window_days": int(window),
                    "start_date": None,
                    "end_date": None,
                    "period_return": float("nan"),
                    "max_drawdown": float("nan"),
                    "n_observations": 0,
                }
            )
            continue

        rolling_return = (1.0 + returns_s).rolling(window).apply(np.prod, raw=True) - 1.0
        end_date = rolling_return.idxmin()
        start_date = returns_s.index[returns_s.index.get_loc(end_date) - window + 1]
        period_returns = returns_s.loc[start_date:end_date]
        rows.append(
            {
                "window_days": int(window),
                "start_date": pd.Timestamp(start_date).date().isoformat(),
                "end_date": pd.Timestamp(end_date).date().isoformat(),
                "period_return": float(rolling_return.loc[end_date]),
                "max_drawdown": RiskAnalytics.maximum_drawdown(period_returns),
                "n_observations": int(len(period_returns)),
            }
        )

    return pd.DataFrame(rows)


def classify_asset_for_stress(ticker: str) -> str:
    """Map a Yahoo ticker to a simple stress bucket."""
    normalized = str(ticker).upper()
    if "GOLDBEES" in normalized:
        return "gold"
    if "SILVERBEES" in normalized:
        return "silver"
    if "LIQUID" in normalized or "SYNTHETIC RISK-FREE" in normalized:
        return "defensive"
    return "equity"


def apply_hypothetical_stress(weights, scenario_returns) -> float:
    """Apply a scenario return map to latest weights."""
    weight_series = _clean_weights(weights)
    if weight_series.empty:
        return float("nan")
    scenario_map = {str(key).lower(): float(value) for key, value in dict(scenario_returns).items()}

    stressed_returns = []
    for asset in weight_series.index:
        asset_key = str(asset).lower()
        bucket_key = classify_asset_for_stress(str(asset))
        stressed_returns.append(scenario_map.get(asset_key, scenario_map.get(bucket_key, 0.0)))

    return float(np.dot(weight_series.values, np.asarray(stressed_returns, dtype=float)))


def calculate_hypothetical_stress_table(
    strategy_weights: dict[str, pd.Series],
    benchmark_name: str,
    scenarios: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Build strategy-vs-benchmark hypothetical stress rows."""
    scenario_definitions = scenarios or DEFAULT_HYPOTHETICAL_SCENARIOS
    if not strategy_weights:
        return pd.DataFrame()

    benchmark_weights = strategy_weights.get(benchmark_name)
    rows = []
    for scenario_name, scenario_returns in scenario_definitions.items():
        benchmark_return = (
            apply_hypothetical_stress(benchmark_weights, scenario_returns)
            if benchmark_weights is not None
            else float("nan")
        )
        scenario_results = {}
        for strategy_name, weights in strategy_weights.items():
            stress_return = apply_hypothetical_stress(weights, scenario_returns)
            scenario_results[strategy_name] = stress_return
            rows.append(
                {
                    "scenario": scenario_name,
                    "strategy": strategy_name,
                    "strategy_stress_return": stress_return,
                    "benchmark_stress_return": benchmark_return,
                    "difference_vs_benchmark": stress_return - benchmark_return
                    if np.isfinite(benchmark_return) and np.isfinite(stress_return)
                    else float("nan"),
                }
            )

        finite_results = {key: value for key, value in scenario_results.items() if np.isfinite(value)}
        if finite_results:
            worst_strategy = min(finite_results, key=finite_results.get)
            most_defensive_strategy = max(finite_results, key=finite_results.get)
            for row in rows:
                if row["scenario"] == scenario_name:
                    row["worst_strategy_under_scenario"] = worst_strategy
                    row["most_defensive_strategy_under_scenario"] = most_defensive_strategy

    return pd.DataFrame(rows)


def calculate_correlation_stress(
    weights,
    returns_df: pd.DataFrame,
    stressed_correlation: float = 0.8,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Estimate volatility if off-diagonal risky-asset correlations rise."""
    weight_series = _clean_weights(weights)
    if weight_series.empty or returns_df.empty:
        return _empty_correlation_stress()

    common_assets = [asset for asset in weight_series.index if asset in returns_df.columns]
    if not common_assets:
        return _empty_correlation_stress()

    aligned_weights = weight_series.reindex(common_assets).fillna(0.0)
    if aligned_weights.sum() <= 0.0:
        return _empty_correlation_stress()
    aligned_weights = aligned_weights / aligned_weights.sum()
    clean_returns = returns_df[common_assets].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean_returns.empty:
        return _empty_correlation_stress()

    normal_covariance = clean_returns.cov().values
    normal_variance = float(aligned_weights.values.T @ normal_covariance @ aligned_weights.values)
    daily_vols = clean_returns.std(ddof=1).values
    n_assets = len(common_assets)
    stressed_corr = np.full((n_assets, n_assets), float(stressed_correlation))
    np.fill_diagonal(stressed_corr, 1.0)
    stressed_covariance = np.diag(daily_vols) @ stressed_corr @ np.diag(daily_vols)
    stressed_variance = float(
        aligned_weights.values.T @ stressed_covariance @ aligned_weights.values
    )

    normal_volatility = float(np.sqrt(max(normal_variance, 0.0)) * np.sqrt(periods_per_year))
    stressed_volatility = float(np.sqrt(max(stressed_variance, 0.0)) * np.sqrt(periods_per_year))
    return {
        "normal_volatility": normal_volatility,
        "correlation_stressed_volatility": stressed_volatility,
        "volatility_increase": stressed_volatility - normal_volatility,
        "stressed_correlation": float(stressed_correlation),
    }


def calculate_stress_period_benchmark_comparison(
    strategy_returns,
    benchmark_returns,
    stress_periods: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Compare a strategy with its benchmark during stress periods."""
    strategy_s = _clean_returns(strategy_returns)
    benchmark_s = _clean_returns(benchmark_returns)
    aligned = pd.concat([strategy_s, benchmark_s], axis=1, join="inner").dropna()
    aligned.columns = ["strategy", "benchmark"]
    periods = stress_periods or DEFAULT_STRESS_PERIODS
    rows = []

    for period_name, (start_date, end_date) in periods.items():
        period = aligned.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
        if period.empty:
            rows.append(
                {
                    "stress_period": period_name,
                    "strategy_stress_return": float("nan"),
                    "benchmark_stress_return": float("nan"),
                    "excess_stress_return": float("nan"),
                    "strategy_max_drawdown": float("nan"),
                    "benchmark_max_drawdown": float("nan"),
                    "drawdown_reduction": float("nan"),
                    "n_observations": 0,
                }
            )
            continue

        strategy_period_return = float((1.0 + period["strategy"]).prod() - 1.0)
        benchmark_period_return = float((1.0 + period["benchmark"]).prod() - 1.0)
        strategy_drawdown = RiskAnalytics.maximum_drawdown(period["strategy"])
        benchmark_drawdown = RiskAnalytics.maximum_drawdown(period["benchmark"])
        rows.append(
            {
                "stress_period": period_name,
                "strategy_stress_return": strategy_period_return,
                "benchmark_stress_return": benchmark_period_return,
                "excess_stress_return": strategy_period_return - benchmark_period_return,
                "strategy_max_drawdown": strategy_drawdown,
                "benchmark_max_drawdown": benchmark_drawdown,
                "drawdown_reduction": abs(benchmark_drawdown) - abs(strategy_drawdown),
                "n_observations": int(len(period)),
            }
        )

    return pd.DataFrame(rows)


class StressTestingFramework:
    """Simple stress scenarios for portfolio diagnostics."""

    @staticmethod
    def historical_scenario(
        portfolio_weights: np.ndarray,
        asset_returns: pd.DataFrame,
        scenario_date: pd.Timestamp,
    ) -> float:
        scenario_returns = asset_returns.loc[scenario_date]
        return float(np.dot(portfolio_weights, scenario_returns.values))

    @staticmethod
    def reverse_stress_test(
        portfolio_weights: np.ndarray,
        target_loss: float,
        asset_correlations: np.ndarray,
    ) -> Dict:
        n_assets = len(portfolio_weights)
        if n_assets == 0:
            return {"required_uniform_move": 0.0, "asset_moves": np.array([])}

        avg_corr = float(np.nanmean(asset_correlations))
        scaling = 1.0 + max(0.0, avg_corr)
        required_move = target_loss / (np.sum(np.abs(portfolio_weights)) * scaling)
        asset_moves = np.full(n_assets, required_move)
        return {
            "required_uniform_move": required_move,
            "asset_moves": asset_moves,
            "assumed_average_correlation": avg_corr,
        }

    @staticmethod
    def correlation_stress_test(
        portfolio_weights: np.ndarray,
        volatilities: np.ndarray,
        correlation_change: float = 0.2,
    ) -> float:
        n_assets = len(portfolio_weights)
        stressed_corr = np.full((n_assets, n_assets), correlation_change)
        np.fill_diagonal(stressed_corr, 1.0)
        stressed_cov = np.outer(volatilities, volatilities) * stressed_corr
        stressed_var = float(portfolio_weights.T @ stressed_cov @ portfolio_weights)
        return float(np.sqrt(max(stressed_var, 0.0)))


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
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()


def _clean_values(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.Series):
        series = values.copy()
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("values must be a Series or single-column DataFrame")
        series = values.iloc[:, 0].copy()
    else:
        series = pd.Series(values)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()


def _clean_weights(weights) -> pd.Series:
    if weights is None:
        return pd.Series(dtype=float)
    if isinstance(weights, pd.Series):
        series = weights.copy()
    else:
        series = pd.Series(weights)
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    series = series.clip(lower=0.0)
    total = float(series.sum())
    if total <= 0.0 or not np.isfinite(total):
        return pd.Series(dtype=float)
    return series / total


def _empty_stress_row(period_name: str, start_date: str, end_date: str) -> dict[str, object]:
    return {
        "stress_period": period_name,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "period_return": float("nan"),
        "max_drawdown": float("nan"),
        "volatility": float("nan"),
        "var_95": float("nan"),
        "es_95": float("nan"),
        "max_drawdown_duration": 0,
        "n_observations": 0,
        "status": "no_data",
    }


def _empty_correlation_stress() -> dict[str, float]:
    return {
        "normal_volatility": float("nan"),
        "correlation_stressed_volatility": float("nan"),
        "volatility_increase": float("nan"),
        "stressed_correlation": float("nan"),
    }
