"""Performance and transition analytics for market regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import PerformanceAnalytics, RiskAnalytics

REGIME_ORDER = ["Calm", "Normal", "Stress", "Crisis", "Unknown"]

OBJECTIVE_COLUMNS = {
    "cagr": "strategy_cagr",
    "sharpe": "strategy_sharpe",
    "sortino": "strategy_sortino",
    "calmar": "strategy_calmar",
    "max_drawdown": "strategy_max_drawdown",
    "final_value": "strategy_final_value",
    "volatility": "strategy_volatility",
}

LOWER_IS_BETTER = {"volatility"}


def _coerce_series(values, name: str) -> pd.Series:
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
    series = series.sort_index()
    return series[~series.index.duplicated(keep="last")]


def _ordered_regimes(values: pd.Series) -> list[str]:
    observed = [str(value) for value in values.dropna().unique()]
    return [regime for regime in REGIME_ORDER if regime in observed] + sorted(
        set(observed) - set(REGIME_ORDER)
    )


def calculate_regime_performance(
    strategy_returns,
    regimes,
    benchmark_returns=None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Calculate strategy and optional benchmark metrics within each regime."""
    strategy = _coerce_series(strategy_returns, "strategy_returns")
    regime_series = _coerce_series(regimes, "regimes").astype("object")
    aligned = pd.concat(
        [strategy.rename("strategy_return"), regime_series.rename("regime")],
        axis=1,
        join="inner",
    ).dropna(subset=["strategy_return", "regime"])

    benchmark = (
        _coerce_series(benchmark_returns, "benchmark_returns")
        if benchmark_returns is not None
        else None
    )
    rows: list[dict[str, object]] = []

    for regime in _ordered_regimes(aligned["regime"]):
        regime_rows = aligned[aligned["regime"].astype(str).eq(regime)]
        returns = regime_rows["strategy_return"].astype(float)
        strategy_cagr = PerformanceAnalytics.cagr(returns, periods_per_year)
        strategy_max_drawdown = RiskAnalytics.maximum_drawdown(returns)
        strategy_metrics = {
            "strategy_cagr": strategy_cagr,
            "strategy_volatility": RiskAnalytics.volatility(returns, periods_per_year),
            "strategy_sharpe": PerformanceAnalytics.sharpe_ratio(
                returns,
                periods_per_year=periods_per_year,
            ),
            "strategy_sortino": PerformanceAnalytics.sortino_ratio(
                returns,
                periods_per_year=periods_per_year,
            ),
            "strategy_max_drawdown": strategy_max_drawdown,
            "strategy_calmar": PerformanceAnalytics.calmar_ratio(
                returns,
                periods_per_year=periods_per_year,
            ),
            "strategy_final_value": float((1.0 + returns).prod()),
        }

        benchmark_metrics = {
            "benchmark_cagr": np.nan,
            "benchmark_volatility": np.nan,
            "benchmark_max_drawdown": np.nan,
            "excess_return": np.nan,
            "hit_ratio_vs_benchmark": np.nan,
        }
        if benchmark is not None:
            comparison = pd.concat(
                [
                    returns.rename("strategy_return"),
                    benchmark.rename("benchmark_return"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            if not comparison.empty:
                benchmark_regime_returns = comparison["benchmark_return"]
                benchmark_cagr = PerformanceAnalytics.cagr(
                    benchmark_regime_returns,
                    periods_per_year,
                )
                benchmark_metrics = {
                    "benchmark_cagr": benchmark_cagr,
                    "benchmark_volatility": RiskAnalytics.volatility(
                        benchmark_regime_returns,
                        periods_per_year,
                    ),
                    "benchmark_max_drawdown": RiskAnalytics.maximum_drawdown(
                        benchmark_regime_returns
                    ),
                    "excess_return": strategy_cagr - benchmark_cagr,
                    "hit_ratio_vs_benchmark": float(
                        (comparison["strategy_return"] > comparison["benchmark_return"]).mean()
                    ),
                }

        rows.append(
            {
                "regime": regime,
                "number_of_days": int(len(returns)),
                **strategy_metrics,
                **benchmark_metrics,
            }
        )

    return pd.DataFrame(rows)


def select_best_strategy_by_regime(
    performance: pd.DataFrame,
    objective: str | None = None,
) -> dict[str, object]:
    """Select the best strategy per regime with a transparent metric fallback."""
    requested = str(objective or "sharpe").strip().lower().replace(" ", "_")
    candidates = [requested, "sharpe", "calmar"]
    selected = next(
        (
            metric
            for metric in candidates
            if OBJECTIVE_COLUMNS.get(metric) in performance.columns
            and pd.to_numeric(
                performance[OBJECTIVE_COLUMNS[metric]],
                errors="coerce",
            )
            .notna()
            .any()
        ),
        None,
    )
    if selected is None or performance.empty:
        return {
            "table": pd.DataFrame(),
            "requested_objective": requested,
            "objective": None,
            "fallback_used": True,
        }

    metric_column = OBJECTIVE_COLUMNS[selected]
    rows: list[dict[str, object]] = []
    for regime, group in performance.groupby("regime", sort=False):
        values = pd.to_numeric(group[metric_column], errors="coerce")
        valid = group.loc[values.notna()].copy()
        if valid.empty:
            continue
        valid_values = pd.to_numeric(valid[metric_column], errors="coerce")
        best_index = valid_values.idxmin() if selected in LOWER_IS_BETTER else valid_values.idxmax()
        best = valid.loc[best_index]
        rows.append(
            {
                "regime": regime,
                "best_strategy": best["strategy"],
                "objective": selected,
                "objective_value": float(best[metric_column]),
            }
        )

    return {
        "table": pd.DataFrame(rows),
        "requested_objective": requested,
        "objective": selected,
        "fallback_used": selected != requested,
    }


def calculate_strategy_regime_summary(
    strategy_returns_dict,
    regimes,
    benchmark_returns=None,
    objective: str | None = None,
) -> dict[str, object]:
    """Build multi-strategy performance, best-strategy, and distribution tables."""
    if not isinstance(strategy_returns_dict, dict):
        raise TypeError("strategy_returns_dict must be a dictionary")

    performance_parts: list[pd.DataFrame] = []
    for strategy_name, returns in strategy_returns_dict.items():
        performance = calculate_regime_performance(
            returns,
            regimes,
            benchmark_returns=benchmark_returns,
        )
        if performance.empty:
            continue
        performance.insert(0, "strategy", strategy_name)
        performance_parts.append(performance)

    performance_table = (
        pd.concat(performance_parts, ignore_index=True) if performance_parts else pd.DataFrame()
    )
    regime_series = _coerce_series(regimes, "regimes").dropna().astype(str)
    counts = regime_series.value_counts()
    distribution = pd.DataFrame(
        {
            "regime": counts.index,
            "number_of_days": counts.values,
            "percentage_of_days": counts.values / max(int(counts.sum()), 1),
        }
    )
    if not distribution.empty:
        distribution["_order"] = distribution["regime"].map(
            {regime: position for position, regime in enumerate(REGIME_ORDER)}
        )
        distribution = (
            distribution.sort_values(["_order", "regime"], na_position="last")
            .drop(columns="_order")
            .reset_index(drop=True)
        )

    selection = select_best_strategy_by_regime(performance_table, objective)
    return {
        "performance": performance_table,
        "best_strategy_by_regime": selection["table"],
        "regime_distribution": distribution,
        "requested_objective": selection["requested_objective"],
        "objective": selection["objective"],
        "fallback_used": selection["fallback_used"],
    }


def calculate_regime_transitions(regimes) -> dict[str, object]:
    """Calculate transition counts/probabilities and contiguous regime durations."""
    regime_series = _coerce_series(regimes, "regimes").dropna().astype(str)
    regime_series = regime_series[~regime_series.eq("Unknown")]
    if regime_series.empty:
        return {
            "transition_count_matrix": pd.DataFrame(),
            "transition_probability_matrix": pd.DataFrame(),
            "average_duration": pd.DataFrame(columns=["regime", "average_duration"]),
            "current_regime": "Unknown",
            "current_regime_duration": 0,
        }

    labels = _ordered_regimes(regime_series)
    previous = regime_series.shift(1)
    transition_pairs = pd.DataFrame({"from_regime": previous, "to_regime": regime_series}).dropna()
    count_matrix = pd.crosstab(
        transition_pairs["from_regime"],
        transition_pairs["to_regime"],
    ).reindex(index=labels, columns=labels, fill_value=0)
    probability_matrix = count_matrix.div(
        count_matrix.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0)

    run_id = regime_series.ne(regime_series.shift()).cumsum()
    runs = regime_series.groupby(run_id).agg(regime="first", duration="size").reset_index(drop=True)
    average_duration = (
        runs.groupby("regime", sort=False)["duration"]
        .mean()
        .reindex(labels)
        .dropna()
        .rename("average_duration")
        .reset_index()
    )

    current_regime = str(regime_series.iloc[-1])
    current_regime_duration = int(runs.iloc[-1]["duration"])
    return {
        "transition_count_matrix": count_matrix,
        "transition_probability_matrix": probability_matrix,
        "average_duration": average_duration,
        "current_regime": current_regime,
        "current_regime_duration": current_regime_duration,
    }
