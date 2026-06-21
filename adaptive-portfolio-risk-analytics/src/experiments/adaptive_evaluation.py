"""Phase 3D adaptive-vs-fixed and stress-period evaluation helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.analytics import PerformanceAnalytics, RiskAnalytics, find_worst_periods

OBJECTIVE_DIRECTIONS = {
    "cagr": True,
    "sharpe": True,
    "sortino": True,
    "calmar": True,
    "max_drawdown": True,
    "final_value": True,
    "volatility": False,
}


def compare_adaptive_vs_fixed(
    adaptive_results,
    fixed_strategy_results,
    benchmark_results=None,
    objective: str | None = None,
) -> dict[str, object]:
    """Compare the best adaptive and fixed strategies without assuming a winner."""
    selected_objective = _normalize_objective(objective)
    adaptive_table = _coerce_metric_table(adaptive_results, "regime_adaptive")
    fixed_table = _coerce_metric_table(fixed_strategy_results, "fixed")
    if adaptive_table.empty:
        return _empty_comparison("No successful adaptive strategy result is available.")
    if fixed_table.empty:
        return _empty_comparison("No successful fixed strategy result is available.")

    best_adaptive = _select_best(adaptive_table, selected_objective)
    best_fixed = _select_best(fixed_table, selected_objective)
    result = {
        "best_adaptive_strategy": str(best_adaptive["strategy"]),
        "best_fixed_strategy": str(best_fixed["strategy"]),
        "selected_objective": selected_objective,
        "adaptive_objective_value": _number(best_adaptive.get(selected_objective)),
        "fixed_objective_value": _number(best_fixed.get(selected_objective)),
        "adaptive_minus_fixed_CAGR": _delta(best_adaptive, best_fixed, "cagr"),
        "adaptive_minus_fixed_Sharpe": _delta(best_adaptive, best_fixed, "sharpe"),
        "adaptive_minus_fixed_Calmar": _delta(best_adaptive, best_fixed, "calmar"),
        "adaptive_minus_fixed_MaxDrawdown": _delta(
            best_adaptive,
            best_fixed,
            "max_drawdown",
        ),
        "adaptive_minus_fixed_FinalValue": _delta(
            best_adaptive,
            best_fixed,
            "final_value",
        ),
        "adaptive_turnover_penalty": _delta(
            best_adaptive,
            best_fixed,
            "total_turnover",
        ),
        "adaptive_cost_penalty": _delta(
            best_adaptive,
            best_fixed,
            "total_transaction_cost",
        ),
        "adaptive_stress_advantage": _delta(
            best_adaptive,
            best_fixed,
            "stress_period_return",
        ),
        "adaptive_cpcv_advantage": _delta(
            best_adaptive,
            best_fixed,
            "cpcv_median_objective",
        ),
    }
    result["interpretation"] = _comparison_interpretation(
        result,
        higher_is_better=OBJECTIVE_DIRECTIONS[selected_objective],
    )
    _ = benchmark_results
    return result


def build_adaptive_stress_comparison(
    adaptive_result: Mapping[str, object],
    fixed_strategy_results: Mapping[str, Mapping[str, object]],
    benchmark_name: str,
    objective: str | None = None,
) -> pd.DataFrame:
    """Compare adaptive, best fixed, and benchmark returns over common stress windows."""
    if not isinstance(adaptive_result, Mapping) or "portfolio_returns" not in adaptive_result:
        return pd.DataFrame()
    if not isinstance(fixed_strategy_results, Mapping) or not fixed_strategy_results:
        return pd.DataFrame()

    fixed_table = _coerce_metric_table(fixed_strategy_results, "fixed")
    if fixed_table.empty:
        return pd.DataFrame()
    selected_objective = _normalize_objective(objective)
    best_fixed_name = str(_select_best(fixed_table, selected_objective)["strategy"])
    benchmark_name = benchmark_name if benchmark_name in fixed_strategy_results else best_fixed_name

    adaptive_returns = _returns_from_result(adaptive_result)
    fixed_returns = _returns_from_result(fixed_strategy_results[best_fixed_name])
    benchmark_returns = _returns_from_result(fixed_strategy_results[benchmark_name])
    periods = {
        "COVID Crash": (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30")),
        "2022 Rate/Inflation Shock": (
            pd.Timestamp("2022-01-01"),
            pd.Timestamp("2022-12-31"),
        ),
    }
    worst_periods = find_worst_periods(benchmark_returns, windows=(21, 63, 126))
    labels = {21: "Worst 1-month", 63: "Worst 3-month", 126: "Worst 6-month"}
    for _, row in worst_periods.iterrows():
        if row.get("start_date") and row.get("end_date"):
            periods[labels[int(row["window_days"])]] = (
                pd.Timestamp(row["start_date"]),
                pd.Timestamp(row["end_date"]),
            )

    rows: list[dict[str, object]] = []
    for period_name, (start_date, end_date) in periods.items():
        adaptive_metrics = _period_metrics(adaptive_returns, start_date, end_date)
        fixed_metrics = _period_metrics(fixed_returns, start_date, end_date)
        benchmark_metrics = _period_metrics(benchmark_returns, start_date, end_date)
        rows.append(
            {
                "stress_period": period_name,
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "adaptive_strategy": adaptive_result.get(
                    "strategy",
                    "Regime-Adaptive",
                ),
                "best_fixed_strategy": best_fixed_name,
                "benchmark_strategy": benchmark_name,
                "adaptive_return": adaptive_metrics["return"],
                "best_fixed_return": fixed_metrics["return"],
                "benchmark_return": benchmark_metrics["return"],
                "adaptive_max_drawdown": adaptive_metrics["max_drawdown"],
                "best_fixed_max_drawdown": fixed_metrics["max_drawdown"],
                "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
                "adaptive_drawdown_reduction_vs_benchmark": _drawdown_reduction(
                    adaptive_metrics["max_drawdown"],
                    benchmark_metrics["max_drawdown"],
                ),
                "adaptive_drawdown_reduction_vs_best_fixed": _drawdown_reduction(
                    adaptive_metrics["max_drawdown"],
                    fixed_metrics["max_drawdown"],
                ),
            }
        )
    return pd.DataFrame(rows)


def _coerce_metric_table(results, strategy_type: str) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        table = results.copy()
    elif isinstance(results, Mapping):
        rows: list[dict[str, object]] = []
        for strategy_name, result in results.items():
            if isinstance(result, Mapping) and "portfolio_returns" in result:
                returns = _returns_from_result(result)
                metrics = PerformanceAnalytics.summary_table(returns)
                performance = dict(result.get("performance_metrics", {}))
                rows.append(
                    {
                        "strategy": strategy_name,
                        **metrics,
                        "final_value": float(result["portfolio_values"].iloc[-1]),
                        "total_turnover": performance.get(
                            "total_turnover",
                            np.nan,
                        ),
                        "total_transaction_cost": performance.get(
                            "total_transaction_cost",
                            np.nan,
                        ),
                        "strategy_type": strategy_type,
                        "status": "success",
                    }
                )
            elif isinstance(result, Mapping):
                rows.append({"strategy": strategy_name, **dict(result)})
        table = pd.DataFrame(rows)
    else:
        return pd.DataFrame()

    if "status" in table.columns:
        table = table[table["status"].eq("success")].copy()
    if "strategy_type" in table.columns:
        table = table[table["strategy_type"].fillna(strategy_type).eq(strategy_type)]
    return table.reset_index(drop=True)


def _select_best(table: pd.DataFrame, objective: str) -> pd.Series:
    if objective not in table.columns:
        raise ValueError(f"objective '{objective}' is unavailable for comparison")
    values = pd.to_numeric(table[objective], errors="coerce")
    valid = table.loc[values.notna()].copy()
    if valid.empty:
        raise ValueError(f"objective '{objective}' has no finite comparison values")
    valid_values = pd.to_numeric(valid[objective], errors="coerce")
    index = valid_values.idxmax() if OBJECTIVE_DIRECTIONS[objective] else valid_values.idxmin()
    return valid.loc[index]


def _normalize_objective(objective: str | None) -> str:
    normalized = str(objective or "calmar").strip().lower().replace(" ", "_")
    if normalized not in OBJECTIVE_DIRECTIONS:
        supported = ", ".join(sorted(OBJECTIVE_DIRECTIONS))
        raise ValueError(f"unsupported objective '{normalized}'. Supported: {supported}")
    return normalized


def _comparison_interpretation(
    comparison: Mapping[str, object],
    *,
    higher_is_better: bool,
) -> str:
    adaptive_objective = _number(comparison.get("adaptive_objective_value"))
    fixed_objective = _number(comparison.get("fixed_objective_value"))
    drawdown_delta = _number(comparison.get("adaptive_minus_fixed_MaxDrawdown"))
    cagr_delta = _number(comparison.get("adaptive_minus_fixed_CAGR"))
    turnover_penalty = _number(comparison.get("adaptive_turnover_penalty"))
    cost_penalty = _number(comparison.get("adaptive_cost_penalty"))

    adaptive_better = (
        np.isfinite(adaptive_objective)
        and np.isfinite(fixed_objective)
        and (
            adaptive_objective > fixed_objective
            if higher_is_better
            else adaptive_objective < fixed_objective
        )
    )
    if adaptive_better:
        message = "Adaptive strategy is better by the selected objective."
    elif np.isfinite(drawdown_delta) and drawdown_delta > 0.0 and cagr_delta < 0.0:
        message = "Adaptive strategy improves drawdown but sacrifices CAGR."
    else:
        message = "Adaptive strategy is not superior under this configuration."

    penalties = []
    if np.isfinite(turnover_penalty) and turnover_penalty > 0.0:
        penalties.append("higher turnover")
    if np.isfinite(cost_penalty) and cost_penalty > 0.0:
        penalties.append("higher transaction cost")
    if penalties:
        message += " It also has " + " and ".join(penalties) + "."
    return message


def _period_metrics(
    returns: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, float]:
    period = returns.loc[start_date:end_date].dropna()
    if period.empty:
        return {"return": np.nan, "max_drawdown": np.nan}
    return {
        "return": float((1.0 + period).prod() - 1.0),
        "max_drawdown": RiskAnalytics.maximum_drawdown(period),
    }


def _returns_from_result(result: Mapping[str, object]) -> pd.Series:
    values = result.get("portfolio_returns")
    if not isinstance(values, pd.Series):
        return pd.Series(dtype=float)
    return pd.to_numeric(values, errors="coerce").dropna().sort_index()


def _drawdown_reduction(strategy_drawdown, comparison_drawdown) -> float:
    strategy_value = _number(strategy_drawdown)
    comparison_value = _number(comparison_drawdown)
    if not np.isfinite(strategy_value) or not np.isfinite(comparison_value):
        return np.nan
    return abs(comparison_value) - abs(strategy_value)


def _delta(left: Mapping[str, object], right: Mapping[str, object], key: str) -> float:
    left_value = _number(left.get(key))
    right_value = _number(right.get(key))
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return np.nan
    return left_value - right_value


def _number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _empty_comparison(message: str) -> dict[str, object]:
    return {
        "best_adaptive_strategy": None,
        "best_fixed_strategy": None,
        "adaptive_minus_fixed_CAGR": np.nan,
        "adaptive_minus_fixed_Sharpe": np.nan,
        "adaptive_minus_fixed_Calmar": np.nan,
        "adaptive_minus_fixed_MaxDrawdown": np.nan,
        "adaptive_minus_fixed_FinalValue": np.nan,
        "adaptive_turnover_penalty": np.nan,
        "adaptive_cost_penalty": np.nan,
        "adaptive_stress_advantage": np.nan,
        "adaptive_cpcv_advantage": np.nan,
        "interpretation": message,
    }
