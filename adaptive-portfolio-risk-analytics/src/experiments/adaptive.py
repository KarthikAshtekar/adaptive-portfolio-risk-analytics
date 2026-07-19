"""Phase 3D adaptive strategy experiment generation and execution."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
import json

import numpy as np
import pandas as pd

from src.adaptive import (
    defensive_source_from_label,
    get_policy_preset,
    run_regime_adaptive_backtest,
)
from src.analytics import PerformanceAnalytics, calculate_drawdown_durations
from src.regime import (
    HMM_AVAILABLE,
    calculate_regime_features,
    calculate_regime_performance,
    classify_rule_based_regime,
    fit_hmm_walk_forward,
)

from .config import (
    AdaptiveExperimentConfig,
    normalize_adaptive_policy_preset,
    normalize_adaptive_regime_source,
)


class AdaptiveExperimentSkipped(RuntimeError):
    """Raised when an optional adaptive experiment cannot run safely."""


def adaptive_strategy_name(regime_source: str, policy_preset: str) -> str:
    """Return a readable first-class strategy label."""
    source = normalize_adaptive_regime_source(regime_source)
    preset = normalize_adaptive_policy_preset(policy_preset)
    source_label = "Rule-Based" if source == "rule_based_lagged" else "HMM Walk-Forward"
    return f"Regime-Adaptive {source_label} — {preset.title()}"


def generate_adaptive_parameter_grid(
    config: AdaptiveExperimentConfig,
    max_adaptive_configs: int | None = None,
) -> pd.DataFrame:
    """Generate a bounded Phase 3D adaptive experiment grid."""
    rows: list[dict[str, object]] = []
    combinations = product(
        config.regime_sources,
        config.policy_presets,
        config.training_windows,
        config.defensive_assets,
        config.transaction_cost_bps,
        config.slippage_bps,
        config.rebalance_frequencies,
    )
    for config_id, (
        regime_source,
        policy_preset,
        training_window,
        defensive_asset,
        cost_bps,
        slippage_bps,
        rebalance_frequency,
    ) in enumerate(combinations):
        source = normalize_adaptive_regime_source(regime_source)
        preset = normalize_adaptive_policy_preset(policy_preset)
        defensive_source, defensive_ticker = defensive_source_from_label(str(defensive_asset))
        rows.append(
            {
                "config_id": f"adaptive_{config_id}",
                "experiment_name": config.experiment_name,
                "strategy": adaptive_strategy_name(source, preset),
                "strategy_type": "regime_adaptive",
                "regime_source": source,
                "policy_preset": preset,
                "training_window": int(training_window),
                "train_window": int(training_window),
                "defensive_asset": defensive_asset,
                "defensive_source": defensive_source,
                "defensive_annual_rate": float(config.defensive_annual_rate),
                "defensive_ticker": defensive_ticker,
                "defensive_fallback": config.defensive_fallback,
                "transaction_cost_bps": float(cost_bps),
                "slippage_bps": float(slippage_bps),
                "rebalance_frequency": str(rebalance_frequency).upper(),
                "hmm_n_states": int(config.hmm_n_states),
                "hmm_min_train_size": int(config.hmm_min_train_size),
                "hmm_refit_frequency": int(config.hmm_refit_frequency),
                "hmm_covariance_type": config.hmm_covariance_type,
                "hmm_decision_lag": int(config.hmm_decision_lag),
                "covariance_method": "dynamic_by_regime",
                "rebalance_mode": "dynamic_by_regime",
                "threshold": None,
                "vol_targeting_enabled": False,
                "target_vol": None,
                "initial_capital": float(config.initial_capital),
            }
        )

    grid = pd.DataFrame(rows)
    if max_adaptive_configs is not None:
        limit = int(max_adaptive_configs)
        if limit <= 0:
            raise ValueError("max_adaptive_configs must be positive when provided")
        grid = grid.head(limit).copy()
    return grid


def build_adaptive_regime_input(
    returns_df: pd.DataFrame,
    config_row: Mapping[str, object],
) -> dict[str, object]:
    """Build a lag-safe rule-based or HMM walk-forward regime input."""
    source = normalize_adaptive_regime_source(
        str(config_row.get("regime_source", "rule_based_lagged"))
    )
    features = calculate_regime_features(returns_df)
    if source == "rule_based_lagged":
        return {
            "regimes": classify_rule_based_regime(features),
            "use_lagged_regimes": True,
            "regime_method_name": "Rule-based observed regimes, lagged internally",
            "features": features,
        }

    if not HMM_AVAILABLE:
        raise AdaptiveExperimentSkipped(
            "HMM walk-forward adaptive experiment skipped because `hmmlearn` is unavailable."
        )
    fitted = fit_hmm_walk_forward(
        features,
        n_states=int(config_row.get("hmm_n_states", 4)),
        min_train_size=int(config_row.get("hmm_min_train_size", 504)),
        refit_frequency=int(config_row.get("hmm_refit_frequency", 21)),
        covariance_type=str(config_row.get("hmm_covariance_type", "diag")),
        decision_lag=int(config_row.get("hmm_decision_lag", 1)),
    )
    decision_regimes = fitted["decision_regimes"].reindex(returns_df.index).fillna("Unknown")
    if decision_regimes.astype(str).eq("Unknown").all():
        raise AdaptiveExperimentSkipped(
            "HMM walk-forward adaptive experiment skipped because the available "
            "history is insufficient to produce decision regimes."
        )
    return {
        "regimes": decision_regimes,
        "use_lagged_regimes": False,
        "regime_method_name": "HMM walk-forward decision regimes",
        "features": features,
        "hmm_result": fitted,
    }


def summarize_adaptive_diagnostics(
    backtest_result: Mapping[str, object],
) -> dict[str, object]:
    """Summarize adaptive exposure, regime, allocator, and covariance usage."""
    diagnostics = backtest_result.get("diagnostics")
    if not isinstance(diagnostics, pd.DataFrame) or diagnostics.empty:
        return _empty_adaptive_diagnostics()

    regime_series = diagnostics["regime"].fillna("Unknown").astype(str)
    regime_counts = regime_series.value_counts(normalize=True)
    policy_signature = (
        diagnostics[
            [
                "regime",
                "allocator",
                "covariance_method",
                "target_volatility",
                "rebalance_mode",
            ]
        ]
        .astype(str)
        .agg("|".join, axis=1)
    )
    allocator_usage = diagnostics["allocator"].astype(str).value_counts(normalize=True)
    covariance_usage = diagnostics["covariance_method"].astype(str).value_counts(normalize=True)

    return {
        "average_risky_exposure": float(diagnostics["risky_exposure"].mean()),
        "minimum_risky_exposure": float(diagnostics["risky_exposure"].min()),
        "maximum_risky_exposure": float(diagnostics["risky_exposure"].max()),
        "average_defensive_weight": float(diagnostics["defensive_weight"].mean()),
        "maximum_defensive_weight": float(diagnostics["defensive_weight"].max()),
        "number_of_policy_switches": int(policy_signature.ne(policy_signature.shift()).sum() - 1),
        "most_common_regime": str(regime_series.mode().iloc[0]),
        "percentage_days_calm": float(regime_counts.get("Calm", 0.0)),
        "percentage_days_normal": float(regime_counts.get("Normal", 0.0)),
        "percentage_days_stress": float(regime_counts.get("Stress", 0.0)),
        "percentage_days_crisis": float(regime_counts.get("Crisis", 0.0)),
        "percentage_days_risk_on": float(regime_counts.get("Risk-On", 0.0)),
        "percentage_days_risk_off": float(regime_counts.get("Risk-Off", 0.0)),
        "allocator_usage_distribution": json.dumps(
            allocator_usage.round(6).to_dict(),
            sort_keys=True,
        ),
        "covariance_method_usage_distribution": json.dumps(
            covariance_usage.round(6).to_dict(),
            sort_keys=True,
        ),
    }


def execute_adaptive_experiment(
    returns_df: pd.DataFrame,
    config_row: Mapping[str, object],
    defensive_returns=None,
    regime_input: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Execute one adaptive configuration and retain its detailed backtest."""
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame")
    if returns_df.empty:
        raise ValueError("returns_df must not be empty")

    row = dict(config_row)
    source = normalize_adaptive_regime_source(str(row.get("regime_source", "rule_based_lagged")))
    preset = normalize_adaptive_policy_preset(str(row.get("policy_preset", "balanced")))
    resolved_regime_input = dict(regime_input or build_adaptive_regime_input(returns_df, row))
    configured_defensive_source, configured_ticker = defensive_source_from_label(
        str(row.get("defensive_asset", "Synthetic Risk-Free"))
    )
    defensive_source = str(row.get("defensive_source") or configured_defensive_source)
    defensive_ticker = row.get("defensive_ticker") or configured_ticker
    backtest = run_regime_adaptive_backtest(
        returns=returns_df,
        regimes=resolved_regime_input["regimes"],
        defensive_returns=defensive_returns,
        initial_value=float(row.get("initial_capital", 1_000_000.0)),
        training_window=int(row.get("training_window", row.get("train_window", 252))),
        rebalance_frequency=str(row.get("rebalance_frequency", "M")),
        transaction_cost_bps=float(row.get("transaction_cost_bps", 10.0)),
        slippage_bps=float(row.get("slippage_bps", 5.0)),
        policy_map=get_policy_preset(preset),
        regime_method_name=str(resolved_regime_input["regime_method_name"]),
        use_lagged_regimes=bool(resolved_regime_input["use_lagged_regimes"]),
        defensive_source=defensive_source,
        defensive_annual_rate=float(row.get("defensive_annual_rate", 0.04)),
        defensive_ticker=(str(defensive_ticker) if defensive_ticker is not None else None),
        defensive_fallback=str(row.get("defensive_fallback", "synthetic")),
    )
    metrics = PerformanceAnalytics.summary_table(backtest["portfolio_returns"])
    durations = calculate_drawdown_durations(backtest["portfolio_values"])
    adaptive_diagnostics = summarize_adaptive_diagnostics(backtest)
    result_row = {
        **row,
        "strategy": row.get("strategy") or adaptive_strategy_name(source, preset),
        "strategy_type": "regime_adaptive",
        "regime_source": source,
        "policy_preset": preset,
        "cagr": metrics["cagr"],
        "volatility": metrics["volatility"],
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
        "calmar": metrics["calmar"],
        "pain_index": metrics["pain_index"],
        "pain_ratio": metrics["pain_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "final_value": float(backtest["portfolio_values"].iloc[-1]),
        "total_turnover": float(backtest["performance_metrics"]["total_turnover"]),
        "average_turnover": float(backtest["performance_metrics"]["average_turnover"]),
        "total_transaction_cost": float(backtest["performance_metrics"]["total_transaction_cost"]),
        "number_of_rebalances": int(backtest["performance_metrics"]["number_of_rebalances"]),
        "var_95": metrics["var_95"],
        "cvar_95": metrics["cvar_95"],
        "max_drawdown_duration": int(durations["max_drawdown_duration"]),
        **backtest["defensive_metadata"],
        **adaptive_diagnostics,
        "status": "success",
        "error": None,
    }
    return {
        "result": result_row,
        "backtest": backtest,
        "regime_input": resolved_regime_input,
    }


def run_adaptive_experiment_grid(
    returns_df: pd.DataFrame,
    config: AdaptiveExperimentConfig,
    defensive_returns=None,
    max_adaptive_configs: int | None = None,
) -> dict[str, object]:
    """Run a bounded adaptive grid, skipping unavailable HMM runs cleanly."""
    grid = generate_adaptive_parameter_grid(
        config,
        max_adaptive_configs=max_adaptive_configs,
    )
    rows: list[dict[str, object]] = []
    backtests: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    regime_cache: dict[tuple[object, ...], dict[str, object]] = {}

    for _, config_series in grid.iterrows():
        row = config_series.to_dict()
        config_id = str(row["config_id"])
        cache_key = (
            row["regime_source"],
            row["hmm_n_states"],
            row["hmm_min_train_size"],
            row["hmm_refit_frequency"],
            row["hmm_covariance_type"],
            row["hmm_decision_lag"],
        )
        try:
            if cache_key not in regime_cache:
                regime_cache[cache_key] = build_adaptive_regime_input(returns_df, row)
            execution = execute_adaptive_experiment(
                returns_df,
                row,
                defensive_returns=defensive_returns,
                regime_input=regime_cache[cache_key],
            )
            rows.append(execution["result"])
            backtests[config_id] = execution["backtest"]
        except AdaptiveExperimentSkipped as exc:
            warning = str(exc)
            warnings.append(warning)
            rows.append(_failed_adaptive_row(row, "skipped", warning))
        except Exception as exc:
            rows.append(_failed_adaptive_row(row, "failed", str(exc)))

    return {
        "results": pd.DataFrame(rows),
        "backtests": backtests,
        "warnings": list(dict.fromkeys(warnings)),
    }


def build_adaptive_attribution(
    backtest_result: Mapping[str, object],
    benchmark_returns=None,
) -> dict[str, pd.DataFrame]:
    """Build regime, policy, exposure, and performance attribution tables."""
    diagnostics = backtest_result.get("diagnostics")
    if not isinstance(diagnostics, pd.DataFrame) or diagnostics.empty:
        empty = pd.DataFrame()
        return {
            "regime_distribution": empty,
            "policy_usage": empty,
            "allocator_usage": empty,
            "covariance_usage": empty,
            "exposure_history": empty,
            "policy_switches": empty,
            "regime_performance": empty,
        }

    frame = diagnostics.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["policy_signature"] = (
        frame[
            [
                "regime",
                "allocator",
                "covariance_method",
                "target_volatility",
                "rebalance_mode",
            ]
        ]
        .astype(str)
        .agg("|".join, axis=1)
    )
    frame["policy_switch"] = frame["policy_signature"].ne(frame["policy_signature"].shift())
    frame.loc[frame.index[0], "policy_switch"] = False

    regime_distribution = _usage_table(frame["regime"], "regime")
    allocator_usage = _usage_table(frame["allocator"], "allocator")
    covariance_usage = _usage_table(
        frame["covariance_method"],
        "covariance_method",
    )
    policy_usage = (
        frame.groupby(
            [
                "regime",
                "allocator",
                "covariance_method",
                "target_volatility",
                "rebalance_mode",
            ],
            dropna=False,
        )
        .size()
        .rename("number_of_days")
        .reset_index()
    )
    policy_usage["percentage_of_days"] = policy_usage["number_of_days"] / max(len(frame), 1)
    exposure_history = frame[
        [
            "date",
            "regime",
            "risky_exposure",
            "defensive_weight",
            "target_volatility",
            "policy_switch",
        ]
    ].copy()
    policy_switches = frame.loc[
        frame["policy_switch"],
        [
            "date",
            "regime",
            "allocator",
            "covariance_method",
            "target_volatility",
            "rebalance_mode",
        ],
    ].copy()
    regime_performance = calculate_regime_performance(
        backtest_result["portfolio_returns"],
        backtest_result["applied_regimes"],
        benchmark_returns=benchmark_returns,
    )
    return {
        "regime_distribution": regime_distribution,
        "policy_usage": policy_usage,
        "allocator_usage": allocator_usage,
        "covariance_usage": covariance_usage,
        "exposure_history": exposure_history,
        "policy_switches": policy_switches,
        "regime_performance": regime_performance,
    }


def _usage_table(values: pd.Series, column_name: str) -> pd.DataFrame:
    counts = values.fillna("Unknown").astype(str).value_counts()
    return pd.DataFrame(
        {
            column_name: counts.index,
            "number_of_days": counts.values,
            "percentage_of_days": counts.values / max(int(counts.sum()), 1),
        }
    )


def _empty_adaptive_diagnostics() -> dict[str, object]:
    return {
        "average_risky_exposure": np.nan,
        "minimum_risky_exposure": np.nan,
        "maximum_risky_exposure": np.nan,
        "average_defensive_weight": np.nan,
        "maximum_defensive_weight": np.nan,
        "number_of_policy_switches": 0,
        "most_common_regime": "Unknown",
        "percentage_days_calm": 0.0,
        "percentage_days_normal": 0.0,
        "percentage_days_stress": 0.0,
        "percentage_days_crisis": 0.0,
        "percentage_days_risk_on": 0.0,
        "percentage_days_risk_off": 0.0,
        "allocator_usage_distribution": "{}",
        "covariance_method_usage_distribution": "{}",
    }


def _failed_adaptive_row(
    row: Mapping[str, object],
    status: str,
    error: str,
) -> dict[str, object]:
    empty_metrics = {
        key: np.nan
        for key in [
            "cagr",
            "volatility",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "final_value",
            "pain_index",
            "pain_ratio",
            "total_turnover",
            "average_turnover",
            "total_transaction_cost",
            "number_of_rebalances",
            "var_95",
            "cvar_95",
            "max_drawdown_duration",
        ]
    }
    return {
        **dict(row),
        **empty_metrics,
        "defensive_source_requested": row.get("defensive_source"),
        "defensive_source_used": None,
        "defensive_annual_rate": row.get("defensive_annual_rate", 0.04),
        "defensive_ticker": row.get("defensive_ticker"),
        "defensive_fallback_used": None,
        "defensive_notes": error,
        **_empty_adaptive_diagnostics(),
        "status": status,
        "error": error,
    }
