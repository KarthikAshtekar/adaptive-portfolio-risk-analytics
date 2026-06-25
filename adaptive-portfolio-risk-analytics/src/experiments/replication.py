"""Phase 3E defensive-sleeve replication and policy-tuning studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, timedelta
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.adaptive import (
    defensive_source_from_label,
    get_defensive_returns,
    get_policy_preset,
    run_regime_adaptive_backtest,
)
from src.analytics import PerformanceAnalytics, calculate_drawdown_durations
from src.backtesting import RollingBacktester
from src.backtesting.transaction_costs import TransactionCostModel
from src.benchmarks import BenchmarkFactory
from src.data_pipeline import DataPreprocessor, YahooFinanceProvider

from .adaptive import (
    AdaptiveExperimentSkipped,
    adaptive_strategy_name,
    build_adaptive_regime_input,
    execute_adaptive_experiment,
)

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 1_000_000.0
FIXED_STRATEGIES = ("Equal Weight", "HRP", "HERC")
SCENARIO_KEYS = (
    "universe",
    "date_window",
    "cost_scenario",
    "defensive_sleeve",
)

DEFAULT_REPLICATION_UNIVERSES: dict[str, list[str]] = {
    "Core Diversified": [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "TCS.NS",
        "INFY.NS",
        "RELIANCE.NS",
        "HINDUNILVR.NS",
        "ITC.NS",
        "SUNPHARMA.NS",
        "LT.NS",
        "BHARTIARTL.NS",
        "GOLDBEES.NS",
    ],
    "Equity Heavy": [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "KOTAKBANK.NS",
        "AXISBANK.NS",
        "TCS.NS",
        "INFY.NS",
        "RELIANCE.NS",
        "LT.NS",
        "BHARTIARTL.NS",
    ],
    "Core without Gold/Silver": [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "TCS.NS",
        "INFY.NS",
        "RELIANCE.NS",
        "HINDUNILVR.NS",
        "ITC.NS",
        "SUNPHARMA.NS",
        "LT.NS",
        "BHARTIARTL.NS",
    ],
}

DEFAULT_DATE_WINDOWS = (
    ("2020-01-01", None),
    ("2018-01-01", None),
    ("2021-01-01", None),
    ("2022-01-01", None),
)

DEFAULT_COST_SCENARIOS = (
    {"name": "0 bps + 0 bps", "base_bps": 0.0, "slippage_bps": 0.0},
    {"name": "10 bps + 5 bps", "base_bps": 10.0, "slippage_bps": 5.0},
    {"name": "25 bps + 10 bps", "base_bps": 25.0, "slippage_bps": 10.0},
    {"name": "50 bps + 25 bps", "base_bps": 50.0, "slippage_bps": 25.0},
)

DEFAULT_DEFENSIVE_SLEEVES = (
    "synthetic_4pct",
    "cash_zero",
    "LIQUIDBEES.NS",
    "LIQUIDETF.NS",
)


def run_replication_study(
    universes=None,
    date_windows=None,
    cost_scenarios=None,
    defensive_sleeves=None,
    policy_presets=None,
    regime_sources=None,
    objective: str = "calmar",
    max_runs: int | None = 70,
    output_dir=None,
) -> pd.DataFrame:
    """Run a bounded, failure-isolated fixed-versus-adaptive replication study.

    A run is one strategy in one scenario. Runtime limiting samples complete
    scenarios so each retained scenario includes all three fixed benchmarks and
    every requested adaptive policy/regime combination.
    """
    selected_universes = _normalize_universes(universes)
    selected_windows = _normalize_date_windows(date_windows)
    selected_costs = _normalize_cost_scenarios(cost_scenarios)
    selected_sleeves = list(defensive_sleeves or DEFAULT_DEFENSIVE_SLEEVES)
    selected_presets = [
        str(value).strip().lower()
        for value in (policy_presets or ("Conservative", "Balanced"))
    ]
    selected_sources = [
        str(value).strip().lower()
        for value in (regime_sources or ("rule_based_lagged", "hmm_walk_forward"))
    ]
    _validate_objective(objective)

    adaptive_variants = [
        (preset, source)
        for preset in selected_presets
        for source in selected_sources
    ]
    runs_per_scenario = len(FIXED_STRATEGIES) + len(adaptive_variants)
    scenarios = [
        {
            "universe": universe_name,
            "date_window": window_label,
            "start_date": start_date,
            "end_date": end_date,
            "cost_scenario": cost["name"],
            "base_bps": cost["base_bps"],
            "slippage_bps": cost["slippage_bps"],
            "defensive_sleeve": sleeve,
        }
        for universe_name in selected_universes
        for window_label, start_date, end_date in selected_windows
        for cost in selected_costs
        for sleeve in selected_sleeves
    ]
    scenarios = _bounded_scenarios(
        scenarios,
        max_runs=max_runs,
        runs_per_scenario=runs_per_scenario,
    )

    universe_cache: dict[str, pd.DataFrame | Exception] = {}
    fixed_cache: dict[tuple[object, ...], dict[str, object] | Exception] = {}
    regime_cache: dict[tuple[object, ...], dict[str, object] | Exception] = {}
    defensive_cache: dict[tuple[object, ...], object] = {}
    rows: list[dict[str, object]] = []
    total_runs = len(scenarios) * runs_per_scenario
    completed = 0

    for scenario_number, scenario in enumerate(scenarios, start=1):
        universe_name = str(scenario["universe"])
        if universe_name not in universe_cache:
            try:
                universe_cache[universe_name] = _load_universe_returns(
                    selected_universes[universe_name],
                    selected_windows,
                )
            except Exception as exc:
                universe_cache[universe_name] = exc
        loaded = universe_cache[universe_name]
        if isinstance(loaded, Exception):
            for strategy_name, strategy_type, preset, source in _scenario_strategies(
                adaptive_variants
            ):
                completed += 1
                rows.append(
                    _failed_row(
                        scenario,
                        strategy_name,
                        strategy_type,
                        preset,
                        source,
                        str(loaded),
                    )
                )
            continue

        context_returns, evaluation_start, evaluation_end = _context_for_scenario(
            loaded,
            scenario,
        )
        if context_returns.empty:
            for strategy_name, strategy_type, preset, source in _scenario_strategies(
                adaptive_variants
            ):
                completed += 1
                rows.append(
                    _failed_row(
                        scenario,
                        strategy_name,
                        strategy_type,
                        preset,
                        source,
                        "No returns are available for the requested date window.",
                    )
                )
            continue

        training_window = _training_window(len(context_returns))
        hmm_min_train_size = _hmm_training_window(len(context_returns), training_window)
        defensive_key = (
            universe_name,
            scenario["date_window"],
            scenario["defensive_sleeve"],
        )
        if defensive_key not in defensive_cache:
            requested_source, annual_rate, ticker = _sleeve_settings(
                str(scenario["defensive_sleeve"])
            )
            defensive_cache[defensive_key] = get_defensive_returns(
                index=context_returns.index,
                source=requested_source,
                annual_rate=annual_rate,
                defensive_ticker=ticker,
                fallback="synthetic",
            )
        logger.info(
            "Replication scenario %s/%s: %s, %s, %s, %s",
            scenario_number,
            len(scenarios),
            scenario["universe"],
            scenario["date_window"],
            scenario["cost_scenario"],
            scenario["defensive_sleeve"],
        )

        for strategy_name in FIXED_STRATEGIES:
            completed += 1
            cache_key = (
                universe_name,
                scenario["date_window"],
                scenario["cost_scenario"],
                strategy_name,
            )
            try:
                if cache_key not in fixed_cache:
                    fixed_cache[cache_key] = _run_fixed_strategy(
                        context_returns,
                        evaluation_start,
                        evaluation_end,
                        strategy_name,
                        training_window,
                        float(scenario["base_bps"]),
                        float(scenario["slippage_bps"]),
                    )
                cached = fixed_cache[cache_key]
                if isinstance(cached, Exception):
                    raise cached
                rows.append(
                    {
                        **scenario,
                        **cached,
                        "policy_preset": None,
                        "regime_source": None,
                        "defensive_source_requested": _sleeve_settings(
                            str(scenario["defensive_sleeve"])
                        )[0],
                        "defensive_source_used": "not_used",
                        "defensive_annual_rate": _sleeve_settings(
                            str(scenario["defensive_sleeve"])
                        )[1],
                        "defensive_ticker": _sleeve_settings(
                            str(scenario["defensive_sleeve"])
                        )[2],
                        "defensive_fallback_used": False,
                        "status": "success",
                        "failure_reason": None,
                    }
                )
            except Exception as exc:
                fixed_cache[cache_key] = exc
                rows.append(
                    _failed_row(
                        scenario,
                        strategy_name,
                        "fixed",
                        None,
                        None,
                        str(exc),
                    )
                )
            _log_progress(completed, total_runs)

        for preset, source in adaptive_variants:
            completed += 1
            strategy_name = adaptive_strategy_name(source, preset)
            regime_key = (
                universe_name,
                scenario["date_window"],
                source,
                training_window,
                hmm_min_train_size,
            )
            try:
                if regime_key not in regime_cache:
                    config_row = _adaptive_config_row(
                        scenario,
                        preset,
                        source,
                        training_window,
                        hmm_min_train_size,
                    )
                    try:
                        regime_cache[regime_key] = build_adaptive_regime_input(
                            context_returns,
                            config_row,
                        )
                    except Exception as exc:
                        regime_cache[regime_key] = exc
                regime_input = regime_cache[regime_key]
                if isinstance(regime_input, Exception):
                    raise regime_input
                execution = execute_adaptive_experiment(
                    context_returns,
                    _adaptive_config_row(
                        scenario,
                        preset,
                        source,
                        training_window,
                        hmm_min_train_size,
                    ),
                    defensive_returns=defensive_cache[defensive_key],
                    regime_input=regime_input,
                )
                rows.append(
                    {
                        **scenario,
                        **_summarize_backtest(
                            execution["backtest"],
                            evaluation_start,
                            evaluation_end,
                            strategy_name,
                            "regime_adaptive",
                        ),
                        "policy_preset": preset,
                        "regime_source": source,
                        **execution["backtest"]["defensive_metadata"],
                        "status": "success",
                        "failure_reason": None,
                    }
                )
            except AdaptiveExperimentSkipped as exc:
                rows.append(
                    _failed_row(
                        scenario,
                        strategy_name,
                        "regime_adaptive",
                        preset,
                        source,
                        str(exc),
                        status="skipped",
                    )
                )
            except Exception as exc:
                rows.append(
                    _failed_row(
                        scenario,
                        strategy_name,
                        "regime_adaptive",
                        preset,
                        source,
                        str(exc),
                    )
                )
            _log_progress(completed, total_runs)

    results = pd.DataFrame(rows)
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path / "replication_results.csv", index=False)
        summarize_replication_results(results, objective=objective).to_csv(
            output_path / "replication_summary.csv",
            index=False,
        )
    return results


def summarize_replication_results(
    results_df: pd.DataFrame,
    objective: str = "calmar",
) -> pd.DataFrame:
    """Summarize adaptive wins, downside protection, costs, and verdict."""
    _validate_objective(objective)
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame")
    if results_df.empty:
        return pd.DataFrame()

    successful = results_df.loc[results_df["status"].eq("success")].copy()
    adaptive = successful.loc[
        successful["strategy_type"].eq("regime_adaptive")
    ].copy()
    fixed = successful.loc[successful["strategy_type"].eq("fixed")].copy()
    if adaptive.empty:
        return pd.DataFrame()

    baselines: list[dict[str, object]] = []
    for scenario_values, group in fixed.groupby(list(SCENARIO_KEYS), dropna=False):
        finite = group.loc[pd.to_numeric(group[objective], errors="coerce").notna()]
        if finite.empty:
            continue
        best = finite.loc[pd.to_numeric(finite[objective], errors="coerce").idxmax()]
        baseline = {
            key: value for key, value in zip(SCENARIO_KEYS, scenario_values)
        }
        baseline.update(
            {
                "fixed_objective": best[objective],
                "fixed_calmar": best["calmar"],
                "fixed_pain_index": best.get("pain_index", np.nan),
                "fixed_pain_ratio": best.get("pain_ratio", np.nan),
                "fixed_max_drawdown": best["max_drawdown"],
                "fixed_final_value": best["final_value"],
                "fixed_cagr": best["cagr"],
                "fixed_stress_period_return": best["stress_period_return"],
            }
        )
        baselines.append(baseline)
    baseline_df = pd.DataFrame(baselines)
    paired = adaptive.merge(
        baseline_df,
        on=list(SCENARIO_KEYS),
        how="left",
        validate="many_to_one",
    )
    paired["objective_win"] = (
        pd.to_numeric(paired[objective], errors="coerce")
        > pd.to_numeric(paired["fixed_objective"], errors="coerce")
    )
    paired["calmar_win"] = paired["calmar"] > paired["fixed_calmar"]
    if "pain_index" not in paired:
        paired["pain_index"] = np.nan
    if "pain_ratio" not in paired:
        paired["pain_ratio"] = np.nan
    paired["pain_index_win"] = pd.to_numeric(
        paired["pain_index"],
        errors="coerce",
    ) < pd.to_numeric(paired["fixed_pain_index"], errors="coerce")
    paired["pain_ratio_win"] = pd.to_numeric(
        paired["pain_ratio"],
        errors="coerce",
    ) > pd.to_numeric(paired["fixed_pain_ratio"], errors="coerce")
    paired["drawdown_win"] = paired["max_drawdown"] > paired["fixed_max_drawdown"]
    paired["final_value_win"] = paired["final_value"] > paired["fixed_final_value"]
    paired["stress_protection_win"] = (
        paired["stress_period_return"] > paired["fixed_stress_period_return"]
    ) | paired["drawdown_win"]
    paired["final_value_ratio"] = (
        paired["final_value"] / paired["fixed_final_value"]
    )

    all_rows = []
    failed = results_df.loc[
        results_df["strategy_type"].eq("regime_adaptive")
        & ~results_df["status"].eq("success")
    ]
    for strategy, group in paired.groupby("strategy", sort=True):
        cost_slope = _cost_sensitivity_slope(group)
        failed_count = int(failed["strategy"].eq(strategy).sum())
        successful_count = int(len(group))
        objective_win_rate = float(group["objective_win"].mean())
        drawdown_win_rate = float(group["drawdown_win"].mean())
        final_value_ratio = float(group["final_value_ratio"].mean())
        high_cost = group.loc[
            pd.to_numeric(group["base_bps"], errors="coerce")
            + pd.to_numeric(group["slippage_bps"], errors="coerce")
            >= 60.0
        ]
        high_cost_stable = (
            True
            if high_cost.empty
            else float(pd.to_numeric(high_cost["calmar"], errors="coerce").median())
            >= 0.75 * float(pd.to_numeric(group["calmar"], errors="coerce").median())
        )
        if (
            objective_win_rate > 0.50
            and final_value_ratio >= 0.95
            and high_cost_stable
        ):
            classification = "First-class main strategy"
        elif drawdown_win_rate > 0.50 and final_value_ratio < 1.0:
            classification = "Risk-control overlay"
        else:
            classification = "Experimental only"

        all_rows.append(
            {
                "strategy": strategy,
                "regime_source": group["regime_source"].iloc[0],
                "policy_preset": group["policy_preset"].iloc[0],
                "selected_objective": objective,
                "selected_objective_win_rate": objective_win_rate,
                "win_rate_by_calmar": float(group["calmar_win"].mean()),
                "win_rate_by_pain_index": float(group["pain_index_win"].mean()),
                "win_rate_by_pain_ratio": float(group["pain_ratio_win"].mean()),
                "win_rate_by_max_drawdown": drawdown_win_rate,
                "win_rate_by_final_value": float(group["final_value_win"].mean()),
                "average_calmar": float(group["calmar"].mean()),
                "average_pain_index": float(
                    pd.to_numeric(group.get("pain_index"), errors="coerce").mean()
                ),
                "average_pain_ratio": float(
                    pd.to_numeric(group.get("pain_ratio"), errors="coerce").mean()
                ),
                "median_calmar": float(group["calmar"].median()),
                "worst_case_calmar": float(group["calmar"].min()),
                "average_max_drawdown": float(group["max_drawdown"].mean()),
                "worst_max_drawdown": float(group["max_drawdown"].min()),
                "average_cagr": float(group["cagr"].mean()),
                "average_final_value": float(group["final_value"].mean()),
                "average_turnover": float(group["turnover"].mean()),
                "average_transaction_cost": float(
                    group["transaction_cost"].mean()
                ),
                "stress_protection_win_rate": float(
                    group["stress_protection_win"].mean()
                ),
                "cost_sensitivity_slope": cost_slope,
                "average_final_value_ratio_vs_best_fixed": final_value_ratio,
                "number_of_successful_runs": successful_count,
                "number_of_failed_runs": failed_count,
                "classification": classification,
            }
        )
    return pd.DataFrame(all_rows).sort_values(
        ["selected_objective_win_rate", "stress_protection_win_rate"],
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


def run_policy_tuning_study(
    returns: pd.DataFrame,
    regime_sources: Sequence[str] = (
        "hmm_walk_forward",
        "rule_based_lagged",
    ),
    defensive_sleeve: str = "synthetic_4pct",
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    evaluation_start=None,
    max_variants: int | None = 4,
) -> pd.DataFrame:
    """Compare Conservative base policies with a bounded faster re-risk variant."""
    clean = _clean_returns(returns)
    training_window = _training_window(len(clean))
    hmm_min_train_size = _hmm_training_window(len(clean), training_window)
    start = (
        pd.Timestamp(evaluation_start)
        if evaluation_start is not None
        else clean.index[min(training_window, len(clean) - 1)]
    )
    source_name, annual_rate, ticker = _sleeve_settings(defensive_sleeve)
    variants = [
        ("Conservative base", get_policy_preset("conservative")),
        (
            "Conservative faster re-risking",
            _faster_rerisk_policy(get_policy_preset("conservative")),
        ),
    ]
    tasks = [
        (source, variant_name, policy_map)
        for source in regime_sources
        for variant_name, policy_map in variants
    ]
    if max_variants is not None:
        if int(max_variants) <= 0:
            raise ValueError("max_variants must be positive when provided")
        tasks = tasks[: int(max_variants)]

    regime_cache: dict[str, dict[str, object] | Exception] = {}
    rows: list[dict[str, object]] = []
    for source, variant_name, policy_map in tasks:
        config_row = {
            "strategy": adaptive_strategy_name(source, "conservative"),
            "strategy_type": "regime_adaptive",
            "regime_source": source,
            "policy_preset": "conservative",
            "training_window": training_window,
            "train_window": training_window,
            "defensive_asset": defensive_sleeve,
            "defensive_source": source_name,
            "defensive_annual_rate": annual_rate,
            "defensive_ticker": ticker,
            "defensive_fallback": "synthetic",
            "transaction_cost_bps": float(transaction_cost_bps),
            "slippage_bps": float(slippage_bps),
            "rebalance_frequency": "M",
            "hmm_n_states": 4,
            "hmm_min_train_size": hmm_min_train_size,
            "hmm_refit_frequency": 21,
            "hmm_covariance_type": "diag",
            "hmm_decision_lag": 1,
            "initial_capital": INITIAL_CAPITAL,
        }
        try:
            if source not in regime_cache:
                try:
                    regime_cache[source] = build_adaptive_regime_input(
                        clean,
                        config_row,
                    )
                except Exception as exc:
                    regime_cache[source] = exc
            regime_input = regime_cache[source]
            if isinstance(regime_input, Exception):
                raise regime_input
            defensive_result = get_defensive_returns(
                index=clean.index,
                source=source_name,
                annual_rate=annual_rate,
                defensive_ticker=ticker,
                fallback="synthetic",
            )
            backtest = run_regime_adaptive_backtest(
                returns=clean,
                regimes=regime_input["regimes"],
                initial_value=INITIAL_CAPITAL,
                training_window=training_window,
                rebalance_frequency="M",
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
                policy_map=policy_map,
                regime_method_name=str(regime_input["regime_method_name"]),
                use_lagged_regimes=bool(regime_input["use_lagged_regimes"]),
                defensive_source=source_name,
                defensive_annual_rate=annual_rate,
                defensive_ticker=ticker,
                defensive_returns=defensive_result,
            )
            summary = _summarize_backtest(
                backtest,
                start,
                clean.index.max(),
                variant_name,
                "regime_adaptive",
            )
            rows.append(
                {
                    "policy_variant": variant_name,
                    "regime_source": source,
                    "cagr": summary["cagr"],
                    "calmar": summary["calmar"],
                    "pain_index": summary["pain_index"],
                    "pain_ratio": summary["pain_ratio"],
                    "max_drawdown": summary["max_drawdown"],
                    "final_value": summary["final_value"],
                    "recovery_duration": summary["recovery_duration"],
                    "turnover": summary["turnover"],
                    "transaction_cost": summary["transaction_cost"],
                    "stress_period_return": summary["stress_period_return"],
                    "status": "success",
                    "failure_reason": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "policy_variant": variant_name,
                    "regime_source": source,
                    "status": "failed",
                    "failure_reason": str(exc),
                }
            )

    result = pd.DataFrame(rows)
    return _flag_policy_tuning_findings(result)


def _normalize_universes(universes) -> dict[str, object]:
    if universes is None:
        return dict(DEFAULT_REPLICATION_UNIVERSES)
    if isinstance(universes, Mapping):
        return {str(name): value for name, value in universes.items()}
    selected = {}
    for name in universes:
        if str(name) not in DEFAULT_REPLICATION_UNIVERSES:
            raise ValueError(f"unknown predefined universe '{name}'")
        selected[str(name)] = DEFAULT_REPLICATION_UNIVERSES[str(name)]
    if not selected:
        raise ValueError("universes must not be empty")
    return selected


def _normalize_date_windows(date_windows) -> list[tuple[str, pd.Timestamp, pd.Timestamp | None]]:
    values = date_windows or DEFAULT_DATE_WINDOWS
    normalized = []
    for value in values:
        if isinstance(value, Mapping):
            start = pd.Timestamp(value["start"])
            end = pd.Timestamp(value["end"]) if value.get("end") else None
            label = str(value.get("name") or _window_label(start, end))
        else:
            start = pd.Timestamp(value[0])
            end = pd.Timestamp(value[1]) if value[1] is not None else None
            label = _window_label(start, end)
        if end is not None and start >= end:
            raise ValueError("date window start must be before end")
        normalized.append((label, start, end))
    if not normalized:
        raise ValueError("date_windows must not be empty")
    return normalized


def _normalize_cost_scenarios(cost_scenarios) -> list[dict[str, float | str]]:
    values = cost_scenarios or DEFAULT_COST_SCENARIOS
    normalized = []
    for value in values:
        if isinstance(value, Mapping):
            base = float(value.get("base_bps", 0.0))
            slippage = float(value.get("slippage_bps", 0.0))
            name = str(value.get("name") or f"{base:g} bps + {slippage:g} bps")
        else:
            base = float(value[0])
            slippage = float(value[1])
            name = f"{base:g} bps + {slippage:g} bps"
        if base < 0.0 or slippage < 0.0:
            raise ValueError("cost scenarios must be non-negative")
        normalized.append(
            {"name": name, "base_bps": base, "slippage_bps": slippage}
        )
    return normalized


def _bounded_scenarios(
    scenarios: list[dict[str, object]],
    *,
    max_runs: int | None,
    runs_per_scenario: int,
) -> list[dict[str, object]]:
    if max_runs is None:
        return scenarios
    limit = int(max_runs)
    if limit <= 0:
        raise ValueError("max_runs must be positive when provided")
    scenario_limit = max(1, limit // max(runs_per_scenario, 1))
    if scenario_limit >= len(scenarios):
        return scenarios
    dimensions = (
        "universe",
        "date_window",
        "cost_scenario",
        "defensive_sleeve",
    )
    selected: list[dict[str, object]] = []
    remaining = list(scenarios)

    def ordered_values(dimension: str) -> list[object]:
        return list(dict.fromkeys(scenario[dimension] for scenario in scenarios))

    def take_matching(**criteria) -> None:
        if len(selected) >= scenario_limit:
            return
        for position, candidate in enumerate(remaining):
            if all(candidate.get(key) == value for key, value in criteria.items()):
                selected.append(remaining.pop(position))
                return

    universe_values = ordered_values("universe")
    window_values = ordered_values("date_window")
    cost_values = ordered_values("cost_scenario")
    sleeve_values = ordered_values("defensive_sleeve")
    baseline_universe = universe_values[0]
    baseline_window = window_values[0]
    baseline_cost = next(
        (
            value
            for value in cost_values
            if str(value).startswith("10 bps + 5 bps")
        ),
        cost_values[0],
    )
    baseline_sleeve = next(
        (
            value
            for value in sleeve_values
            if str(value).lower() == "synthetic_4pct"
        ),
        sleeve_values[0],
    )

    # Preserve matched one-factor sweeps before adding broad coverage.
    for cost in cost_values:
        take_matching(
            universe=baseline_universe,
            date_window=baseline_window,
            cost_scenario=cost,
            defensive_sleeve=baseline_sleeve,
        )
    for sleeve in sleeve_values:
        take_matching(
            universe=baseline_universe,
            date_window=baseline_window,
            cost_scenario=baseline_cost,
            defensive_sleeve=sleeve,
        )

    missing_universes = [
        value
        for value in universe_values
        if value not in {scenario["universe"] for scenario in selected}
    ]
    missing_windows = [
        value
        for value in window_values
        if value not in {scenario["date_window"] for scenario in selected}
    ]
    coverage_count = max(len(missing_universes), len(missing_windows))
    for position in range(coverage_count):
        take_matching(
            universe=(
                missing_universes[position % len(missing_universes)]
                if missing_universes
                else baseline_universe
            ),
            date_window=(
                missing_windows[position % len(missing_windows)]
                if missing_windows
                else baseline_window
            ),
            cost_scenario=baseline_cost,
            defensive_sleeve=baseline_sleeve,
        )

    uncovered = {
        dimension: {
            scenario[dimension]
            for scenario in scenarios
            if scenario[dimension]
            not in {selected_row[dimension] for selected_row in selected}
        }
        for dimension in dimensions
    }
    while remaining and len(selected) < scenario_limit:
        best_position = max(
            range(len(remaining)),
            key=lambda position: (
                sum(
                    remaining[position][dimension] in uncovered[dimension]
                    for dimension in dimensions
                ),
                position / max(len(remaining) - 1, 1),
            ),
        )
        selected_scenario = remaining.pop(best_position)
        selected.append(selected_scenario)
        for dimension in dimensions:
            uncovered[dimension].discard(selected_scenario[dimension])
        if all(not values for values in uncovered.values()):
            break
    if len(selected) < scenario_limit and remaining:
        positions = np.linspace(
            0,
            len(remaining) - 1,
            scenario_limit - len(selected),
            dtype=int,
        )
        selected.extend(
            remaining[int(position)]
            for position in dict.fromkeys(positions)
        )
    return selected[:scenario_limit]


def _load_universe_returns(
    universe,
    windows: Sequence[tuple[str, pd.Timestamp, pd.Timestamp | None]],
) -> pd.DataFrame:
    if isinstance(universe, pd.DataFrame):
        return _clean_returns(universe)
    tickers = [str(value) for value in universe]
    earliest_start = min(value[1] for value in windows) - pd.DateOffset(years=3)
    latest_end = max(
        [
            value[2]
            for value in windows
            if value[2] is not None
        ]
        or [pd.Timestamp(date.today())]
    )
    download_end = (latest_end.date() + timedelta(days=1)).isoformat()
    market_data = YahooFinanceProvider().get_market_data(
        symbols=tickers,
        start_date=earliest_start.date().isoformat(),
        end_date=download_end,
    )
    clean_prices, _ = DataPreprocessor.handle_missing_values(market_data.prices_df)
    outputs = DataPreprocessor.build_returns_risk_outputs(clean_prices)
    return _clean_returns(outputs.returns_df)


def _context_for_scenario(
    returns: pd.DataFrame,
    scenario: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    evaluation_start = pd.Timestamp(scenario["start_date"])
    requested_end = scenario.get("end_date")
    evaluation_end = (
        pd.Timestamp(requested_end)
        if requested_end is not None
        else returns.index.max()
    )
    context_start = evaluation_start - pd.DateOffset(years=3)
    context = returns.loc[context_start:evaluation_end].copy()
    return context, evaluation_start, evaluation_end


def _run_fixed_strategy(
    returns: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    strategy_name: str,
    training_window: int,
    base_bps: float,
    slippage_bps: float,
) -> dict[str, object]:
    allocator = BenchmarkFactory.get_allocator(
        strategy_name=strategy_name,
        covariance_method="sample",
    )
    backtest = RollingBacktester(
        allocator=allocator,
        train_window=training_window,
        rebalance_frequency="M",
        initial_capital=INITIAL_CAPITAL,
        rebalance_mode="calendar",
        threshold=0.05,
        transaction_cost_model=TransactionCostModel(
            base_bps=base_bps,
            slippage_bps=slippage_bps,
        ),
    ).run(returns)
    return _summarize_backtest(
        backtest,
        evaluation_start,
        evaluation_end,
        BenchmarkFactory.normalize_strategy_name(strategy_name),
        "fixed",
    )


def _adaptive_config_row(
    scenario: Mapping[str, object],
    preset: str,
    source: str,
    training_window: int,
    hmm_min_train_size: int,
) -> dict[str, object]:
    defensive_source, annual_rate, ticker = _sleeve_settings(
        str(scenario["defensive_sleeve"])
    )
    return {
        "strategy": adaptive_strategy_name(source, preset),
        "strategy_type": "regime_adaptive",
        "regime_source": source,
        "policy_preset": preset,
        "training_window": training_window,
        "train_window": training_window,
        "defensive_asset": scenario["defensive_sleeve"],
        "defensive_source": defensive_source,
        "defensive_annual_rate": annual_rate,
        "defensive_ticker": ticker,
        "defensive_fallback": "synthetic",
        "transaction_cost_bps": float(scenario["base_bps"]),
        "slippage_bps": float(scenario["slippage_bps"]),
        "rebalance_frequency": "M",
        "hmm_n_states": 4,
        "hmm_min_train_size": hmm_min_train_size,
        "hmm_refit_frequency": 21,
        "hmm_covariance_type": "diag",
        "hmm_decision_lag": 1,
        "initial_capital": INITIAL_CAPITAL,
    }


def _summarize_backtest(
    backtest: Mapping[str, object],
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    strategy_name: str,
    strategy_type: str,
) -> dict[str, object]:
    returns = pd.to_numeric(
        backtest["portfolio_returns"],
        errors="coerce",
    ).loc[evaluation_start:evaluation_end].dropna()
    if returns.empty:
        raise ValueError("backtest produced no returns in the evaluation window")
    values = _normalized_values(returns)
    metrics = PerformanceAnalytics.summary_table(returns)
    durations = calculate_drawdown_durations(values)

    if strategy_type == "regime_adaptive":
        diagnostics = backtest.get("diagnostics", pd.DataFrame()).copy()
        if not diagnostics.empty:
            diagnostics["date"] = pd.to_datetime(diagnostics["date"])
            diagnostics = diagnostics.loc[
                diagnostics["date"].between(evaluation_start, evaluation_end)
            ]
        turnover = float(diagnostics.get("turnover", pd.Series(dtype=float)).sum())
        transaction_cost = float(
            diagnostics.get("transaction_cost", pd.Series(dtype=float)).sum()
        )
        rebalances = int(
            diagnostics.get("rebalanced", pd.Series(dtype=bool)).sum()
        )
        average_risky = float(diagnostics["risky_exposure"].mean())
        average_defensive = float(diagnostics["defensive_weight"].mean())
        applied = backtest.get("applied_regimes", pd.Series(dtype=object))
        stress_period_return = _stress_period_return(returns, applied)
    else:
        rebalance_log = backtest.get("rebalance_log", pd.DataFrame()).copy()
        if not rebalance_log.empty:
            rebalance_log["rebalance_date"] = pd.to_datetime(
                rebalance_log["rebalance_date"]
            )
            rebalance_log = rebalance_log.loc[
                rebalance_log["rebalance_date"].between(
                    evaluation_start,
                    evaluation_end,
                )
            ]
        turnover = float(
            rebalance_log.get("turnover", pd.Series(dtype=float)).sum()
        )
        transaction_cost = float(
            rebalance_log.get("transaction_cost", pd.Series(dtype=float)).sum()
        )
        rebalances = int(len(rebalance_log))
        average_risky = 1.0
        average_defensive = 0.0
        stress_period_return = _fixed_stress_period_return(returns)

    return {
        "strategy": strategy_name,
        "strategy_type": strategy_type,
        "cagr": metrics["cagr"],
        "volatility": metrics["volatility"],
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
        "calmar": metrics["calmar"],
        "pain_index": metrics["pain_index"],
        "pain_ratio": metrics["pain_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "final_value": float(values.iloc[-1]),
        "var_95": metrics["var_95"],
        "cvar_95": metrics["cvar_95"],
        "drawdown_duration": int(durations["max_drawdown_duration"]),
        "recovery_duration": _maximum_drawdown_recovery_duration(returns),
        "turnover": turnover,
        "transaction_cost": transaction_cost,
        "number_of_rebalances": rebalances,
        "average_risky_exposure": average_risky,
        "average_defensive_weight": average_defensive,
        "stress_period_return": stress_period_return,
        "evaluation_start": returns.index.min(),
        "evaluation_end": returns.index.max(),
        "n_observations": int(len(returns)),
    }


def _failed_row(
    scenario: Mapping[str, object],
    strategy: str,
    strategy_type: str,
    policy_preset,
    regime_source,
    reason: str,
    *,
    status: str = "failed",
) -> dict[str, object]:
    source, annual_rate, ticker = _sleeve_settings(
        str(scenario["defensive_sleeve"])
    )
    return {
        **dict(scenario),
        "strategy": strategy,
        "strategy_type": strategy_type,
        "policy_preset": policy_preset,
        "regime_source": regime_source,
        "defensive_source_requested": source,
        "defensive_source_used": None,
        "defensive_annual_rate": annual_rate,
        "defensive_ticker": ticker,
        "defensive_fallback_used": None,
        "status": status,
        "failure_reason": reason,
    }


def _scenario_strategies(
    adaptive_variants: Sequence[tuple[str, str]],
) -> list[tuple[str, str, str | None, str | None]]:
    fixed = [(name, "fixed", None, None) for name in FIXED_STRATEGIES]
    adaptive = [
        (
            adaptive_strategy_name(source, preset),
            "regime_adaptive",
            preset,
            source,
        )
        for preset, source in adaptive_variants
    ]
    return fixed + adaptive


def _sleeve_settings(sleeve: str) -> tuple[str, float, str | None]:
    normalized = str(sleeve).strip()
    if normalized.lower() in {"synthetic_4pct", "synthetic", "synthetic risk-free"}:
        return "synthetic", 0.04, None
    source, ticker = defensive_source_from_label(normalized)
    return source, 0.04, ticker


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a DatetimeIndex")
    clean = returns.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    clean = clean.sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]
    if clean.empty or clean.shape[1] < 2:
        raise ValueError("returns must contain at least two assets and valid rows")
    return clean


def _training_window(n_observations: int) -> int:
    if n_observations <= 20:
        raise ValueError("at least 21 return observations are required")
    return min(252, max(20, n_observations // 3))


def _hmm_training_window(n_observations: int, training_window: int) -> int:
    if n_observations >= 756:
        return 504
    return min(max(training_window, 63), max(training_window, n_observations // 2))


def _window_label(start: pd.Timestamp, end: pd.Timestamp | None) -> str:
    return (
        f"{start.date().isoformat()} to "
        f"{end.date().isoformat() if end is not None else 'latest'}"
    )


def _normalized_values(returns: pd.Series) -> pd.Series:
    anchor = returns.index[0] - pd.Timedelta(nanoseconds=1)
    return pd.concat(
        [
            pd.Series([INITIAL_CAPITAL], index=[anchor], dtype=float),
            (1.0 + returns).cumprod() * INITIAL_CAPITAL,
        ]
    )


def _maximum_drawdown_recovery_duration(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    trough_date = drawdown.idxmin()
    peak_value = float(running_peak.loc[trough_date])
    recovered = wealth.loc[trough_date:]
    recovered = recovered.loc[recovered >= peak_value]
    if recovered.empty:
        return np.nan
    return float(len(wealth.loc[trough_date : recovered.index[0]]) - 1)


def _stress_period_return(
    returns: pd.Series,
    regimes,
) -> float:
    if not isinstance(regimes, pd.Series):
        return _fixed_stress_period_return(returns)
    aligned = regimes.reindex(returns.index).fillna("Unknown").astype(str)
    mask = aligned.str.lower().isin(
        {"stress", "crisis", "risk-off", "risk_off", "risk off"}
    )
    stress_returns = returns.loc[mask]
    return (
        float((1.0 + stress_returns).prod() - 1.0)
        if not stress_returns.empty
        else np.nan
    )


def _fixed_stress_period_return(returns: pd.Series) -> float:
    covid = returns.loc["2020-02-01":"2020-04-30"]
    if not covid.empty:
        return float((1.0 + covid).prod() - 1.0)
    window = min(63, len(returns))
    rolling = (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0
    return float(rolling.min()) if rolling.notna().any() else np.nan


def _cost_sensitivity_slope(group: pd.DataFrame) -> float:
    slopes: list[float] = []
    for _, matched in group.groupby(
        ["universe", "date_window", "defensive_sleeve"],
        dropna=False,
    ):
        x = (
            pd.to_numeric(matched["base_bps"], errors="coerce")
            + pd.to_numeric(matched["slippage_bps"], errors="coerce")
        )
        y = pd.to_numeric(matched["calmar"], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 2 or x.loc[valid].nunique() < 2:
            continue
        slopes.append(float(np.polyfit(x.loc[valid], y.loc[valid], 1)[0]))
    return float(np.mean(slopes)) if slopes else np.nan


def _faster_rerisk_policy(policy_map):
    tuned = dict(policy_map)
    tuned["Normal"] = replace(
        tuned["Normal"],
        target_volatility=0.10,
        notes=f"{tuned['Normal'].notes} Faster re-risking variant.",
    )
    tuned["Stress"] = replace(
        tuned["Stress"],
        risky_exposure_cap=0.80,
        defensive_weight_floor=0.15,
        notes=f"{tuned['Stress'].notes} Faster re-risking variant.",
    )
    tuned["Crisis"] = replace(
        tuned["Crisis"],
        risky_exposure_cap=0.50,
        defensive_weight_floor=0.30,
        notes=f"{tuned['Crisis'].notes} Faster re-risking variant.",
    )
    return tuned


def _flag_policy_tuning_findings(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    frame = results.copy()
    for column in (
        "best_drawdown_preserving_improvement",
        "best_recovery_improvement",
        "best_calmar_improvement",
        "overfit_or_drawdown_cost",
    ):
        frame[column] = False
    successful = frame.loc[frame["status"].eq("success")].copy()
    if successful.empty:
        return frame

    comparisons = []
    for source, group in successful.groupby("regime_source"):
        base = group.loc[group["policy_variant"].eq("Conservative base")]
        faster = group.loc[
            group["policy_variant"].eq("Conservative faster re-risking")
        ]
        if base.empty or faster.empty:
            continue
        base_row = base.iloc[0]
        faster_row = faster.iloc[0]
        comparisons.append(
            {
                "index": faster.index[0],
                "calmar_gain": faster_row["calmar"] - base_row["calmar"],
                "recovery_gain": (
                    base_row["recovery_duration"] - faster_row["recovery_duration"]
                ),
                "final_value_gain": faster_row["final_value"] - base_row["final_value"],
                "drawdown_change": (
                    abs(faster_row["max_drawdown"])
                    - abs(base_row["max_drawdown"])
                ),
                "drawdown_relative_change": (
                    (
                        abs(faster_row["max_drawdown"])
                        / abs(base_row["max_drawdown"])
                    )
                    - 1.0
                    if abs(base_row["max_drawdown"]) > 0.0
                    else np.nan
                ),
            }
        )
    if not comparisons:
        return frame
    comparison = pd.DataFrame(comparisons).set_index("index")
    drawdown_safe = comparison.loc[
        (comparison["drawdown_change"] <= 0.02)
        & (comparison["final_value_gain"] > 0.0)
    ]
    if not drawdown_safe.empty:
        frame.loc[
            drawdown_safe["final_value_gain"].idxmax(),
            "best_drawdown_preserving_improvement",
        ] = True
    finite_recovery = comparison.loc[
        comparison["recovery_gain"] > 0.0,
        "recovery_gain",
    ].dropna()
    if not finite_recovery.empty:
        frame.loc[
            finite_recovery.idxmax(),
            "best_recovery_improvement",
        ] = True
    finite_calmar = comparison.loc[
        comparison["calmar_gain"] > 0.0,
        "calmar_gain",
    ].dropna()
    if not finite_calmar.empty:
        frame.loc[
            finite_calmar.idxmax(),
            "best_calmar_improvement",
        ] = True
    frame.loc[
        comparison.index[
            (comparison["drawdown_change"] > 0.02)
            | (comparison["drawdown_relative_change"] > 0.10)
        ],
        "overfit_or_drawdown_cost",
    ] = True
    return frame


def _validate_objective(objective: str) -> None:
    if str(objective).strip().lower() not in {
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "pain_ratio",
        "max_drawdown",
        "final_value",
    }:
        raise ValueError(f"unsupported replication objective '{objective}'")


def _log_progress(completed: int, total: int) -> None:
    logger.info("Replication progress: %s/%s runs completed", completed, total)
