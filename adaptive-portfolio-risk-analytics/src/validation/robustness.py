"""Phase 3A fold summaries, stability scoring, and CPCV experiment validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.adaptive import defensive_source_from_label, get_defensive_returns
from src.analytics import PerformanceAnalytics
from src.backtesting import (
    RollingBacktester,
    VolatilityTargetingConfig,
    apply_volatility_targeting,
)
from src.backtesting.transaction_costs import TransactionCostModel
from src.benchmarks import BenchmarkFactory
from src.experiments.adaptive import execute_adaptive_experiment
from src.experiments.config import (
    ExperimentConfig,
    normalize_adaptive_regime_source,
)
from src.experiments.runner import generate_parameter_grid

from .cpcv import generate_cpcv_splits, generate_time_blocks

METRIC_DIRECTIONS = {
    "cumulative_return": True,
    "cagr": True,
    "sharpe": True,
    "sortino": True,
    "volatility": False,
    "max_drawdown": True,
    "var_95": True,
    "cvar_95": True,
    "calmar": True,
    "final_value": True,
}

DEFAULT_METRICS = list(METRIC_DIRECTIONS)

CONFIG_COLUMNS = [
    "experiment_name",
    "strategy",
    "strategy_type",
    "regime_source",
    "policy_preset",
    "covariance_method",
    "rebalance_mode",
    "threshold",
    "transaction_cost_bps",
    "slippage_bps",
    "vol_targeting_enabled",
    "target_vol",
    "defensive_asset",
    "defensive_source",
    "defensive_annual_rate",
    "defensive_ticker",
    "defensive_fallback",
    "defensive_source_requested",
    "defensive_source_used",
    "defensive_fallback_used",
    "defensive_notes",
    "train_window",
    "training_window",
    "rebalance_frequency",
    "hmm_n_states",
    "hmm_min_train_size",
    "hmm_refit_frequency",
    "hmm_covariance_type",
    "hmm_decision_lag",
    "initial_capital",
]

SUMMARY_COLUMNS = [
    "metric",
    "n_folds",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "best_fold",
    "worst_fold",
    "stability_score",
    "higher_is_better",
]


def _normalize_objective(objective: str | None) -> str:
    """Normalize objective names, falling back to Calmar only when absent."""
    if objective is None or not str(objective).strip():
        return "calmar"
    return str(objective).strip().lower().replace(" ", "_")


def calculate_stability_score(values, higher_is_better: bool = True) -> float:
    """Score fold consistency from 0 to 1.

    The score rewards low dispersion and a worst fold close to the mean. A
    sign-consistency factor penalizes return and risk-adjusted metrics that
    alternate between positive and negative outcomes.
    """
    clean = np.asarray(list(values), dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return 0.0
    if clean.size == 1:
        return 1.0

    mean = float(clean.mean())
    std = float(clean.std(ddof=0))
    scale = max(abs(mean), float(np.mean(np.abs(clean))), 1e-12)

    dispersion_component = 1.0 / (1.0 + std / scale)
    worst_value = float(clean.min() if higher_is_better else clean.max())
    adverse_gap = mean - worst_value if higher_is_better else worst_value - mean
    worst_fold_component = 1.0 / (1.0 + max(0.0, adverse_gap) / scale)

    positive_count = int((clean > 0.0).sum())
    negative_count = int((clean < 0.0).sum())
    if positive_count and negative_count:
        sign_component = max(positive_count, negative_count) / clean.size
    else:
        sign_component = 1.0

    score = 0.55 * dispersion_component + 0.30 * worst_fold_component + 0.15 * sign_component
    return float(np.clip(score, 0.0, 1.0))


def summarize_fold_metrics(
    fold_results,
    metric_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize fold-level metric dictionaries into a long-form table."""
    frame = (
        fold_results.copy()
        if isinstance(fold_results, pd.DataFrame)
        else pd.DataFrame(fold_results)
    )
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    if "status" in frame.columns:
        frame = frame[frame["status"].eq("success")].copy()
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    if metric_names is None:
        metric_names = [
            metric
            for metric in DEFAULT_METRICS
            if metric in frame.columns and pd.api.types.is_numeric_dtype(frame[metric])
        ]

    fold_labels = (
        frame["split_id"]
        if "split_id" in frame.columns
        else pd.Series(frame.index, index=frame.index)
    )
    rows: list[dict[str, object]] = []

    for metric in metric_names:
        if metric not in frame.columns:
            continue

        numeric = pd.to_numeric(frame[metric], errors="coerce")
        valid_mask = np.isfinite(numeric.to_numpy(dtype=float))
        values = numeric[valid_mask]
        if values.empty:
            continue

        labels = fold_labels[valid_mask]
        higher_is_better = METRIC_DIRECTIONS.get(str(metric).lower(), True)
        best_position = values.idxmax() if higher_is_better else values.idxmin()
        worst_position = values.idxmin() if higher_is_better else values.idxmax()

        rows.append(
            {
                "metric": metric,
                "n_folds": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
                "best_fold": labels.loc[best_position],
                "worst_fold": labels.loc[worst_position],
                "stability_score": calculate_stability_score(
                    values.to_numpy(),
                    higher_is_better=higher_is_better,
                ),
                "higher_is_better": higher_is_better,
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def rank_by_robustness(
    summary: pd.DataFrame,
    objective: str | None = None,
) -> pd.DataFrame:
    """Rank configurations using median, worst-fold, and stability components.

    Median performance receives 50% weight, worst-fold performance 30%, and
    stability 20%. Median and worst-fold values are converted to favorable
    percentile ranks so the score remains transparent across metric scales.
    """
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("summary must be a pandas DataFrame")
    if summary.empty:
        return pd.DataFrame()
    if "metric" not in summary.columns:
        raise ValueError("summary must contain a 'metric' column")

    selected_objective = _normalize_objective(objective)
    objective_rows = summary[
        summary["metric"].astype(str).str.lower().eq(selected_objective)
    ].copy()
    if objective_rows.empty:
        raise ValueError(f"objective '{selected_objective}' is not present in summary")

    higher_is_better = bool(objective_rows["higher_is_better"].iloc[0])
    objective_rows["objective_median"] = pd.to_numeric(objective_rows["median"], errors="coerce")
    objective_rows["objective_worst"] = pd.to_numeric(
        objective_rows["min" if higher_is_better else "max"],
        errors="coerce",
    )
    objective_rows["median_component"] = objective_rows["objective_median"].rank(
        method="average",
        pct=True,
        ascending=higher_is_better,
    )
    objective_rows["worst_fold_component"] = objective_rows["objective_worst"].rank(
        method="average",
        pct=True,
        ascending=higher_is_better,
    )
    objective_rows["robustness_score"] = (
        0.50 * objective_rows["median_component"].fillna(0.0)
        + 0.30 * objective_rows["worst_fold_component"].fillna(0.0)
        + 0.20 * pd.to_numeric(objective_rows["stability_score"], errors="coerce").fillna(0.0)
    )

    sort_columns = ["robustness_score", "objective_median", "stability_score"]
    ascending = [False, not higher_is_better, False]
    return objective_rows.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
    ).reset_index(drop=True)


def _coerce_experiment_configs(experiment_configs) -> pd.DataFrame:
    if isinstance(experiment_configs, ExperimentConfig):
        frame = generate_parameter_grid(experiment_configs)
    elif isinstance(experiment_configs, pd.DataFrame):
        frame = experiment_configs.copy()
    elif isinstance(experiment_configs, Mapping):
        frame = pd.DataFrame([dict(experiment_configs)])
    else:
        frame = pd.DataFrame(list(experiment_configs))

    if "status" in frame.columns:
        frame = frame[frame["status"].eq("success")].copy()
    return frame.reset_index(drop=True)


def _config_value(row: Mapping[str, object], key: str, default):
    value = row.get(key, default)
    if value is None:
        return default
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    return default if isinstance(missing, (bool, np.bool_)) and missing else value


def _defensive_settings(
    row: Mapping[str, object],
) -> tuple[str, float, str | None, str]:
    label_source, label_ticker = defensive_source_from_label(
        str(_config_value(row, "defensive_asset", "Synthetic Risk-Free"))
    )
    source = str(_config_value(row, "defensive_source", label_source))
    annual_rate = float(_config_value(row, "defensive_annual_rate", 0.04))
    ticker_value = _config_value(row, "defensive_ticker", label_ticker)
    ticker = str(ticker_value) if ticker_value not in (None, "") else None
    fallback = str(_config_value(row, "defensive_fallback", "synthetic"))
    return source, annual_rate, ticker, fallback


def _run_test_block(
    returns: pd.DataFrame,
    split: Mapping[str, object],
    test_index: pd.DatetimeIndex,
    config_row: Mapping[str, object],
    defensive_returns=None,
) -> pd.Series:
    train_window = int(_config_value(config_row, "train_window", 252))
    strategy_type = str(_config_value(config_row, "strategy_type", "fixed")).strip().lower()
    is_adaptive = strategy_type == "regime_adaptive"
    history_requirement = train_window
    if is_adaptive:
        source = normalize_adaptive_regime_source(
            str(_config_value(config_row, "regime_source", "rule_based_lagged"))
        )
        feature_warmup = 126
        history_requirement = train_window + feature_warmup
        if source == "hmm_walk_forward":
            history_requirement = (
                max(
                    train_window,
                    int(_config_value(config_row, "hmm_min_train_size", 504)),
                )
                + feature_warmup
            )

    test_start = test_index[0]
    permitted_train = split["train_index"]
    permitted_past = permitted_train[permitted_train < test_start]
    if len(permitted_past) < history_requirement:
        raise ValueError(
            f"insufficient permitted history before {test_start.date()}: "
            f"need {history_requirement}, found {len(permitted_past)}"
        )

    context_index = permitted_past[-history_requirement:].append(test_index)
    context_returns = returns.loc[returns.index.isin(context_index)].sort_index()

    if is_adaptive:
        execution = execute_adaptive_experiment(
            context_returns,
            config_row,
            defensive_returns=defensive_returns,
        )
        effective_returns = execution["backtest"]["portfolio_returns"]
        observed_test_returns = effective_returns.reindex(test_index).dropna()
        if observed_test_returns.empty:
            raise ValueError("adaptive backtest produced no returns for the test block")
        observed_test_returns.attrs["defensive_metadata"] = execution["backtest"].get(
            "defensive_metadata",
            {},
        )
        return observed_test_returns

    allocator = BenchmarkFactory.get_allocator(
        strategy_name=str(_config_value(config_row, "strategy", "Equal Weight")),
        covariance_method=str(_config_value(config_row, "covariance_method", "sample")),
    )
    transaction_cost_model = TransactionCostModel(
        base_bps=float(_config_value(config_row, "transaction_cost_bps", 10.0)),
        slippage_bps=float(_config_value(config_row, "slippage_bps", 5.0)),
    )
    backtest = RollingBacktester(
        allocator=allocator,
        train_window=train_window,
        rebalance_frequency="M",
        initial_capital=float(_config_value(config_row, "initial_capital", 1_000_000.0)),
        rebalance_mode=str(_config_value(config_row, "rebalance_mode", "calendar")),
        threshold=float(_config_value(config_row, "threshold", 0.05)),
        transaction_cost_model=transaction_cost_model,
    ).run(context_returns)

    effective_returns = backtest["portfolio_returns"]
    if bool(_config_value(config_row, "vol_targeting_enabled", False)):
        source, annual_rate, ticker, fallback = _defensive_settings(config_row)
        defensive_result = get_defensive_returns(
            index=effective_returns.index,
            source=source,
            annual_rate=annual_rate,
            defensive_ticker=ticker,
            returns=defensive_returns,
            fallback=fallback,
        )
        overlay = apply_volatility_targeting(
            risky_returns=effective_returns,
            defensive_returns=defensive_result.returns,
            config=VolatilityTargetingConfig(
                base_target_vol=float(_config_value(config_row, "target_vol", 0.10))
            ),
        )
        effective_returns = overlay["targeted_returns"]

    observed_test_returns = effective_returns.reindex(test_index).dropna()
    if observed_test_returns.empty:
        raise ValueError("backtest produced no returns for the test block")
    if bool(_config_value(config_row, "vol_targeting_enabled", False)):
        observed_test_returns.attrs["defensive_metadata"] = defensive_result.metadata
    return observed_test_returns


def run_cpcv_validation(
    returns,
    experiment_configs,
    benchmark_returns=None,
    n_blocks: int = 6,
    n_test_blocks: int = 2,
    embargo_pct: float = 0.01,
    purge_window: int = 0,
    objective: str | None = None,
    max_configs: int | None = None,
    max_adaptive_configs: int | None = None,
    defensive_returns=None,
) -> dict[str, pd.DataFrame]:
    """Evaluate experiment configurations across CPCV-style walk-forward folds."""
    selected_objective = _normalize_objective(objective)
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.empty:
        empty = pd.DataFrame()
        return {
            "fold_results": empty,
            "summary_table": empty,
            "robustness_ranking": empty,
            "split_diagnostics": empty,
        }
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a DatetimeIndex")

    clean_returns = returns.dropna(how="any").sort_index()
    clean_returns = clean_returns[~clean_returns.index.duplicated(keep="last")]
    configs = _coerce_experiment_configs(experiment_configs)
    adaptive_mask = (
        configs["strategy_type"].fillna("fixed").astype(str).eq("regime_adaptive")
        if "strategy_type" in configs.columns
        else pd.Series(False, index=configs.index)
    )
    if adaptive_mask.any():
        if "regime_source" not in configs.columns:
            raise ValueError("adaptive configurations must include regime_source")
        for source in configs.loc[adaptive_mask, "regime_source"].dropna():
            normalize_adaptive_regime_source(str(source))

    if max_adaptive_configs is not None:
        adaptive_limit = int(max_adaptive_configs)
        if adaptive_limit <= 0:
            raise ValueError("max_adaptive_configs must be positive when provided")
        adaptive_configs = configs.loc[adaptive_mask].head(adaptive_limit)
        fixed_configs = configs.loc[~adaptive_mask]
        if max_configs is not None:
            if int(max_configs) <= 0:
                raise ValueError("max_configs must be positive when provided")
            fixed_configs = fixed_configs.head(int(max_configs))
        configs = pd.concat(
            [fixed_configs, adaptive_configs],
            ignore_index=True,
        )
    elif max_configs is not None:
        if int(max_configs) <= 0:
            raise ValueError("max_configs must be positive when provided")
        configs = configs.head(int(max_configs)).copy()

    splits = generate_cpcv_splits(
        clean_returns.index,
        n_blocks=n_blocks,
        n_test_blocks=n_test_blocks,
        embargo_pct=embargo_pct,
        purge_window=purge_window,
    )
    blocks = generate_time_blocks(clean_returns.index, n_blocks)
    split_diagnostics = pd.DataFrame(
        [
            {key: value for key, value in split.items() if key not in {"train_index", "test_index"}}
            for split in splits
        ]
    )

    fold_rows: list[dict[str, object]] = []
    for config_id, config_series in configs.iterrows():
        config_row = config_series.to_dict()
        config_metadata = {
            column: config_row.get(column) for column in CONFIG_COLUMNS if column in config_row
        }

        for split in splits:
            base_row = {
                "config_id": int(config_id),
                **config_metadata,
                "split_id": split["split_id"],
                "test_block_ids": split["test_block_ids"],
                "n_train": split["n_train"],
                "n_test": split["n_test"],
            }
            try:
                block_returns: list[pd.Series] = []
                block_defensive_metadata: list[dict[str, object]] = []
                for block_id in split["test_block_ids"]:
                    test_index = blocks[int(block_id)]["dates"]
                    block_result = _run_test_block(
                        returns=clean_returns,
                        split=split,
                        test_index=test_index,
                        config_row=config_row,
                        defensive_returns=defensive_returns,
                    )
                    block_returns.append(block_result)
                    metadata = block_result.attrs.get("defensive_metadata")
                    if isinstance(metadata, Mapping):
                        block_defensive_metadata.append(dict(metadata))

                fold_returns = pd.concat(block_returns).sort_index()
                fold_returns = fold_returns[~fold_returns.index.duplicated(keep="last")]
                metrics = PerformanceAnalytics.summary_table(fold_returns)
                initial_capital = float(_config_value(config_row, "initial_capital", 1_000_000.0))
                metrics["final_value"] = float(initial_capital * (1.0 + fold_returns).prod())

                if benchmark_returns is not None:
                    benchmark_series = (
                        pd.Series(benchmark_returns).reindex(fold_returns.index).dropna()
                    )
                    if not benchmark_series.empty:
                        metrics["benchmark_cagr"] = PerformanceAnalytics.cagr(benchmark_series)
                        metrics["excess_cagr"] = metrics["cagr"] - metrics["benchmark_cagr"]

                fold_rows.append(
                    {
                        **base_row,
                        **metrics,
                        **(block_defensive_metadata[0] if block_defensive_metadata else {}),
                        "n_test_observed": len(fold_returns),
                        "status": "success",
                        "error": None,
                    }
                )
            except Exception as exc:
                fold_rows.append(
                    {
                        **base_row,
                        "n_test_observed": 0,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    fold_results = pd.DataFrame(fold_rows)
    summary_parts: list[pd.DataFrame] = []
    if not fold_results.empty:
        for config_id, group in fold_results.groupby("config_id", sort=True):
            config_summary = summarize_fold_metrics(group, metric_names=DEFAULT_METRICS)
            if config_summary.empty:
                continue
            config_summary.insert(0, "config_id", config_id)
            first_row = group.iloc[0]
            insert_at = 1
            for column in CONFIG_COLUMNS:
                if column in group.columns:
                    config_summary.insert(insert_at, column, first_row[column])
                    insert_at += 1
            summary_parts.append(config_summary)

    summary_table = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    robustness_ranking = (
        rank_by_robustness(summary_table, objective=selected_objective)
        if not summary_table.empty
        and summary_table["metric"].astype(str).str.lower().eq(selected_objective).any()
        else pd.DataFrame()
    )

    return {
        "fold_results": fold_results,
        "summary_table": summary_table,
        "robustness_ranking": robustness_ranking,
        "split_diagnostics": split_diagnostics,
    }
