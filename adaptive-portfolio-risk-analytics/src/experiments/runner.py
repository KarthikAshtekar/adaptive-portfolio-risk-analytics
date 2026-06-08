"""Experiment grid execution for Phase 2D."""

from __future__ import annotations

from itertools import product

import pandas as pd

from src.analytics import PerformanceAnalytics
from src.backtesting import (
    RollingBacktester,
    VolatilityTargetingConfig,
    apply_volatility_targeting,
)
from src.backtesting.transaction_costs import TransactionCostModel
from src.benchmarks import BenchmarkFactory
from src.data_pipeline import get_defensive_asset_returns

from .config import ExperimentConfig
from .reporting import log_experiment_to_mlflow


def run_single_experiment(
    returns_df: pd.DataFrame,
    config_row,
    defensive_returns=None,
) -> dict[str, object]:
    """Run one experiment configuration against the existing platform components."""
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame")
    if returns_df.empty:
        raise ValueError("returns_df must not be empty")

    row = dict(config_row)
    allocator = BenchmarkFactory.get_allocator(
        strategy_name=str(row["strategy"]),
        covariance_method=str(row["covariance_method"]),
    )
    transaction_cost_model = TransactionCostModel(
        base_bps=float(row["transaction_cost_bps"]),
        slippage_bps=float(row["slippage_bps"]),
    )
    threshold = float(row["threshold"]) if pd.notna(row.get("threshold")) else 0.05

    backtest_results = RollingBacktester(
        allocator=allocator,
        train_window=int(row["train_window"]),
        rebalance_frequency="M",
        initial_capital=float(row["initial_capital"]),
        rebalance_mode=str(row["rebalance_mode"]),
        threshold=threshold,
        transaction_cost_model=transaction_cost_model,
    ).run(returns_df)

    effective_returns = backtest_results["portfolio_returns"]
    final_value = float(backtest_results["portfolio_values"].iloc[-1])
    vol_targeting_enabled = bool(row["vol_targeting_enabled"])

    if vol_targeting_enabled:
        defensive_series = _resolve_defensive_returns(
            returns_df=returns_df,
            config_row=row,
            defensive_returns=defensive_returns,
        )
        target_vol = float(row["target_vol"]) if pd.notna(row.get("target_vol")) else 0.10
        overlay_results = apply_volatility_targeting(
            risky_returns=effective_returns,
            defensive_returns=defensive_series,
            config=VolatilityTargetingConfig(base_target_vol=target_vol),
        )
        effective_returns = overlay_results["targeted_returns"]
        final_value = float((1.0 + effective_returns).cumprod().iloc[-1] * float(row["initial_capital"]))

    metrics = PerformanceAnalytics.summary_table(effective_returns)

    result = {
        "experiment_name": row.get("experiment_name"),
        "strategy": row["strategy"],
        "covariance_method": row["covariance_method"],
        "rebalance_mode": row["rebalance_mode"],
        "threshold": row.get("threshold"),
        "transaction_cost_bps": row["transaction_cost_bps"],
        "slippage_bps": row["slippage_bps"],
        "vol_targeting_enabled": vol_targeting_enabled,
        "target_vol": row.get("target_vol"),
        "defensive_asset": row.get("defensive_asset"),
        "cagr": metrics["cagr"],
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
        "volatility": metrics["volatility"],
        "max_drawdown": metrics["max_drawdown"],
        "calmar": metrics["calmar"],
        "final_value": final_value,
        "total_turnover": float(backtest_results["performance_metrics"]["total_turnover"]),
        "average_turnover": float(backtest_results["performance_metrics"]["average_turnover"]),
        "total_transaction_cost": float(
            backtest_results["performance_metrics"]["total_transaction_cost"]
        ),
        "number_of_rebalances": int(backtest_results["performance_metrics"]["number_of_rebalances"]),
        "status": "success",
        "error": None,
    }
    return result


def generate_parameter_grid(
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Build a local-friendly parameter grid from an ExperimentConfig."""
    rows: list[dict[str, object]] = []

    for strategy, covariance_method, rebalance_mode, cost_bps, slippage_bps, vol_target in product(
        config.strategies,
        config.covariance_methods,
        config.rebalance_modes,
        config.transaction_cost_bps,
        config.slippage_bps,
        config.enable_vol_targeting,
    ):
        thresholds = config.thresholds if rebalance_mode == "threshold" else [None]
        target_vols = config.target_vols if vol_target else [None]
        defensive_assets = config.defensive_assets if vol_target else [None]

        for threshold, target_vol, defensive_asset in product(
            thresholds,
            target_vols,
            defensive_assets,
        ):
            rows.append(
                {
                    "experiment_name": config.experiment_name,
                    "strategy": strategy,
                    "covariance_method": covariance_method,
                    "rebalance_mode": rebalance_mode,
                    "threshold": threshold,
                    "transaction_cost_bps": float(cost_bps),
                    "slippage_bps": float(slippage_bps),
                    "vol_targeting_enabled": bool(vol_target),
                    "target_vol": target_vol,
                    "defensive_asset": defensive_asset,
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                    "train_window": int(config.train_window),
                    "initial_capital": float(config.initial_capital),
                }
            )

    return pd.DataFrame(rows)


def run_experiment_grid(
    returns_df: pd.DataFrame,
    config: ExperimentConfig,
    defensive_returns=None,
    max_runs=None,
) -> pd.DataFrame:
    """Run an experiment grid and keep going when individual runs fail."""
    grid = generate_parameter_grid(config)
    if max_runs is not None:
        max_runs = int(max_runs)
        if max_runs <= 0:
            raise ValueError("max_runs must be positive when provided")
        grid = grid.head(max_runs).copy()

    results: list[dict[str, object]] = []

    for _, config_row in grid.iterrows():
        try:
            result = run_single_experiment(
                returns_df=returns_df,
                config_row=config_row.to_dict(),
                defensive_returns=defensive_returns,
            )
            results.append(result)
            log_experiment_to_mlflow(config_row.to_dict(), result)
        except Exception as exc:
            failed_row = config_row.to_dict()
            failed_row.update(
                {
                    "cagr": None,
                    "sharpe": None,
                    "sortino": None,
                    "volatility": None,
                    "max_drawdown": None,
                    "calmar": None,
                    "final_value": None,
                    "total_turnover": None,
                    "average_turnover": None,
                    "total_transaction_cost": None,
                    "number_of_rebalances": None,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            results.append(failed_row)

    return pd.DataFrame(results)


def _resolve_defensive_returns(
    returns_df: pd.DataFrame,
    config_row: dict[str, object],
    defensive_returns,
) -> pd.Series:
    if isinstance(defensive_returns, pd.Series):
        return defensive_returns
    if isinstance(defensive_returns, dict):
        asset_name = config_row.get("defensive_asset")
        if asset_name in defensive_returns:
            return defensive_returns[asset_name]
        raise ValueError(f"defensive_returns does not contain asset '{asset_name}'")

    asset_name = config_row.get("defensive_asset")
    if asset_name in (None, "", "Synthetic Risk-Free"):
        defensive_series, _ = get_defensive_asset_returns(
            start_date=returns_df.index.min(),
            end_date=returns_df.index.max(),
            preferred_ticker=None,
            fallback_tickers=[],
        )
        return defensive_series

    defensive_series, _ = get_defensive_asset_returns(
        start_date=returns_df.index.min(),
        end_date=returns_df.index.max(),
        preferred_ticker=str(asset_name),
        fallback_tickers=[],
    )
    return defensive_series
