"""Reporting and export helpers for experiment studies."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .sensitivity import rank_experiments


def build_experiment_summary_table(
    experiment_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a compact summary of successful experiments."""
    if not isinstance(experiment_results_df, pd.DataFrame):
        raise TypeError("experiment_results_df must be a pandas DataFrame")
    if experiment_results_df.empty:
        raise ValueError("experiment_results_df must not be empty")

    columns = [
        "strategy",
        "covariance_method",
        "rebalance_mode",
        "threshold",
        "vol_targeting_enabled",
        "cagr",
        "sharpe",
        "sortino",
        "volatility",
        "max_drawdown",
        "calmar",
        "final_value",
        "status",
    ]
    selected = [column for column in columns if column in experiment_results_df.columns]
    return experiment_results_df[selected].copy()


def build_top_n_table(
    experiment_results_df: pd.DataFrame,
    metric: str = "calmar",
    n: int = 10,
) -> pd.DataFrame:
    """Return the top N successful configurations by the chosen metric."""
    if n <= 0:
        raise ValueError("n must be positive")
    return rank_experiments(experiment_results_df, objective=metric).head(n).copy()


def build_parameter_pivot(
    experiment_results_df: pd.DataFrame,
    index,
    columns,
    values,
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Build a pivot table for experiment sensitivity reporting."""
    if not isinstance(experiment_results_df, pd.DataFrame):
        raise TypeError("experiment_results_df must be a pandas DataFrame")
    if experiment_results_df.empty:
        raise ValueError("experiment_results_df must not be empty")

    successful = (
        experiment_results_df[experiment_results_df["status"] == "success"]
        if "status" in experiment_results_df.columns
        else experiment_results_df
    )
    return pd.pivot_table(
        successful,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
    )


def export_experiment_results(
    experiment_results_df: pd.DataFrame,
    experiment_name: str,
    output_dir: str | Path = "outputs/experiments",
    export_json: bool = False,
) -> dict[str, str]:
    """Export experiment summary tables under outputs/experiments."""
    if not isinstance(experiment_results_df, pd.DataFrame):
        raise TypeError("experiment_results_df must be a pandas DataFrame")
    if experiment_results_df.empty:
        raise ValueError("experiment_results_df must not be empty")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = experiment_name.strip().replace(" ", "_")
    csv_path = output_path / f"{base_name}.csv"
    experiment_results_df.to_csv(csv_path, index=False)

    exported = {"csv": str(csv_path)}
    if export_json:
        json_path = output_path / f"{base_name}.json"
        experiment_results_df.to_json(json_path, orient="records", indent=2)
        exported["json"] = str(json_path)

    return exported


def log_experiment_to_mlflow(
    config_row,
    metrics,
    artifacts: Iterable[str] | None = None,
) -> bool:
    """Optionally log experiment parameters and metrics to MLflow."""
    try:
        import mlflow
    except ImportError:
        return False

    config_dict = dict(config_row)
    metrics_dict = dict(metrics)

    with mlflow.start_run(run_name=str(config_dict.get("experiment_name", "phase2d_run"))):
        mlflow.log_params(
            {
                key: value
                for key, value in config_dict.items()
                if value is not None and not isinstance(value, (dict, list, tuple, set))
            }
        )
        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in metrics_dict.items()
                if value is not None and isinstance(value, (int, float))
            }
        )
        for artifact in artifacts or []:
            mlflow.log_artifact(str(artifact))

    return True
