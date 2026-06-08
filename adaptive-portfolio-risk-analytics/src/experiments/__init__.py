"""Experiment orchestration exports for Phase 2D."""

from .config import ExperimentConfig, default_phase2d_config
from .reporting import (
    build_experiment_summary_table,
    build_parameter_pivot,
    build_top_n_table,
    export_experiment_results,
    log_experiment_to_mlflow,
)
from .runner import generate_parameter_grid, run_experiment_grid, run_single_experiment
from .sensitivity import (
    compute_parameter_sensitivity,
    rank_experiments,
    summarize_by_parameter,
)

__all__ = [
    "ExperimentConfig",
    "default_phase2d_config",
    "generate_parameter_grid",
    "run_single_experiment",
    "run_experiment_grid",
    "rank_experiments",
    "summarize_by_parameter",
    "compute_parameter_sensitivity",
    "build_experiment_summary_table",
    "build_top_n_table",
    "build_parameter_pivot",
    "export_experiment_results",
    "log_experiment_to_mlflow",
]
