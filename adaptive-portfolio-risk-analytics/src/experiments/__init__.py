"""Experiment orchestration exports for Phase 2D and Phase 3D."""

from .adaptive import (
    AdaptiveExperimentSkipped,
    adaptive_strategy_name,
    build_adaptive_attribution,
    build_adaptive_regime_input,
    execute_adaptive_experiment,
    generate_adaptive_parameter_grid,
    run_adaptive_experiment_grid,
    summarize_adaptive_diagnostics,
)
from .adaptive_evaluation import (
    build_adaptive_stress_comparison,
    compare_adaptive_vs_fixed,
)
from .config import (
    ADAPTIVE_POLICY_PRESETS,
    ADAPTIVE_REGIME_SOURCES,
    FULL_SAMPLE_HMM_ERROR,
    AdaptiveExperimentConfig,
    ExperimentConfig,
    default_phase2d_config,
    normalize_adaptive_policy_preset,
    normalize_adaptive_regime_source,
)
from .reporting import (
    build_experiment_summary_table,
    build_parameter_pivot,
    build_top_n_table,
    export_experiment_results,
    log_experiment_to_mlflow,
)
from .runner import generate_parameter_grid, run_experiment_grid, run_single_experiment
from .replication import (
    DEFAULT_REPLICATION_UNIVERSES,
    run_policy_tuning_study,
    run_replication_study,
    summarize_replication_results,
)
from .sensitivity import (
    compute_parameter_sensitivity,
    rank_experiments,
    summarize_by_parameter,
)

__all__ = [
    "ExperimentConfig",
    "AdaptiveExperimentConfig",
    "ADAPTIVE_REGIME_SOURCES",
    "ADAPTIVE_POLICY_PRESETS",
    "FULL_SAMPLE_HMM_ERROR",
    "normalize_adaptive_regime_source",
    "normalize_adaptive_policy_preset",
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
    "AdaptiveExperimentSkipped",
    "adaptive_strategy_name",
    "generate_adaptive_parameter_grid",
    "build_adaptive_regime_input",
    "execute_adaptive_experiment",
    "run_adaptive_experiment_grid",
    "summarize_adaptive_diagnostics",
    "build_adaptive_attribution",
    "compare_adaptive_vs_fixed",
    "build_adaptive_stress_comparison",
    "DEFAULT_REPLICATION_UNIVERSES",
    "run_replication_study",
    "summarize_replication_results",
    "run_policy_tuning_study",
]
