"""Phase 3B market regime detection and analytics exports."""

from .analytics import (
    calculate_regime_performance,
    calculate_regime_transitions,
    calculate_strategy_regime_summary,
    select_best_strategy_by_regime,
)
from .features import calculate_regime_features
from .hmm_regime import (
    DEFAULT_HMM_FEATURE_COLUMNS,
    HMM_AVAILABLE,
    calculate_hmm_transition_matrix,
    compare_regime_methods,
    fit_hmm_full_sample,
    fit_hmm_walk_forward,
    map_hmm_states_to_regimes,
    prepare_hmm_features,
)
from .rule_based import (
    calculate_regime_state_table,
    classify_rule_based_regime,
    lag_regime_labels,
)

__all__ = [
    "calculate_regime_features",
    "classify_rule_based_regime",
    "calculate_regime_state_table",
    "lag_regime_labels",
    "calculate_regime_performance",
    "calculate_strategy_regime_summary",
    "select_best_strategy_by_regime",
    "calculate_regime_transitions",
    "HMM_AVAILABLE",
    "DEFAULT_HMM_FEATURE_COLUMNS",
    "prepare_hmm_features",
    "fit_hmm_full_sample",
    "fit_hmm_walk_forward",
    "map_hmm_states_to_regimes",
    "calculate_hmm_transition_matrix",
    "compare_regime_methods",
]
