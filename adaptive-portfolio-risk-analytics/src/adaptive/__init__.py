"""Phase 3C regime-aware adaptive allocation exports."""

from .backtest import run_regime_adaptive_backtest
from .controller import RegimeAdaptiveController
from .defensive import (
    DefensiveReturnResult,
    defensive_source_from_label,
    format_defensive_source,
    get_defensive_returns,
)
from .policies import (
    DEFAULT_REGIME_POLICY,
    RegimePolicy,
    get_policy_for_regime,
    get_policy_preset,
    policy_map_to_dataframe,
    validate_policy_map,
)

__all__ = [
    "RegimePolicy",
    "DEFAULT_REGIME_POLICY",
    "get_policy_for_regime",
    "validate_policy_map",
    "policy_map_to_dataframe",
    "get_policy_preset",
    "RegimeAdaptiveController",
    "run_regime_adaptive_backtest",
    "DefensiveReturnResult",
    "get_defensive_returns",
    "defensive_source_from_label",
    "format_defensive_source",
]
