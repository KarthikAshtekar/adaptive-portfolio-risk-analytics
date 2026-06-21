"""Scenario-based strategy-selection public API."""

from src.selection.config import (
    COST_ASSUMPTIONS,
    COST_ASSUMPTION_NAMES,
    INVESTOR_PROFILES,
    PROFILE_NAMES,
    SCENARIO_CATEGORIES,
)
from src.selection.gates import GateResult, GateStatus, evaluate_selection_gates
from src.selection.playbook import build_strategy_playbook
from src.selection.selector import (
    StrategyRecommendation,
    classify_scenarios,
    load_selection_artifacts,
    select_strategy_for_profile,
)

__all__ = [
    "COST_ASSUMPTIONS",
    "COST_ASSUMPTION_NAMES",
    "INVESTOR_PROFILES",
    "PROFILE_NAMES",
    "SCENARIO_CATEGORIES",
    "GateResult",
    "GateStatus",
    "StrategyRecommendation",
    "build_strategy_playbook",
    "classify_scenarios",
    "evaluate_selection_gates",
    "load_selection_artifacts",
    "select_strategy_for_profile",
]
