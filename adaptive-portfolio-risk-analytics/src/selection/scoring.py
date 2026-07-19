"""Profile-aware scoring and role classification."""

from __future__ import annotations

from math import isfinite
from typing import Mapping

import pandas as pd

from src.selection.config import (
    BENCHMARK_ROLE,
    EQUAL_WEIGHT,
    EXPERIMENTAL_ROLE,
    HERC,
    HMM_CONSERVATIVE,
    INVESTOR_PROFILES,
    MAIN_GROWTH_ROLE,
    REJECTED_ROLE,
    RISK_CONTROL_ROLE,
    ROBUSTNESS_ROLE,
    RULE_CONSERVATIVE,
)
from src.selection.gates import GateResult, GateStatus


def _series(frame: pd.DataFrame, column: str, *, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(
        frame.get(column, pd.Series(index=frame.index, dtype=float)), errors="coerce"
    )
    if values.notna().sum() <= 1 or values.max() == values.min():
        result = pd.Series(0.5, index=frame.index, dtype=float)
        return result.where(values.notna())
    result = (values - values.min()) / (values.max() - values.min())
    return result if higher_is_better else 1.0 - result


def classify_strategy_roles(
    candidates: pd.DataFrame,
    gates_by_strategy: Mapping[str, list[GateResult]] | None = None,
) -> dict[str, str]:
    """Assign operational roles without treating adaptive as a HERC replacement."""

    gates_by_strategy = gates_by_strategy or {}
    roles: dict[str, str] = {}
    for strategy, row in candidates.iterrows():
        failed = any(
            gate.status == GateStatus.FAIL for gate in gates_by_strategy.get(str(strategy), [])
        )
        if failed:
            roles[str(strategy)] = REJECTED_ROLE
        elif str(strategy) == EQUAL_WEIGHT:
            roles[str(strategy)] = BENCHMARK_ROLE
        elif str(strategy) == HERC:
            roles[str(strategy)] = MAIN_GROWTH_ROLE
        elif str(strategy) == HMM_CONSERVATIVE:
            roles[str(strategy)] = RISK_CONTROL_ROLE
        elif str(strategy) == RULE_CONSERVATIVE:
            roles[str(strategy)] = ROBUSTNESS_ROLE
        elif str(row.get("strategy_type", "")).lower() == "fixed":
            roles[str(strategy)] = EXPERIMENTAL_ROLE
        else:
            roles[str(strategy)] = EXPERIMENTAL_ROLE
    return roles


def score_candidates(
    candidates: pd.DataFrame,
    *,
    profile_name: str,
    gates_by_strategy: Mapping[str, list[GateResult]],
    cpcv_by_strategy: Mapping[str, Mapping[str, object]] | None = None,
    scenario_categories: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Score candidates on net growth, protection, robustness, and cost."""

    if profile_name not in INVESTOR_PROFILES:
        raise ValueError(f"Unknown investor profile: {profile_name}")
    if candidates.empty:
        return pd.DataFrame()

    frame = candidates.copy()
    frame.index = frame.index.astype(str)
    profile = INVESTOR_PROFILES[profile_name]
    growth = 0.55 * _series(frame, "cagr") + 0.45 * _series(frame, "final_value")
    protection = 0.45 * _series(frame, "calmar") + 0.55 * _series(frame, "max_drawdown")
    stress = _series(frame, "stress_period_return")
    protection = protection.fillna(0.5) * 0.75 + stress.fillna(0.5) * 0.25
    cost = 0.55 * _series(frame, "total_turnover", higher_is_better=False) + 0.45 * _series(
        frame, "total_transaction_cost", higher_is_better=False
    )

    cpcv_by_strategy = cpcv_by_strategy or {}
    robustness_values: dict[str, float] = {}
    for strategy in frame.index:
        cpcv = cpcv_by_strategy.get(strategy, {})
        try:
            score = float(cpcv.get("robustness_score"))
        except (TypeError, ValueError):
            score = 0.5
        robustness_values[strategy] = score if isfinite(score) else 0.5
    robustness = pd.Series(robustness_values, dtype=float).clip(0.0, 1.0)

    score = (
        profile.growth_weight * growth.fillna(0.5)
        + profile.drawdown_weight * protection.fillna(0.5)
        + profile.robustness_weight * robustness
        + profile.cost_weight * cost.fillna(0.5)
    )

    penalties = pd.Series(0.0, index=frame.index)
    for strategy in frame.index:
        gates = gates_by_strategy.get(strategy, [])
        penalties.loc[strategy] += 0.30 * sum(g.status == GateStatus.FAIL for g in gates)
        penalties.loc[strategy] += 0.06 * sum(g.status == GateStatus.WARN for g in gates)
        penalties.loc[strategy] += 0.02 * sum(g.status == GateStatus.NOT_AVAILABLE for g in gates)

    if "High Cost" in scenario_categories:
        penalties += (1.0 - cost.fillna(0.5)) * 0.15
    if "Stress" in scenario_categories or "Crisis" in scenario_categories:
        score += protection.fillna(0.5) * 0.10
    if "Low CPCV Confidence" in scenario_categories:
        score += robustness * 0.08

    result = pd.DataFrame(
        {
            "growth_score": growth,
            "protection_score": protection,
            "robustness_score": robustness,
            "cost_score": cost,
            "gate_penalty": penalties,
            "selection_score": (score - penalties).clip(lower=0.0, upper=1.0),
        }
    )
    return result.sort_values("selection_score", ascending=False)
