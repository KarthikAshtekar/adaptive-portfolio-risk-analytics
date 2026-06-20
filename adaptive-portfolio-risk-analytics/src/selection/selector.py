"""Scenario-based, evidence-gated strategy selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.selection.config import (
    EQUAL_WEIGHT,
    HERC,
    HMM_CONSERVATIVE,
    INVESTOR_PROFILES,
    PROFILE_NAMES,
    REJECTED_ROLE,
    RISK_CONTROL_ROLE,
    ROBUSTNESS_ROLE,
    RULE_CONSERVATIVE,
)
from src.selection.explanations import build_recommendation_explanation
from src.selection.gates import GateResult, GateStatus, evaluate_selection_gates, gate_summary
from src.selection.scoring import classify_strategy_roles, score_candidates


@dataclass
class StrategyRecommendation:
    investor_profile: str
    scenario_categories: tuple[str, ...]
    main_strategy: str
    overlay_strategy: str | None
    overlay_role: str | None
    confidence: str
    confidence_score: float
    evidence: dict[str, object]
    gate_results: dict[str, list[GateResult]]
    role_assignments: dict[str, str]
    candidate_scores: pd.DataFrame
    assumptions: list[str]
    explanation: str
    warnings: list[str]
    artifact_diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["candidate_scores"] = self.candidate_scores.reset_index().to_dict("records")
        result["gate_results"] = {
            strategy: [gate.to_dict() for gate in gates]
            for strategy, gates in self.gate_results.items()
        }
        return result


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def load_selection_artifacts(repo_root: str | Path | None = None) -> dict[str, object]:
    """Load Phase 3E evidence, falling back to post-P0 artifacts without crashing."""

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    phase3e = root / "outputs" / "reports" / "phase_3e_replication"
    post_p0 = root / "outputs" / "reports" / "post_p0_adaptive_validation"
    replication_results = _read_csv(phase3e / "replication_results.csv")
    replication_summary = _read_csv(phase3e / "replication_summary.csv")
    fallback_used = replication_results.empty
    metrics = _read_csv(post_p0 / "metrics_comparison.csv")
    cpcv = _read_csv(post_p0 / "cpcv_summary.csv")
    stress = _read_csv(post_p0 / "stress_period_comparison.csv")
    warnings: list[str] = []
    if fallback_used:
        warnings.append("Phase 3E replication artifacts were unavailable; post-P0 evidence was used.")
    if metrics.empty and replication_results.empty:
        warnings.append("No persisted candidate metrics were available.")
    return {
        "repo_root": root,
        "replication_results": replication_results,
        "replication_summary": replication_summary,
        "metrics_comparison": metrics,
        "cpcv_summary": cpcv,
        "stress_comparison": stress,
        "phase3e_available": not replication_results.empty,
        "fallback_used": fallback_used,
        "warnings": warnings,
        "paths": {
            "phase3e": str(phase3e),
            "post_p0": str(post_p0),
        },
    }


def _baseline_candidates(artifacts: Mapping[str, object]) -> pd.DataFrame:
    replication = artifacts.get("replication_results", pd.DataFrame())
    if isinstance(replication, pd.DataFrame) and not replication.empty:
        preferred = replication.copy()
        for column, value in {
            "universe": "Core Diversified",
            "date_window": "2020-01-01 to latest",
            "cost_scenario": "10 bps + 5 bps",
            "defensive_sleeve": "synthetic_4pct",
            "status": "success",
        }.items():
            if column in preferred:
                preferred = preferred.loc[preferred[column].astype(str).eq(value)]
        if not preferred.empty:
            preferred = preferred.drop_duplicates("strategy", keep="last").copy()
            preferred["return_basis"] = "net"
            preferred["total_turnover"] = preferred.get("turnover")
            preferred["total_transaction_cost"] = preferred.get("transaction_cost")
            preferred["total_cost_bps"] = preferred.get("base_bps", 0) + preferred.get("slippage_bps", 0)
            summary = artifacts.get("replication_summary", pd.DataFrame())
            if isinstance(summary, pd.DataFrame) and not summary.empty:
                preferred = preferred.merge(
                    summary[["strategy", "classification", "stress_protection_win_rate", "cost_sensitivity_slope"]],
                    on="strategy",
                    how="left",
                )
            return preferred.set_index("strategy")

    metrics = artifacts.get("metrics_comparison", pd.DataFrame())
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        baseline = metrics.drop_duplicates("strategy", keep="last").copy()
        baseline["total_cost_bps"] = 15.0
        baseline["n_observations"] = 1603
        return baseline.set_index("strategy")
    return pd.DataFrame()


def _merge_candidates(
    candidate_metrics: pd.DataFrame | None,
    artifacts: Mapping[str, object],
    *,
    total_cost_bps: float,
) -> pd.DataFrame:
    baseline = _baseline_candidates(artifacts)
    if candidate_metrics is None or candidate_metrics.empty:
        result = baseline.copy()
    else:
        current = candidate_metrics.copy()
        if "strategy" in current:
            current = current.set_index("strategy")
        current.index = current.index.astype(str)
        result = current.combine_first(baseline)
        result.update(current)
    if result.empty:
        return result
    result.index.name = "strategy"
    result["total_cost_bps"] = float(total_cost_bps)
    if "return_basis" not in result:
        result["return_basis"] = "net"
    return result


def _records_by_strategy(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "strategy" not in frame:
        return {}
    return {
        str(row["strategy"]): row.to_dict()
        for _, row in frame.drop_duplicates("strategy", keep="first").iterrows()
    }


def _stress_by_strategy(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "strategy" not in frame:
        return {}
    grouped = (
        frame.groupby("strategy", as_index=False)
        .agg(period_return=("period_return", "mean"), max_drawdown=("max_drawdown", "min"))
    )
    return _records_by_strategy(grouped)


def classify_scenarios(
    *,
    current_regime: str | None,
    total_cost_bps: float,
    hmm_valid: bool,
    cpcv_summary: pd.DataFrame,
    n_observations: int | None,
) -> tuple[str, ...]:
    """Map available runtime evidence to the supported scenario categories."""

    regime = str(current_regime or "").strip().lower()
    scenarios: list[str] = []
    if n_observations is not None and n_observations < 504:
        scenarios.append("Insufficient Data")
    if "crisis" in regime:
        scenarios.append("Crisis")
    elif "stress" in regime:
        scenarios.append("Stress")
    elif "high volatility" in regime or "high_volatility" in regime:
        scenarios.append("High Volatility")
    elif "calm" in regime or "growth" in regime or "low volatility" in regime:
        scenarios.append("Calm / Growth")
    else:
        scenarios.append("Normal")
    if total_cost_bps >= 50:
        scenarios.append("High Cost")
    if not hmm_valid:
        scenarios.append("HMM Unstable")
    if isinstance(cpcv_summary, pd.DataFrame) and not cpcv_summary.empty:
        successful = pd.to_numeric(cpcv_summary.get("successful_folds"), errors="coerce")
        failed = pd.to_numeric(cpcv_summary.get("failed_folds"), errors="coerce").fillna(0)
        coverage = successful / (successful + failed)
        adaptive_mask = (
            cpcv_summary.get("strategy_type", pd.Series("", index=cpcv_summary.index))
            .astype(str)
            .eq("regime_adaptive")
        )
        assessed = coverage.loc[adaptive_mask] if adaptive_mask.any() else coverage
        if not assessed.dropna().empty and (assessed.dropna() < 0.60).any():
            scenarios.append("Low CPCV Confidence")
    return tuple(dict.fromkeys(scenarios))


def _confidence(
    selected: list[str],
    gates_by_strategy: Mapping[str, list[GateResult]],
    *,
    artifacts_available: bool,
) -> tuple[str, float]:
    gates = [gate for strategy in selected for gate in gates_by_strategy.get(strategy, [])]
    counts = gate_summary(gates)
    score = 0.90
    score -= counts[GateStatus.FAIL.value] * 0.25
    score -= counts[GateStatus.WARN.value] * 0.05
    score -= counts[GateStatus.NOT_AVAILABLE.value] * 0.015
    if not artifacts_available:
        score -= 0.15
    score = max(0.0, min(1.0, score))
    label = "High" if score >= 0.75 else "Moderate" if score >= 0.50 else "Low"
    return label, score


def select_strategy_for_profile(
    investor_profile: str = "Balanced",
    *,
    candidate_metrics: pd.DataFrame | None = None,
    current_regime: str | None = None,
    base_bps: float = 10.0,
    slippage_bps: float = 5.0,
    hmm_walk_forward_valid: bool = True,
    n_observations: int | None = None,
    artifacts: Mapping[str, object] | None = None,
    repo_root: str | Path | None = None,
) -> StrategyRecommendation:
    """Select a fixed core and optional adaptive overlay for an investor profile."""

    if investor_profile not in INVESTOR_PROFILES:
        raise ValueError(f"Unknown investor profile: {investor_profile}. Expected one of {PROFILE_NAMES}.")
    artifacts = dict(artifacts or load_selection_artifacts(repo_root))
    cpcv_frame = artifacts.get("cpcv_summary", pd.DataFrame())
    stress_frame = artifacts.get("stress_comparison", pd.DataFrame())
    if not isinstance(cpcv_frame, pd.DataFrame):
        cpcv_frame = pd.DataFrame()
    if not isinstance(stress_frame, pd.DataFrame):
        stress_frame = pd.DataFrame()
    total_cost_bps = float(base_bps) + float(slippage_bps)
    candidates = _merge_candidates(candidate_metrics, artifacts, total_cost_bps=total_cost_bps)
    if candidates.empty:
        raise ValueError("No strategy evidence is available for selection.")

    if n_observations is None and "n_observations" in candidates:
        values = pd.to_numeric(candidates["n_observations"], errors="coerce").dropna()
        n_observations = int(values.max()) if not values.empty else None
    scenarios = classify_scenarios(
        current_regime=current_regime,
        total_cost_bps=total_cost_bps,
        hmm_valid=hmm_walk_forward_valid,
        cpcv_summary=cpcv_frame,
        n_observations=n_observations,
    )
    cpcv_by_strategy = _records_by_strategy(cpcv_frame)
    stress_by_strategy = _stress_by_strategy(stress_frame)
    gates_by_strategy: dict[str, list[GateResult]] = {}
    for strategy, row in candidates.iterrows():
        values = row.to_dict()
        if str(strategy) == HMM_CONSERVATIVE:
            values["hmm_walk_forward_valid"] = bool(hmm_walk_forward_valid)
        gates_by_strategy[str(strategy)] = evaluate_selection_gates(
            values,
            cpcv=cpcv_by_strategy.get(str(strategy)),
            stress=stress_by_strategy.get(str(strategy)),
        )

    roles = classify_strategy_roles(candidates, gates_by_strategy)
    scores = score_candidates(
        candidates,
        profile_name=investor_profile,
        gates_by_strategy=gates_by_strategy,
        cpcv_by_strategy=cpcv_by_strategy,
        scenario_categories=scenarios,
    )
    scores["role"] = pd.Series(roles)

    eligible_fixed = [
        strategy
        for strategy in scores.index
        if str(candidates.loc[strategy].get("strategy_type", "")).lower() == "fixed"
        and roles.get(strategy) != REJECTED_ROLE
        and strategy != EQUAL_WEIGHT
    ]
    main_strategy = HERC if HERC in eligible_fixed else (eligible_fixed[0] if eligible_fixed else EQUAL_WEIGHT)

    overlay_strategy: str | None = None
    overlay_role: str | None = None
    insufficient = "Insufficient Data" in scenarios
    hmm_eligible = HMM_CONSERVATIVE in candidates.index and roles.get(HMM_CONSERVATIVE) != REJECTED_ROLE
    rule_eligible = RULE_CONSERVATIVE in candidates.index and roles.get(RULE_CONSERVATIVE) != REJECTED_ROLE
    if not insufficient:
        if investor_profile == "Robustness First" and rule_eligible:
            overlay_strategy, overlay_role = RULE_CONSERVATIVE, ROBUSTNESS_ROLE
        elif hmm_eligible:
            overlay_strategy, overlay_role = HMM_CONSERVATIVE, RISK_CONTROL_ROLE
        elif rule_eligible:
            overlay_strategy, overlay_role = RULE_CONSERVATIVE, ROBUSTNESS_ROLE

    selected = [main_strategy] + ([overlay_strategy] if overlay_strategy else [])
    confidence, confidence_score = _confidence(
        selected,
        gates_by_strategy,
        artifacts_available=bool(artifacts.get("phase3e_available") or not artifacts.get("metrics_comparison", pd.DataFrame()).empty),
    )
    warnings = list(artifacts.get("warnings", []))
    if "Low CPCV Confidence" in scenarios:
        warnings.append("CPCV successful-fold coverage is limited; robustness claims are confidence-adjusted.")
    if "High Cost" in scenarios:
        warnings.append("High cost assumptions can materially reduce the benefit of an active overlay.")
    if "HMM Unstable" in scenarios:
        warnings.append("HMM walk-forward evidence is unavailable or unstable; full-sample HMM is not used for selection.")
    if overlay_strategy is None:
        warnings.append("No adaptive overlay cleared the current evidence and safety gates.")
    assumptions = [
        "All performance comparisons use net return metrics.",
        "HERC remains the strategic core unless current fixed-strategy evidence invalidates it.",
        "Adaptive strategies are overlays or robustness references, not automatic HERC replacements.",
        f"Trading costs are assumed to be {base_bps:.0f} bps base plus {slippage_bps:.0f} bps slippage.",
    ]
    comparison_columns = [
        column
        for column in [
            "strategy_type",
            "cagr",
            "calmar",
            "max_drawdown",
            "final_value",
            "total_turnover",
            "total_transaction_cost",
            "stress_period_return",
        ]
        if column in candidates
    ]
    evidence = {
        "comparison_table": candidates.loc[
            [name for name in [HERC, HMM_CONSERVATIVE, RULE_CONSERVATIVE, EQUAL_WEIGHT] if name in candidates.index],
            comparison_columns,
        ].reset_index(),
        "cpcv_summary": cpcv_frame,
        "phase3e_available": bool(artifacts.get("phase3e_available")),
    }
    explanation = build_recommendation_explanation(
        profile_name=investor_profile,
        main_strategy=main_strategy,
        overlay_strategy=overlay_strategy,
        scenarios=scenarios,
        confidence=confidence,
    )
    return StrategyRecommendation(
        investor_profile=investor_profile,
        scenario_categories=scenarios,
        main_strategy=main_strategy,
        overlay_strategy=overlay_strategy,
        overlay_role=overlay_role,
        confidence=confidence,
        confidence_score=confidence_score,
        evidence=evidence,
        gate_results=gates_by_strategy,
        role_assignments=roles,
        candidate_scores=scores,
        assumptions=assumptions,
        explanation=explanation,
        warnings=list(dict.fromkeys(warnings)),
        artifact_diagnostics={
            "phase3e_available": bool(artifacts.get("phase3e_available")),
            "fallback_used": bool(artifacts.get("fallback_used")),
            "paths": artifacts.get("paths", {}),
            "candidate_count": int(len(candidates)),
        },
    )
