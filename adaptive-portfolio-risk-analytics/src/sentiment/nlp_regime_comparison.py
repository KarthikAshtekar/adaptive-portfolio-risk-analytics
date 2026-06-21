"""Compare lagged composite NLP risk with trading-safe quantitative regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def nlp_agrees_with_regime(nlp_label: object, regime: object) -> bool | float:
    label = str(nlp_label or "insufficient_nlp_data").strip().lower()
    quantitative = str(regime or "unknown").strip().lower()
    if label == "insufficient_nlp_data" or quantitative in {
        "",
        "unknown",
        "nan",
        "none",
    }:
        return np.nan
    if label == "nlp_risk_off":
        return quantitative in {"stress", "crisis", "risk-off", "risk_off"}
    if label == "nlp_risk_on":
        return quantitative in {"calm", "normal", "risk-on", "risk_on"}
    if label == "nlp_neutral":
        return quantitative == "normal"
    return False


def _rate(values: pd.Series) -> float:
    valid = values.dropna()
    return float(valid.astype(bool).mean()) if not valid.empty else np.nan


def _pre_stress_count(
    labels: pd.Series,
    regimes: pd.Series,
    lead_window: int,
) -> int:
    stress = regimes.astype(str).str.lower().isin(
        {"stress", "crisis", "risk-off", "risk_off"}
    )
    onsets = stress & ~stress.shift(1, fill_value=False)
    count = 0
    for stress_date in regimes.index[onsets]:
        position = regimes.index.get_loc(stress_date)
        preceding = labels.iloc[max(0, position - lead_window) : position]
        count += int(preceding.eq("nlp_risk_off").any())
    return count


def compare_composite_nlp_to_regimes(
    composite_index: pd.DataFrame,
    rule_based_regimes: pd.Series,
    hmm_walk_forward_regimes: pd.Series | None = None,
    *,
    strategy_recommendation: object | None = None,
    lead_window: int = 10,
) -> dict[str, object]:
    """Compute descriptive agreement without claiming predictiveness."""
    if "decision_composite_nlp_label" not in composite_index:
        raise ValueError(
            "composite_index must contain decision_composite_nlp_label"
        )
    labels = composite_index["decision_composite_nlp_label"].astype(str)
    rule = pd.Series(rule_based_regimes).reindex(composite_index.index)
    hmm = (
        pd.Series(hmm_walk_forward_regimes).reindex(composite_index.index)
        if hmm_walk_forward_regimes is not None
        else pd.Series(index=composite_index.index, dtype="object")
    )
    comparison = pd.DataFrame(index=composite_index.index)
    comparison.index.name = "date"
    comparison["composite_nlp_label"] = labels
    comparison["composite_nlp_risk_score"] = composite_index.get(
        "decision_composite_nlp_risk_score"
    )
    comparison["coverage_score"] = composite_index.get(
        "decision_coverage_score", 0.0
    )
    comparison["rule_based_regime"] = rule
    comparison["hmm_walk_forward_regime"] = hmm
    comparison["agreement_rule_based"] = comparison.apply(
        lambda row: nlp_agrees_with_regime(
            row["composite_nlp_label"], row["rule_based_regime"]
        ),
        axis=1,
    )
    comparison["agreement_hmm"] = comparison.apply(
        lambda row: nlp_agrees_with_regime(
            row["composite_nlp_label"], row["hmm_walk_forward_regime"]
        ),
        axis=1,
    )
    covered = labels.ne("insufficient_nlp_data")
    risk_off = labels.eq("nlp_risk_off")
    rule_stress = rule.astype(str).str.lower().isin(
        {"stress", "crisis", "risk-off", "risk_off"}
    )
    hmm_stress = hmm.astype(str).str.lower().isin(
        {"stress", "crisis", "risk-off", "risk_off"}
    )
    stress_union = rule_stress | hmm_stress
    false_risk_off = risk_off & ~stress_union
    source_columns = [
        "decision_rbi_macro_risk_score",
        "decision_earnings_sector_risk_score",
        "decision_news_geopolitical_risk_score",
    ]
    source_contribution = pd.DataFrame(
        [
            {
                "source": column.removeprefix("decision_").removesuffix(
                    "_risk_score"
                ),
                "mean_absolute_risk_contribution": float(
                    pd.to_numeric(
                        composite_index.get(column), errors="coerce"
                    ).abs().mean()
                ),
            }
            for column in source_columns
        ]
    )
    current_regime = (
        str(hmm.dropna().iloc[-1])
        if not hmm.dropna().empty
        else str(rule.dropna().iloc[-1])
        if not rule.dropna().empty
        else "Unknown"
    )
    current_label = labels.iloc[-1] if len(labels) else "insufficient_nlp_data"
    current_confirmation = (
        "Insufficient NLP Data"
        if current_label == "insufficient_nlp_data"
        else "Confirms Quantitative Stress"
        if current_label == "nlp_risk_off"
        and current_regime.lower() in {"stress", "crisis", "risk-off", "risk_off"}
        else "Confirms Quantitative Risk-On"
        if current_label == "nlp_risk_on"
        and current_regime.lower() in {"calm", "normal", "risk-on", "risk_on"}
        else "Quant-NLP Disagreement"
    )
    return {
        "comparison_table": comparison,
        "agreement_with_rule_based": _rate(
            comparison["agreement_rule_based"]
        ),
        "agreement_with_hmm_walk_forward": _rate(
            comparison["agreement_hmm"]
        ),
        "risk_off_confirmation_rule_based": float(
            risk_off.loc[rule_stress & covered].mean()
        )
        if (rule_stress & covered).any()
        else np.nan,
        "risk_off_confirmation_hmm": float(
            risk_off.loc[hmm_stress & covered].mean()
        )
        if (hmm_stress & covered).any()
        else np.nan,
        "pre_stress_warning_count": _pre_stress_count(
            labels, rule, int(lead_window)
        ),
        "false_risk_off_count": int(false_risk_off.sum()),
        "coverage_ratio": float(covered.mean()) if len(covered) else 0.0,
        "source_level_contribution": source_contribution,
        "current_quantitative_regime": current_regime,
        "current_nlp_label": current_label,
        "current_confirmation": current_confirmation,
        "strategy_main": getattr(
            strategy_recommendation, "main_strategy", None
        ),
        "strategy_overlay": getattr(
            strategy_recommendation, "overlay_strategy", None
        ),
        "predictiveness_claim": False,
    }
