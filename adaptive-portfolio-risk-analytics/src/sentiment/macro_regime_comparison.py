"""Comparison of lagged RBI macro labels with quantitative regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def macro_agrees_with_regime(
    macro_label: object,
    quantitative_regime: object,
) -> bool | float:
    """Return whether one macro label confirms one quantitative regime."""
    macro = str(macro_label or "insufficient_macro_data").strip().lower()
    regime = str(quantitative_regime or "unknown").strip().lower()
    if macro == "insufficient_macro_data" or regime in {
        "",
        "unknown",
        "nan",
        "none",
    }:
        return np.nan
    if macro == "risk_off_macro":
        return regime in {"stress", "crisis", "risk-off", "risk_off"}
    if macro == "risk_on_macro":
        return regime in {"calm", "normal", "risk-on", "risk_on"}
    if macro == "neutral_macro":
        return regime == "normal"
    return False


def _coerce_regimes(values: pd.Series | None, name: str) -> pd.Series:
    if values is None:
        return pd.Series(dtype="object", name=name)
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError(f"{name} index must be a DatetimeIndex")
    result = values.sort_index().astype("object")
    result.name = name
    return result


def _agreement_rate(values: pd.Series) -> float:
    valid = values.dropna()
    return float(valid.astype(bool).mean()) if not valid.empty else np.nan


def _lead_lag_table(
    labels: pd.Series,
    rule_regimes: pd.Series,
    hmm_regimes: pd.Series,
    max_shift: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shift in range(-int(max_shift), int(max_shift) + 1):
        shifted = labels.shift(shift)
        row: dict[str, object] = {"macro_shift_market_days": shift}
        for name, regimes in (
            ("rule_based", rule_regimes),
            ("hmm", hmm_regimes),
        ):
            paired = pd.concat(
                [shifted.rename("macro"), regimes.rename("regime")],
                axis=1,
                join="inner",
            )
            agreement = paired.apply(
                lambda item: macro_agrees_with_regime(
                    item["macro"],
                    item["regime"],
                ),
                axis=1,
            )
            row[f"{name}_agreement_rate"] = _agreement_rate(agreement)
        rows.append(row)
    return pd.DataFrame(rows)


def _pre_stress_dates(
    macro_labels: pd.Series,
    regimes: pd.Series,
    *,
    lead_window: int,
) -> pd.DataFrame:
    normalized = regimes.astype(str).str.lower()
    stress = normalized.isin({"stress", "crisis", "risk-off", "risk_off"})
    onsets = stress & ~stress.shift(1, fill_value=False)
    rows: list[dict[str, object]] = []
    for stress_date in regimes.index[onsets]:
        position = regimes.index.get_loc(stress_date)
        start = max(0, position - int(lead_window))
        preceding = macro_labels.iloc[start:position]
        risk_off = preceding[preceding.astype(str).eq("risk_off_macro")]
        if not risk_off.empty:
            macro_date = risk_off.index[-1]
            rows.append(
                {
                    "stress_date": stress_date,
                    "preceding_macro_risk_off_date": macro_date,
                    "lead_market_days": position - regimes.index.get_loc(macro_date),
                }
            )
    return pd.DataFrame(rows)


def compare_macro_to_regimes(
    macro_index: pd.DataFrame,
    rule_based_regimes: pd.Series,
    hmm_walk_forward_regimes: pd.Series | None = None,
    *,
    max_shift: int = 10,
    lead_window: int = 10,
) -> dict[str, object]:
    """Compare lagged macro labels with trading-safe quantitative regimes."""
    if "decision_macro_label" not in macro_index:
        raise ValueError("macro_index must contain decision_macro_label")
    labels = macro_index["decision_macro_label"].astype("object")
    rule = _coerce_regimes(
        rule_based_regimes,
        "rule_based_regime",
    ).reindex(macro_index.index)
    hmm = _coerce_regimes(
        hmm_walk_forward_regimes,
        "hmm_walk_forward_regime",
    ).reindex(macro_index.index)

    comparison = pd.DataFrame(index=macro_index.index)
    comparison.index.name = "date"
    comparison["macro_label"] = labels
    comparison["rule_based_regime"] = rule
    comparison["hmm_walk_forward_regime"] = hmm
    comparison["sentence_count"] = macro_index.get("decision_sentence_count", 0)
    comparison["agreement_rule_based"] = comparison.apply(
        lambda row: macro_agrees_with_regime(
            row["macro_label"],
            row["rule_based_regime"],
        ),
        axis=1,
    )
    comparison["agreement_hmm"] = comparison.apply(
        lambda row: macro_agrees_with_regime(
            row["macro_label"],
            row["hmm_walk_forward_regime"],
        ),
        axis=1,
    )

    def risk_off_confirmation(regimes: pd.Series) -> float:
        stress = regimes.astype(str).str.lower().isin(
            {"stress", "crisis", "risk-off", "risk_off"}
        )
        covered = stress & labels.ne("insufficient_macro_data")
        return (
            float(labels.loc[covered].eq("risk_off_macro").mean())
            if covered.any()
            else np.nan
        )

    disagreement_mask = (
        comparison[["agreement_rule_based", "agreement_hmm"]]
        .eq(False)
        .any(axis=1)
    )
    disagreements = comparison.loc[disagreement_mask].copy()
    disagreements.insert(0, "date", disagreements.index)
    covered = labels.ne("insufficient_macro_data")
    sentence_count = pd.to_numeric(
        macro_index.get(
            "decision_sentence_count",
            pd.Series(0, index=macro_index.index),
        ),
        errors="coerce",
    ).fillna(0)

    lead_rule = _pre_stress_dates(
        labels,
        rule,
        lead_window=int(lead_window),
    )
    lead_hmm = _pre_stress_dates(
        labels,
        hmm,
        lead_window=int(lead_window),
    )
    lead_dates = pd.concat(
        [
            lead_rule.assign(regime_method="rule_based"),
            lead_hmm.assign(regime_method="hmm_walk_forward"),
        ],
        ignore_index=True,
    )
    return {
        "agreement_with_rule_based": _agreement_rate(
            comparison["agreement_rule_based"]
        ),
        "agreement_with_hmm_walk_forward": _agreement_rate(
            comparison["agreement_hmm"]
        ),
        "stress_crisis_risk_off_confirmation_rule_based": risk_off_confirmation(
            rule
        ),
        "stress_crisis_risk_off_confirmation_hmm": risk_off_confirmation(hmm),
        "lead_lag_diagnostics": _lead_lag_table(
            labels,
            rule,
            hmm,
            int(max_shift),
        ),
        "macro_risk_off_before_stress_dates": lead_dates,
        "dates_of_major_disagreement": disagreements.reset_index(drop=True),
        "coverage_ratio": float(covered.mean()) if len(covered) else 0.0,
        "sentence_day_coverage_ratio": float(sentence_count.gt(0).mean())
        if len(sentence_count)
        else 0.0,
        "comparison_table": comparison,
    }
