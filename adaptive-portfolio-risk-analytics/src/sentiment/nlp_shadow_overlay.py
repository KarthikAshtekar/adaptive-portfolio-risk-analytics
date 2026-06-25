"""Decision-lagged NLP shadow overlays for reporting-only experiments.

This module does not change production allocation or strategy selection. It
builds alternative regime labels for controlled shadow backtests and emits the
audit fields needed to prove that only lagged NLP signals were used.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.adaptive import get_policy_preset


CONFIRMATION_VARIANT = "confirmation"
EARLY_WARNING_VARIANT = "early_warning"
DEFAULT_RISK_OFF_LABELS = (
    "nlp_risk_off",
    "risk_off",
    "stress",
    "negative_macro",
)
DEFAULT_SOURCE_MIX = ("rbi_and_news", "rbi_only", "news_only")
QUALITY_THRESHOLDS = {
    "low": 0.0,
    "medium": 1.0 / 3.0,
    "high": 2.0 / 3.0,
}


@dataclass(frozen=True)
class NLPShadowOverlayConfig:
    """Configuration for reporting-only NLP shadow overlays."""

    nlp_confirm_required: bool = True
    partial_defensive_weight: float = 0.50
    risk_off_labels: tuple[str, ...] = DEFAULT_RISK_OFF_LABELS
    min_source_quality: str = "medium"
    min_source_mix: tuple[str, ...] = DEFAULT_SOURCE_MIX
    decision_lag_days: int = 1
    nlp_persistence_days: int = 3
    requires_market_confirmation: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.partial_defensive_weight) <= 1.0:
            raise ValueError("partial_defensive_weight must be between 0 and 1")
        if int(self.decision_lag_days) < 1:
            raise ValueError("decision_lag_days must be at least 1")
        if int(self.nlp_persistence_days) < 1:
            raise ValueError("nlp_persistence_days must be at least 1")
        if str(self.min_source_quality).lower() not in QUALITY_THRESHOLDS:
            raise ValueError("min_source_quality must be low, medium, or high")


def normalize_nlp_label(label: object) -> str:
    return str(label or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_regime_label(regime: object) -> str:
    raw = str(regime or "Unknown").strip().lower().replace("_", "-")
    aliases = {
        "risk-off": "Stress",
        "risk off": "Stress",
        "stress": "Stress",
        "crisis": "Crisis",
        "risk-on": "Calm",
        "risk on": "Calm",
        "calm": "Calm",
        "normal": "Normal",
        "unknown": "Unknown",
    }
    return aliases.get(raw, "Unknown")


def is_defensive_regime(regime: object) -> bool:
    return normalize_regime_label(regime) in {"Stress", "Crisis"}


def is_risk_off_nlp_label(
    label: object,
    risk_off_labels: Iterable[str] = DEFAULT_RISK_OFF_LABELS,
) -> bool:
    normalized = normalize_nlp_label(label)
    accepted = {normalize_nlp_label(value) for value in risk_off_labels}
    return normalized in accepted


def source_mix_allowed(source_mix: object, allowed: Iterable[str]) -> bool:
    return str(source_mix or "none").strip().lower() in {
        str(value).strip().lower() for value in allowed
    }


def source_quality_label(coverage_score: object) -> str:
    try:
        score = float(coverage_score)
    except (TypeError, ValueError):
        score = 0.0
    if score >= QUALITY_THRESHOLDS["high"]:
        return "high"
    if score >= QUALITY_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def source_quality_allowed(coverage_score: object, min_quality: str) -> bool:
    label = str(min_quality).strip().lower()
    threshold = QUALITY_THRESHOLDS.get(label, QUALITY_THRESHOLDS["medium"])
    try:
        score = float(coverage_score)
    except (TypeError, ValueError):
        score = 0.0
    return bool(score >= threshold)


def build_nlp_signal_alignment(
    nlp_signal: pd.DataFrame,
    market_index: pd.DatetimeIndex,
    *,
    decision_lag_days: int = 1,
) -> pd.DataFrame:
    """Align decision-lagged NLP signals to market decision dates."""
    if int(decision_lag_days) < 1:
        raise ValueError("decision_lag_days must be at least 1")
    if not isinstance(market_index, pd.DatetimeIndex):
        raise TypeError("market_index must be a DatetimeIndex")
    index = pd.DatetimeIndex(market_index).sort_values().drop_duplicates()
    if index.empty:
        raise ValueError("market_index must not be empty")

    if isinstance(nlp_signal, pd.DataFrame) and not nlp_signal.empty:
        frame = nlp_signal.copy()
        if "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"]).set_index("date")
        elif not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("nlp_signal must have a date column or DatetimeIndex")
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame = frame.sort_index()
    else:
        frame = pd.DataFrame(index=index)

    aligned = frame.reindex(index).copy()
    aligned["decision_date"] = index
    aligned["nlp_label"] = (
        aligned.get(
            "decision_nlp_label",
            pd.Series("insufficient_nlp_data", index=index),
        )
        .fillna("insufficient_nlp_data")
        .astype(str)
    )
    aligned["source_mix"] = (
        aligned.get("source_mix", pd.Series("none", index=index))
        .fillna("none")
        .astype(str)
    )
    aligned["coverage_score"] = pd.to_numeric(
        aligned.get("coverage_score", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    if "decision_source_date" in aligned:
        source_dates = pd.to_datetime(
            aligned["decision_source_date"],
            errors="coerce",
        )
    elif "decision_source_timestamp" in aligned:
        source_dates = pd.to_datetime(
            aligned["decision_source_timestamp"],
            errors="coerce",
        )
    else:
        source_dates = pd.Series(pd.NaT, index=index)
    source_dates = pd.Series(source_dates, index=index).dt.tz_localize(None)
    aligned["nlp_signal_date_used"] = source_dates

    lag = int(decision_lag_days)
    aligned["latest_allowed_signal_date"] = index - pd.Timedelta(days=lag)
    label_is_missing = aligned["nlp_label"].eq("insufficient_nlp_data")
    aligned["lookahead_check_passed"] = np.where(
        aligned["nlp_signal_date_used"].isna(),
        label_is_missing,
        aligned["nlp_signal_date_used"] <= aligned["latest_allowed_signal_date"],
    )
    aligned["source_quality"] = aligned["coverage_score"].map(source_quality_label)
    aligned["decision_lag_days"] = lag
    return aligned[
        [
            "decision_date",
            "nlp_signal_date_used",
            "latest_allowed_signal_date",
            "nlp_label",
            "source_mix",
            "coverage_score",
            "source_quality",
            "decision_lag_days",
            "lookahead_check_passed",
        ]
    ].reset_index(drop=True)


def build_overlay_decisions(
    regime_series: pd.Series,
    nlp_alignment: pd.DataFrame,
    *,
    variant: str,
    config: NLPShadowOverlayConfig | None = None,
    features: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return overlay regime labels and an auditable decision table."""
    if not isinstance(regime_series, pd.Series):
        raise TypeError("regime_series must be a pandas Series")
    if not isinstance(regime_series.index, pd.DatetimeIndex):
        raise ValueError("regime_series index must be a DatetimeIndex")
    if not isinstance(nlp_alignment, pd.DataFrame):
        raise TypeError("nlp_alignment must be a DataFrame")
    normalized_variant = str(variant).strip().lower()
    if normalized_variant not in {CONFIRMATION_VARIANT, EARLY_WARNING_VARIANT}:
        raise ValueError("variant must be confirmation or early_warning")

    cfg = config or NLPShadowOverlayConfig()
    index = pd.DatetimeIndex(regime_series.index).sort_values().drop_duplicates()
    regimes = regime_series.reindex(index).fillna("Unknown")
    alignment = nlp_alignment.copy()
    alignment["decision_date"] = pd.to_datetime(
        alignment["decision_date"],
        errors="coerce",
    )
    alignment = alignment.dropna(subset=["decision_date"]).set_index("decision_date")
    alignment = alignment.reindex(index)
    alignment["nlp_label"] = alignment["nlp_label"].fillna("insufficient_nlp_data")
    alignment["source_mix"] = alignment["source_mix"].fillna("none")
    alignment["coverage_score"] = pd.to_numeric(
        alignment["coverage_score"],
        errors="coerce",
    ).fillna(0.0)
    alignment["lookahead_check_passed"] = alignment[
        "lookahead_check_passed"
    ].fillna(False)

    risk_off = alignment.apply(
        lambda row: (
            is_risk_off_nlp_label(row["nlp_label"], cfg.risk_off_labels)
            and source_mix_allowed(row["source_mix"], cfg.min_source_mix)
            and source_quality_allowed(row["coverage_score"], cfg.min_source_quality)
            and bool(row["lookahead_check_passed"])
        ),
        axis=1,
    )
    risk_off = pd.Series(risk_off.to_numpy(), index=index, dtype=bool)
    persistent_risk_off = (
        risk_off.astype(int)
        .rolling(int(cfg.nlp_persistence_days), min_periods=int(cfg.nlp_persistence_days))
        .sum()
        .ge(int(cfg.nlp_persistence_days))
    )
    deterioration = _market_deterioration_flags(features, index)
    if not cfg.requires_market_confirmation:
        deterioration = pd.Series(True, index=index, dtype=bool)

    rows: list[dict[str, object]] = []
    overlay_values: list[str] = []
    for date_value in index:
        regime = normalize_regime_label(regimes.loc[date_value])
        defensive_before = is_defensive_regime(regime)
        eligible_risk_off = bool(risk_off.loc[date_value])
        market_confirmed = bool(deterioration.loc[date_value])

        overlay_regime = regime
        action = "no_overlay"
        allocation_before = "defensive" if defensive_before else "core"
        allocation_after = allocation_before

        if normalized_variant == CONFIRMATION_VARIANT:
            if defensive_before and eligible_risk_off:
                action = "confirmed_defensive"
                overlay_regime = regime
                allocation_after = "defensive"
            elif defensive_before and cfg.nlp_confirm_required:
                action = "defensive_not_confirmed_core"
                overlay_regime = "Normal"
                allocation_after = "core"
            elif defensive_before:
                action = "defensive_unconfirmed_partial"
                overlay_regime = "Stress"
                allocation_after = "partial_defensive"
        else:
            if defensive_before:
                action = "hmm_defensive_preserved"
                overlay_regime = regime
                allocation_after = "defensive"
            elif bool(persistent_risk_off.loc[date_value]) and market_confirmed:
                action = "early_warning_partial_defensive"
                overlay_regime = "Unknown"
                allocation_after = "partial_defensive"

        overlay_values.append(overlay_regime)
        rows.append(
            {
                "decision_date": date_value,
                "nlp_signal_date_used": alignment.loc[
                    date_value,
                    "nlp_signal_date_used",
                ],
                "source_mix": alignment.loc[date_value, "source_mix"],
                "nlp_label": alignment.loc[date_value, "nlp_label"],
                "regime_label": regime,
                "allocation_before_overlay": allocation_before,
                "allocation_after_overlay": allocation_after,
                "overlay_action": action,
                "overlay_variant": normalized_variant,
                "nlp_risk_off_eligible": eligible_risk_off,
                "nlp_risk_off_persistent": bool(persistent_risk_off.loc[date_value]),
                "market_confirmation": market_confirmed,
                "lookahead_check_passed": bool(
                    alignment.loc[date_value, "lookahead_check_passed"]
                ),
            }
        )

    overlay_regimes = pd.Series(
        overlay_values,
        index=index,
        name=f"{normalized_variant}_overlay_regime",
        dtype="object",
    )
    decisions = pd.DataFrame(rows)
    return overlay_regimes, decisions


def build_shadow_policy_map(
    variant: str,
    *,
    config: NLPShadowOverlayConfig | None = None,
    base_preset: str = "Conservative",
) -> Mapping[str, object]:
    """Return a policy map for shadow backtests only."""
    cfg = config or NLPShadowOverlayConfig()
    policies = dict(get_policy_preset(base_preset))
    if str(variant).strip().lower() == EARLY_WARNING_VARIANT:
        partial = float(cfg.partial_defensive_weight)
        policies["Unknown"] = replace(
            policies["Unknown"],
            regime="Unknown",
            allocator="herc",
            covariance_method="ledoit_wolf",
            target_volatility=max(0.04, float(policies["Stress"].target_volatility)),
            defensive_weight_floor=partial,
            risky_exposure_cap=1.0 - partial,
            notes=(
                policies["Unknown"].notes
                + " NLP shadow early-warning partial defensive policy."
            ),
        )
    return policies


def _market_deterioration_flags(
    features: pd.DataFrame | None,
    index: pd.DatetimeIndex,
) -> pd.Series:
    if features is None or not isinstance(features, pd.DataFrame) or features.empty:
        return pd.Series(False, index=index, dtype=bool)
    frame = features.reindex(index)
    drawdown = pd.to_numeric(
        frame.get("rolling_drawdown", pd.Series(np.nan, index=index)),
        errors="coerce",
    )
    vol_pct = pd.to_numeric(
        frame.get("volatility_percentile", pd.Series(np.nan, index=index)),
        errors="coerce",
    )
    momentum = pd.to_numeric(
        frame.get("benchmark_return_21d", pd.Series(np.nan, index=index)),
        errors="coerce",
    )
    deteriorating = (
        drawdown.le(-0.02).fillna(False)
        | vol_pct.ge(0.60).fillna(False)
        | momentum.lt(0.0).fillna(False)
    )
    return pd.Series(deteriorating.to_numpy(), index=index, dtype=bool)
