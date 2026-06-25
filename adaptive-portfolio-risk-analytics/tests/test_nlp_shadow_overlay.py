"""Tests for reporting-only NLP shadow overlay rules."""

from __future__ import annotations

import pandas as pd

from src.sentiment import (
    CONFIRMATION_VARIANT,
    EARLY_WARNING_VARIANT,
    NLPShadowOverlayConfig,
    build_nlp_signal_alignment,
    build_overlay_decisions,
)


def _signal(index: pd.DatetimeIndex, labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": index,
            "decision_nlp_label": labels,
            "source_mix": ["news_only"] * len(index),
            "coverage_score": [1.0 / 3.0] * len(index),
            "decision_source_date": [pd.NaT] + list(index[:-1]),
        }
    )


def test_confirmation_overlay_uses_defensive_only_when_hmm_and_nlp_agree() -> None:
    index = pd.bdate_range("2026-04-01", periods=4)
    regimes = pd.Series(["Stress", "Stress", "Normal", "Crisis"], index=index)
    signal = _signal(
        index,
        ["insufficient_nlp_data", "nlp_risk_off", "nlp_neutral", "nlp_risk_on"],
    )
    alignment = build_nlp_signal_alignment(signal, index, decision_lag_days=1)

    overlay_regimes, decisions = build_overlay_decisions(
        regimes,
        alignment,
        variant=CONFIRMATION_VARIANT,
        config=NLPShadowOverlayConfig(decision_lag_days=1),
    )

    assert decisions.loc[1, "overlay_action"] == "confirmed_defensive"
    assert decisions.loc[1, "allocation_after_overlay"] == "defensive"
    assert decisions.loc[3, "overlay_action"] == "defensive_not_confirmed_core"
    assert decisions.loc[3, "allocation_after_overlay"] == "core"
    assert overlay_regimes.loc[index[3]] == "Normal"


def test_early_warning_overlay_only_partially_derisks() -> None:
    index = pd.bdate_range("2026-04-01", periods=5)
    regimes = pd.Series("Normal", index=index)
    signal = _signal(
        index,
        [
            "insufficient_nlp_data",
            "nlp_risk_off",
            "nlp_risk_off",
            "nlp_risk_off",
            "nlp_risk_off",
        ],
    )
    features = pd.DataFrame(
        {
            "rolling_drawdown": [-0.01, -0.01, -0.03, -0.04, -0.05],
            "volatility_percentile": [0.2, 0.3, 0.7, 0.8, 0.8],
            "benchmark_return_21d": [0.02, 0.01, -0.01, -0.02, -0.03],
        },
        index=index,
    )
    alignment = build_nlp_signal_alignment(signal, index, decision_lag_days=1)

    _, decisions = build_overlay_decisions(
        regimes,
        alignment,
        variant=EARLY_WARNING_VARIANT,
        config=NLPShadowOverlayConfig(decision_lag_days=1, nlp_persistence_days=3),
        features=features,
    )

    triggered = decisions.loc[
        decisions["overlay_action"].eq("early_warning_partial_defensive")
    ]
    assert not triggered.empty
    assert triggered["allocation_after_overlay"].eq("partial_defensive").all()


def test_nlp_signal_alignment_is_decision_lagged() -> None:
    index = pd.bdate_range("2026-04-01", periods=3)
    signal = _signal(index, ["insufficient_nlp_data", "nlp_risk_off", "nlp_risk_off"])

    alignment = build_nlp_signal_alignment(signal, index, decision_lag_days=1)

    assert alignment.loc[1, "nlp_signal_date_used"] == index[0]
    assert alignment.loc[1, "latest_allowed_signal_date"] == index[0]
    assert bool(alignment.loc[1, "lookahead_check_passed"]) is True


def test_lookahead_diagnostic_catches_future_signal_use() -> None:
    index = pd.bdate_range("2026-04-01", periods=3)
    signal = _signal(index, ["insufficient_nlp_data", "nlp_risk_off", "nlp_risk_off"])
    signal.loc[1, "decision_source_date"] = index[2]

    alignment = build_nlp_signal_alignment(signal, index, decision_lag_days=1)

    assert bool(alignment.loc[1, "lookahead_check_passed"]) is False
