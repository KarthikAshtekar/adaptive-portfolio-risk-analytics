"""Scenario playbook generation for manager and research surfaces."""

from __future__ import annotations

import pandas as pd

from src.selection.config import HERC, HMM_CONSERVATIVE, INVESTOR_PROFILES, RULE_CONSERVATIVE


def build_strategy_playbook() -> pd.DataFrame:
    """Return the operational profile/scenario recommendation map."""

    rows: list[dict[str, object]] = []
    overlay_by_profile = {
        "Growth": HMM_CONSERVATIVE,
        "Balanced": HMM_CONSERVATIVE,
        "Capital Preservation": HMM_CONSERVATIVE,
        "Stress Protection": HMM_CONSERVATIVE,
        "Robustness First": RULE_CONSERVATIVE,
    }
    for profile_name, profile in INVESTOR_PROFILES.items():
        rows.append(
            {
                "investor_profile": profile_name,
                "scenario": "Normal",
                "core_strategy": HERC,
                "overlay_or_reference": overlay_by_profile[profile_name],
                "overlay_usage": profile.overlay_preference,
                "decision_rule": profile.description,
            }
        )

    rows.extend(
        [
            {
                "investor_profile": "All",
                "scenario": "Stress / Crisis",
                "core_strategy": HERC,
                "overlay_or_reference": HMM_CONSERVATIVE,
                "overlay_usage": "recommended if walk-forward gates pass",
                "decision_rule": "Prioritize drawdown protection; retain HERC as the strategic core.",
            },
            {
                "investor_profile": "All",
                "scenario": "HMM Unstable",
                "core_strategy": HERC,
                "overlay_or_reference": RULE_CONSERVATIVE,
                "overlay_usage": "fallback / robustness reference",
                "decision_rule": "Do not use full-sample or failed walk-forward HMM evidence.",
            },
            {
                "investor_profile": "All",
                "scenario": "High Cost",
                "core_strategy": HERC,
                "overlay_or_reference": HMM_CONSERVATIVE,
                "overlay_usage": "optional; reduce switching",
                "decision_rule": "Require net evidence and prefer the lower-turnover eligible overlay.",
            },
            {
                "investor_profile": "All",
                "scenario": "Low CPCV Confidence",
                "core_strategy": HERC,
                "overlay_or_reference": RULE_CONSERVATIVE,
                "overlay_usage": "reference only unless coverage improves",
                "decision_rule": "Reduce confidence when successful-fold coverage is limited.",
            },
            {
                "investor_profile": "All",
                "scenario": "Insufficient Data",
                "core_strategy": HERC,
                "overlay_or_reference": None,
                "overlay_usage": "none",
                "decision_rule": "Use the fixed core only; adaptive selection is not available.",
            },
        ]
    )
    return pd.DataFrame(rows)
