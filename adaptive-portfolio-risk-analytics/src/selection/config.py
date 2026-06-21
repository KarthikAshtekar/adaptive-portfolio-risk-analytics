"""Configuration contracts for the strategy-selection layer."""

from __future__ import annotations

from dataclasses import dataclass

EQUAL_WEIGHT = "Equal Weight"
INVERSE_VOLATILITY = "Inverse Volatility"
HRP = "HRP"
HERC = "HERC"
HMM_CONSERVATIVE = "Regime-Adaptive HMM Walk-Forward — Conservative"
RULE_CONSERVATIVE = "Regime-Adaptive Rule-Based — Conservative"

MAIN_GROWTH_ROLE = "Main Growth Strategy"
RISK_CONTROL_ROLE = "Risk-Control Overlay"
ROBUSTNESS_ROLE = "Robustness Reference"
EXPERIMENTAL_ROLE = "Experimental Candidate"
REJECTED_ROLE = "Rejected"
BENCHMARK_ROLE = "Benchmark"


@dataclass(frozen=True)
class InvestorProfile:
    """Manager-facing objective mapped to explicit selection priorities."""

    name: str
    description: str
    growth_weight: float
    drawdown_weight: float
    robustness_weight: float
    cost_weight: float
    overlay_preference: str


INVESTOR_PROFILES = {
    "Growth": InvestorProfile(
        "Growth",
        "Prioritize terminal wealth while retaining a separately identified risk overlay.",
        0.60,
        0.15,
        0.15,
        0.10,
        "optional",
    ),
    "Balanced": InvestorProfile(
        "Balanced",
        "Balance net growth, drawdown control, evidence quality, and implementation cost.",
        0.35,
        0.30,
        0.25,
        0.10,
        "recommended",
    ),
    "Capital Preservation": InvestorProfile(
        "Capital Preservation",
        "Emphasize drawdown reduction and stable net outcomes over maximum terminal wealth.",
        0.15,
        0.55,
        0.20,
        0.10,
        "recommended",
    ),
    "Stress Protection": InvestorProfile(
        "Stress Protection",
        "Emphasize stress-period protection and drawdown control.",
        0.10,
        0.55,
        0.25,
        0.10,
        "recommended",
    ),
    "Robustness First": InvestorProfile(
        "Robustness First",
        "Prioritize CPCV and replication evidence before point-estimate performance.",
        0.15,
        0.20,
        0.55,
        0.10,
        "recommended",
    ),
}

PROFILE_NAMES = tuple(INVESTOR_PROFILES)

COST_ASSUMPTIONS = {
    "Low": (0.0, 0.0),
    "Moderate": (10.0, 5.0),
    "High": (50.0, 25.0),
}
COST_ASSUMPTION_NAMES = (*COST_ASSUMPTIONS, "Custom")

SCENARIO_CATEGORIES = (
    "Calm / Growth",
    "Normal",
    "Stress",
    "Crisis",
    "High Volatility",
    "High Cost",
    "HMM Unstable",
    "Low CPCV Confidence",
    "Insufficient Data",
)

