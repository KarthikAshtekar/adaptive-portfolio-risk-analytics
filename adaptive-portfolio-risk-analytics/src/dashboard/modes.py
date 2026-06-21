"""Dashboard mode, objective, and presentation contracts.

This module is intentionally independent from Streamlit so the dashboard UX
contract can be tested without rendering the application.
"""

from __future__ import annotations

from collections.abc import Mapping

MANAGER_VIEW = "Manager View"
RESEARCH_VIEW = "Research View"
DEVELOPER_VIEW = "Developer / Debug View"

DASHBOARD_MODES = (MANAGER_VIEW, RESEARCH_VIEW, DEVELOPER_VIEW)
DEFAULT_DASHBOARD_MODE = MANAGER_VIEW

RESEARCH_OBJECTIVES = {
    "Net CAGR": "cagr",
    "Net Sharpe": "sharpe",
    "Net Sortino": "sortino",
    "Net Calmar": "calmar",
    "Max Drawdown": "max_drawdown",
    "Net Final Value": "final_value",
}
DEFAULT_RESEARCH_OBJECTIVE = "Net Calmar"

MANAGER_PROFILE_OBJECTIVES = {
    "Growth": "Net Final Value",
    "Balanced": "Net Calmar",
    "Capital Preservation": "Max Drawdown",
    "Stress Protection": "Net Calmar",
    "Robustness First": "Net Calmar",
}

MANAGER_INPUT_LABELS = (
    "Portfolio Universe",
    "Investment Amount",
    "Start Date",
    "End Date",
    "Investor Objective",
    "Cost Assumption",
    "Run Recommendation",
)

MANAGER_HIDDEN_ADVANCED_LABELS = (
    "Strategy",
    "Benchmark Comparison Strategies",
    "Benchmark Strategy",
    "Research Objective",
    "Covariance Method",
    "Rebalance Mode",
    "Threshold",
    "Regime Method",
    "Adaptive Regime Source",
    "Adaptive Policy Preset",
    "HMM States",
    "HMM Refit Frequency",
    "Sentiment Source",
    "Sentiment Lookback Window",
    "Sentiment Decision Lag",
    "RBI Corpus Mode",
    "RBI Manifest Path",
    "RBI Scorer Method",
    "Macro Lookback Window",
    "Macro Decision Lag",
    "Minimum Coverage Threshold",
    "NLP Provider Selector",
    "NLP Query Preset",
    "NLP Scoring Method",
    "NLP Decision Lag",
)

DEFAULT_MANAGER_ADAPTIVE_OVERLAY = {
    "regime_source": "HMM walk-forward decision regime",
    "policy_preset": "Conservative",
    "regime_method": "HMM walk-forward experimental",
    "display_name": "Regime-Adaptive HMM Walk-Forward — Conservative",
    "role": "Best Adaptive Risk-Control Overlay",
}
RULE_BASED_ROBUSTNESS_REFERENCE = "Regime-Adaptive Rule-Based — Conservative"

MODE_NOTES = {
    RESEARCH_VIEW: (
        "Research View exposes methodology controls and validation diagnostics. "
        "These are intended for model research, not first-pass investment decision review."
    ),
    DEVELOPER_VIEW: (
        "Developer View is for audit, debugging, and implementation validation. "
        "Most users should not need these tables."
    ),
}

NET_METRIC_LABELS = {
    "cumulative_return": "Net Cumulative Return",
    "return": "Net Return",
    "period_return": "Net Period Return",
    "stress_return": "Net Stress Return",
    "excess_stress_return": "Net Excess Stress Return",
    "cagr": "Net CAGR",
    "sharpe": "Net Sharpe",
    "sortino": "Net Sortino",
    "calmar": "Net Calmar",
    "volatility": "Net Volatility",
    "max_drawdown": "Max Drawdown",
    "final_value": "Net Final Value",
    "var_95": "Net Return VaR 95",
    "cvar_95": "Net Return CVaR 95",
    "excess_cagr": "Net Excess CAGR",
    "excess_sharpe": "Net Excess Sharpe",
    "volatility_difference": "Net Volatility Difference",
    "final_value_difference": "Net Final Value Difference",
}

MANAGER_SECTIONS = (
    "Portfolio Universe",
    "Recommendation Inputs",
    "Run Recommendation",
    "Strategy Recommendation",
    "Tradeoff Table",
    "Why This Recommendation",
    "Sentiment Confirmation",
    "RBI Macro-Sentiment Confirmation",
    "NLP Risk Confirmation",
    "Warnings and Assumptions",
)

RESEARCH_SECTIONS = (
    "Advanced Strategy Controls",
    "Volatility Targeting",
    "Phase 3B — Regime Detection",
    "Phase 3C — Adaptive Allocation Policy",
    "Phase 4A — Sentiment Confirmation",
    "Phase 4A.3 — Real RBI Corpus Validation",
    "Phase 4A.5 — API-Based Ex-Ante NLP Risk Monitoring",
    "Experiment Sensitivity",
    "Phase 3A — CPCV Robustness Validation",
    "Regime Attribution",
    "Liquidity and Cost Diagnostics",
    "Strategy Selection Diagnostics",
    "Selection Gates",
    "Scenario Playbook",
)

DEVELOPER_SECTIONS = (
    "Raw HMM Diagnostics",
    "Raw Sentiment Records",
    "Sentiment Alignment Checks",
    "Raw RBI Documents",
    "RBI Sentence Scores",
    "RBI Macro Construction Diagnostics",
    "Real RBI Manifest Validation",
    "Invalid RBI Documents",
    "Raw Provider Records",
    "Provider API/Cache Diagnostics",
    "Ex-Ante Validation",
    "Possible Reaction-Data Records",
    "FinBERT Fallback Metadata",
    "Composite NLP Risk Index",
    "Raw CPCV Diagnostics",
    "Full Adaptive Decision Log",
    "Full Weight History",
    "Net/Gross Reconciliation",
    "Defensive Return Reconciliation",
    "Internal Config Dump",
    "Raw Strategy Recommendation",
    "Selection Gate Results",
    "Selection Artifact Diagnostics",
    "Selection Scoring Trace",
)


def objective_metric(label: str | None) -> str:
    """Resolve the single global research objective with a Calmar fallback."""
    return RESEARCH_OBJECTIVES.get(
        str(label or DEFAULT_RESEARCH_OBJECTIVE),
        RESEARCH_OBJECTIVES[DEFAULT_RESEARCH_OBJECTIVE],
    )


def research_objective_label(metric: str | None) -> str:
    """Return the explicit net label for an internal objective metric."""
    normalized = str(metric or "").strip().lower()
    return next(
        (
            label
            for label, objective in RESEARCH_OBJECTIVES.items()
            if objective == normalized
        ),
        net_metric_label(normalized),
    )


def net_metric_label(metric: str) -> str:
    """Return an explicit user-facing net label for a metric key."""
    raw_metric = str(metric)
    normalized = raw_metric.strip().lower()
    if normalized in NET_METRIC_LABELS:
        return NET_METRIC_LABELS[normalized]

    role_prefixes = {
        "strategy_": "Strategy",
        "benchmark_": "Benchmark",
        "adaptive_": "Adaptive",
        "best_fixed_": "Best Fixed",
    }
    for prefix, role in role_prefixes.items():
        if normalized.startswith(prefix):
            suffix = normalized.removeprefix(prefix)
            if suffix in NET_METRIC_LABELS:
                return f"{role} {NET_METRIC_LABELS[suffix]}"

    return raw_metric.replace("_", " ").title()


def adaptive_overlay_name(regime_source: str, policy_preset: str) -> str:
    """Build the first-class display name used by manager comparisons."""
    source = str(regime_source).strip().lower()
    source_label = (
        "HMM Walk-Forward"
        if "hmm" in source
        else "Rule-Based"
    )
    preset = str(policy_preset).strip().replace("_", " ").title()
    if preset == "Balanced Default":
        preset = "Balanced"
    return f"Regime-Adaptive {source_label} — {preset}"


def classify_recommended_use(
    fixed_metrics: Mapping[str, object],
    adaptive_metrics: Mapping[str, object] | None,
) -> str:
    """Classify adaptive use from current net metrics without hard-coded values."""
    if not adaptive_metrics:
        return "Experimental Only"

    def number(values: Mapping[str, object], key: str) -> float:
        try:
            return float(values.get(key))
        except (TypeError, ValueError):
            return float("nan")

    fixed_cagr = number(fixed_metrics, "cagr")
    fixed_calmar = number(fixed_metrics, "calmar")
    fixed_drawdown = number(fixed_metrics, "max_drawdown")
    fixed_final = number(fixed_metrics, "final_value")
    adaptive_cagr = number(adaptive_metrics, "cagr")
    adaptive_calmar = number(adaptive_metrics, "calmar")
    adaptive_drawdown = number(adaptive_metrics, "max_drawdown")
    adaptive_final = number(adaptive_metrics, "final_value")

    improves_drawdown = adaptive_drawdown > fixed_drawdown
    improves_calmar = adaptive_calmar > fixed_calmar
    sacrifices_growth = adaptive_cagr < fixed_cagr or adaptive_final < fixed_final
    leads_growth = adaptive_cagr >= fixed_cagr and adaptive_final >= fixed_final

    if improves_drawdown and improves_calmar and sacrifices_growth:
        return "Risk-Control Overlay"
    if leads_growth and improves_calmar:
        return "Main Growth Strategy"
    return "Experimental Only"
