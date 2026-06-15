"""Analytics exports."""

from .risk_metrics import RiskAnalytics
from .performance_metrics import PerformanceAnalytics
from .active_risk_metrics import (
    calculate_active_risk_metrics,
    calculate_beta,
    calculate_concentration_metrics,
    calculate_drawdown_durations,
    calculate_hit_ratio,
    calculate_information_ratio,
    calculate_jensens_alpha,
    calculate_simple_alpha,
    calculate_tracking_error,
)
from .liquidity_diagnostics import (
    calculate_liquidity_diagnostics,
    summarize_liquidity_diagnostics,
)
from .risk_contribution import (
    compare_risk_contributions,
    marginal_risk_contribution,
    percentage_risk_contribution,
    portfolio_volatility,
    risk_contribution_table,
    total_risk_contribution,
)
from .stress_testing import (
    DEFAULT_HYPOTHETICAL_SCENARIOS,
    DEFAULT_STRESS_PERIODS,
    StressTestingFramework,
    apply_hypothetical_stress,
    calculate_correlation_stress,
    calculate_historical_stress_performance,
    calculate_hypothetical_stress_table,
    calculate_stress_period_benchmark_comparison,
    classify_asset_for_stress,
    find_worst_periods,
)
from .var_es import (
    calculate_historical_es,
    calculate_historical_var,
    calculate_var_exceptions,
)

__all__ = [
    "RiskAnalytics",
    "PerformanceAnalytics",
    "calculate_active_risk_metrics",
    "calculate_beta",
    "calculate_concentration_metrics",
    "calculate_drawdown_durations",
    "calculate_hit_ratio",
    "calculate_information_ratio",
    "calculate_jensens_alpha",
    "calculate_simple_alpha",
    "calculate_tracking_error",
    "calculate_liquidity_diagnostics",
    "summarize_liquidity_diagnostics",
    "portfolio_volatility",
    "marginal_risk_contribution",
    "total_risk_contribution",
    "percentage_risk_contribution",
    "risk_contribution_table",
    "compare_risk_contributions",
    "StressTestingFramework",
    "DEFAULT_HYPOTHETICAL_SCENARIOS",
    "DEFAULT_STRESS_PERIODS",
    "apply_hypothetical_stress",
    "calculate_correlation_stress",
    "calculate_historical_stress_performance",
    "calculate_hypothetical_stress_table",
    "calculate_stress_period_benchmark_comparison",
    "classify_asset_for_stress",
    "find_worst_periods",
    "calculate_historical_es",
    "calculate_historical_var",
    "calculate_var_exceptions",
]
