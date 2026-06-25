"""Run Phase 4A.13 NLP shadow-impact experiment.

The experiment is reporting-only. It compares fixed/adaptive baselines against
two decision-lagged NLP shadow overlays without changing production allocation,
strategy selection, evidence gates, or backtests outside this script.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adaptive import get_policy_preset, run_regime_adaptive_backtest
from src.analytics import (
    PerformanceAnalytics,
    calculate_drawdown_durations,
    compute_drawdown_series,
)
from src.backtesting import RollingBacktester
from src.backtesting.transaction_costs import TransactionCostModel
from src.benchmarks import BenchmarkFactory
from src.data_pipeline import DataPreprocessor, YahooFinanceProvider
from src.experiments.replication import DEFAULT_REPLICATION_UNIVERSES
from src.regime import (
    HMM_AVAILABLE,
    calculate_regime_features,
    classify_rule_based_regime,
    fit_hmm_walk_forward,
)
from src.sentiment.nlp_shadow_overlay import (
    CONFIRMATION_VARIANT,
    EARLY_WARNING_VARIANT,
    NLPShadowOverlayConfig,
    build_nlp_signal_alignment,
    build_overlay_decisions,
    build_shadow_policy_map,
)


OUTPUT_DIR = REPO_ROOT / "outputs" / "reports" / "phase_4a13_nlp_shadow_impact"
PHASE_4A8_SIGNAL = (
    REPO_ROOT / "outputs" / "reports" / "phase_4a8_multisource_nlp_monitoring" / "daily_nlp_signal.csv"
)
PHASE_4A6_SIGNAL = (
    REPO_ROOT / "outputs" / "reports" / "phase_4a6_real_nlp_validation" / "daily_nlp_signal.csv"
)
INITIAL_CAPITAL = 1_000_000.0


@dataclass(frozen=True)
class ExperimentResult:
    output_dir: Path
    strategy_metrics: pd.DataFrame
    pain_ratio_comparison: pd.DataFrame
    drawdown_comparison: pd.DataFrame
    overlay_decisions: pd.DataFrame
    nlp_signal_alignment: pd.DataFrame
    lookahead_diagnostics: pd.DataFrame
    summary: dict[str, object]


def run_nlp_shadow_impact_experiment(
    *,
    start_date: str,
    end_date: str,
    include_transaction_costs: bool = True,
    decision_lag_days: int = 1,
    output_dir: Path | str = OUTPUT_DIR,
    returns_df: pd.DataFrame | None = None,
    nlp_signal: pd.DataFrame | None = None,
) -> ExperimentResult:
    """Run the Phase 4A.13 shadow experiment and write report artifacts."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start >= end:
        raise ValueError("start_date must be before end_date")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if returns_df is None:
        returns, data_source, data_warning = _load_or_generate_returns(start, end)
    else:
        returns = _clean_returns(returns_df)
        data_source = "provided_returns"
        data_warning = ""

    evaluation_returns = returns.loc[start:end]
    if evaluation_returns.empty:
        raise ValueError("no returns are available in the requested evaluation window")
    training_window = _training_window(len(returns))
    transaction_cost_bps = 10.0 if include_transaction_costs else 0.0
    slippage_bps = 5.0 if include_transaction_costs else 0.0
    cost_model = TransactionCostModel(
        base_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )

    features = calculate_regime_features(returns)
    rule_regimes = classify_rule_based_regime(features)
    hmm_regimes, hmm_status = _build_hmm_decision_regimes(
        returns,
        features,
        training_window=training_window,
        decision_lag_days=decision_lag_days,
    )
    nlp_frame = nlp_signal.copy() if isinstance(nlp_signal, pd.DataFrame) else _load_nlp_signal()
    nlp_alignment = build_nlp_signal_alignment(
        nlp_frame,
        returns.index,
        decision_lag_days=decision_lag_days,
    )

    shadow_config = NLPShadowOverlayConfig(decision_lag_days=decision_lag_days)
    confirmation_regimes, confirmation_decisions = build_overlay_decisions(
        hmm_regimes,
        nlp_alignment,
        variant=CONFIRMATION_VARIANT,
        config=shadow_config,
        features=features,
    )
    early_warning_regimes, early_warning_decisions = build_overlay_decisions(
        hmm_regimes,
        nlp_alignment,
        variant=EARLY_WARNING_VARIANT,
        config=shadow_config,
        features=features,
    )

    backtests: dict[str, dict[str, object]] = {}
    backtests["Fixed HERC"] = RollingBacktester(
        allocator=BenchmarkFactory.get_allocator("HERC", covariance_method="sample"),
        train_window=training_window,
        rebalance_frequency="M",
        initial_capital=INITIAL_CAPITAL,
        rebalance_mode="calendar",
        transaction_cost_model=cost_model,
    ).run(returns)
    backtests["HMM Conservative"] = run_regime_adaptive_backtest(
        returns=returns,
        regimes=hmm_regimes,
        initial_value=INITIAL_CAPITAL,
        training_window=training_window,
        rebalance_frequency="M",
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        policy_map=get_policy_preset("Conservative"),
        regime_method_name="HMM walk-forward decision regimes",
        use_lagged_regimes=False,
        defensive_source="synthetic",
        defensive_annual_rate=0.04,
    )
    backtests["Rule Conservative"] = run_regime_adaptive_backtest(
        returns=returns,
        regimes=rule_regimes,
        initial_value=INITIAL_CAPITAL,
        training_window=training_window,
        rebalance_frequency="M",
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        policy_map=get_policy_preset("Conservative"),
        regime_method_name="Rule-based observed regimes, lagged internally",
        use_lagged_regimes=True,
        defensive_source="synthetic",
        defensive_annual_rate=0.04,
    )
    backtests["HMM + NLP Confirmation Overlay"] = run_regime_adaptive_backtest(
        returns=returns,
        regimes=confirmation_regimes,
        initial_value=INITIAL_CAPITAL,
        training_window=training_window,
        rebalance_frequency="M",
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        policy_map=build_shadow_policy_map(CONFIRMATION_VARIANT, config=shadow_config),
        regime_method_name="HMM + NLP confirmation shadow overlay",
        use_lagged_regimes=False,
        defensive_source="synthetic",
        defensive_annual_rate=0.04,
    )
    backtests["HMM + NLP Early-Warning Overlay"] = run_regime_adaptive_backtest(
        returns=returns,
        regimes=early_warning_regimes,
        initial_value=INITIAL_CAPITAL,
        training_window=training_window,
        rebalance_frequency="M",
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        policy_map=build_shadow_policy_map(EARLY_WARNING_VARIANT, config=shadow_config),
        regime_method_name="HMM + NLP early-warning shadow overlay",
        use_lagged_regimes=False,
        defensive_source="synthetic",
        defensive_annual_rate=0.04,
    )

    strategy_metrics = pd.DataFrame(
        [
            _summarize_backtest(
                strategy,
                backtest,
                start,
                end,
                shadow_experimental=strategy.startswith("HMM + NLP"),
            )
            for strategy, backtest in backtests.items()
        ]
    )
    strategy_metrics["data_source"] = data_source
    strategy_metrics["hmm_status"] = hmm_status
    strategy_metrics["transaction_costs_included"] = bool(include_transaction_costs)

    pain_ratio_comparison = _build_metric_comparison(
        strategy_metrics,
        ["pain_ratio", "pain_index", "calmar", "max_drawdown", "cagr"],
        baseline_strategy="HMM Conservative",
    )
    drawdown_comparison = _build_drawdown_comparison(backtests, start, end)
    overlay_decisions = pd.concat(
        [
            confirmation_decisions,
            early_warning_decisions,
        ],
        ignore_index=True,
    )
    overlay_decisions = overlay_decisions.loc[
        pd.to_datetime(overlay_decisions["decision_date"]).between(start, end)
    ].reset_index(drop=True)
    nlp_signal_alignment = nlp_alignment.loc[
        pd.to_datetime(nlp_alignment["decision_date"]).between(start, end)
    ].reset_index(drop=True)
    lookahead_diagnostics = nlp_signal_alignment[
        [
            "decision_date",
            "nlp_signal_date_used",
            "latest_allowed_signal_date",
            "nlp_label",
            "source_mix",
            "decision_lag_days",
            "lookahead_check_passed",
        ]
    ].copy()
    summary = _build_summary(
        strategy_metrics=strategy_metrics,
        lookahead_diagnostics=lookahead_diagnostics,
        overlay_decisions=overlay_decisions,
        start=start,
        end=end,
        data_source=data_source,
        data_warning=data_warning,
        hmm_status=hmm_status,
    )

    strategy_metrics.to_csv(out_dir / "strategy_metrics.csv", index=False)
    pain_ratio_comparison.to_csv(out_dir / "pain_ratio_comparison.csv", index=False)
    drawdown_comparison.to_csv(out_dir / "drawdown_comparison.csv", index=False)
    overlay_decisions.to_csv(out_dir / "overlay_decisions.csv", index=False)
    nlp_signal_alignment.to_csv(out_dir / "nlp_signal_alignment.csv", index=False)
    lookahead_diagnostics.to_csv(out_dir / "lookahead_diagnostics.csv", index=False)
    _write_summary(out_dir / "summary.md", summary, strategy_metrics, pain_ratio_comparison)
    _write_limitations(out_dir / "limitations.md", summary)
    _write_html_report(
        out_dir / "report.html",
        summary,
        strategy_metrics,
        pain_ratio_comparison,
        overlay_decisions,
        lookahead_diagnostics,
    )

    return ExperimentResult(
        output_dir=out_dir,
        strategy_metrics=strategy_metrics,
        pain_ratio_comparison=pain_ratio_comparison,
        drawdown_comparison=drawdown_comparison,
        overlay_decisions=overlay_decisions,
        nlp_signal_alignment=nlp_signal_alignment,
        lookahead_diagnostics=lookahead_diagnostics,
        summary=summary,
    )


def _load_or_generate_returns(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, str, str]:
    context_start = start - pd.DateOffset(years=3)
    tickers = DEFAULT_REPLICATION_UNIVERSES["Core Diversified"]
    try:
        market_data = YahooFinanceProvider().get_market_data(
            symbols=tickers,
            start_date=context_start.date().isoformat(),
            end_date=(end.date() + timedelta(days=1)).isoformat(),
        )
        clean_prices, _ = DataPreprocessor.handle_missing_values(market_data.prices_df)
        returns = DataPreprocessor.build_returns_risk_outputs(clean_prices).returns_df
        returns = _clean_returns(returns)
        if returns.loc[start:end].empty or returns.index.max() < end - pd.offsets.BDay(5):
            raise ValueError("downloaded data does not cover the requested evaluation window")
        return returns, "yahoo_finance", ""
    except Exception as exc:
        warning = (
            "Live market data unavailable or incomplete; deterministic offline "
            f"returns were used. Cause: {exc}"
        )
        return _offline_returns(context_start, end, tickers), "deterministic_offline_fallback", warning


def _offline_returns(
    start: pd.Timestamp,
    end: pd.Timestamp,
    tickers: list[str],
) -> pd.DataFrame:
    index = pd.bdate_range(start, end)
    rng = np.random.default_rng(20260625)
    market = rng.normal(0.00025, 0.0075, len(index))
    stress_mask = (index >= pd.Timestamp("2026-05-04")) & (index <= pd.Timestamp("2026-05-22"))
    market[stress_mask] += rng.normal(-0.0035, 0.006, int(stress_mask.sum()))
    columns: dict[str, np.ndarray] = {}
    for position, ticker in enumerate(tickers):
        beta = 0.75 + 0.05 * (position % 6)
        idiosyncratic = rng.normal(0.00005, 0.006 + 0.0005 * (position % 5), len(index))
        columns[ticker] = np.clip(beta * market + idiosyncratic, -0.12, 0.12)
    return pd.DataFrame(columns, index=index)


def _load_nlp_signal() -> pd.DataFrame:
    for path in (PHASE_4A8_SIGNAL, PHASE_4A6_SIGNAL):
        if path.is_file():
            return pd.read_csv(path)
    return pd.DataFrame()


def _build_hmm_decision_regimes(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    *,
    training_window: int,
    decision_lag_days: int,
) -> tuple[pd.Series, str]:
    if not HMM_AVAILABLE:
        rule = classify_rule_based_regime(features).shift(int(decision_lag_days)).fillna("Unknown")
        return rule.reindex(returns.index).fillna("Unknown"), "fallback_rule_based_hmmlearn_unavailable"
    try:
        hmm_min_train = min(max(training_window, 63), max(training_window, len(returns) // 2))
        fitted = fit_hmm_walk_forward(
            features,
            n_states=4,
            min_train_size=hmm_min_train,
            refit_frequency=21,
            covariance_type="diag",
            decision_lag=int(decision_lag_days),
        )
        regimes = fitted["decision_regimes"].reindex(returns.index).fillna("Unknown")
        if regimes.astype(str).eq("Unknown").all():
            raise ValueError("HMM produced only Unknown decision regimes")
        return regimes, "hmm_walk_forward"
    except Exception as exc:
        rule = classify_rule_based_regime(features).shift(int(decision_lag_days)).fillna("Unknown")
        return (
            rule.reindex(returns.index).fillna("Unknown"),
            f"fallback_rule_based_hmm_failed:{exc}",
        )


def _summarize_backtest(
    strategy: str,
    backtest: dict[str, object],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    shadow_experimental: bool,
) -> dict[str, object]:
    returns = pd.to_numeric(backtest["portfolio_returns"], errors="coerce").loc[start:end].dropna()
    if returns.empty:
        raise ValueError(f"{strategy} produced no evaluation-window returns")
    values = (1.0 + returns).cumprod() * INITIAL_CAPITAL
    values = pd.concat(
        [pd.Series([INITIAL_CAPITAL], index=[returns.index[0] - pd.Timedelta(nanoseconds=1)]), values]
    )
    metrics = PerformanceAnalytics.summary_table(returns)
    durations = calculate_drawdown_durations(values)
    turnover, transaction_cost, rebalances = _activity_metrics(backtest, start, end)
    return {
        "strategy": strategy,
        "strategy_role": _strategy_role(strategy),
        "shadow_experimental": bool(shadow_experimental),
        "production_allocation_active": False if shadow_experimental else None,
        "cagr": metrics["cagr"],
        "volatility": metrics["volatility"],
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
        "max_drawdown": metrics["max_drawdown"],
        "calmar": metrics["calmar"],
        "pain_index": metrics["pain_index"],
        "pain_ratio": metrics["pain_ratio"],
        "var_95": metrics["var_95"],
        "cvar_95": metrics["cvar_95"],
        "final_value": float(values.iloc[-1]),
        "max_drawdown_duration": int(durations["max_drawdown_duration"]),
        "recovery_duration": _maximum_drawdown_recovery_duration(returns),
        "total_turnover": turnover,
        "average_turnover": turnover / max(rebalances, 1),
        "total_transaction_cost": transaction_cost,
        "transaction_cost_drag": transaction_cost / INITIAL_CAPITAL,
        "number_of_rebalances": rebalances,
        "evaluation_start": returns.index.min().date().isoformat(),
        "evaluation_end": returns.index.max().date().isoformat(),
        "n_observations": int(len(returns)),
    }


def _activity_metrics(
    backtest: dict[str, object],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float, float, int]:
    if "diagnostics" in backtest and isinstance(backtest["diagnostics"], pd.DataFrame):
        diagnostics = backtest["diagnostics"].copy()
        if diagnostics.empty:
            return 0.0, 0.0, 0
        diagnostics["date"] = pd.to_datetime(diagnostics["date"], errors="coerce")
        diagnostics = diagnostics.loc[diagnostics["date"].between(start, end)]
        return (
            float(pd.to_numeric(diagnostics.get("turnover"), errors="coerce").fillna(0.0).sum()),
            float(
                pd.to_numeric(diagnostics.get("transaction_cost"), errors="coerce")
                .fillna(0.0)
                .sum()
            ),
            int(diagnostics.get("rebalanced", pd.Series(dtype=bool)).fillna(False).sum()),
        )
    rebalance_log = backtest.get("rebalance_log", pd.DataFrame())
    if not isinstance(rebalance_log, pd.DataFrame) or rebalance_log.empty:
        return 0.0, 0.0, 0
    log = rebalance_log.copy()
    log["rebalance_date"] = pd.to_datetime(log["rebalance_date"], errors="coerce")
    log = log.loc[log["rebalance_date"].between(start, end)]
    return (
        float(pd.to_numeric(log.get("turnover"), errors="coerce").fillna(0.0).sum()),
        float(pd.to_numeric(log.get("transaction_cost"), errors="coerce").fillna(0.0).sum()),
        int(len(log)),
    )


def _strategy_role(strategy: str) -> str:
    roles = {
        "Fixed HERC": "strategic growth core",
        "HMM Conservative": "risk-control overlay baseline",
        "Rule Conservative": "robustness/fallback reference",
        "HMM + NLP Confirmation Overlay": "shadow/experimental NLP confirmation",
        "HMM + NLP Early-Warning Overlay": "shadow/experimental NLP early warning",
    }
    return roles.get(strategy, "experiment")


def _build_metric_comparison(
    metrics: pd.DataFrame,
    columns: list[str],
    *,
    baseline_strategy: str,
) -> pd.DataFrame:
    baseline = metrics.loc[metrics["strategy"].eq(baseline_strategy)]
    baseline_row = baseline.iloc[0] if not baseline.empty else pd.Series(dtype=float)
    rows = []
    for _, row in metrics.iterrows():
        output = {"strategy": row["strategy"]}
        for column in columns:
            output[column] = row.get(column, np.nan)
            output[f"{column}_delta_vs_{baseline_strategy}"] = (
                row.get(column, np.nan) - baseline_row.get(column, np.nan)
            )
        rows.append(output)
    return pd.DataFrame(rows)


def _build_drawdown_comparison(
    backtests: dict[str, dict[str, object]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for strategy, backtest in backtests.items():
        returns = pd.to_numeric(backtest["portfolio_returns"], errors="coerce").loc[start:end].dropna()
        drawdown = compute_drawdown_series(returns)
        rows.append(
            pd.DataFrame(
                {
                    "date": drawdown.index,
                    "strategy": strategy,
                    "drawdown": drawdown.to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_summary(
    *,
    strategy_metrics: pd.DataFrame,
    lookahead_diagnostics: pd.DataFrame,
    overlay_decisions: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_source: str,
    data_warning: str,
    hmm_status: str,
) -> dict[str, object]:
    indexed = strategy_metrics.set_index("strategy")
    hmm = indexed.loc["HMM Conservative"]
    nlp_rows = indexed.loc[
        [
            "HMM + NLP Confirmation Overlay",
            "HMM + NLP Early-Warning Overlay",
        ]
    ]
    best_nlp_name = str(nlp_rows["pain_ratio"].idxmax())
    best_nlp = nlp_rows.loc[best_nlp_name]
    pain_ratio_delta = float(best_nlp["pain_ratio"] - hmm["pain_ratio"])
    pain_index_delta = float(best_nlp["pain_index"] - hmm["pain_index"])
    calmar_delta = float(best_nlp["calmar"] - hmm["calmar"])
    max_drawdown_delta = float(best_nlp["max_drawdown"] - hmm["max_drawdown"])
    cagr_delta = float(best_nlp["cagr"] - hmm["cagr"])
    turnover_delta = float(best_nlp["total_turnover"] - hmm["total_turnover"])
    cost_delta = float(best_nlp["transaction_cost_drag"] - hmm["transaction_cost_drag"])
    lookahead_passed = bool(lookahead_diagnostics["lookahead_check_passed"].all())
    overlay_action_count = int(
        overlay_decisions["overlay_action"].ne("no_overlay").sum()
        if not overlay_decisions.empty
        else 0
    )
    positive_shadow = pain_ratio_delta > 1e-9
    if positive_shadow and int(hmm["n_observations"]) < 126:
        verdict = "Positive shadow impact, not production-ready."
    elif positive_shadow and cagr_delta < -0.01:
        verdict = "Risk-control benefit with opportunity-cost tradeoff."
    elif positive_shadow:
        verdict = "Positive shadow impact, evidence still monitoring-only."
    else:
        verdict = "NLP remains useful for monitoring/context but not allocation."
    return {
        "phase": "Phase 4A.13",
        "version_label": "v1.3.0 — Final Integrated Portfolio Risk Analytics Release",
        "evaluation_start": start.date().isoformat(),
        "evaluation_end": end.date().isoformat(),
        "data_source": data_source,
        "data_warning": data_warning,
        "hmm_status": hmm_status,
        "best_nlp_shadow_strategy": best_nlp_name,
        "pain_ratio_delta_vs_hmm": pain_ratio_delta,
        "pain_index_delta_vs_hmm": pain_index_delta,
        "calmar_delta_vs_hmm": calmar_delta,
        "max_drawdown_delta_vs_hmm": max_drawdown_delta,
        "cagr_delta_vs_hmm": cagr_delta,
        "turnover_delta_vs_hmm": turnover_delta,
        "transaction_cost_drag_delta_vs_hmm": cost_delta,
        "lookahead_passed": lookahead_passed,
        "overlay_action_count": overlay_action_count,
        "positive_shadow_impact": bool(positive_shadow),
        "production_allocation_active": False,
        "promotion_recommendation": "No",
        "verdict": verdict,
    }


def _write_summary(
    path: Path,
    summary: dict[str, object],
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 4A.13 NLP Shadow Impact Summary",
        "",
        f"Version: {summary['version_label']}",
        f"Evaluation window: {summary['evaluation_start']} to {summary['evaluation_end']}",
        f"Data source: {summary['data_source']}",
        f"HMM status: {summary['hmm_status']}",
        "",
        f"Verdict: {summary['verdict']}",
        "",
        "NLP production allocation active: No.",
        "NLP variants are shadow/experimental only.",
        "",
        "## Key answers",
        "",
        f"- Did NLP improve Pain Ratio? {'Yes' if summary['pain_ratio_delta_vs_hmm'] > 0 else 'No'}",
        f"- Did NLP reduce Pain Index? {'Yes' if summary['pain_index_delta_vs_hmm'] < 0 else 'No'}",
        f"- Did NLP improve Calmar? {'Yes' if summary['calmar_delta_vs_hmm'] > 0 else 'No'}",
        f"- Did NLP reduce Max Drawdown? {'Yes' if summary['max_drawdown_delta_vs_hmm'] > 0 else 'No'}",
        f"- Did NLP reduce turnover? {'Yes' if summary['turnover_delta_vs_hmm'] < 0 else 'No'}",
        f"- Did NLP reduce transaction cost drag? {'Yes' if summary['transaction_cost_drag_delta_vs_hmm'] < 0 else 'No'}",
        f"- Look-ahead diagnostics passed? {'Yes' if summary['lookahead_passed'] else 'No'}",
        "- Should NLP be promoted to production allocation? No.",
        "",
        "## Strategy metrics",
        "",
        _markdown_table(metrics),
        "",
        "## Pain Ratio comparison",
        "",
        _markdown_table(comparison),
    ]
    if summary.get("data_warning"):
        lines.extend(["", "## Data warning", "", str(summary["data_warning"])])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_limitations(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Phase 4A.13 Limitations",
        "",
        "- NLP remains monitoring-only in production.",
        "- The two NLP variants are shadow/experimental and do not alter production allocation.",
        "- The evidence window is short and should not be treated as proof of predictive power.",
        "- NLP signals are decision-lagged and look-ahead diagnostics are reported, but source coverage can remain sparse.",
        "- Transaction costs use the project transaction-cost model and are not full market-impact estimates.",
        "- HMM fitting can fall back to rule-based decision labels if HMM inference is unavailable or unstable.",
        "- Promotion to allocation would require separate, explicit evidence gates and robust out-of-sample validation.",
    ]
    if summary.get("data_warning"):
        lines.append(f"- {summary['data_warning']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html_report(
    path: Path,
    summary: dict[str, object],
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    overlay_decisions: pd.DataFrame,
    lookahead_diagnostics: pd.DataFrame,
) -> None:
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Phase 4A.13 NLP Shadow Impact</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f2f4f8; }}
    .verdict {{ padding: 12px; background: #eef6ff; border-left: 4px solid #3366cc; }}
  </style>
</head>
<body>
  <h1>Phase 4A.13 — Pain Ratio and NLP Shadow Impact Analysis</h1>
  <p class="verdict"><strong>Verdict:</strong> {summary['verdict']}</p>
  <p><strong>Production allocation active:</strong> No. NLP variants are shadow/experimental only.</p>
  <p><strong>Look-ahead diagnostics passed:</strong> {summary['lookahead_passed']}</p>
  <p><strong>Data source:</strong> {summary['data_source']}</p>
  <h2>Strategy metrics</h2>
  {metrics.to_html(index=False)}
  <h2>Pain Ratio comparison</h2>
  {comparison.to_html(index=False)}
  <h2>Overlay decision sample</h2>
  {overlay_decisions.head(50).to_html(index=False)}
  <h2>Look-ahead diagnostics sample</h2>
  {lookahead_diagnostics.head(50).to_html(index=False)}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
            )
    headers = [str(column) for column in display.columns]
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(rows)


def _maximum_drawdown_recovery_duration(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    trough_date = drawdown.idxmin()
    peak_value = float(running_peak.loc[trough_date])
    recovered = wealth.loc[trough_date:]
    recovered = recovered.loc[recovered >= peak_value]
    if recovered.empty:
        return np.nan
    return float(len(wealth.loc[trough_date : recovered.index[0]]) - 1)


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a DataFrame")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a DatetimeIndex")
    clean = returns.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(how="any").sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]
    if clean.empty or clean.shape[1] < 2:
        raise ValueError("returns must contain at least two assets")
    return clean


def _training_window(n_observations: int) -> int:
    if n_observations <= 60:
        raise ValueError("at least 61 observations are required")
    return min(252, max(40, n_observations // 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--include-transaction-costs", action="store_true")
    parser.add_argument("--decision-lag-days", type=int, default=1)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_nlp_shadow_impact_experiment(
        start_date=args.start_date,
        end_date=args.end_date,
        include_transaction_costs=bool(args.include_transaction_costs),
        decision_lag_days=int(args.decision_lag_days),
        output_dir=Path(args.output_dir),
    )
    print(f"Wrote Phase 4A.13 artifacts to {result.output_dir}")
    print(f"Verdict: {result.summary['verdict']}")
    print(f"Look-ahead diagnostics passed: {result.summary['lookahead_passed']}")
    print(f"Production allocation active: {result.summary['production_allocation_active']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
