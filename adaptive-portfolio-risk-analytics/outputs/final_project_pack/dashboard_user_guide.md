# Dashboard User Guide

**v1.0 — Regime-Aware Portfolio Risk Analytics Platform**

Launch the application from the repository root:

```bash
streamlit run src/dashboard/app.py
```

## Manager View

Manager View is the simplified recommendation interface. It accepts portfolio universe, investment amount, date range, investor objective, and cost assumption, then shows:

- strategic core,
- risk-control overlay or robustness fallback,
- recommendation confidence,
- net performance and drawdown trade-offs,
- explanation, warnings, and assumptions.

The default manager settings are Balanced objective, Moderate costs, and HMM Walk-Forward Conservative as the adaptive risk-control candidate. The evidence gates can reduce confidence or fall back to Rule-based Conservative.

## Research View

Research View exposes model and validation detail:

- covariance and rebalancing settings,
- volatility targeting,
- rule-based and HMM regime controls,
- adaptive policy settings,
- sensitivity analysis,
- CPCV-style validation,
- policy and regime attribution,
- liquidity and cost diagnostics,
- selection gates, scores, and scenario playbook.

Use this view to explain methodology or test assumptions. It is not required for a first-pass recommendation.

## Developer / Debug View

Developer / Debug View exposes audit and implementation detail:

- raw HMM diagnostics,
- CPCV split and fold details,
- daily adaptive decision logs,
- full weight histories,
- internal configuration,
- net/gross and defensive-return reconciliation,
- raw recommendation payload,
- gate results and scoring trace.

Use this view when checking implementation behavior, failed folds, data alignment, or accounting consistency.

## Short demo flow

1. Select the universe and date range.
2. Choose the investor objective.
3. Select the cost assumption and run the recommendation.
4. Read the executive recommendation card.
5. Compare HERC, HMM Conservative, Rule-based Conservative, and Equal Weight.
6. Open Research View only if methodology or validation details are needed.

## How to explain the recommendation

Use role language:

- HERC is the strategic growth core.
- HMM Conservative is the drawdown-control overlay.
- Rule-based Conservative is the robustness reference and HMM fallback.
- Equal Weight is the benchmark.

Do not describe adaptive as the best strategy overall. State that confidence is Moderate because adaptive CPCV successful-fold coverage is limited.

## Reading metric labels

Headline performance metrics are net of configured transaction costs. Gross values are shown only in reconciliation or cost-drag diagnostics. Maximum drawdown is negative, so a less-negative value is better.

Experiment `var_95` and `cvar_95` are signed daily return-tail statistics. Dashboard historical VaR and ES are displayed as positive loss amounts or rates; their signs should not be compared directly.

## Troubleshooting

- If market data cannot be downloaded, confirm internet access and ticker availability.
- If HMM is unavailable, confirm `hmmlearn` is installed; the recommendation must not substitute full-sample HMM.
- If early CPCV folds fail, check the available warm-up history and model-training requirements.
- If the dashboard is slow, reduce the universe or avoid running HMM and CPCV diagnostics during a short demo.

