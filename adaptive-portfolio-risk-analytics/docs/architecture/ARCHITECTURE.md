# Architecture

This document describes the current, tested repository. Historical scaffolding is archived under
`docs/archive/`; implementation code and tests take precedence over older stage reports.

## Runtime entrypoints

- `src/dashboard/app.py`: primary Streamlit application.
- `main.py`: Phase 1 Yahoo Finance pipeline and fixed-strategy backtest runner.
- `scripts/final_smoke_test.py`: lightweight release/import/artifact guard.
- `scripts/*.py`: governed NLP corpus ingestion, validation, monitoring, and shadow experiments.
- `notebooks/`: stage-oriented research companions; several are intentionally unexecuted.

## End-to-end flow

```text
Yahoo Finance / governed local text inputs
                |
                v
Data inspection and centralized preprocessing
                |
                v
Returns -> covariance/correlation -> clustering -> portfolio weights
                |
                v
Rolling net/gross backtests -> risk, stress, liquidity, and active-risk analytics
                |
                +--> sensitivity and CPCV-style robustness
                |
                +--> lagged regimes -> adaptive policy and defensive sleeve
                |
                +--> timestamped sentiment/NLP monitoring and shadow overlays
                |
                v
Strategy-selection evidence gates -> Manager / Research / Developer dashboard views
```

## Package map

| Package | Responsibility | Current boundary |
| --- | --- | --- |
| `src/data_pipeline` | Yahoo Finance ingestion, inspection, missingness, returns, outliers, defensive returns | Alpha Vantage market-price ingestion remains an extension point |
| `src/covariance` | Sample, Ledoit-Wolf, EWMA, EWMA plus Ledoit-Wolf, correlation and distance | Gerber covariance is not implemented |
| `src/clustering` | Linkage, cluster membership, dendrograms, HERC engine, legacy HRP helpers | Canonical HERC is `herc_allocator.py` |
| `src/optimization` | Equal Weight, Inverse Volatility, HRP, HERC export, Mean-Variance | Dynamic allocator class is a non-functional future extension point |
| `src/benchmarks` | Four-strategy fixed benchmark factory and comparison tables | Mean-Variance is not a first-class benchmark/dashboard strategy |
| `src/backtesting` | Rolling backtest, drift-aware rebalance rules, turnover, costs, volatility targeting | `CPCVBacktester` is compatibility-only; use `src.validation` |
| `src/analytics` | Performance, drawdown, Pain Ratio, VaR/ES, risk contribution, stress, liquidity, active risk | Market-impact modeling is diagnostic, not an execution simulator |
| `src/regime` | Rule-based and HMM full-sample/walk-forward regime research | Full-sample HMM is historical-only |
| `src/adaptive` | Lag-safe regime policies, defensive returns, adaptive backtest | Research backtest, not live allocation |
| `src/experiments` | Fixed/adaptive grids, sensitivity, replication, reporting | Some report/export branches have lighter coverage |
| `src/validation` | Purged/embargoed CPCV-style splits and robustness ranking | Pragmatic split combinations, not complete independent-path CPCV |
| `src/selection` | Investor profiles, evidence gates, role scoring, recommendations | Consumes saved research evidence; it is not personalized advice |
| `src/sentiment` | Corpus governance, providers, lexicon/optional FinBERT scoring, monitoring, shadow overlay | NLP does not drive production-active weights or gates |
| `src/dashboard` | Streamlit orchestration, plots, components, three audience modes | Large monolithic app; browser-level automation is limited |

## Data contracts

- Portfolio inputs are wide `pandas.DataFrame` objects of simple daily returns with a
  `DatetimeIndex` and one column per asset.
- Backtests expose net and gross return/value series separately. Costs are reflected in net
  returns and net final value.
- Weights selected at decision date `t` apply to returns at `t+1`.
- Target weights update on their configured schedule; threshold rebalancing compares naturally
  drifted weights with the latest stored target.
- Experiment and CPCV result tables use one row per configuration or configuration/fold.
- Sentiment records must carry publication/availability timing and are lagged before comparison.

## Paths and portability

`src/paths.py` owns the repository root and common data/config/output paths. Runtime modules use
these paths instead of machine-specific absolute paths. Scripts retain a small file-relative
bootstrap so they can be launched directly from any working directory.

## Compatibility and historical material

- `src.optimization.HERCAllocator` remains a public re-export of the clustering implementation.
- `src.backtesting.CPCVBacktester` remains a compatibility stub; canonical validation functions
  are exported by `src.validation`.
- Historical stage/audit/bootstrap reports are retained under `docs/stage_reports`,
  `docs/audits`, and `docs/archive` rather than presented as current architecture.

## Operational boundary

The repository is a research and decision-support system. It has no broker integration, order
management, live-trading controls, production model governance, or complete market-impact engine.
