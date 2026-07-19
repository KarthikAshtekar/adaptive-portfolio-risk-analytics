# Adaptive Portfolio Risk Analytics

An evidence-gated Python and Streamlit research platform for portfolio construction, realistic
backtesting, downside-risk analysis, regime-aware overlays, robustness validation, and governed
sentiment monitoring.

Current release label: **v1.3.0 — Final Integrated Portfolio Risk Analytics Release**

> Research software only. The repository does not place orders, connect to a broker, or provide
> individualized investment advice.

## Problem statement

Simple portfolio comparisons often mix inconsistent data windows, ignore trading frictions, tune
many settings on one history, and report only the best return metric. This project builds a common
daily-return research pipeline so fixed and adaptive policies can be compared on allocation,
drawdown, turnover, cost drag, stress behavior, and fold-level robustness.

The project does not claim one universal winner. Saved evidence assigns methods different research
roles: HERC as a growth-oriented core, conservative adaptive policies as downside-control overlays,
Rule-based Conservative as an explainable fallback, and Equal Weight as the transparent baseline.

## What is actually implemented

| Capability | Evidence | Status | Boundary |
| --- | --- | --- | --- |
| Yahoo Finance prices, adjusted-close fallback, volume inspection | `src/data_pipeline/ingest.py`, data-pipeline tests | Fully implemented and tested | Network/source revisions affect reruns |
| Missingness, anomaly, returns, winsorization, and data-quality diagnostics | `src/data_pipeline/preprocess.py`, `tests/test_data_pipeline.py` | Fully implemented and tested | Cleaning is centralized before optimization |
| Sample, Ledoit-Wolf, EWMA, and EWMA plus Ledoit-Wolf covariance | `src/covariance`, covariance tests | Fully implemented and tested | Gerber covariance is not implemented |
| Correlation distance, linkage, dendrogram, HRP, and HERC | `src/clustering`, `src/optimization/hrp_allocator.py`, allocation tests | Fully implemented and tested | Long-only research allocators |
| Equal Weight, Inverse Volatility, HRP, HERC benchmark comparison | `src/benchmarks`, benchmark tests | Fully implemented and tested | Mean-Variance is standalone, not a dashboard benchmark |
| Mean-Variance / max-Sharpe allocator | `src/optimization/mean_variance.py`, optimization tests | Fully implemented and tested | Sensitive to expected-return estimates and corner solutions |
| Rolling net/gross backtests, drift-aware rebalancing, turnover, costs | `src/backtesting`, backtesting tests | Fully implemented and tested | Simplified cost model, not a market-impact simulator |
| VaR/ES, drawdown, Pain Ratio, risk contribution, stress, liquidity, active risk | `src/analytics`, analytics tests | Fully implemented and tested | VaR sign convention depends on the API |
| Volatility targeting and defensive sleeve | `src/backtesting/volatility_targeting.py`, defensive tests | Fully implemented and tested | Historical overlay research only |
| Rule-based and HMM walk-forward regimes | `src/regime`, regime tests | Fully implemented and tested | Full-sample HMM is historical visualization only |
| Regime-adaptive policies and backtest | `src/adaptive`, adaptive tests | Fully implemented and tested | Uses lagged decisions; not live allocation |
| Sensitivity grids and CPCV-style purge/embargo robustness | `src/experiments`, `src/validation`, validation tests | Fully implemented and tested | Pragmatic split combinations, not complete independent-path CPCV |
| Strategy selection and evidence gates | `src/selection`, selection tests | Fully implemented and tested | Decision support, not personalized advice |
| RBI/news ingestion, source quality, lexicon scoring, monitoring | `src/sentiment`, sentiment/provider tests | Fully implemented and tested as monitoring | Does not drive production-active weights |
| Optional local FinBERT scoring | `src/sentiment/finbert_scoring.py`, one focused test | Implemented but weakly tested | Falls back to lexicon when local model files are unavailable |
| Streamlit Manager/Research/Developer views | `src/dashboard`, dashboard source/import tests | Implemented but weakly tested | Limited browser-level and plot coverage; app remains large |
| Technical/macro feature engineering | `src/data_pipeline/feature_engineering.py` | Partially implemented / scaffolded | Rolling volatility exists; several feature methods are incomplete |
| Alpha Vantage market-price ingestion | `src/data_pipeline/ingest.py` | Partially implemented / scaffolded | Class exists but raises `NotImplementedError` |
| Dynamic allocator compatibility class | `src/optimization/dynamic_allocation.py` | Partially implemented / scaffolded | Canonical adaptive allocation lives in `src/adaptive` |

The complete evidence map is in [docs/PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md).

## Pipeline overview

```text
Market/text inputs
      |
      v
Inspection -> centralized preprocessing -> daily returns
      |
      v
Covariance/correlation -> clustering -> portfolio allocation
      |
      v
Rolling backtest -> net/gross performance -> costs and risk analytics
      |
      +--> sensitivity + CPCV-style robustness
      +--> lagged regimes + adaptive defensive policy
      +--> lagged sentiment/NLP monitoring + shadow overlays
      |
      v
Evidence gates and dashboard views
```

## Repository structure

```text
adaptive-portfolio-risk-analytics/
|-- config/                    # Portfolio and optional provider configuration
|-- data/                      # Tracked samples/templates; private/raw data ignored
|-- docs/
|   |-- architecture/         # Current architecture and references
|   |-- methodology/          # Current methodology
|   |-- stage_reports/        # Historical Stage 1-14 reports
|   |-- audits/               # Historical focused audits
|   `-- archive/              # Superseded bootstrap/release documents
|-- notebooks/                # Stage-oriented research notebooks
|-- outputs/
|   |-- final_project_pack/   # Tracked final narrative pack
|   `-- reports/              # Selected tracked evidence artifacts
|-- scripts/                  # Smoke, corpus, monitoring, and shadow-study commands
|-- src/
|   |-- data_pipeline/        # Market data and centralized cleaning
|   |-- covariance/           # Estimator factory and matrix utilities
|   |-- clustering/           # Hierarchy and HERC
|   |-- optimization/         # Fixed allocators and public exports
|   |-- benchmarks/           # Fixed-strategy comparison
|   |-- backtesting/          # Rolling simulation, costs, overlays
|   |-- analytics/            # FRM and active-risk metrics
|   |-- regime/               # Rule-based and HMM regimes
|   |-- adaptive/             # Regime policies and adaptive simulation
|   |-- experiments/          # Grids, sensitivity, replication
|   |-- validation/           # CPCV-style splits and robustness
|   |-- selection/            # Evidence gates and recommendations
|   |-- sentiment/            # Governed NLP monitoring and shadow logic
|   `-- dashboard/            # Streamlit app, modes, plots, components
|-- tests/                    # Unit and integration tests
|-- main.py                   # Phase 1 market-data pipeline
|-- project_explainer.html    # Offline interview/project explainer
|-- pyproject.toml            # Build and Ruff configuration
`-- verify_setup.py           # Structure/dependency verification
```

## Installation

Python 3.10 or later is required. The current local suite also passes on Python 3.13.

Windows PowerShell:

```powershell
cd D:\path\to\adaptive-portfolio-risk-analytics
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_setup.py
```

macOS/Linux:

```bash
cd /path/to/adaptive-portfolio-risk-analytics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_setup.py
```

Use `python -m pip`, not the `.venv/Scripts/pip` launcher, if a Windows environment was moved from
another drive and its launcher still contains an obsolete absolute Python path.

## Quickstart

Run the dashboard:

```powershell
python -m streamlit run src/dashboard/app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

Run release checks:

```powershell
python verify_setup.py
python scripts\final_smoke_test.py
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

## Dashboard guide

The dashboard has three modes:

- **Manager View** asks for universe, dates, investor objective, and cost assumption, then shows a
  role-based recommendation and its evidence/caveats.
- **Research View** exposes fixed/adaptive strategies, covariance choice, rebalance controls,
  risk/stress diagnostics, sensitivity, CPCV, regimes, and NLP monitoring.
- **Developer / Debug View** exposes raw diagnostics, look-ahead checks, source mix, gate traces,
  configuration, and reconciliation tables.

Suggested research workflow:

1. Choose at least two assets and a date range long enough for the selected training window.
2. Fix one objective before running sensitivity or CPCV.
3. Compare Equal Weight, Inverse Volatility, HRP, and HERC on the same data and costs.
4. Review net and gross performance together to understand cost drag.
5. Inspect turnover, rebalance reasons, weight drift, drawdown, stress, and concentration.
6. Treat adaptive results as overlays and check failed CPCV folds before accepting a ranking.
7. Treat sentiment/NLP as monitoring or shadow evidence only.

The dashboard contains a collapsed **How to use this dashboard** guide and contextual help beside
the controls that materially affect interpretation.

## Strategy guide

| Method | Suitable research use | Main trade-off |
| --- | --- | --- |
| Equal Weight | Transparent benchmark and low-assumption baseline | Ignores volatility and dependence |
| Inverse Volatility | Simple risk reduction without clustering | Ignores cross-asset covariance |
| HRP | Cluster-aware diversification with variance-based recursive allocation | Sensitive to hierarchy/linkage choices |
| HERC | Equal risk budgets across actual cluster-tree branches | Can retain larger drawdowns than defensive overlays |
| Mean-Variance | Expected-return-aware max-Sharpe research | Estimation error and concentrated solutions |
| Conservative adaptive | Downside-control overlay in stress/crisis states | Lower growth, model/policy complexity, more assumptions |

## Covariance and methodology guide

- `sample`: direct history; simple but unstable with short samples or many assets.
- `ledoit_wolf`: shrinkage; generally better conditioned and less sample-sensitive.
- `ewma`: emphasizes recent observations; faster response with more parameter sensitivity.
- `ewma_ledoit_wolf`: combines recency and shrinkage.

HRP and HERC accept all four methods through `CovarianceFactory`. For a fair comparison, keep the
asset universe, date range, covariance method, linkage, training window, rebalance rule, and costs
fixed.

## Programmatic backtest example

```python
from src.backtesting import RollingBacktester
from src.benchmarks import BenchmarkFactory

allocator = BenchmarkFactory.get_allocator(
    "HERC",
    covariance_method="ledoit_wolf",
)
backtester = RollingBacktester(
    allocator=allocator,
    train_window=252,
    rebalance_mode="threshold",
    threshold=0.05,
    target_update_frequency="M",
)
result = backtester.run(returns_df)

print(result["performance_metrics"])
print(result["rebalance_log"].head())
```

`returns_df` must be a non-empty daily simple-return DataFrame with a `DatetimeIndex` and one
column per asset.

## Experiments and robustness

Fixed and adaptive experiment grids are available through `src.experiments`. CPCV-style validation
is available through `src.validation`:

```python
from src.validation import generate_cpcv_splits, run_cpcv_validation
```

The validator uses ordered blocks, purge, and embargo, then reports fold medians, worst folds,
stability, failures, and robustness ranking. It does not implement complete independent-path CPCV
and does not guarantee future results.

## Saved results

The matched primary scenario in
[final_results_summary.md](outputs/final_project_pack/final_results_summary.md) uses the 12-asset
Core Diversified preset, January 2020 through June 19, 2026, initial capital 1,000,000, 10 bps base
cost plus 5 bps slippage, and a synthetic 4% defensive sleeve for adaptive policies.

| Strategy | Research role | Net CAGR | Calmar | Max drawdown | Net final value |
| --- | --- | ---: | ---: | ---: | ---: |
| HERC | Strategic growth core | 15.01% | 0.794 | -18.91% | 2,434,518 |
| HMM Walk-Forward Conservative | Risk-control overlay | 11.84% | 1.521 | -7.78% | 2,037,180 |
| Rule-based Conservative | Explainable fallback | 10.80% | 1.114 | -9.69% | 1,920,070 |
| Equal Weight | Benchmark | 12.56% | 0.381 | -32.94% | 2,122,050 |

These values belong to that saved snapshot. A later handoff table and the short Phase 4A.13 NLP
shadow window contain different values because they use different evaluation contexts. Do not mix
them into one ranking.

Current saved CPCV evidence ranks Rule-based Conservative first on Calmar robustness, but only 6
of 15 folds succeeded; HMM Conservative succeeded on 3 of 15. Limited fold coverage is a central
reason to keep confidence moderate.

The June 25 Phase 4A.6 monitoring artifact reports 34 real RBI documents, 50 real GDELT/news
records, and 98.3% decision-label coverage, with the verdict **useful for monitoring only**. The
older June 21 Phase 4A.3 artifact used synthetic fallback because its real corpus was unavailable.
Both are retained for provenance; neither supports production-active NLP allocation.

## Reports and artifacts

- [Offline project explainer](project_explainer.html)
- [Current project audit](docs/PROJECT_AUDIT.md)
- [Final project pack](outputs/final_project_pack/INDEX.md)
- [Team report handoff pack](outputs/reports/team_report_handoff_pack/README_FOR_TEAM.md)
- [Stage reports](docs/stage_reports/README.md)
- [Historical audits](docs/audits/README.md)
- [Architecture](docs/architecture/ARCHITECTURE.md)
- [Methodology](docs/methodology/METHODOLOGY.md)
- [Roadmap](docs/ROADMAP.md)

## Regenerating results

- `python main.py` reruns the Phase 1 Yahoo Finance workflow using `config/portfolio_config.yaml`;
  it requires network access and current source data.
- The notebooks provide stage-specific research workflows. Several retained notebooks are
  unexecuted, so they are code companions rather than frozen evidence.
- `python scripts/run_nlp_shadow_impact_experiment.py` regenerates the configured shadow study.
- `python scripts/validate_real_nlp_signal.py --help` documents real-provider validation options.
- Saved reports should record universe, dates, costs, defensive source, objective, and corpus
  metadata because these materially change outputs.

## Validation status

The final post-cleanup suite collected 505 tests and produced **503 passed, 2 skipped** with 66%
total coverage. The skipped tests are optional HMM branches. The pre-cleanup baseline was 504
passed and 2 skipped with 65% coverage. Exact commands and the test-inventory reconciliation are
recorded in [docs/PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md).

Dashboard plotting/orchestration and optional live-provider branches remain less covered than the
core allocation, backtesting, covariance, risk, regime, and selection layers.

## Limitations

- Historical backtests are sample-dependent and do not predict future performance.
- Yahoo Finance and external APIs can revise, throttle, or omit data.
- The validated universes and windows are bounded; results may not generalize.
- Adaptive CPCV loses early folds to training/warm-up requirements, and the current rank does not
  directly penalize low successful-fold coverage.
- HMM results are probabilistic and specification-sensitive.
- Defensive-sleeve assumptions materially affect adaptive results.
- Costs are turnover-based and omit a calibrated nonlinear market-impact/capacity model.
- Dashboard historical VaR/ES and experiment VaR/CVaR use different sign presentations.
- FinBERT is optional and locally dependent; the lexicon path is the deterministic baseline.
- Real NLP history is short, older corpus snapshots differ, and shadow gains are not allocation
  approval.
- The Streamlit app is large and has limited browser-level automated coverage.

## Future work

- Penalize or gate CPCV configurations with low successful-fold coverage.
- Replicate across broader independent universes and longer histories.
- Add versioned market/corpus snapshots and experiment manifests.
- Implement robust covariance alternatives such as Gerber under the factory contract.
- Add nonlinear liquidity-aware market impact and capacity analysis.
- Modularize dashboard orchestration and add browser interaction tests.
- Expand real timestamped NLP history and define explicit promotion gates.
- Add service/container/broker layers only if the project moves beyond research.

## Interview-ready summary

The project starts with cleaned daily market data, compares four first-class fixed strategies under
four covariance estimators, and evaluates them through rolling net-of-cost backtests and FRM risk
diagnostics. It then adds lag-safe rule-based/HMM regimes, adaptive defensive policies, sensitivity
grids, CPCV-style robustness, and evidence-gated recommendations. Saved results show a clear
trade-off: HERC delivered stronger growth in the matched primary scenario, while conservative
adaptive policies reduced drawdown. NLP is intentionally kept as governed monitoring and shadow
evidence because its history and validation are not strong enough to influence active allocation.

## Academic and research disclaimer

This repository is for education, research, and portfolio-risk analysis. Results are historical,
depend on assumptions and data availability, and are not investment recommendations. Independent
review, governance, and operational controls would be required before any real-money use.

## License

Released under the [MIT License](LICENSE).
