# Adaptive Portfolio Allocation and Risk Analytics under Dynamic Correlation and Market Regimes

Quantitative portfolio optimization and risk analytics platform built for portfolio construction, research, risk diagnostics, and backtesting.

## 📋 Overview

This platform currently provides:

- **Adaptive Portfolio Allocation**: Multiple allocation methods (Equal Weight, Mean-Variance, Hierarchical Risk Parity, Hierarchical Equal Risk Contribution)
- **Robust Covariance Estimation**: Sample covariance, Ledoit-Wolf shrinkage, EWMA, and EWMA plus Ledoit-Wolf
- **Dynamic Clustering**: Linkage-based clustering, dendrogram analysis, hierarchical portfolio construction
- **Volatility Targeting**: Rule-based volatility-state overlay with a defensive sleeve
- **Backtesting Framework**: Rolling-window validation, threshold rebalancing, transaction costs, and turnover diagnostics
- **Phase 3A — Robustness Validation / CPCV-Style Validation**: Time-series splits, purge and embargo controls, fold stability scoring, and robustness ranking
- **Phase 3B — Market Regime Detection & Regime Analytics**: Explainable rule-based regimes and optional experimental Gaussian HMM regimes
- **Phase 3C — Regime-Aware Adaptive Allocation**: Lagged regime policies dynamically select allocation, covariance, volatility targets, and defensive exposure
- **Phase 3D — Adaptive Strategy Experimentation & Robustness Evaluation**: Adaptive sensitivity grids, fixed-strategy comparisons, attribution, stress analysis, and optional CPCV evaluation
- **Phase 3E.1 — Defensive Return Consistency + Robustness Replication Harness**: Central defensive-sleeve handling, controlled replication, policy tuning, and a decision-ready validation report
- **Phase 3F — Scenario-Based Strategy Selection**: Evidence-gated profile recommendations, scenario playbooks, and a simplified manager workflow
- **Risk Analytics**: VaR, ES/CVaR, VaR exceptions, stress testing, drawdown duration, concentration, and active-risk diagnostics
- **Interactive Dashboard**: Streamlit-based visualization and analytics interface

Future work includes broader Markov-switching models, NLP and macro-sentiment integration, production model governance, and liquidity-aware market-impact modeling.

## Project Phases

- **Phase 1:** Core portfolio construction, risk analytics, backtesting, and dashboard
- **Phase 2:** FRM risk layer: active-risk metrics, VaR/ES, stress testing, and liquidity diagnostics
- **Phase 3A:** Robustness Validation / CPCV-Style Validation
- **Phase 3B.1:** Explainable Rule-Based Market Regime Detection
- **Phase 3B.2:** HMM-Based Probabilistic Regime Detection
- **Phase 3C:** Regime-Aware Adaptive Allocation Controller — implemented
- **Phase 3D:** Adaptive Strategy Experimentation & Robustness Evaluation — implemented
- **Phase 3E.1:** Defensive Return Consistency + Robustness Replication Harness — implemented
- **Phase 3F:** Scenario-Based Strategy Selection Engine + Simplified Manager Frontend — implemented

## 🏗️ Architecture

```
adaptive-portfolio-risk-analytics/
├── src/
│   ├── data_pipeline/          # Data ingestion and preprocessing
│   ├── covariance/             # Covariance estimation methods
│   ├── clustering/             # Hierarchical clustering algorithms
│   ├── regime/                 # Phase 3B rule-based and optional HMM analytics
│   ├── adaptive/               # Phase 3C adaptive controller and backtest
│   ├── regime_detection/       # Legacy/future Markov-switching extension points
│   ├── nlp/                    # NLP and sentiment analysis
│   ├── optimization/           # Portfolio allocation methods
│   ├── backtesting/            # Backtesting and validation frameworks
│   ├── experiments/            # Sensitivity and Phase 3D adaptive evaluation
│   ├── validation/             # Phase 3A CPCV-style robustness validation
│   ├── selection/              # Phase 3F profiles, gates, scoring, selector, playbook
│   ├── analytics/              # Risk metrics and performance analysis
│   ├── dashboard/              # Streamlit dashboard
│   └── utils/                  # Common utilities and helpers
├── data/                       # Raw, processed, and interim data
├── notebooks/                  # Exploratory analysis notebooks
├── tests/                      # Unit and integration tests
├── config/                     # Configuration management
├── outputs/                    # Weights, reports, metrics, figures
└── docs/                       # Architecture, methodology, research notes
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adaptive-portfolio-risk-analytics.git
cd adaptive-portfolio-risk-analytics
```

2. Create and activate Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

1. Create a `.env` file in the project root:
```
DATA_PATH=./data
RESULTS_PATH=./outputs
LOG_LEVEL=INFO
```

2. Configure allocation parameters in `config/portfolio_config.yaml`

### Running the Dashboard

```bash
streamlit run src/dashboard/app.py
```

The dashboard opens in **Manager View**. This surface exposes only Portfolio
Universe, Investment Amount, Date Range, Investor Objective, Cost Assumption,
and **Run Recommendation**. The default objective is **Balanced**, the default
cost assumption is **Moderate**, and the default risk-control candidate is
**Regime-Adaptive HMM Walk-Forward — Conservative**. Manager output contains
the selected fixed core, optional overlay/reference, confidence, a net tradeoff
table, explanation, warnings, and assumptions.

Use **Research View** for covariance, rebalancing, volatility-targeting, regime,
adaptive-policy, sensitivity, CPCV controls, selection gates, candidate scores,
profile mapping, ranking, and the scenario playbook. Use **Developer / Debug
View** for collapsed raw diagnostics, full weight and decision logs,
reconciliation, raw recommendation payloads, artifact availability, and scoring
traces.

One global **Research Objective** drives dashboard takeaways, regime selection,
sensitivity ranking, adaptive comparison, and CPCV ranking. It defaults to
**Net Calmar**. Headline return and performance metrics are net of configured
transaction costs; gross values appear only in the cost-drag reconciliation.
Large diagnostic tables are available as CSV downloads.

### Running Tests

```bash
pytest tests/ -v --cov=src
```

## 📦 Core Components

### Data Pipeline
- Market data ingestion primarily through Yahoo Finance, with Alpha Vantage kept as an extension point
- Preprocessing and feature engineering
- Data validation and quality checks

### Covariance Estimation
- **Ledoit-Wolf Shrinkage**: Regularized covariance with optimal shrinkage intensity
- **EWMA Covariance**: Time-decayed covariance estimation
- **EWMA + Ledoit-Wolf**: Time-decayed shrinkage covariance estimation

### Portfolio Optimization
- **Equal Weight**: Naive equal-weighted allocation
- **Mean-Variance**: Markowitz efficient frontier optimization
- **HRP**: Hierarchical Risk Parity with optimal tree structure
- **HERC**: Hierarchical Equal Risk Contribution
- **Regime-Adaptive**: Phase 3C controller that changes portfolio behavior using lagged market regimes

### Phase 3B — Market Regime Detection & Regime Analytics

Phase 3B.1 rule-based implementation:

- Regime feature engineering
- Rule-based regime classification
- Lagged decision regimes to avoid look-ahead
- Regime timeline
- Strategy performance by regime
- Regime transition matrix
- Regime duration analytics

#### Phase 3B.2 — HMM-Based Probabilistic Regime Detection

Implemented as an experimental method:

- Gaussian HMM regime detection
- Full-sample HMM for historical visualization
- Walk-forward HMM for time-series-safe regime inference
- Mapping hidden states to Calm/Normal/Stress/Crisis
- Two-state Risk-On/Risk-Off mapping
- HMM transition matrix and duration analytics
- Rule-based versus HMM regime comparison
- Strategy performance by HMM regime

HMM regime detection uses `hmmlearn`:

```bash
pip install hmmlearn
```

Important caveats:

- Full-sample HMM uses the complete feature history and must not support trading-safe performance claims.
- Walk-forward HMM refits on expanding historical windows and lags decision regimes to reduce look-ahead bias.
- HMM regimes are probabilistic and experimental.
- Hidden state numbers have no inherent meaning; readable labels are inferred from state-level volatility, drawdown, return, momentum, and correlation characteristics.
- The rule-based method remains the explainable default.

If `hmmlearn` is unavailable, the dashboard disables HMM controls gracefully.

Regime features:

- Realized volatility percentile
- Drawdown
- Trend
- Momentum
- Average correlation

Regime states:

- Calm
- Normal
- Stress
- Crisis

Not yet implemented:

- Broader Markov-switching models and production HMM governance
- NLP/macro sentiment integration

### Phase 3C — Regime-Aware Adaptive Allocation Controller

Implemented:

- Regime-to-policy mapping
- Policy presets: Conservative, Balanced default, and Aggressive
- Dynamic allocator selection by regime
- Dynamic covariance estimator selection by regime
- Regime-dependent volatility targets
- Risky exposure caps and defensive sleeve floors
- Adaptive strategy diagnostics
- Dashboard comparison against baseline strategies

The default balanced policy uses HERC in Calm/Normal markets, HRP in
Stress/Crisis markets, and an Equal Weight fallback when the regime is Unknown.
Two-state HMM labels map safely from Risk-On to a Calm-style policy and from
Risk-Off to a Stress-style policy.

Important caveats:

- Adaptive strategy decisions use lagged decision regimes to reduce look-ahead bias.
- Full-sample HMM regimes are rejected for trading-safe adaptive backtests.
- Phase 3C is a research controller, not a production execution system.
- NLP and macro-sentiment signals remain future work.

Phase 3C adaptive returns use the standard analytics interface and can be
evaluated with the existing Phase 3A robustness tools through Phase 3D.

### Phase 3D — Adaptive Strategy Experimentation & Robustness Evaluation

Phase 3D evaluates whether lagged regime-aware allocation improves:

- CAGR, volatility, Sharpe, Sortino, Calmar, and maximum drawdown
- Historical VaR and ES/CVaR
- Stress-period return and drawdown duration
- Turnover, transaction costs, and number of rebalances
- CPCV median selected objective, worst-fold selected objective, and stability score

Implemented:

- Adaptive strategy included as a first-class experiment type
- Rule-based lagged and HMM walk-forward adaptive configurations
- Conservative, Balanced, and Aggressive adaptive policy sensitivity
- Adaptive versus fixed-strategy comparison
- Exposure, regime, policy, allocator, and covariance diagnostics
- Regime and policy attribution
- Adaptive stress-period comparison
- Optional, bounded adaptive CPCV robustness evaluation

Important caveats:

- Adaptive results are research backtests, not live trading advice.
- Regime signals are lagged to reduce look-ahead bias.
- Full-sample HMM is excluded from adaptive trading-safe experiments.
- HMM walk-forward experiments can be computationally expensive.
- CPCV uses the currently selected dashboard objective; Calmar is only the default fallback.

### Phase 3E.1 — Defensive Return Consistency + Robustness Replication Harness

Implemented:

- Central defensive-return utility with synthetic, zero-cash, ticker, and
  provided-series sources
- Consistent defensive sleeve handling across dashboard, experiments, fixed
  overlays, and CPCV
- Defensive source, rate, ticker, fallback, and notes metadata in adaptive outputs
- Controlled replication across universes, date windows, transaction costs,
  defensive sleeves, policy presets, and trading-safe regime sources
- Conservative policy-tuning mini-grid for faster re-risking
- Decision-ready report under `outputs/reports/phase_3e_replication/`

Important caveats:

- Results remain historical backtests, not live trading claims.
- Yahoo Finance data can change.
- HMM walk-forward fitting can be computationally expensive.
- The replication grid is bounded by runtime limits and records skipped or
  failed combinations instead of stopping the full study.
- Full-sample HMM remains historical visualization only.

### Phase 3F — Scenario-Based Strategy Selection Engine

Implemented:

- Investor profiles: Growth, Balanced, Capital Preservation, Stress Protection,
  and Robustness First
- Scenario classification for calm/growth, normal, stress, crisis, high
  volatility, high cost, unstable HMM, low CPCV confidence, and insufficient data
- PASS/WARN/FAIL/NOT_AVAILABLE gates for net return basis, full-sample HMM
  exclusion, CPCV coverage and worst fold, turnover/cost, stress evidence,
  defensive metadata, sufficient history, and HMM walk-forward validity
- Role separation between Main Growth Strategy, Risk-Control Overlay,
  Robustness Reference, Experimental Candidate, Rejected, and benchmark
- Phase 3E artifact loading with post-P0 fallback and explicit warnings
- Scenario playbook and profile-aware recommendation explanations
- Manager, Research, and Developer dashboard surfaces backed by the same
  selection result
- Decision artifacts under
  `outputs/reports/phase_3f_strategy_selection/`

Selection guardrails:

- Return-derived evidence must be net of configured transaction costs.
- Full-sample HMM may be used for historical diagnostics only.
- Adaptive strategies are treated as overlays or robustness references unless
  repeated net evidence supports a different role.
- Low CPCV successful-fold coverage reduces confidence even when the
  successful-fold objective is strong.
- The selector does not infer a live-trading guarantee from historical
  backtests.

### NLP Sentiment Analysis

NLP and macro-sentiment integration remain future work.

### Backtesting
- Rolling-window in-sample validation
- Threshold and calendar rebalancing
- Transaction cost modeling
- Performance metrics and stress testing

### Phase 3A — Robustness Validation / CPCV-Style Validation

- CPCV-style time-series splits
- Purge and embargo logic
- Fold-level performance comparison
- Stability scoring
- Robustness ranking

The existing sensitivity analysis identifies strong configurations in one backtest setting. Robustness validation checks whether those configurations remain stable across multiple time partitions.

## 📊 Key Metrics

- **Risk Metrics**: VaR, ES/CVaR, VaR exceptions, Volatility, Maximum Drawdown, Calmar Ratio
- **Performance Metrics**: Sharpe Ratio, Sortino Ratio, Information Ratio, CAGR
- **Portfolio Metrics**: HHI, Effective N, Turnover, ADTV, Participation Rate

## FRM Alignment

The platform maps portfolio analytics to FRM concepts: market risk, systematic risk, unsystematic risk, tail risk, liquidity trading risk, model risk, concentration risk, and benchmark active risk.

Core formulas:

```text
Historical VaR at confidence c = negative of the (1-c) quantile of returns

Historical ES/CVaR = average loss conditional on loss exceeding VaR

Expected exceptions = number of observations * (1 - confidence level)

Exception ratio = actual exceptions / expected exceptions

ADTV = average daily volume * latest price

Participation Rate = trade value / ADTV

Stress Return = sum(weight_i * scenario_return_i)
```

## 🔧 Technologies

- **Python 3.10+**: Core language
- **NumPy & SciPy**: Numerical computing
- **Pandas**: Data manipulation
- **scikit-learn**: Machine learning
- **Statsmodels**: Statistical modeling
- **scikit-portfolio & Riskfolio-lib**: Portfolio optimization
- **Transformers**: NLP models
- **Streamlit**: Dashboard framework
- **Plotly**: Interactive visualizations

## 📚 Documentation

- [Architecture Overview](docs/architecture/README.md)
- [Methodology](docs/methodology/README.md)
- [Research Notes](docs/research_notes/README.md)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please follow PEP8 standards and include unit tests.

## 📧 Contact

For questions or suggestions, contact: team@example.com

---

**Built as a quantitative finance research platform for portfolio analytics, risk diagnostics, regime analysis, and adaptive allocation experiments.**
