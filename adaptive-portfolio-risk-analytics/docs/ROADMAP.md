# Roadmap

This roadmap starts from the current implementation rather than the original scaffold plan.

## Implemented and tested

- Centralized Yahoo Finance ingestion, inspection, preprocessing, and missingness diagnostics.
- Sample, Ledoit-Wolf, EWMA, and EWMA plus Ledoit-Wolf covariance estimation.
- Equal Weight, Inverse Volatility, Mean-Variance, HRP, and HERC allocation engines.
- Rolling net/gross backtests with drift-aware rebalancing, turnover, transaction costs, and
  slippage.
- Risk contribution, VaR/ES, drawdown, Pain Ratio, stress, liquidity, and active-risk analytics.
- Fixed/adaptive sensitivity experiments and CPCV-style purge/embargo validation.
- Rule-based and HMM walk-forward regime research with lag-safe adaptive policies.
- Strategy-selection evidence gates and Manager/Research/Developer dashboard modes.
- Governed sentiment ingestion, RBI/news monitoring, optional local FinBERT fallback, and NLP
  shadow overlays.

## Hardening priorities

- Split the large Streamlit orchestration module into tested page/component modules without
  changing the current entrypoint.
- Add browser-level dashboard smoke/interaction tests and broader plotting tests.
- Add eligibility penalties for low CPCV successful-fold coverage.
- Expand replication across independent universes, regions, and longer histories.
- Version market data and experiment inputs so saved metrics can be reproduced exactly.
- Strengthen live-source tests for official RBI/GDELT/Alpha Vantage adapters with recorded,
  timestamped fixtures and explicit network-test markers.
- Execute and freeze the currently unexecuted stage notebooks where deterministic local inputs are
  available.

## Partial or scaffolded work

- `FeatureEngineer` implements rolling volatility features, while technical, macro, and sentiment
  feature methods remain incomplete.
- Alpha Vantage market-price ingestion is an extension point; sentiment-provider support is more
  developed but live-key validation is environment-dependent.
- `DynamicAllocationAllocator` remains a non-functional legacy extension point; regime-adaptive
  allocation is implemented separately under `src/adaptive`.
- The compatibility `CPCVBacktester` is not the implemented validator API.
- Optional FinBERT execution depends on locally available model files; deterministic lexicon
  fallback is the reliable baseline.

## Future research and engineering

- Gerber or other robust covariance estimators under the same factory contract.
- Liquidity-aware nonlinear market-impact and capacity modeling.
- Independent live-data governance, model registry, drift monitoring, and reproducible data
  snapshots.
- Broker/order-management integration only after research controls, review, and compliance design.
- Longer real NLP histories and explicit promotion gates before any allocation influence.
- API/container deployment if a maintained service boundary becomes a project requirement.
