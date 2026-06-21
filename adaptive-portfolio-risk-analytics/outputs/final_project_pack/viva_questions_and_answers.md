# Viva Questions and Answers

**v1.0 — Regime-Aware Portfolio Risk Analytics Platform**

## 1. What is the main objective of the project?

To build a research platform that combines portfolio construction, FRM risk diagnostics, regime-aware overlays, robustness validation, and evidence-gated strategy recommendations.

## 2. What is the final strategic conclusion?

HERC is the strategic growth core, HMM Conservative is the drawdown-control overlay, Rule-based Conservative is the robustness reference and HMM fallback, and Equal Weight is the benchmark.

## 3. Why did HERC become the growth core?

In the latest matched primary scenario, HERC produced the highest net CAGR and terminal value among the role candidates. Its growth advantage remained material even though adaptive overlays had lower drawdown.

## 4. Why use HRP or HERC instead of Markowitz optimization?

Markowitz portfolios are sensitive to expected-return and covariance estimation errors. HRP and HERC use hierarchical correlation structure, avoid direct covariance inversion, and do not require unstable expected-return forecasts.

## 5. What is the difference between HRP and HERC?

HRP recursively allocates between hierarchical clusters using cluster variance. HERC explicitly targets risk contribution across clusters, allowing cluster-level equal-risk allocation.

## 6. Why is adaptive not the main strategy?

Adaptive strategies improved drawdown and Calmar but generally produced lower CAGR and final value than HERC. Their evidence supports a risk-control role, not a universal growth replacement.

## 7. Why use an HMM?

An HMM can infer latent market states and capture persistent probabilistic regime behavior that fixed thresholds may miss. It provides a complementary, data-driven regime model.

## 8. Why keep a rule-based fallback?

The rule-based method is transparent, cheaper to run, and less dependent on probabilistic fitting. It also achieved the strongest adaptive rank in the current CPCV artifact, making it a useful robustness reference.

## 9. How did you avoid look-ahead bias?

Rule-based observed labels are shifted before decisions. HMM adaptive analysis trains on expanding prior history, refits walk-forward, and applies a decision lag. Weights chosen at date `t` apply to returns at `t+1`.

## 10. Why can full-sample HMM not be used for trading-safe claims?

It estimates states using the complete history, including information that would not have been available at earlier dates. It is therefore restricted to historical visualization.

## 11. What is walk-forward HMM inference?

The model is repeatedly fit using only data available up to each refit date. It then produces out-of-sample state estimates, which are lagged before portfolio decisions.

## 12. What regime features are used?

Rolling volatility and volatility percentile, drawdown, trend, momentum, average correlation, and benchmark return and volatility features.

## 13. What are the rule-based regime states?

Calm, Normal, Stress, Crisis, and Unknown. Unknown primarily covers insufficient warm-up history.

## 14. What changes when the adaptive policy changes regime?

The allocator, covariance estimator, target volatility, rebalance rule, threshold, risky-exposure cap, and defensive floor can change.

## 15. What is the defensive sleeve?

It is the low-risk allocation used when risky exposure is reduced. Its return can come from a synthetic rate, zero cash, a ticker, or a provided series, with source metadata recorded.

## 16. Why centralize defensive-return handling?

Different defensive assumptions can materially change adaptive results. Centralization makes dashboard, experiment, replication, and CPCV paths consistent and auditable.

## 17. What is transaction-cost modeling doing?

It converts rebalance turnover into base cost plus slippage drag. Net returns and final value include this drag, while gross series are retained for reconciliation.

## 18. Why are net labels important?

An active strategy can look attractive before costs but not after costs. Explicit net labels prevent gross results from being mistaken for implementable historical outcomes.

## 19. What is turnover?

Turnover is half the sum of absolute portfolio-weight changes at a rebalance. Higher turnover usually creates higher trading cost and implementation burden.

## 20. What is the Sharpe ratio?

Annualized excess return divided by annualized volatility. It measures return per unit of total risk.

## 21. What is the Sortino ratio?

Annualized excess return divided by downside deviation. It penalizes harmful volatility rather than all volatility.

## 22. What is the Calmar ratio?

CAGR divided by the absolute value of maximum drawdown. It measures annualized growth relative to the worst historical peak-to-trough loss.

## 23. What is maximum drawdown?

The largest peak-to-trough decline in compounded portfolio value. It is reported as a negative number, so a less-negative value is better.

## 24. What is VaR?

Value at Risk is a loss threshold expected not to be exceeded at a stated confidence level under the chosen historical method. It does not describe the average severity beyond that threshold.

## 25. What is ES or CVaR?

Expected Shortfall averages losses beyond the VaR threshold. It is more informative about tail severity than VaR alone.

## 26. Why are there two VaR/ES sign conventions?

Experiment outputs store signed tail returns, while dashboard historical VaR/ES displays positive losses. The implementation documents the source API so signs are not compared incorrectly.

## 27. What is stress testing?

Stress testing evaluates portfolio behavior during severe historical windows or under specified hypothetical shocks. It complements full-period metrics by showing scenario-specific vulnerability.

## 28. Why use CPCV-style validation?

It tests whether a configuration remains stable across multiple time-block combinations instead of relying on one train/test split. Purge and embargo controls reduce leakage around test periods.

## 29. Is this full CPCV?

No. It is a pragmatic CPCV-style time-block framework and does not construct complete independent backtest paths. That limitation is documented.

## 30. What does the CPCV robustness score contain?

It combines the selected objective's median percentile, adverse worst-fold percentile, and stability score. The selected dashboard objective is preserved through the ranking.

## 31. What is the main CPCV limitation?

Adaptive warm-up requirements cause many early folds to fail. The current ranking reports coverage but does not directly penalize missing folds.

## 32. Why is recommendation confidence only Moderate?

Rule-based Conservative succeeded on 6 of 15 CPCV folds and HMM Conservative on 3 of 15. Favorable successful folds are not enough to justify High confidence with that coverage.

## 33. What does evidence-gated selection mean?

Candidates must pass or survive explicit checks for net metrics, data sufficiency, HMM safety, CPCV evidence, turnover, cost, stress protection, defensive metadata, and replication classification.

## 34. What investor profiles are supported?

Growth, Balanced, Capital Preservation, Stress Protection, and Robustness First. Each profile changes the weights on growth, drawdown, robustness, and cost.

## 35. Does the profile score override strategy roles?

No. Role guardrails keep HERC as the core when valid and treat adaptive candidates as overlays or references unless repeated net growth evidence supports promotion.

## 36. What is the strongest result for HMM Conservative?

In the latest matched primary scenario it reduced maximum drawdown to -7.78% versus -18.91% for HERC and raised Calmar to 1.521 versus 0.794.

## 37. What is the main weakness of HMM Conservative?

It produced lower CAGR and terminal value than HERC, requires more modeling assumptions, and has limited successful CPCV fold coverage.

## 38. What is the main weakness of Rule-based Conservative?

Its matched turnover was materially higher than HMM Conservative and HERC, making it more cost-sensitive despite its simplicity and CPCV rank.

## 39. What did replication add beyond one backtest?

It compared strategies across matched universes, date windows, cost levels, defensive sleeves, policies, and trading-safe regime sources. This showed that downside wins persisted while final-value wins did not.

## 40. What is the biggest project limitation?

The strongest limitation is incomplete adaptive robustness coverage. Broader data and explicit fold-coverage penalties are needed before stronger generalization claims.

## 41. Why was NLP not implemented?

Adding sentiment would create a new data, timing, leakage, and validation problem. The v1.0 priority was to finish and validate the portfolio, risk, regime, and selection pipeline first.

## 42. What would you improve next?

Add CPCV fold-coverage eligibility or penalties, broaden market replication, strengthen data and model versioning, and improve liquidity-aware transaction-cost modeling.

## 43. Is the project live-trading ready?

No. It is a historical research and decision-support platform. It does not include production execution, complete market-impact modeling, operational controls, or live model governance.

## 44. How would you explain the project in one sentence?

It is an evidence-gated portfolio research platform that separates a hierarchical growth core from regime-aware drawdown-control overlays and validates the recommendation through stress, replication, and CPCV-style tests.

