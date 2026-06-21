# Final Results Summary

**v1.0 — Regime-Aware Portfolio Risk Analytics Platform**

## Evidence basis

Point estimates below use the latest matched Phase 3E primary scenario:

- universe: Core Diversified, 12 assets;
- evaluation: January 1, 2020 through June 19, 2026;
- initial capital: 1,000,000;
- costs: 10 bps base plus 5 bps slippage;
- adaptive defensive sleeve: synthetic 4% annual return;
- return basis: net of configured transaction costs.

Robustness numbers use the current post-P0 CPCV artifact because it is the latest fold-level validation output used by the Phase 3F selector.

## Matched primary results

| Strategy | Role | Net CAGR | Volatility | Sharpe | Sortino | Calmar | Max drawdown | Net final value | Turnover |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HERC | Strategic growth core | 15.01% | 12.18% | 1.045 | 1.152 | 0.794 | -18.91% | 2,434,518 | 7.31 |
| HMM Walk-Forward Conservative | Risk-control overlay | 11.84% | 7.35% | 1.286 | 1.570 | 1.521 | -7.78% | 2,037,180 | 8.12 |
| Rule-based Conservative | Robustness reference / fallback | 10.80% | 7.69% | 1.112 | 1.317 | 1.114 | -9.69% | 1,920,070 | 20.73 |
| Equal Weight | Benchmark | 12.56% | 15.59% | 0.709 | 0.772 | 0.381 | -32.94% | 2,122,050 | 1.69 |

## HERC

- **Role:** strategic growth core.
- **Strength:** highest net CAGR and terminal value among the role candidates.
- **Weakness:** larger drawdown than both conservative adaptive overlays.
- **Interpretation:** HERC remains the core because adaptive strategies did not match its growth outcome, even when they improved downside efficiency.

## HMM Conservative

- **Role:** risk-control overlay.
- **Strength:** lowest matched maximum drawdown, highest matched Calmar, and strong Phase 3E drawdown replication.
- **Weakness:** lower CAGR and final value than HERC; HMM fitting is more complex and CPCV coverage is limited.
- **Interpretation:** suitable when drawdown control is worth a growth sacrifice. It is not positioned as the growth replacement.

## Rule-based Conservative

- **Role:** robustness reference and HMM fallback.
- **Strength:** simpler, explainable, and first in the current adaptive CPCV robustness ranking.
- **Weakness:** higher turnover in the matched scenario and lower growth than HERC.
- **Interpretation:** preferred when model transparency, robustness-first selection, or HMM instability matters.

## Equal Weight

- **Role:** baseline benchmark.
- **Strength:** simple and lowest-turnover reference.
- **Weakness:** weakest matched drawdown and Calmar among the four role candidates.
- **Interpretation:** useful for judging whether added portfolio and regime complexity produces measurable value.

## CPCV-style robustness

| Rank | Strategy | Median Calmar | Worst successful-fold Calmar | Stability | Successful / attempted folds |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Rule-based Conservative | 2.010 | 0.956 | 0.721 | 6 / 15 |
| 2 | HRP | 1.510 | 0.249 | 0.651 | 10 / 15 |
| 3 | HERC | 1.476 | 0.530 | 0.663 | 10 / 15 |
| 4 | HMM Conservative | 1.176 | 0.955 | 0.817 | 3 / 15 |
| 6 | Equal Weight | 0.950 | 0.088 | 0.615 | 10 / 15 |

Successful adaptive folds were favorable, but coverage was only 40% for Rule-based Conservative and 20% for HMM Conservative. The current ranking does not directly penalize missing folds, so confidence remains Moderate.

## Replication findings

- Phase 3E completed 70 successful runs with no failed or skipped runs in the retained bounded grid.
- HMM Conservative recorded a 100% drawdown win rate, 90% Calmar win rate, and 0% final-value win rate against the paired best fixed strategy.
- Rule-based Conservative completed 10 matched replication runs with worst-case Calmar of approximately 0.60.
- Adaptive Calmar declined as transaction costs increased, confirming implementation sensitivity.
- Faster HMM re-risking shortened recovery by nine trading days in the tuning study but reduced Calmar and final value.

## Final conclusion

This project does not identify a universal winner. HERC is the strategic growth core; HMM Conservative is the drawdown-control overlay; Rule-based Conservative is the robustness reference and HMM fallback; Equal Weight is the benchmark. Adaptive improves downside behavior but does not replace HERC's growth role. The recommendation is Moderate confidence because adaptive CPCV coverage remains limited.

## Sources

- `outputs/reports/phase_3e_replication/replication_results.csv`
- `outputs/reports/phase_3e_replication/replication_summary.csv`
- `outputs/reports/phase_3e_replication/summary.md`
- `outputs/reports/post_p0_adaptive_validation/cpcv_summary.csv`
- `outputs/reports/phase_3f_strategy_selection/summary.md`

