# Ready-to-Use Tables, Figures, Results, and Insights

This file aggregates the material report writers should use directly. The source CSVs and PNGs have been copied into:

- `outputs/reports/team_report_handoff_pack/ready_to_use_tables/`
- `outputs/reports/team_report_handoff_pack/ready_to_use_figures/`

Use these files for the final academic report instead of hunting across phase folders.

## 1. Executive summary material

Use these result bullets in the executive summary:

- HERC remains the strategic growth core.
- HMM Conservative remains the primary model-based risk-control overlay.
- Rule Conservative remains the robustness/fallback reference.
- Pain Ratio is now a production reporting metric alongside Calmar.
- RBI + GDELT/news NLP remains monitoring and shadow-confirmation only.
- NLP Confirmation Overlay showed positive shadow impact on Pain Ratio, CAGR, Calmar, turnover, and cost drag versus HMM Conservative, but worsened Pain Index and maximum drawdown.
- Production NLP allocation is not approved.

## 2. Tables to include in the final report

| Report section | Use this table | Local handoff path | Main insight |
| --- | --- | --- | --- |
| Final recommendation | Strategy ranking table | `ready_to_use_tables/strategy_ranking_table.csv` | HERC is core, HMM Conservative is overlay, Rule Conservative is fallback, NLP overlays are shadow-only. |
| Evidence gates | Evidence matrix | `ready_to_use_tables/evidence_matrix.csv` | Production decisions are separated from monitoring and shadow evidence. |
| Full strategy metrics | Final metrics table | `ready_to_use_tables/final_metrics_table.csv` | Shows production/reference and NLP shadow strategies in one compact table. |
| Fixed vs adaptive | Fixed/adaptive metrics comparison | `ready_to_use_tables/fixed_vs_adaptive_metrics_comparison.csv` | Adaptive improves drawdown efficiency but sacrifices growth versus HERC. |
| Adaptive summary | Adaptive vs fixed summary | `ready_to_use_tables/adaptive_vs_fixed_summary.csv` | HMM Conservative improves max drawdown and Calmar but not terminal wealth. |
| Robustness validation | CPCV robustness summary | `ready_to_use_tables/cpcv_robustness_summary.csv` | Rule Conservative has strong robustness evidence but fold coverage remains limited. |
| Stress testing | Stress-period comparison | `ready_to_use_tables/stress_period_comparison.csv` | Adaptive helped in severe stress windows, especially drawdown control. |
| Replication | Replication summary | `ready_to_use_tables/replication_summary.csv` | Repeated checks support role-based interpretation rather than a single universal winner. |
| Policy tuning | Policy tuning results | `ready_to_use_tables/policy_tuning_results.csv` | Conservative adaptive policies are strongest for risk-control framing. |
| Strategy selection | Selection gate results | `ready_to_use_tables/selection_gate_results.csv` | Evidence gates explain why confidence remains moderate and role-based. |
| Investor profiles | Recommendation examples | `ready_to_use_tables/recommendation_examples.csv` | Demonstrates profile-specific recommendations from the selection engine. |
| Scenario framing | Scenario playbook | `ready_to_use_tables/scenario_playbook.csv` | Useful for explaining when each strategy role matters. |
| NLP source mix | Source mix summary | `ready_to_use_tables/source_mix_summary.csv` | Real NLP source mix supports monitoring, not allocation. |
| RBI corpus | RBI corpus summary | `ready_to_use_tables/rbi_corpus_summary.csv` | RBI policy corpus is sufficient for monitoring context in this final state. |
| News signal | News signal summary | `ready_to_use_tables/news_signal_summary.csv` | News/GDELT contributes to the composite monitoring signal. |
| Pain Ratio | Pain Ratio comparison | `ready_to_use_tables/pain_ratio_comparison.csv` | NLP Confirmation improves Pain Ratio versus HMM Conservative but worsens Pain Index/max drawdown. |
| NLP shadow | NLP shadow impact table | `ready_to_use_tables/nlp_shadow_impact_table.csv` | Positive shadow impact, not production-active. |
| NLP shadow metrics | Phase 4A.13 strategy metrics | `ready_to_use_tables/phase_4a13_strategy_metrics.csv` | Full shadow-window metrics with cost and look-ahead flags. |
| Look-ahead validation | Phase 4A.13 look-ahead diagnostics | `ready_to_use_tables/phase_4a13_lookahead_diagnostics.csv` | 56/56 look-ahead diagnostics passed. |
| Overlay audit | Phase 4A.13 overlay decisions | `ready_to_use_tables/phase_4a13_overlay_decisions.csv` | Shows each shadow overlay action and confirms shadow-only behavior. |

## 3. Figures to include in the final report

| Report section | Use this figure | Local handoff path | Suggested caption |
| --- | --- | --- | --- |
| Strategy comparison | Calmar ranking | `ready_to_use_figures/calmar_ranking.png` | Calmar ranking highlights the drawdown-efficiency advantage of adaptive risk-control strategies. |
| Stress testing | Stress drawdown comparison | `ready_to_use_figures/stress_drawdown_comparison.png` | Adaptive overlays reduce drawdown in severe stress windows but may sacrifice growth/recovery speed. |
| Replication | Replication win rates | `ready_to_use_figures/replication_win_rates.png` | Replication checks support a role-based strategy interpretation. |
| Cost sensitivity | Cost sensitivity heatmap | `ready_to_use_figures/cost_sensitivity_heatmap.png` | Strategy conclusions should be interpreted net of transaction-cost assumptions. |
| Strategy selection | Profile candidate scores | `ready_to_use_figures/profile_candidate_scores.png` | Candidate scoring varies by investor profile, but HERC remains the core strategy. |
| NLP validation | NLP coverage threshold attainment | `ready_to_use_figures/nlp_coverage_threshold_attainment.png` | NLP coverage and freshness are monitored explicitly before any interpretation. |

## 4. Ready-to-write result paragraphs

### Fixed versus adaptive strategies

Fixed HERC remains the strategic growth core because it produced stronger long-run CAGR and final wealth. HMM Conservative materially improves drawdown control and Calmar Ratio, so it is best interpreted as a risk-control overlay rather than a replacement for HERC. Rule Conservative remains important because it is explainable, robust in short windows, and suitable as a fallback when HMM evidence is unstable.

### CPCV and robustness

CPCV-style validation reduces overfitting risk by evaluating strategies across multiple purged and embargoed time partitions. The current evidence supports adaptive allocation as useful for risk control, but limited successful-fold coverage means the interpretation should remain conservative. This is why the final recommendation is role-based rather than “adaptive always wins.”

### Pain Ratio

Pain Ratio complements Calmar Ratio. Calmar evaluates return relative to the worst drawdown, while Pain Ratio evaluates return relative to average drawdown pain. This is useful because a defensive strategy should be judged not only by the single worst event, but also by the time spent in drawdown.

### NLP shadow impact

The NLP Confirmation Overlay improved Pain Ratio, CAGR, Calmar Ratio, turnover, and transaction-cost drag relative to HMM Conservative in the shadow experiment. However, it worsened Pain Index and maximum drawdown, and the validation window is short. Therefore, NLP is a positive shadow-confirmation signal, not a production allocation input.

### Final strategy recommendation

The final recommendation is: HERC as the strategic growth core, HMM Conservative as the primary model-based risk-control overlay, Rule Conservative as the robustness/fallback reference, Pain Ratio as a production reporting metric, and RBI + news NLP as monitoring/shadow confirmation only. Production NLP allocation is not approved.

## 5. Tables and figures to avoid overclaiming

Do not use the NLP tables to claim that NLP improves the production strategy. The NLP variants are shadow-only. Do not use CPCV to claim guaranteed future performance. Do not use adaptive stress-window success to claim adaptive allocation always beats HERC. The correct claim is narrower: adaptive allocation improves risk-control evidence, while HERC remains the growth core.

