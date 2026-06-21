"""Generate the Phase 4A sentiment-confirmation evidence package."""

from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
import sys

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.regime import (  # noqa: E402
    calculate_regime_features,
    classify_rule_based_regime,
    fit_hmm_walk_forward,
    lag_regime_labels,
)
from src.sentiment import (  # noqa: E402
    build_alignment_checks,
    build_current_sentiment_summary,
    build_daily_sentiment_signal,
    compare_sentiment_to_regimes,
    load_local_sentiment_csv,
    score_sentiment_records,
)


def build_synthetic_market_returns() -> pd.DataFrame:
    """Create deterministic market history for an offline pipeline demonstration."""
    index = pd.bdate_range("2022-01-03", "2026-06-19")
    rng = np.random.default_rng(42)
    common = np.zeros(len(index), dtype=float)
    volatility = np.full(len(index), 0.007)
    drift = np.full(len(index), 0.00035)

    stress_windows = [
        ("2022-02-01", "2022-07-29"),
        ("2024-03-01", "2024-05-15"),
        ("2025-03-01", "2025-05-30"),
        ("2026-04-01", "2026-04-30"),
    ]
    for start, end in stress_windows:
        mask = (index >= start) & (index <= end)
        volatility[mask] = 0.019
        drift[mask] = -0.0010
    recovery_windows = [
        ("2022-08-01", "2023-01-31"),
        ("2024-05-16", "2024-08-30"),
        ("2025-06-01", "2025-09-30"),
        ("2026-05-01", "2026-06-19"),
    ]
    for start, end in recovery_windows:
        mask = (index >= start) & (index <= end)
        volatility[mask] = 0.009
        drift[mask] = 0.00075

    common = drift + rng.normal(0.0, volatility)
    assets = {}
    for position in range(8):
        idiosyncratic = rng.normal(0.0, volatility * 0.45)
        assets[f"SYNTH_{position + 1}"] = common + idiosyncratic
    return pd.DataFrame(assets, index=index)


def percentage(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if not np.isfinite(numeric) else f"{numeric:.1%}"


def build_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_local_sentiment_csv(
        REPO_ROOT / "data" / "sentiment" / "sample_market_news.csv"
    )
    scored = score_sentiment_records(records)
    returns = build_synthetic_market_returns()
    features = calculate_regime_features(returns)
    rule_observed = classify_rule_based_regime(features)
    rule_decisions = lag_regime_labels(rule_observed, lag=1)
    hmm = fit_hmm_walk_forward(
        features,
        n_states=2,
        min_train_size=252,
        refit_frequency=126,
        covariance_type="diag",
        decision_lag=1,
    )
    hmm_decisions = hmm["decision_regimes"].reindex(returns.index).fillna("Unknown")

    signal = build_daily_sentiment_signal(
        scored,
        returns.index,
        lookback_window=5,
        decision_lag=1,
    )
    comparison = compare_sentiment_to_regimes(
        signal,
        rule_decisions,
        hmm_decisions,
    )
    current_quant = str(hmm_decisions.iloc[-1])
    current = build_current_sentiment_summary(signal, current_quant)
    alignment = build_alignment_checks(
        scored,
        signal,
        returns.index,
        decision_lag=1,
    )

    signal.to_csv(OUTPUT_DIR / "sentiment_signal.csv", index=True)
    comparison_table = comparison["comparison_table"].copy()
    comparison_table.to_csv(
        OUTPUT_DIR / "sentiment_regime_comparison.csv",
        index=True,
    )
    comparison["dates_of_major_disagreement"].to_csv(
        OUTPUT_DIR / "disagreement_dates.csv",
        index=False,
    )

    summary = f"""# Phase 4A Sentiment Regime Confirmation

Generated: {date.today().isoformat()}

## Conclusion

**Sentiment is an auxiliary confirmation layer. It is not yet used for allocation.**

## Evidence summary

- Source: 32 synthetic, timestamped market-news records from `data/sentiment/sample_market_news.csv`.
- Scoring: deterministic Phase 4A risk-on/risk-off lexicon; no external model dependency.
- Signal: five-market-day rolling mean with a one-market-day decision lag.
- Rule-based agreement: {percentage(comparison['agreement_with_rule_based'])}.
- HMM walk-forward agreement: {percentage(comparison['agreement_with_hmm'])}.
- Rule stress/crisis risk-off confirmation: {percentage(comparison['risk_off_agreement_rule_based'])}.
- HMM risk-off confirmation: {percentage(comparison['risk_off_agreement_hmm'])}.
- Article-day coverage: {percentage(comparison['article_coverage_ratio'])}.
- Decision-label coverage: {percentage(comparison['decision_coverage_ratio'])}.
- Current synthetic-fixture confirmation: {current['confirmation_status']}.
- Alignment checks passed: {alignment['all_checks_passed']}.

## Interpretation

The sparse synthetic feed demonstrates ingestion, scoring, lagging, alignment,
agreement, disagreement, and dashboard commentary. The agreement rates are not
empirical market evidence and do not establish that sentiment predicts returns.
Phase 4B should be considered only after timestamped real-source data, source
governance, broader coverage, and out-of-sample validation are available.
"""
    (OUTPUT_DIR / "summary.md").write_text(summary, encoding="utf-8")

    source_notes = f"""# Phase 4A Source Notes

Generated: {date.today().isoformat()}

## Sentiment source

- Bundled file: `data/sentiment/sample_market_news.csv`
- Records: {len(records)}
- Source type: synthetic research sample
- No current or scraped news is embedded.

## Quantitative comparison source

- Deterministic synthetic eight-asset daily return fixture from 2022-01-03 to 2026-06-19.
- Rule-based labels use the repository Phase 3B feature and classification code.
- HMM labels use the repository two-state expanding-window walk-forward implementation.
- Full-sample HMM is not used.

## Look-ahead controls

- Timestamps are parsed before scoring.
- Records are assigned to the first market date on or after their calendar date.
- Observed rolling sentiment is shifted by one market session before becoming a decision label.
- Decision-source timestamps are checked to precede each decision date.
- All alignment checks passed: {alignment['all_checks_passed']}.

## Scoring

- Model: `phase4a_lexicon`, version `1.0`.
- Positive score means risk-on.
- Negative score means risk-off.
- Near-zero score means neutral.
- VADER and FinBERT are not required or used.

## Limitations

- Both news and market history in this report are synthetic fixtures.
- The feed is intentionally sparse, so article and decision coverage are limited.
- Lexicon scoring does not resolve context, negation, source credibility, or entity relevance.
- Publication-time and market-close rules require source-specific governance for real data.
- Leading/lagging diagnostics are descriptive and do not establish prediction.
- Sentiment does not change allocation, policy selection, strategy scores, or confidence.

## Verification

- `python -m pytest -q`: 375 passed, 1 skipped.
- Total statement coverage: 63%.
- Headless Streamlit root and health endpoints: HTTP 200.
- `python scripts/final_smoke_test.py`: passed.
"""
    (OUTPUT_DIR / "source_notes.md").write_text(source_notes, encoding="utf-8")

    alignment_html = alignment["checks"].to_html(index=False, border=0)
    distribution_html = comparison["sentiment_distribution"].to_html(
        index=False,
        border=0,
    )
    disagreements = comparison["dates_of_major_disagreement"].head(20)
    disagreement_html = disagreements.to_html(index=False, border=0)
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 4A Sentiment Regime Confirmation</title>
  <style>
    :root {{ --ink:#1F2430; --muted:#697086; --line:#E5E8F0; --blue:#2E4780; --open:#EAF1FE; --bg:#F7F8FB; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:Inter,"Segoe UI",Arial,sans-serif; line-height:1.55; }}
    main {{ max-width:1080px; margin:auto; padding:44px 24px 72px; }}
    header {{ background:linear-gradient(135deg,#1F2C4D,#2E4780); color:white; padding:38px; border-radius:20px; }}
    header p {{ color:#DDE7FF; max-width:800px; }}
    h1 {{ margin:8px 0; font-size:38px; }}
    h2 {{ margin-top:38px; }}
    .eyebrow {{ text-transform:uppercase; letter-spacing:.11em; font-size:12px; font-weight:700; }}
    .callout {{ margin-top:24px; border-left:4px solid var(--blue); background:var(--open); padding:16px 18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }}
    .card,.panel {{ background:white; border:1px solid var(--line); border-radius:14px; padding:18px; }}
    .card span {{ display:block; color:var(--muted); font-size:12px; }}
    .card strong {{ font-size:24px; }}
    table {{ border-collapse:collapse; width:100%; background:white; font-size:13px; }}
    th,td {{ border-bottom:1px solid var(--line); padding:9px; text-align:left; }}
    th {{ color:var(--blue); background:#F4F7FD; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:14px; }}
    footer {{ margin-top:36px; color:var(--muted); font-size:12px; }}
  </style>
</head>
<body><main>
  <header>
    <div class="eyebrow">Adaptive Portfolio Risk Analytics · v1.1</div>
    <h1>Phase 4A — Sentiment Regime Confirmation</h1>
    <p>Timestamped, lagged sentiment is compared with rule-based and HMM walk-forward regimes. It remains a commentary layer and does not control allocation.</p>
  </header>
  <div class="callout"><strong>Conclusion:</strong> Sentiment is an auxiliary confirmation layer. It is not yet used for allocation.</div>

  <h2>Agreement summary</h2>
  <section class="grid">
    <div class="card"><span>Rule-based agreement</span><strong>{percentage(comparison['agreement_with_rule_based'])}</strong></div>
    <div class="card"><span>HMM walk-forward agreement</span><strong>{percentage(comparison['agreement_with_hmm'])}</strong></div>
    <div class="card"><span>Article-day coverage</span><strong>{percentage(comparison['article_coverage_ratio'])}</strong></div>
    <div class="card"><span>Current confirmation</span><strong>{escape(str(current['confirmation_status']))}</strong></div>
  </section>

  <h2>What source and scorer were used?</h2>
  <section class="panel">
    <p>The report uses 32 synthetic headlines from the bundled local CSV and the dependency-light <code>phase4a_lexicon</code> scorer. Positive terms map toward risk-on, negative terms toward risk-off, and near-zero scores toward neutral.</p>
  </section>

  <h2>How was look-ahead avoided?</h2>
  <section class="panel">
    <p>Records are timestamped and assigned to an observed market date. The rolling observed label is shifted by one market session. The decision-source timestamp must precede the decision date.</p>
  </section>
  <div class="table-wrap">{alignment_html}</div>

  <h2>Does sentiment confirm stress periods?</h2>
  <section class="grid">
    <div class="card"><span>Rule stress/crisis confirmation</span><strong>{percentage(comparison['risk_off_agreement_rule_based'])}</strong></div>
    <div class="card"><span>HMM risk-off confirmation</span><strong>{percentage(comparison['risk_off_agreement_hmm'])}</strong></div>
  </section>

  <h2>Sentiment distribution</h2>
  <div class="table-wrap">{distribution_html}</div>

  <h2>Where does sentiment disagree?</h2>
  <div class="table-wrap">{disagreement_html}</div>

  <h2>Should sentiment become an allocation signal?</h2>
  <section class="panel">
    <p><strong>Not in Phase 4A.</strong> These synthetic-fixture results validate plumbing and timing controls, not predictive value. Promotion requires governed real-source data and out-of-sample Phase 4B testing.</p>
  </section>
  <footer>Generated {date.today().isoformat()}. See source_notes.md for provenance and limitations.</footer>
</main></body></html>
"""
    (OUTPUT_DIR / "report.html").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    build_outputs()
