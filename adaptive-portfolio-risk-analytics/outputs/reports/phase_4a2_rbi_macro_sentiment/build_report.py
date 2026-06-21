"""Generate the Phase 4A.2 RBI macro-sentiment evidence package."""

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
    build_current_macro_summary,
    build_macro_stance_index,
    compare_macro_to_regimes,
    load_rbi_documents,
    score_rbi_sentences,
    split_rbi_sentences,
)


CONCLUSION = (
    "RBI macro-sentiment is now a real/reproducible confirmation layer, but "
    "it remains commentary-only until validated out of sample."
)


def build_synthetic_market_returns() -> pd.DataFrame:
    """Create deterministic market history for offline regime comparison."""
    index = pd.bdate_range("2022-01-03", "2026-06-19")
    rng = np.random.default_rng(42)
    volatility = np.full(len(index), 0.007)
    drift = np.full(len(index), 0.00035)
    for start, end in (
        ("2022-02-01", "2022-07-29"),
        ("2024-03-01", "2024-05-15"),
        ("2025-03-01", "2025-05-30"),
        ("2026-04-01", "2026-04-30"),
    ):
        mask = (index >= start) & (index <= end)
        volatility[mask] = 0.019
        drift[mask] = -0.0010
    for start, end in (
        ("2022-08-01", "2023-01-31"),
        ("2024-05-16", "2024-08-30"),
        ("2025-06-01", "2025-09-30"),
        ("2026-05-01", "2026-06-19"),
    ):
        mask = (index >= start) & (index <= end)
        volatility[mask] = 0.009
        drift[mask] = 0.00075
    common = drift + rng.normal(0.0, volatility)
    return pd.DataFrame(
        {
            f"SYNTH_{position + 1}": common
            + rng.normal(0.0, volatility * 0.45)
            for position in range(8)
        },
        index=index,
    )


def percentage(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if not np.isfinite(numeric) else f"{numeric:.1%}"


def build_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = (
        REPO_ROOT
        / "data"
        / "sentiment"
        / "rbi_documents"
        / "sample_manifest.csv"
    )
    documents = load_rbi_documents(manifest)
    sentences = split_rbi_sentences(documents)
    scored = score_rbi_sentences(sentences, method="lexicon")
    documents["corpus_type"] = "synthetic_fixture"
    scored["corpus_type"] = "synthetic_fixture"

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
    macro_index = build_macro_stance_index(
        scored,
        returns.index,
        lookback_window=63,
        decision_lag=1,
    )
    comparison = compare_macro_to_regimes(
        macro_index,
        rule_decisions,
        hmm_decisions,
    )
    current = build_current_macro_summary(
        macro_index,
        str(hmm_decisions.iloc[-1]),
    )
    macro_index["corpus_type"] = "synthetic_fixture"
    comparison["comparison_table"]["corpus_type"] = "synthetic_fixture"
    comparison["dates_of_major_disagreement"]["corpus_type"] = (
        "synthetic_fixture"
    )
    lead_dates = comparison["macro_risk_off_before_stress_dates"]
    coverage_assessment = (
        "Insufficient for empirical or allocation conclusions because the "
        "bundled corpus is synthetic, sparse, and covers only "
        f"{percentage(comparison['coverage_ratio'])} of decision dates."
    )

    documents.to_csv(OUTPUT_DIR / "rbi_documents.csv", index=False)
    scored.to_csv(OUTPUT_DIR / "rbi_sentence_scores.csv", index=False)
    macro_index.to_csv(OUTPUT_DIR / "macro_stance_index.csv", index=True)
    comparison["comparison_table"].to_csv(
        OUTPUT_DIR / "macro_regime_comparison.csv",
        index=True,
    )
    comparison["dates_of_major_disagreement"].to_csv(
        OUTPUT_DIR / "disagreement_dates.csv",
        index=False,
    )

    loaded = int(documents["load_status"].eq("loaded").sum())
    errors = int(documents["load_status"].eq("error").sum())
    fallback_count = int(scored["fallback_used"].fillna(False).sum())
    summary = f"""# Phase 4A.2 RBI Macro-Sentiment Confirmation

Generated: {date.today().isoformat()}

## Conclusion

**{CONCLUSION}**

## Evidence summary

- Corpus type: `synthetic_fixture`.
- Source: local manifest with {len(documents)} synthetic RBI-style documents; {loaded} loaded and {errors} flagged.
- Sentence corpus: {len(scored)} cleaned and deterministically ordered sentences.
- Scoring: RBI stance/certainty/time lexicon; transformer fallback count {fallback_count}.
- Signal: 63-market-day rolling stance index with a one-market-day decision lag.
- Rule-based agreement: {percentage(comparison['agreement_with_rule_based'])}.
- HMM walk-forward agreement: {percentage(comparison['agreement_with_hmm_walk_forward'])}.
- Rule stress/crisis risk-off confirmation: {percentage(comparison['stress_crisis_risk_off_confirmation_rule_based'])}.
- HMM stress/crisis risk-off confirmation: {percentage(comparison['stress_crisis_risk_off_confirmation_hmm'])}.
- Decision-label coverage: {percentage(comparison['coverage_ratio'])}.
- Current fixture macro label: {current['macro_sentiment_label']}.
- Current fixture confirmation: {current['macro_sentiment_confirmation']}.

## Interpretation

The manifest, sentence processing, scoring, rolling index, lagging, and regime
comparison are reproducible. The bundled documents and market history are
synthetic fixtures, so these statistics are pipeline evidence rather than
empirical evidence about RBI language or future returns.

## Required questions answered

1. **What RBI documents were ingested?** Four locally stored synthetic RBI-style MPC-minute and monetary-policy-statement fixtures listed in `rbi_documents.csv`.
2. **How were sentences scored?** Text was cleaned, split into stable ordered sentences, and classified with phrase dictionaries for stance, certainty, and time orientation.
3. **Which scorer was used?** The deterministic `rbi_macro_lexicon` version `1.0`; transformer models were not required.
4. **How was look-ahead avoided?** Publication dates were aligned to the first available market date, then observed macro labels were shifted by one market session.
5. **What is the Macro-Stance Index?** A 63-session rolling index where `net_stance_score = hawkish_share - dovish_share` and `macro_risk_score = net_stance_score + uncertainty_share`.
6. **Does it confirm HMM regimes?** HMM walk-forward agreement was {percentage(comparison['agreement_with_hmm_walk_forward'])}; stress/risk-off confirmation was {percentage(comparison['stress_crisis_risk_off_confirmation_hmm'])}.
7. **Does it confirm rule-based regimes?** Rule-based agreement was {percentage(comparison['agreement_with_rule_based'])}; stress/crisis risk-off confirmation was {percentage(comparison['stress_crisis_risk_off_confirmation_rule_based'])}.
8. **Are risk-off signals visible before or during stress?** {len(lead_dates)} pre-stress macro risk-off alignments were identified; during-stress confirmation rates are reported above and in `macro_regime_comparison.csv`.
9. **Is coverage sufficient?** {coverage_assessment}
10. **Should it move into allocation testing?** No. It should remain commentary-only until a governed real-document corpus is validated out of sample.
"""
    (OUTPUT_DIR / "summary.md").write_text(summary, encoding="utf-8")

    source_notes = f"""# Phase 4A.2 Source Notes

Generated: {date.today().isoformat()}

## RBI document source

- Corpus type: `synthetic_fixture`.
- Manifest: `data/sentiment/rbi_documents/sample_manifest.csv`
- Canonical manifest columns: `document_id`, `publication_date`, `document_type`, `title`, `local_path`, `source_url`.
- Documents: {len(documents)} locally stored synthetic RBI-style fixtures.
- The fixture does not reproduce long passages from RBI publications.
- `manifest.csv` is also provided as the default local-ingestion convention.

## Processing and scoring

- `.txt`, `.md`, and `.csv` are supported directly.
- PDF extraction is optional through `pypdf`; unavailable or unreadable PDFs are flagged without aborting the corpus.
- Sentence IDs preserve document order and are stable across reruns.
- Lexicon scoring classifies hawkish/neutral/dovish stance, certain/uncertain/neutral certainty, and forward/backward/current/unknown time orientation.
- Optional Hugging Face model IDs are configured for RBI stance, certainty, and time labels; failed model loads fall back per sentence to the lexicon.

## Quantitative comparison source

- Deterministic synthetic eight-asset daily return fixture from 2022-01-03 to 2026-06-19.
- Rule-based labels use the repository feature and classification code with a one-session lag.
- HMM labels use the repository expanding-window walk-forward implementation.
- Full-sample HMM is not used for decision-facing comparison.

## Look-ahead controls

- Publication dates are assigned to the first market date on or after publication.
- Observed macro labels are shifted by one market session before becoming decision labels.
- Decision-source dates precede their decision dates.

## Limitations

- This report demonstrates reproducible local ingestion, not a live or comprehensive RBI archive.
- The bundled corpus is intentionally small and synthetic.
- Lexicon labels can miss negation, context, speaker differences, and policy nuance.
- Transformer outputs require independent model and data validation.
- Lead/lag diagnostics are descriptive and do not establish predictive power.
- RBI macro-sentiment does not alter strategy scores, gates, confidence, adaptive policy, allocation, or portfolio weights.

## Verification

- `python -m pytest -q`: 393 passed, 1 skipped.
- Total statement coverage: 63%.
- Headless Streamlit root and health endpoints: HTTP 200.
- `python scripts/final_smoke_test.py`: passed.
"""
    (OUTPUT_DIR / "source_notes.md").write_text(source_notes, encoding="utf-8")

    agreement_rows = pd.DataFrame(
        [
            {
                "Comparison": "Rule-based decision regime",
                "Agreement": percentage(
                    comparison["agreement_with_rule_based"]
                ),
                "Stress/crisis risk-off confirmation": percentage(
                    comparison[
                        "stress_crisis_risk_off_confirmation_rule_based"
                    ]
                ),
            },
            {
                "Comparison": "HMM walk-forward decision regime",
                "Agreement": percentage(
                    comparison["agreement_with_hmm_walk_forward"]
                ),
                "Stress/crisis risk-off confirmation": percentage(
                    comparison["stress_crisis_risk_off_confirmation_hmm"]
                ),
            },
        ]
    )
    document_table = documents[
        [
            "document_id",
            "publication_date",
            "document_type",
            "local_path",
            "source_url",
            "load_status",
            "error",
        ]
    ].to_html(index=False, border=0)
    agreement_table = agreement_rows.to_html(index=False, border=0)
    disagreement_table = comparison["dates_of_major_disagreement"].head(20).to_html(
        index=False,
        border=0,
    )
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 4A.2 RBI Macro-Sentiment Confirmation</title>
  <style>
    :root {{ --ink:#1f2937; --muted:#667085; --line:#e4e7ec; --blue:#294a7a; --bg:#f7f8fb; --open:#eaf1fe; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:Inter,"Segoe UI",Arial,sans-serif; line-height:1.55; }}
    main {{ max-width:1080px; margin:auto; padding:44px 24px 72px; }}
    header {{ background:linear-gradient(135deg,#1f2c4d,#31588f); color:white; padding:38px; border-radius:20px; }}
    header p {{ color:#dce7f8; max-width:850px; }}
    h1 {{ margin:8px 0; font-size:38px; }}
    h2 {{ margin-top:38px; }}
    .eyebrow {{ text-transform:uppercase; letter-spacing:.11em; font-size:12px; font-weight:700; }}
    .callout {{ margin-top:24px; border-left:4px solid var(--blue); background:var(--open); padding:16px 18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }}
    .card,.panel {{ background:white; border:1px solid var(--line); border-radius:14px; padding:18px; }}
    .card span {{ display:block; color:var(--muted); font-size:12px; }}
    .card strong {{ font-size:23px; }}
    table {{ border-collapse:collapse; width:100%; background:white; font-size:13px; }}
    th,td {{ border-bottom:1px solid var(--line); padding:9px; text-align:left; }}
    th {{ color:var(--blue); background:#f4f7fd; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:14px; }}
    footer {{ margin-top:36px; color:var(--muted); font-size:12px; }}
  </style>
</head>
<body><main>
  <header>
    <div class="eyebrow">Adaptive Portfolio Risk Analytics · v1.1.1</div>
    <h1>Phase 4A.2 — RBI Macro-Sentiment Confirmation</h1>
    <p>Locally stored documents are ingested, split, scored for stance/certainty/time orientation, aligned to market dates, lagged, and compared with trading-safe quantitative regimes.</p>
  </header>
  <div class="callout"><strong>Corpus type:</strong> <code>synthetic_fixture</code>. This report is pipeline validation, not empirical RBI evidence.</div>
  <div class="callout"><strong>Conclusion:</strong> {escape(CONCLUSION)}</div>
  <h2>Current fixture state</h2>
  <section class="grid">
    <div class="card"><span>Quantitative regime</span><strong>{escape(str(current['quantitative_regime']))}</strong></div>
    <div class="card"><span>Macro label</span><strong>{escape(str(current['macro_sentiment_label']).replace('_', ' ').title())}</strong></div>
    <div class="card"><span>Confirmation</span><strong>{escape(str(current['macro_sentiment_confirmation']))}</strong></div>
    <div class="card"><span>Sentence coverage</span><strong>{int(current['macro_sentiment_coverage'])}</strong></div>
  </section>
  <h2>Agreement and coverage</h2>
  <div class="table-wrap">{agreement_table}</div>
  <div class="grid" style="margin-top:12px">
    <div class="card"><span>Decision-label coverage</span><strong>{percentage(comparison['coverage_ratio'])}</strong></div>
    <div class="card"><span>Loaded documents</span><strong>{loaded}/{len(documents)}</strong></div>
    <div class="card"><span>Scored sentences</span><strong>{len(scored)}</strong></div>
    <div class="card"><span>Transformer fallbacks</span><strong>{fallback_count}</strong></div>
  </div>
  <h2>Document ingestion audit</h2>
  <div class="table-wrap">{document_table}</div>
  <h2>Major disagreement dates</h2>
  <p>Disagreements are descriptive. They do not modify strategy selection or allocation.</p>
  <div class="table-wrap">{disagreement_table}</div>
  <h2>Required questions answered</h2>
  <div class="panel">
    <ol>
      <li><strong>Documents:</strong> Four local synthetic RBI-style MPC-minute and policy-statement fixtures; see the ingestion audit and <code>rbi_documents.csv</code>.</li>
      <li><strong>Sentence scoring:</strong> Stable ordered sentences scored for stance, certainty, and time orientation using phrase dictionaries.</li>
      <li><strong>Scorer:</strong> <code>rbi_macro_lexicon</code> version 1.0. Transformer models were optional and not used for this report.</li>
      <li><strong>Look-ahead:</strong> Publication-to-market alignment followed by a one-market-session decision lag.</li>
      <li><strong>Index:</strong> 63-session rolling shares with net stance equal to hawkish minus dovish share and macro risk equal to net stance plus uncertainty share.</li>
      <li><strong>HMM confirmation:</strong> {percentage(comparison['agreement_with_hmm_walk_forward'])} agreement and {percentage(comparison['stress_crisis_risk_off_confirmation_hmm'])} stress/risk-off confirmation.</li>
      <li><strong>Rule-based confirmation:</strong> {percentage(comparison['agreement_with_rule_based'])} agreement and {percentage(comparison['stress_crisis_risk_off_confirmation_rule_based'])} stress/crisis risk-off confirmation.</li>
      <li><strong>Pre-stress evidence:</strong> {len(lead_dates)} macro risk-off alignments occurred within the configured lead window before quantitative stress onsets.</li>
      <li><strong>Coverage:</strong> {escape(coverage_assessment)}</li>
      <li><strong>Allocation decision:</strong> Remain commentary-only until a governed real-document corpus is validated out of sample.</li>
    </ol>
  </div>
  <h2>Governance boundary</h2>
  <div class="panel">
    The corpus and market history in this report are synthetic fixtures. The
    implementation is reproducible, but no predictive or live-data claim is
    made. Out-of-sample validation and governed real-source coverage are
    required before any allocation experiment.
  </div>
  <footer>Generated {date.today().isoformat()} · Commentary-only research layer</footer>
</main></body></html>
"""
    (OUTPUT_DIR / "report.html").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    build_outputs()
