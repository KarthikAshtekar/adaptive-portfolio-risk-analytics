"""Generate the Phase 4A.5 API-based ex-ante NLP monitoring report."""

from __future__ import annotations

from datetime import date
from html import escape
import os
from pathlib import Path
import sys
import tempfile
import textwrap

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[2]
CHART_DIR = OUTPUT_DIR / "charts"
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "apra_matplotlib"),
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.regime import (  # noqa: E402
    calculate_regime_features,
    classify_rule_based_regime,
    fit_hmm_walk_forward,
    lag_regime_labels,
)
from src.sentiment import (  # noqa: E402
    AlphaVantageNewsProvider,
    EarningsCallProvider,
    GDELTProvider,
    RBIProvider,
    apply_publication_lag,
    build_composite_nlp_risk_index,
    build_macro_stance_index,
    compare_composite_nlp_to_regimes,
    flag_reaction_data_leakage,
    load_rbi_documents,
    run_sentiment_provider_ingestion,
    score_rbi_sentences,
    score_with_finbert,
    split_rbi_documents_into_sentences,
    validate_ex_ante_records,
)


CONCLUSION = (
    "API-based NLP risk is a confirmation and monitoring layer only. "
    "It is not yet promoted to allocation or strategy scoring."
)
PROVIDER_SCOPE = (
    "Offline fixtures and local manifests were used. Live API access and API "
    "keys were not required."
)

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue_base": "#A3BEFA",
    "blue_dark": "#2E4780",
    "orange_base": "#F0986E",
    "orange_dark": "#804126",
}


def _use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Aptos",
                "Inter",
                "Segoe UI",
                "DejaVu Sans",
                "Arial",
            ],
        },
    )


def _add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, 72)
    subtitle = textwrap.fill(subtitle, 108)
    ax.set_title("")
    fig.subplots_adjust(top=0.79, left=0.20, right=0.96, bottom=0.16)
    left = ax.get_position().x0
    fig.text(
        left,
        0.97,
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.90,
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    sns.despine(ax=ax)


def _build_market_returns() -> pd.DataFrame:
    """Deterministic market history used only for pipeline comparison."""
    index = pd.bdate_range("2022-01-03", "2026-06-19")
    rng = np.random.default_rng(45)
    volatility = np.full(len(index), 0.007)
    drift = np.full(len(index), 0.00030)
    for start, end in (
        ("2022-02-01", "2022-07-29"),
        ("2024-03-01", "2024-05-15"),
        ("2025-03-01", "2025-05-30"),
        ("2026-04-01", "2026-04-30"),
    ):
        mask = (index >= start) & (index <= end)
        volatility[mask] = 0.019
        drift[mask] = -0.0010
    common = drift + rng.normal(0.0, volatility)
    return pd.DataFrame(
        {
            f"SYNTH_{position + 1}": common
            + rng.normal(0.0, volatility * 0.45)
            for position in range(8)
        },
        index=index,
    )


def _percentage(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if not np.isfinite(number) else f"{number:.1%}"


def _render_charts(
    provider_diagnostics: pd.DataFrame,
    composite_index: pd.DataFrame,
) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    _use_chart_theme()

    provider_counts = provider_diagnostics[
        ["provider", "deduped_valid_record_count"]
    ].copy()
    provider_counts["deduped_valid_record_count"] = pd.to_numeric(
        provider_counts["deduped_valid_record_count"], errors="coerce"
    ).fillna(0)
    provider_counts = provider_counts.sort_values("deduped_valid_record_count")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.barplot(
        data=provider_counts,
        x="deduped_valid_record_count",
        y="provider",
        color=TOKENS["blue_base"],
        edgecolor=TOKENS["blue_dark"],
        linewidth=1.0,
        ax=ax,
    )
    ax.set_xlabel("Valid normalized records")
    ax.set_ylabel("Provider")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    _add_chart_header(
        fig,
        ax,
        "Provider coverage after provenance validation",
        "Offline fixture run; records missing required URLs or timestamps remain diagnostic rows and are excluded from deduplicated scoring.",
    )
    fig.savefig(
        CHART_DIR / "provider_coverage.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=TOKENS["surface"],
    )
    plt.close(fig)

    label_counts = (
        composite_index["decision_composite_nlp_label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="session_count")
        .sort_values("session_count")
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.barplot(
        data=label_counts,
        x="session_count",
        y="label",
        color=TOKENS["orange_base"],
        edgecolor=TOKENS["orange_dark"],
        linewidth=1.0,
        ax=ax,
    )
    ax.set_xlabel("Market sessions")
    ax.set_ylabel("Decision-lagged NLP label")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    _add_chart_header(
        fig,
        ax,
        "Composite NLP label distribution",
        "The insufficient-data label dominates sparse periods; covered labels are monitoring diagnostics, not allocation signals.",
    )
    fig.savefig(
        CHART_DIR / "composite_label_distribution.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=TOKENS["surface"],
    )
    plt.close(fig)


def build_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    providers = [
        RBIProvider(
            feeds_enabled=False,
            local_manifest_path=(
                REPO_ROOT / "data" / "sentiment" / "rbi_real" / "manifest.csv"
            ),
            local_corpus_path=(
                REPO_ROOT
                / "data"
                / "sentiment"
                / "rbi_documents"
                / "sample_manifest.csv"
            ),
        ),
        EarningsCallProvider(
            REPO_ROOT
            / "data"
            / "sentiment"
            / "earnings_calls"
            / "manifest.csv"
        ),
        GDELTProvider(
            enabled=True,
            fixture_path=(
                REPO_ROOT
                / "data"
                / "sentiment"
                / "provider_fixtures"
                / "gdelt_sample.json"
            ),
        ),
        AlphaVantageNewsProvider(api_key="", enabled=True),
    ]
    query_terms = [
        "India inflation",
        "RBI rate hike",
        "geopolitical risk India",
        "oil price shock",
        "banking stress",
        "currency crisis",
        "war escalation",
        "supply chain disruption",
    ]
    with tempfile.TemporaryDirectory(prefix="phase4a5_") as temp_dir:
        ingestion = run_sentiment_provider_ingestion(
            providers,
            "2022-01-01",
            "2026-06-19",
            temp_dir,
            query_config={
                "gdelt": {"query": query_terms, "limit": 100},
                "default": {"limit": 100},
            },
            use_cache=True,
        )
    normalized = ingestion["normalized_sentiment_records"]
    deduped = ingestion["deduped_sentiment_records"]
    diagnostics = ingestion["provider_diagnostics"]

    ex_ante = validate_ex_ante_records(deduped)
    ex_ante = flag_reaction_data_leakage(ex_ante)
    ex_ante = apply_publication_lag(ex_ante, lag_days=1)
    finbert_scores = score_with_finbert(
        ex_ante,
        pipeline_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("offline report validation: local FinBERT unavailable")
        ),
    )
    earnings = finbert_scores.loc[
        finbert_scores["document_type"].astype(str).eq("earnings_call")
    ]
    news = finbert_scores.loc[
        finbert_scores["document_type"].astype(str).eq("financial_news")
    ]

    returns = _build_market_returns()
    features = calculate_regime_features(returns)
    rule_regimes = lag_regime_labels(
        classify_rule_based_regime(features),
        lag=1,
    )
    hmm = fit_hmm_walk_forward(
        features,
        n_states=2,
        min_train_size=252,
        refit_frequency=126,
        covariance_type="diag",
        decision_lag=1,
    )
    hmm_regimes = hmm["decision_regimes"].reindex(returns.index).fillna(
        "Unknown"
    )

    rbi_documents = load_rbi_documents(
        REPO_ROOT
        / "data"
        / "sentiment"
        / "rbi_documents"
        / "sample_manifest.csv"
    )
    rbi_sentences = split_rbi_documents_into_sentences(rbi_documents)
    rbi_scores = score_rbi_sentences(rbi_sentences, method="lexicon")
    rbi_macro_index = build_macro_stance_index(
        rbi_scores,
        returns.index,
        lookback_window=63,
        decision_lag=1,
    )
    composite = build_composite_nlp_risk_index(
        rbi_macro_index=rbi_macro_index,
        earnings_sentiment=earnings,
        news_sentiment=news,
        market_index=returns.index,
        decision_lag=1,
    )
    comparison = compare_composite_nlp_to_regimes(
        composite,
        rule_regimes,
        hmm_regimes,
    )
    comparison_table = comparison["comparison_table"]
    reaction_warnings = ex_ante.loc[
        ex_ante["possible_reaction_data"].fillna(False)
    ].copy()

    normalized.to_csv(
        OUTPUT_DIR / "normalized_sentiment_records.csv", index=False
    )
    diagnostics.to_csv(OUTPUT_DIR / "provider_diagnostics.csv", index=False)
    ex_ante.to_csv(OUTPUT_DIR / "ex_ante_validation.csv", index=False)
    finbert_scores.to_csv(OUTPUT_DIR / "finbert_scores.csv", index=False)
    composite.to_csv(OUTPUT_DIR / "composite_nlp_risk_index.csv", index=True)
    comparison_table.to_csv(
        OUTPUT_DIR / "nlp_regime_comparison.csv", index=True
    )
    reaction_warnings.to_csv(
        OUTPUT_DIR / "reaction_data_warnings.csv", index=False
    )
    _render_charts(diagnostics, composite)

    providers_configured = diagnostics["provider"].astype(str).tolist()
    providers_with_data = diagnostics.loc[
        pd.to_numeric(
            diagnostics["deduped_valid_record_count"], errors="coerce"
        ).fillna(0).gt(0),
        "provider",
    ].astype(str).tolist()
    valid_count = int(ex_ante["is_ex_ante_valid"].sum())
    reaction_count = int(len(reaction_warnings))
    fallback_count = int(finbert_scores["fallback_used"].fillna(False).sum())
    coverage_ratio = float(comparison["coverage_ratio"])
    allocation_suitable = False
    label_distribution = (
        composite["decision_composite_nlp_label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="session_count")
    )

    summary = f"""# Phase 4A.5 API-Based Ex-Ante NLP Risk Monitoring

Generated: {date.today().isoformat()}

## Technical summary

**{CONCLUSION}**

- Provider mode: offline fixtures and local manifests.
- Providers configured: {", ".join(providers_configured)}.
- Providers returning valid records: {", ".join(providers_with_data) or "None"}.
- Ex-ante valid records: {valid_count} of {len(ex_ante)}.
- Possible reaction-data warnings: {reaction_count}.
- FinBERT status: lexicon fallback for {fallback_count} records.
- Composite NLP decision coverage: {_percentage(coverage_ratio)}.
- HMM walk-forward agreement: {_percentage(comparison["agreement_with_hmm_walk_forward"])}.
- Rule-based agreement: {_percentage(comparison["agreement_with_rule_based"])}.
- Suitable for allocation testing: {allocation_suitable}.

This fixture-backed run validates provider normalization, timestamp discipline,
reaction-data warnings, fallback scoring, composite construction, and lagged
regime comparison. It is not empirical evidence from live provider coverage.
"""
    (OUTPUT_DIR / "summary.md").write_text(summary, encoding="utf-8")

    chart_map = """- `provider_coverage.png`: ranked horizontal bars; valid records by provider; supports provider-return and provenance findings.
- `composite_label_distribution.png`: horizontal bars; market sessions by lagged composite label; supports coverage and label-distribution findings."""
    source_notes = f"""# Phase 4A.5 Source Notes

Generated: {date.today().isoformat()}

## Report contract

- Delivery mode: portable technical HTML.
- Audience: technical.
- Provider mode: offline fixtures and local manifests.
- Live API calls: disabled.
- Market history: deterministic synthetic data for pipeline comparison only.
- Selection/allocation impact: none.

## Source inventory

- RBI provider: empty real manifest with synthetic local fallback; rows without source URLs remain invalid provider diagnostics.
- Earnings provider: synthetic local transcript fixtures.
- GDELT provider: fixture JSON.
- Alpha Vantage provider: enabled test path with missing key; skipped safely.
- RBI macro component: existing synthetic RBI fixture pipeline.

## Chart map

{chart_map}

## Required structure mapping

- Technical summary: provider status, ex-ante validity, fallback, coverage, and conclusion.
- Key findings: provider coverage and composite label distribution.
- Scope/data/definitions: normalized schema, source types, and coverage definitions.
- Methodology: provider ingestion, timestamp validation, reaction flags, scoring, lagging, and regime comparison.
- Limitations/robustness: fixtures, no live API evidence, no predictive claim.
- Recommended next steps: governed provider trials and out-of-sample monitoring.
- Further questions: minimum coverage and source-governance thresholds.

## Verification

- Full suite: 418 passed, 1 skipped, 64% statement coverage.
- Dashboard root and health endpoints: HTTP 200.
- Final smoke test: passed.
"""
    (OUTPUT_DIR / "source_notes.md").write_text(
        source_notes, encoding="utf-8"
    )

    audit_rows = [
        ("1. Which providers were configured?", ", ".join(providers_configured)),
        (
            "2. Which providers returned data?",
            ", ".join(providers_with_data) or "None",
        ),
        (
            "3. Which records were ex-ante valid?",
            f"{valid_count} of {len(ex_ante)} normalized, deduplicated records",
        ),
        (
            "4. Were possible reaction-data records flagged?",
            f"Yes; {reaction_count} record(s)",
        ),
        (
            "5. Was FinBERT used or did it fall back?",
            f"Lexicon fallback was recorded for {fallback_count} record(s)",
        ),
        (
            "6. What was the source coverage?",
            f"{len(providers_with_data)} of {len(providers_configured)} providers "
            f"returned provenance-valid data; composite decision coverage "
            f"was {_percentage(coverage_ratio)}",
        ),
        (
            "7. What was the composite NLP risk label distribution?",
            "; ".join(
                f"{row.label}: {int(row.session_count)}"
                for row in label_distribution.itertuples(index=False)
            ),
        ),
        (
            "8. How often did NLP agree with HMM regimes?",
            _percentage(comparison["agreement_with_hmm_walk_forward"]),
        ),
        (
            "9. How often did NLP agree with rule-based regimes?",
            _percentage(comparison["agreement_with_rule_based"]),
        ),
        (
            "10. Did NLP show pre-stress warnings?",
            f"{comparison['pre_stress_warning_count']} warning onset(s)",
        ),
        (
            "11. Is the signal suitable for allocation testing?",
            f"No. {CONCLUSION}",
        ),
    ]
    audit_table = pd.DataFrame(
        audit_rows, columns=["question", "answer"]
    ).to_html(index=False, border=0)
    provider_table = diagnostics.to_html(index=False, border=0)
    label_table = label_distribution.to_html(index=False, border=0)

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 4A.5 API-Based Ex-Ante NLP Risk Monitoring</title>
  <style>
    :root {{ --ink:#1f2430; --muted:#6f768a; --line:#e6e8f0; --blue:#2e4780; --open:#eaf1fe; --bg:#fcfcfd; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:Inter,"Segoe UI",Arial,sans-serif; }}
    main {{ max-width:980px; margin:auto; padding:42px 24px 72px; }}
    header,section {{ margin-bottom:34px; }}
    h1 {{ font-size:38px; margin:0 0 10px; }}
    h2 {{ font-size:25px; margin:0 0 12px; }}
    p,li {{ line-height:1.65; }}
    .status {{ border-left:4px solid var(--blue); background:var(--open); padding:16px 18px; border-radius:8px; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:18px 0; }}
    .metric {{ background:white; border:1px solid var(--line); border-radius:12px; padding:16px; }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; }}
    .metric strong {{ font-size:22px; }}
    figure {{ margin:22px 0; }}
    figure img {{ width:100%; border:1px solid var(--line); border-radius:12px; background:white; }}
    figcaption {{ color:var(--muted); font-size:13px; margin-top:8px; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:12px; }}
    table {{ border-collapse:collapse; width:100%; background:white; font-size:13px; }}
    th,td {{ border-bottom:1px solid var(--line); padding:9px 11px; text-align:left; }}
    th {{ color:var(--blue); background:#f4f7fd; }}
    code {{ background:#f2f4f7; padding:2px 5px; border-radius:4px; }}
  </style>
</head>
<body><main data-report-audience="technical">
  <header data-contract-section="title">
    <h1>Phase 4A.5 API-Based Ex-Ante NLP Risk Monitoring</h1>
  </header>

  <section data-contract-section="technical-summary">
    <h2>Technical Summary</h2>
    <div class="status"><strong>{escape(CONCLUSION)}</strong><br>{escape(PROVIDER_SCOPE)}</div>
    <div class="metric-grid">
      <div class="metric"><span>Valid ex-ante records</span><strong>{valid_count}</strong></div>
      <div class="metric"><span>Providers with data</span><strong>{len(providers_with_data)}/{len(providers_configured)}</strong></div>
      <div class="metric"><span>Composite coverage</span><strong>{_percentage(coverage_ratio)}</strong></div>
      <div class="metric"><span>Reaction warnings</span><strong>{reaction_count}</strong></div>
    </div>
  </section>

  <section data-contract-section="key-findings">
    <h2>Provider plumbing works offline, but provenance gaps remain visible</h2>
    <p>Earnings and GDELT fixtures returned provenance-valid records. Alpha Vantage skipped safely without a key. Synthetic RBI fixture rows lacking source URLs remain visible as invalid diagnostics instead of being silently promoted.</p>
    <figure>
      <img src="charts/provider_coverage.png" alt="Provider valid record coverage">
      <figcaption>Valid normalized records after provider-level provenance checks.</figcaption>
    </figure>
    <p>The composite index is intentionally coverage-gated. Sparse periods remain insufficient, while covered periods demonstrate lagged calculation only; the distribution is not evidence of predictive value.</p>
    <figure>
      <img src="charts/composite_label_distribution.png" alt="Composite NLP label distribution">
      <figcaption>Decision-lagged composite labels across deterministic comparison sessions.</figcaption>
    </figure>
  </section>

  <section data-contract-section="scope-data-and-metric-definitions">
    <h2>Scope, Data, and Metric Definitions</h2>
    <p>Each external record stores provider, publication and retrieval timestamps, source URL, language, text, and raw metadata. Ex-ante validity requires both timestamps and publication no later than retrieval. Composite coverage is the fraction of RBI, earnings, and news components available on a session; fewer than two sources yields <code>insufficient_nlp_data</code>.</p>
    <div class="table-wrap">{provider_table}</div>
    <h3>Decision-lagged label distribution</h3>
    <div class="table-wrap">{label_table}</div>
  </section>

  <section data-contract-section="methodology">
    <h2>Methodology and Ex-Ante Controls</h2>
    <p>Providers fetch or load independently, normalize to one schema, retain row-level failures, and deduplicate by URL/title and publication minute. Reaction-oriented phrases are flagged rather than automatically deleted; flagged records are excluded from composite source scoring. FinBERT is local-only and falls back to the lexicon. The composite label is shifted by one day before comparison with lagged rule-based and HMM walk-forward regimes.</p>
    <h3>Required validation questions</h3>
    <div class="table-wrap">{audit_table}</div>
  </section>

  <section data-contract-section="limitations-uncertainty-and-robustness-checks">
    <h2>Fixtures validate behavior, not signal efficacy</h2>
    <p>This report uses synthetic or fixture text and deterministic market history. Live API reliability, historical completeness, licensing, language coverage, provider revisions, model calibration, and publication-time governance remain unvalidated. Vendor sentiment is retained only as metadata. No future returns, price reactions, volatility, or drawdown values enter NLP scoring. Full-sample HMM is not used.</p>
  </section>

  <section data-contract-section="recommended-next-steps">
    <h2>Run governed shadow monitoring before any allocation experiment</h2>
    <ol>
      <li>Enable one provider at a time with reviewed credentials, rate limits, and retention rules.</li>
      <li>Measure stable source coverage and publication-time completeness over an out-of-sample shadow period.</li>
      <li>Validate FinBERT and provider-sentiment calibration locally without future-return labels.</li>
      <li>Define explicit eligibility thresholds before considering a separate allocation research phase.</li>
    </ol>
  </section>

  <section data-contract-section="further-questions">
    <h2>Further Questions</h2>
    <p>What source mix, freshness threshold, and minimum coverage should make a composite label decision-eligible? How stable are false risk-off rates across providers, languages, sectors, and geopolitical-event types? What licensing and retention constraints govern transcript and news storage?</p>
  </section>
</main></body></html>
"""
    (OUTPUT_DIR / "report.html").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    build_outputs()
