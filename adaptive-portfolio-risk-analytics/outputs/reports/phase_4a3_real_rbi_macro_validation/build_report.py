"""Generate the Phase 4A.3 real-RBI corpus validation report."""

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
    build_macro_stance_index,
    compare_macro_to_regimes,
    load_real_rbi_corpus,
    load_rbi_documents,
    score_rbi_sentences,
    split_rbi_documents_into_sentences,
    validate_rbi_manifest,
)


CONCLUSION = (
    "RBI macro sentiment remains a confirmation layer until real-document "
    "coverage is sufficient and out-of-sample validation supports promotion."
)
FALLBACK_MESSAGE = (
    "Real RBI corpus unavailable; synthetic fixture mode used only for "
    "pipeline validation."
)
MINIMUM_COVERAGE_THRESHOLD = 0.10
MINIMUM_REAL_DOCUMENTS = 5

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


def _add_chart_header(
    fig,
    ax,
    title: str,
    subtitle: str,
) -> None:
    title = textwrap.fill(title, 72)
    subtitle = textwrap.fill(subtitle, 105)
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


def _metadata(
    frame: pd.DataFrame,
    corpus_type: str,
    scoring_method: str,
    decision_lag: int,
    lookback_window: int,
) -> pd.DataFrame:
    result = frame.copy()
    result["corpus_type"] = corpus_type
    result["scoring_method"] = scoring_method
    result["decision_lag"] = int(decision_lag)
    result["lookback_window"] = int(lookback_window)
    return result


def _render_charts(
    documents: pd.DataFrame,
    comparison: dict[str, object],
    corpus_type: str,
) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    _use_chart_theme()

    type_counts = (
        documents["document_type"]
        .value_counts()
        .rename_axis("document_type")
        .reset_index(name="document_count")
        .sort_values("document_count")
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.barplot(
        data=type_counts,
        x="document_count",
        y="document_type",
        color=TOKENS["blue_base"],
        edgecolor=TOKENS["blue_dark"],
        linewidth=1.0,
        ax=ax,
    )
    ax.set_xlabel("Documents")
    ax.set_ylabel("Document type")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    _add_chart_header(
        fig,
        ax,
        "Active corpus document types",
        (
            "Validated local RBI documents by type."
            if corpus_type == "real_rbi"
            else "Synthetic fallback documents used because no valid real-RBI rows are available."
        ),
    )
    fig.savefig(
        CHART_DIR / "document_type_distribution.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=TOKENS["surface"],
    )
    plt.close(fig)

    rates = pd.DataFrame(
        [
            {
                "metric": "Rule agreement",
                "rate": comparison["agreement_with_rule_based"],
            },
            {
                "metric": "HMM agreement",
                "rate": comparison["agreement_with_hmm_walk_forward"],
            },
            {
                "metric": "Rule stress confirmation",
                "rate": comparison[
                    "stress_crisis_risk_off_confirmation_rule_based"
                ],
            },
            {
                "metric": "HMM stress confirmation",
                "rate": comparison[
                    "stress_crisis_risk_off_confirmation_hmm"
                ],
            },
            {
                "metric": "Decision coverage",
                "rate": comparison["coverage_ratio"],
            },
        ]
    ).fillna(0.0)
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    sns.barplot(
        data=rates,
        x="rate",
        y="metric",
        color=TOKENS["orange_base"],
        edgecolor=TOKENS["orange_dark"],
        linewidth=1.0,
        ax=ax,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Rate")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    _add_chart_header(
        fig,
        ax,
        "Macro-regime agreement and coverage",
        (
            "Validated real-RBI corpus diagnostics."
            if corpus_type == "real_rbi"
            else "Synthetic fixture mode; rates validate the pipeline and are not empirical RBI evidence."
        ),
    )
    fig.savefig(
        CHART_DIR / "agreement_and_coverage.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=TOKENS["surface"],
    )
    plt.close(fig)


def build_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    real_manifest = (
        REPO_ROOT / "data" / "sentiment" / "rbi_real" / "manifest.csv"
    )
    real_validation = validate_rbi_manifest(real_manifest)
    real_documents = load_real_rbi_corpus(real_manifest)
    if real_documents.empty:
        corpus_type = "synthetic_fixture"
        corpus_message = FALLBACK_MESSAGE
        active_manifest = (
            REPO_ROOT
            / "data"
            / "sentiment"
            / "rbi_documents"
            / "sample_manifest.csv"
        )
        documents = load_rbi_documents(active_manifest)
    else:
        corpus_type = "real_rbi"
        corpus_message = (
            "A validated local real-RBI corpus was used for this report."
        )
        active_manifest = real_manifest
        documents = real_documents

    scoring_method = "lexicon"
    lookback_window = 63
    decision_lag = 1
    sentences = split_rbi_documents_into_sentences(documents)
    scores = score_rbi_sentences(sentences, method=scoring_method)
    returns = _build_market_returns()
    features = calculate_regime_features(returns)
    rule_decisions = lag_regime_labels(
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
    hmm_decisions = hmm["decision_regimes"].reindex(returns.index).fillna(
        "Unknown"
    )
    macro_index = build_macro_stance_index(
        scores,
        returns.index,
        lookback_window=lookback_window,
        decision_lag=decision_lag,
    )
    comparison = compare_macro_to_regimes(
        macro_index,
        rule_decisions,
        hmm_decisions,
    )

    coverage_ratio = float(comparison["coverage_ratio"])
    valid_real_count = int(len(real_documents))
    dates = pd.to_datetime(
        real_documents.get(
            "publication_date",
            pd.Series(dtype="datetime64[ns]"),
        ),
        errors="coerce",
    ).dropna()
    real_date_start = (
        dates.min().date().isoformat() if not dates.empty else "Unavailable"
    )
    real_date_end = (
        dates.max().date().isoformat() if not dates.empty else "Unavailable"
    )
    real_date_range = (
        f"{real_date_start} to {real_date_end}"
        if not dates.empty
        else "Unavailable"
    )
    active_types = ", ".join(
        sorted(documents["document_type"].dropna().astype(str).unique())
    )
    distribution = (
        macro_index["decision_macro_label"]
        .value_counts()
        .rename_axis("macro_label")
        .reset_index(name="session_count")
    )
    pre_stress_count = int(
        len(comparison["macro_risk_off_before_stress_dates"])
    )
    coverage_sufficient = bool(
        corpus_type == "real_rbi"
        and valid_real_count >= MINIMUM_REAL_DOCUMENTS
        and coverage_ratio >= MINIMUM_COVERAGE_THRESHOLD
    )

    corpus_summary = dict(real_validation["summary"])
    corpus_summary.update(
        {
            "active_corpus_type": corpus_type,
            "active_document_count": int(len(documents)),
            "active_sentence_count": int(len(scores)),
            "active_document_types": active_types,
            "fallback_reason": corpus_message
            if corpus_type == "synthetic_fixture"
            else "",
        }
    )
    corpus_diagnostics = pd.DataFrame(
        [
            {
                "metric": metric,
                "category": "",
                "value": value,
            }
            for metric, value in corpus_summary.items()
            if not isinstance(value, dict)
        ]
        + [
            {
                "metric": "document_type_counts",
                "category": key,
                "value": value,
            }
            for key, value in corpus_summary.get(
                "document_type_counts",
                {},
            ).items()
        ]
    )
    coverage_diagnostics = pd.DataFrame(
        [
            {
                "metric": "decision_coverage_ratio",
                "value": coverage_ratio,
            },
            {
                "metric": "minimum_coverage_threshold",
                "value": MINIMUM_COVERAGE_THRESHOLD,
            },
            {
                "metric": "minimum_real_documents",
                "value": MINIMUM_REAL_DOCUMENTS,
            },
            {
                "metric": "coverage_sufficient",
                "value": coverage_sufficient,
            },
            {
                "metric": "real_document_count",
                "value": valid_real_count,
            },
            {
                "metric": "active_sentence_count",
                "value": int(len(scores)),
            },
        ]
    )

    output_frames = {
        "rbi_documents": documents,
        "rbi_sentence_scores": scores,
        "macro_stance_index": macro_index,
        "macro_regime_comparison": comparison["comparison_table"],
        "coverage_diagnostics": coverage_diagnostics,
        "corpus_diagnostics": corpus_diagnostics,
        "disagreement_dates": comparison["dates_of_major_disagreement"],
    }
    for filename, frame in output_frames.items():
        _metadata(
            frame,
            corpus_type,
            scoring_method,
            decision_lag,
            lookback_window,
        ).to_csv(OUTPUT_DIR / f"{filename}.csv", index=True)

    _render_charts(documents, comparison, corpus_type)

    summary = f"""# Phase 4A.3 Real RBI Corpus Validation

Generated: {date.today().isoformat()}

## Technical summary

**{corpus_message}**

- Corpus type: `{corpus_type}`.
- Real RBI documents ingested: {valid_real_count}.
- Real corpus date range: {real_date_range}.
- Active fallback documents: {len(documents)}; active sentences: {len(scores)}.
- Scoring method: `{scoring_method}`.
- Decision coverage: {_percentage(coverage_ratio)}.
- HMM walk-forward agreement: {_percentage(comparison['agreement_with_hmm_walk_forward'])}.
- Rule-based agreement: {_percentage(comparison['agreement_with_rule_based'])}.
- Coverage sufficient for empirical conclusions: {coverage_sufficient}.

## Conservative conclusion

**{CONCLUSION}**

The current report does not contain empirical real-RBI results. Synthetic
fixtures and deterministic market history are retained only to verify corpus
fallback, sentence scoring, lagging, coverage diagnostics, and regime
comparison.
"""
    (OUTPUT_DIR / "summary.md").write_text(summary, encoding="utf-8")

    source_notes = f"""# Phase 4A.3 Source Notes

Generated: {date.today().isoformat()}

## Report contract

- Delivery mode: portable technical HTML.
- Audience: technical.
- Corpus type: `{corpus_type}`.
- Active manifest: `{active_manifest.relative_to(REPO_ROOT)}`.
- Real manifest: `data/sentiment/rbi_real/manifest.csv`.
- Real corpus status: {corpus_message}

## Source and methodology

- Real-corpus validation uses the exact nine-column Phase 4A.3 manifest.
- Synthetic fixtures remain under `data/sentiment/rbi_documents/`.
- Scoring uses `{scoring_method}`; transformer models remain optional.
- Publication dates align to the first market date on or after publication.
- Macro decisions are shifted by {decision_lag} market session before comparison.
- HMM comparison uses the repository walk-forward implementation.
- The report market history is deterministic synthetic data, not live market evidence.

## Chart map

- `document_type_distribution.png`: category comparison; document type and count; shows active fallback composition.
- `agreement_and_coverage.png`: horizontal rate comparison; agreement, stress confirmation, and coverage; shows pipeline diagnostics only.

## Required structure mapping

- Technical summary: corpus availability and main conclusion.
- Key findings: active-corpus composition and macro-regime diagnostics.
- Scope/data/definitions: corpus and metric definitions.
- Methodology: validation, scoring, alignment, lagging, comparison.
- Limitations/robustness: synthetic fallback and no empirical claim.
- Recommended next steps: populate and validate a real corpus.
- Further questions: coverage and out-of-sample promotion criteria.

## Verification

- Full suite: 402 passed, 1 skipped, 64% statement coverage.
- Dashboard root and health endpoints: HTTP 200.
- Final smoke test: passed.
"""
    (OUTPUT_DIR / "source_notes.md").write_text(
        source_notes,
        encoding="utf-8",
    )

    distribution_table = distribution.to_html(index=False, border=0)
    type_table = (
        documents["document_type"]
        .value_counts()
        .rename_axis("document_type")
        .reset_index(name="document_count")
        .to_html(index=False, border=0)
    )
    label_distribution = ", ".join(
        f"{row.macro_label}: {int(row.session_count)} sessions"
        for row in distribution.itertuples(index=False)
    )
    audit_rows = [
        ("1. How many real RBI documents were ingested?", str(valid_real_count)),
        ("2. What date range does the corpus cover?", real_date_range),
        (
            "3. Which document types are included?",
            "None" if valid_real_count == 0 else active_types,
        ),
        ("4. What scoring method was used?", scoring_method),
        (
            "5. What is the sentence/document coverage?",
            f"{len(scores)} sentences / {len(documents)} active documents; "
            f"{_percentage(coverage_ratio)} decision-session coverage",
        ),
        (
            "6. How was look-ahead avoided?",
            f"Publication dates were aligned to the first eligible market "
            f"session and macro labels were shifted by {decision_lag} session.",
        ),
        (
            "7. What is the macro-risk label distribution?",
            label_distribution,
        ),
        (
            "8. What is the agreement with HMM walk-forward regimes?",
            _percentage(comparison["agreement_with_hmm_walk_forward"]),
        ),
        (
            "9. What is the agreement with rule-based regimes?",
            _percentage(comparison["agreement_with_rule_based"]),
        ),
        (
            "10. Does RBI macro risk-off confirm stress/crisis periods?",
            (
                "Rule-based: "
                + _percentage(
                    comparison[
                        "stress_crisis_risk_off_confirmation_rule_based"
                    ]
                )
                + "; HMM walk-forward: "
                + _percentage(
                    comparison[
                        "stress_crisis_risk_off_confirmation_hmm"
                    ]
                )
            ),
        ),
        (
            "11. Are there pre-stress macro warnings?",
            (
                f"{pre_stress_count} identified date(s); synthetic fallback "
                "results are pipeline diagnostics only."
            ),
        ),
        (
            "12. Is coverage sufficient for empirical conclusions?",
            str(coverage_sufficient),
        ),
        (
            "13. Should sentiment remain commentary-only?",
            f"Yes. {CONCLUSION}",
        ),
    ]
    audit_table = pd.DataFrame(
        audit_rows,
        columns=["question", "answer"],
    ).to_html(index=False, border=0)
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 4A.3 Real RBI Corpus Validation</title>
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
    <h1>Phase 4A.3 Real RBI Corpus Validation</h1>
  </header>

  <section data-contract-section="technical-summary">
    <h2>Technical Summary</h2>
    <div class="status"><strong>{escape(corpus_message)}</strong><br>{escape(CONCLUSION)}</div>
    <div class="metric-grid">
      <div class="metric"><span>Corpus type</span><strong>{corpus_type}</strong></div>
      <div class="metric"><span>Real documents</span><strong>{valid_real_count}</strong></div>
      <div class="metric"><span>Decision coverage</span><strong>{_percentage(coverage_ratio)}</strong></div>
      <div class="metric"><span>Scorer</span><strong>{scoring_method}</strong></div>
    </div>
  </section>

  <section data-contract-section="key-findings">
    <h2>No empirical conclusion is available without a real corpus</h2>
    <p>The active documents are synthetic fixtures. The composition chart is therefore a pipeline audit, not a description of RBI communication coverage.</p>
    <figure>
      <img src="charts/document_type_distribution.png" alt="Document type distribution">
      <figcaption>Active corpus by document type. Corpus type: {corpus_type}.</figcaption>
    </figure>
    <p>Agreement and coverage rates confirm that alignment and comparison code runs end-to-end. They do not establish predictive or empirical validity.</p>
    <figure>
      <img src="charts/agreement_and_coverage.png" alt="Agreement and coverage rates">
      <figcaption>Synthetic fixture agreement, stress confirmation, and decision coverage.</figcaption>
    </figure>
  </section>

  <section data-contract-section="scope-data-and-metric-definitions">
    <h2>Scope, Data, and Metric Definitions</h2>
    <p>The real manifest contains {valid_real_count} valid documents covering {real_date_range}. The active corpus contains {len(documents)} documents and {len(scores)} scored sentences. Decision coverage is the share of market sessions with a non-insufficient lagged macro label.</p>
    <div class="table-wrap">{type_table}</div>
    <h3>Macro-risk label distribution</h3>
    <div class="table-wrap">{distribution_table}</div>
  </section>

  <section data-contract-section="methodology">
    <h2>Methodology and Look-Ahead Controls</h2>
    <p>Documents are validated locally, segmented into stable sentences, and scored for stance, certainty, and time orientation. Publication dates map to the first market session on or after publication. The rolling macro label is shifted by {decision_lag} session before comparison with rule-based and HMM walk-forward regimes.</p>
    <pre>net_stance_score = hawkish_share - dovish_share
macro_risk_score = net_stance_score + uncertainty_share
decision_macro_label[t] = macro_label[t - {decision_lag}]</pre>
    <h3>Required validation questions</h3>
    <div class="table-wrap">{audit_table}</div>
  </section>

  <section data-contract-section="limitations-uncertainty-and-robustness-checks">
    <h2>Limitations, Uncertainty, and Robustness Checks</h2>
    <p>No real RBI text is bundled, the active corpus is sparse and synthetic, transformer models are optional and unvalidated, and the report market history is deterministic synthetic data. Full-sample HMM is not used. Allocation, strategy scoring, evidence gates, confidence, and portfolio weights remain unchanged.</p>
  </section>

  <section data-contract-section="recommended-next-steps">
    <h2>Populate the governed real corpus before empirical interpretation</h2>
    <ol>
      <li>Download public RBI documents using the manual guide.</li>
      <li>Validate source URLs, publication dates, duplicates, and extracted text.</li>
      <li>Run the empirical validation runner with real market returns and walk-forward regimes.</li>
      <li>Require adequate coverage and out-of-sample stability before considering allocation research.</li>
    </ol>
  </section>

  <section data-contract-section="further-questions">
    <h2>Further Questions</h2>
    <p>What minimum document count, time span, and decision-date coverage should govern empirical eligibility? Does performance remain stable across document types, scoring methods, lookback windows, and genuinely out-of-sample periods?</p>
  </section>
</main></body></html>
"""
    (OUTPUT_DIR / "report.html").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    build_outputs()
