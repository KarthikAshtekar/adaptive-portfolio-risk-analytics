"""Validate whether collected real NLP evidence is allocation-test eligible."""

from __future__ import annotations

import argparse
from datetime import date
from html import escape
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "apra_matplotlib"),
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.regime import (  # noqa: E402
    calculate_regime_features,
    classify_rule_based_regime,
    fit_hmm_walk_forward,
    lag_regime_labels,
)
from src.sentiment import (  # noqa: E402
    VALID_COMPOSITE_NLP_LABELS,
    apply_publication_lag,
    build_composite_nlp_risk_index,
    build_daily_nlp_signal,
    build_macro_stance_index,
    calculate_nlp_coverage,
    compare_composite_nlp_to_regimes,
    flag_reaction_data_leakage,
    load_real_rbi_corpus,
    load_provider_config,
    score_rbi_sentences,
    score_sentiment_records,
    score_source_quality,
    score_with_finbert,
    split_rbi_sentences,
    validate_ex_ante_records,
    validate_nlp_corpus_intake,
    validate_rbi_manifest,
)
from scripts.check_rbi_corpus_status import build_rbi_corpus_status  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "reports" / "phase_4a6_real_nlp_validation"
)
DEFAULT_MULTI_SOURCE_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "reports"
    / "phase_4a8_multisource_nlp_monitoring"
)
DEFAULT_RBI_MANIFEST = REPO_ROOT / "data" / "sentiment" / "rbi_real" / "manifest.csv"
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "deduped_sentiment_records.csv"
VERDICT_A = "A. Sufficient for future allocation testing"
VERDICT_B = "B. Useful for monitoring only"
VERDICT_C = "C. Insufficient real-data coverage"
MONITORING_CAVEAT = (
    "NLP signal is monitoring-only due to insufficient real-data coverage."
)
MONITORING_ONLY_CAVEAT = (
    "GDELT/news-only NLP is a real news monitoring signal, not an "
    "allocation signal. Allocation impact: None."
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
}


def _read_records(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Collected NLP records not found: {input_path}")
    try:
        return pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _build_market_returns(start_date, end_date) -> pd.DataFrame:
    """Deterministic comparator history; it is never an NLP input."""
    index = pd.bdate_range(start_date, end_date)
    rng = np.random.default_rng(46)
    volatility = np.full(len(index), 0.008)
    drift = np.full(len(index), 0.00025)
    for stress_start, stress_end in (
        ("2020-02-17", "2020-05-29"),
        ("2022-02-01", "2022-07-29"),
        ("2024-03-01", "2024-05-15"),
        ("2026-04-01", "2026-04-30"),
    ):
        mask = (index >= stress_start) & (index <= stress_end)
        volatility[mask] = 0.020
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


def _rbi_component(
    scored: pd.DataFrame,
    market_index: pd.DatetimeIndex,
) -> pd.DataFrame | None:
    if scored.empty:
        return None
    rbi = scored.loc[
        scored.get("provider", pd.Series("", index=scored.index))
        .astype(str)
        .eq("rbi")
    ].copy()
    if rbi.empty:
        return None
    dates = pd.to_datetime(
        rbi.get("decision_available_date"), errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()
    positions = market_index.searchsorted(dates.to_numpy(), side="left")
    valid = dates.notna() & (positions < len(market_index))
    if not valid.any():
        return None
    daily = pd.DataFrame(
        {
            "date": market_index.take(positions[valid]),
            "macro_risk_score": -pd.to_numeric(
                rbi.loc[valid, "sentiment_score"], errors="coerce"
            ).to_numpy(),
        }
    ).dropna().groupby("date")["macro_risk_score"].mean()
    return daily.reindex(market_index).rolling(21, min_periods=1).mean().to_frame()


def _build_real_rbi_macro_payload(
    *,
    manifest_path: str | Path | None,
    market_index: pd.DatetimeIndex,
    decision_lag: int,
) -> dict[str, object]:
    """Build a decision-lagged real-RBI macro component when local docs exist."""
    path = Path(manifest_path) if manifest_path else DEFAULT_RBI_MANIFEST
    validation = validate_rbi_manifest(path)
    documents = load_real_rbi_corpus(path)
    if documents.empty:
        return {
            "manifest_path": path,
            "manifest_validation": validation,
            "documents": documents,
            "sentences": pd.DataFrame(),
            "scored_sentences": pd.DataFrame(),
            "macro_index": pd.DataFrame(index=market_index),
            "available": False,
        }
    sentences = split_rbi_sentences(documents)
    scored_sentences = score_rbi_sentences(sentences, method="lexicon")
    macro_index = build_macro_stance_index(
        scored_sentences,
        market_index,
        lookback_window=63,
        decision_lag=max(1, int(decision_lag)),
    )
    return {
        "manifest_path": path,
        "manifest_validation": validation,
        "documents": documents,
        "sentences": sentences,
        "scored_sentences": scored_sentences,
        "macro_index": macro_index,
        "available": not macro_index.empty,
    }


def _component_label_from_score(value: object) -> str:
    score = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(score):
        return "insufficient_nlp_data"
    if float(score) >= 0.15:
        return "nlp_risk_off"
    if float(score) <= -0.15:
        return "nlp_risk_on"
    return "nlp_neutral"


def _source_mix_diagnostics(composite: pd.DataFrame) -> pd.DataFrame:
    if composite.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "source_mix",
                "composite_nlp_label",
                "rbi_macro_label",
                "news_geopolitical_label",
                "rbi_news_agree",
            ]
        )
    frame = pd.DataFrame(index=composite.index)
    frame.index.name = "date"
    frame["source_mix"] = (
        composite.get("decision_source_mix", pd.Series(index=composite.index))
        .fillna("none")
        .astype(str)
    )
    frame["composite_nlp_label"] = composite[
        "decision_composite_nlp_label"
    ].fillna("insufficient_nlp_data")
    frame["rbi_macro_score"] = composite.get("decision_rbi_macro_risk_score")
    frame["news_geopolitical_score"] = composite.get(
        "decision_news_geopolitical_risk_score"
    )
    frame["rbi_macro_label"] = [
        _component_label_from_score(value) for value in frame["rbi_macro_score"]
    ]
    frame["news_geopolitical_label"] = [
        _component_label_from_score(value)
        for value in frame["news_geopolitical_score"]
    ]
    frame["rbi_news_agree"] = np.where(
        frame["source_mix"].eq("rbi_and_news"),
        frame["rbi_macro_label"].eq(frame["news_geopolitical_label"]),
        np.nan,
    )
    return frame.reset_index()


def _score_records(
    records: pd.DataFrame,
    scoring_config: dict[str, object],
) -> tuple[pd.DataFrame, str]:
    finbert_enabled = bool(scoring_config.get("finbert_enabled", False))
    method = str(scoring_config.get("method", "lexicon")).lower()
    if finbert_enabled or method == "finbert":
        scored = score_with_finbert(
            records,
            model_name=str(
                scoring_config.get("finbert_model", "ProsusAI/finbert")
            ),
            local_files_only=True,
        )
        used = (
            "FinBERT local model"
            if not scored.empty
            and scored.get(
                "scoring_method_used",
                pd.Series("", index=scored.index),
            ).eq("finbert").all()
            else "FinBERT unavailable; lexicon fallback"
        )
        return scored, used
    scored = score_sentiment_records(records, method="lexicon")
    scored["scoring_method_used"] = "lexicon"
    scored["fallback_used"] = False
    scored["fallback_reason"] = pd.NA
    scored["finbert_label"] = pd.NA
    scored["finbert_score"] = np.nan
    return scored, "Lexicon configured; FinBERT not used"


def _percentage(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if not np.isfinite(number) else f"{number:.1%}"


def _diagnostic_rows(
    coverage: dict[str, object],
    *,
    reaction_rate: float,
    max_reaction_rate: float,
    valid_decision_label_dates: int = 0,
) -> pd.DataFrame:
    rows = [
        {
            "metric": "record_count",
            "actual": coverage["record_count"],
            "threshold": coverage["threshold_min_records"],
            "passes": coverage["record_count"]
            >= coverage["threshold_min_records"],
        },
        {
            "metric": "distinct_publication_dates",
            "actual": coverage["distinct_publication_dates"],
            "threshold": coverage["threshold_min_distinct_dates"],
            "passes": coverage["distinct_publication_dates"]
            >= coverage["threshold_min_distinct_dates"],
        },
        {
            "metric": "decision_label_coverage",
            "actual": coverage["decision_label_coverage"],
            "threshold": coverage["threshold_min_coverage_ratio"],
            "passes": coverage["decision_label_coverage"]
            >= coverage["threshold_min_coverage_ratio"],
        },
        {
            "metric": "provider_coverage",
            "actual": coverage["provider_coverage"],
            "threshold": 1.0,
            "passes": coverage["provider_coverage"] >= 1.0,
        },
        {
            "metric": "reaction_warning_rate",
            "actual": reaction_rate,
            "threshold": max_reaction_rate,
            "passes": reaction_rate <= max_reaction_rate,
        },
        {
            "metric": "valid_decision_label_dates",
            "actual": int(valid_decision_label_dates),
            "threshold": 1,
            "passes": int(valid_decision_label_dates) > 0,
        },
        {
            "metric": "source_family_count",
            "actual": coverage.get("source_family_count", 0),
            "threshold": 2,
            "passes": int(coverage.get("source_family_count", 0) or 0) >= 2,
        },
    ]
    return pd.DataFrame(rows)


def _signal_construction_diagnostics(
    *,
    eligible_records: pd.DataFrame,
    scored_records: pd.DataFrame,
    composite_records: pd.DataFrame,
    daily_signal: pd.DataFrame,
) -> pd.DataFrame:
    sentiment_scores = pd.to_numeric(
        scored_records.get(
            "sentiment_score",
            pd.Series(np.nan, index=scored_records.index),
        ),
        errors="coerce",
    )
    valid_labels = (
        daily_signal.get(
            "decision_nlp_label",
            pd.Series(dtype="string"),
        )
        .fillna("insufficient_nlp_data")
        .astype(str)
        .isin(VALID_COMPOSITE_NLP_LABELS)
    )
    raw_valid_labels = (
        daily_signal.get("raw_nlp_label", pd.Series(dtype="string"))
        .fillna("insufficient_nlp_data")
        .astype(str)
        .isin(VALID_COMPOSITE_NLP_LABELS)
    )
    rolling = pd.to_numeric(
        daily_signal.get(
            "rolling_article_day_coverage",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )
    reason_counts = (
        daily_signal.get(
            "insufficient_reason",
            pd.Series(dtype="string"),
        )
        .replace("", pd.NA)
        .dropna()
        .astype(str)
        .value_counts()
        .to_dict()
    )
    rows = [
        ("records_entering_scoring", int(len(eligible_records))),
        ("records_with_sentiment_scores", int(sentiment_scores.notna().sum())),
        ("records_entering_composite_index", int(len(composite_records))),
        (
            "article_days_with_at_least_one_valid_record",
            int(
                pd.to_numeric(
                    daily_signal.get(
                        "valid_record_count",
                        pd.Series(dtype=float),
                    ),
                    errors="coerce",
                )
                .fillna(0)
                .gt(0)
                .sum()
            ),
        ),
        (
            "rolling_coverage_by_day_mean",
            float(rolling.mean()) if not rolling.dropna().empty else 0.0,
        ),
        (
            "rolling_coverage_by_day_max",
            float(rolling.max()) if not rolling.dropna().empty else 0.0,
        ),
        ("raw_daily_nlp_label_count", int(raw_valid_labels.sum())),
        ("decision_lagged_nlp_label_count", int(valid_labels.sum())),
        (
            "insufficient_reason_counts",
            json.dumps(reason_counts, sort_keys=True),
        ),
    ]
    return pd.DataFrame(rows, columns=["diagnostic", "value"])


def _render_coverage_chart(
    diagnostics: pd.DataFrame,
    chart_path: Path,
) -> None:
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = diagnostics.loc[
        diagnostics["metric"].ne("reaction_warning_rate")
    ].copy()
    denominator = pd.to_numeric(plot_df["threshold"], errors="coerce").replace(
        0, np.nan
    )
    plot_df["threshold_attainment"] = (
        pd.to_numeric(plot_df["actual"], errors="coerce") / denominator
    ).fillna(0).clip(0, 1.2)
    plot_df["metric_label"] = plot_df["metric"].map(
        {
            "record_count": "Real records",
            "distinct_publication_dates": "Publication dates",
            "decision_label_coverage": "Decision coverage",
            "provider_coverage": "Enabled provider coverage",
            "valid_decision_label_dates": "Valid decision-label dates",
            "source_family_count": "Source families",
        }
    )
    plot_df = plot_df.sort_values("threshold_attainment")
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "grid.color": TOKENS["grid"],
            "font.family": "sans-serif",
        },
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.barplot(
        data=plot_df,
        x="threshold_attainment",
        y="metric_label",
        color=TOKENS["blue_base"],
        edgecolor=TOKENS["blue_dark"],
        linewidth=1.0,
        ax=ax,
    )
    ax.scatter(
        plot_df["threshold_attainment"],
        plot_df["metric_label"],
        color=TOKENS["blue_dark"],
        s=28,
        zorder=4,
    )
    for row in plot_df.itertuples(index=False):
        ax.text(
            max(0.015, float(row.threshold_attainment) + 0.015),
            row.metric_label,
            f"{float(row.threshold_attainment):.0%}",
            va="center",
            ha="left",
            fontsize=8,
            color=TOKENS["ink"],
        )
    ax.axvline(1.0, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.set_xlabel("Share of minimum threshold attained")
    ax.set_ylabel("")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_xlim(0, 1.2)
    ax.set_title("")
    fig.subplots_adjust(top=0.78, left=0.24, right=0.96, bottom=0.16)
    left = ax.get_position().x0
    fig.text(
        left,
        0.97,
        "Real NLP evidence relative to validation thresholds",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.90,
        textwrap.fill(
            "Threshold attainment for real records, distinct publication "
            "dates, decision-label coverage, and enabled-provider coverage.",
            108,
        ),
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    sns.despine(ax=ax)
    fig.savefig(
        chart_path,
        dpi=180,
        bbox_inches="tight",
        facecolor=TOKENS["surface"],
    )
    plt.close(fig)


def _build_report(
    output: Path,
    *,
    summary: dict[str, object],
    question_rows: list[tuple[str, str]],
    diagnostics: pd.DataFrame,
) -> None:
    question_table = pd.DataFrame(
        question_rows, columns=["question", "answer"]
    ).to_html(index=False, border=0, escape=True)
    diagnostic_table = diagnostics.to_html(index=False, border=0)
    insufficient = summary["verdict"] == VERDICT_C
    monitoring_only = summary["verdict"] == VERDICT_B
    findings_heading = (
        "Real-data thresholds are not yet met"
        if insufficient
        else "Real news monitoring signal is active"
        if monitoring_only
        else "Real-data evidence meets the configured coverage gate"
    )
    findings_text = (
        "Fixture and placeholder rows were retained for diagnostics but "
        "excluded from empirical coverage, source-quality distributions, "
        "composite validation, and regime-agreement claims."
        if insufficient
        else (
            "Valid decision-lagged news labels are available for monitoring, "
            "but source diversity and allocation-testing evidence remain limited."
        )
        if monitoring_only
        else (
            "Only records that passed real-data provenance, ex-ante timestamp, "
            "and reaction-warning filters entered composite validation."
        )
    )
    limitation_heading = (
        "Insufficient records prevent empirical claims"
        if insufficient
        else "Monitoring signal is not allocation evidence"
        if monitoring_only
        else "Passing thresholds does not establish predictiveness"
    )
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 4A.6 Real NLP Signal Validation</title>
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
  </style>
</head>
<body><main data-report-audience="technical">
  <header data-contract-section="title">
    <h1>Phase 4A.6 Real NLP Signal Validation</h1>
  </header>
  <section data-contract-section="technical-summary">
    <h2>Technical Summary</h2>
    <div class="status"><strong>{escape(str(summary["verdict"]))}</strong><br>{escape(str(summary["caveat"]))}</div>
    <div class="metric-grid">
      <div class="metric"><span>Real records</span><strong>{summary["real_record_count"]}</strong></div>
      <div class="metric"><span>Coverage quality</span><strong>{escape(str(summary["coverage_quality"]).title())}</strong></div>
      <div class="metric"><span>Reaction warnings</span><strong>{_percentage(summary["reaction_warning_rate"])}</strong></div>
      <div class="metric"><span>Decision coverage</span><strong>{_percentage(summary["coverage_ratio"])}</strong></div>
      <div class="metric"><span>Valid label dates</span><strong>{summary["valid_decision_label_dates"]}</strong></div>
      <div class="metric"><span>Allocation impact</span><strong>None</strong></div>
    </div>
  </section>
  <section data-contract-section="key-findings">
    <h2>{escape(findings_heading)}</h2>
    <p>The harness found {summary["real_record_count"]} real provider records. {escape(findings_text)}</p>
    <p>GDELT-only NLP is a real news monitoring signal, not an allocation signal. Source mix: {escape(json.dumps(summary.get("source_mix", {}), sort_keys=True))}. Insufficient reasons: {escape(json.dumps(summary.get("insufficient_reason_counts", {}), sort_keys=True))}.</p>
    <p>The chart shows each evidence dimension relative to its configured minimum. Values below the reference line are not decision-eligible.</p>
    <figure>
      <img src="charts/coverage_threshold_attainment.png" alt="NLP validation threshold attainment">
      <figcaption>Real-data evidence relative to the configured minimum validation thresholds.</figcaption>
    </figure>
  </section>
  <section data-contract-section="scope-data-and-metric-definitions">
    <h2>Scope, Data, and Metric Definitions</h2>
    <p>Real evidence excludes rows marked by explicit fixture metadata, placeholder domains, or synthetic-test notes. Coverage uses real records only. Decision-label coverage is the share of market sessions with a coverage-gated composite label; source quality evaluates provenance and timestamp fields without market outcomes.</p>
    <div class="table-wrap">{diagnostic_table}</div>
  </section>
  <section data-contract-section="methodology">
    <h2>Ex-Ante Scoring and Validation Method</h2>
    <p>Records are timestamp-validated, reaction-oriented language is flagged, publication availability is lagged, and scoring uses the configured local method. The composite combines available RBI, earnings, and news risk components only after filtering invalid or reaction-warning rows. News-only composite labels are permitted for monitoring, with source diversity marked limited. Quantitative comparison uses lagged rule-based regimes and HMM walk-forward only; full-sample HMM is never used.</p>
    <h3>Required validation questions</h3>
    <div class="table-wrap">{question_table}</div>
  </section>
  <section data-contract-section="limitations-uncertainty-and-robustness-checks">
    <h2>{escape(limitation_heading)}</h2>
    <p>The bundled transcripts and news fixtures validate software behavior, not real signal efficacy. The quantitative comparison history is deterministic and synthetic, used only to exercise safe lagged interfaces. No return, volatility, drawdown, or post-event market reaction is used as an NLP feature. Agreement and pre-stress statistics are exploratory monitoring diagnostics when coverage or source diversity is limited.</p>
  </section>
  <section data-contract-section="recommended-next-steps">
    <h2>Collect governed real text before allocation research</h2>
    <ol>
      <li>Populate reviewed RBI and earnings manifests with genuine publication timestamps and source URLs.</li>
      <li>Enable one cached API provider at a time under documented rate, licensing, and retention rules.</li>
      <li>Re-run shadow monitoring until volume, freshness, source quality, and reaction-warning thresholds are stable.</li>
      <li>Keep NLP commentary-only until a separate future allocation-testing phase is explicitly approved.</li>
    </ol>
  </section>
  <section data-contract-section="further-questions">
    <h2>Further Questions</h2>
    <p>Which provider mix can sustain the minimum daily coverage? How should source-quality thresholds vary by provider type? What out-of-sample period is long enough to evaluate false risk-off warnings without optimizing on future returns?</p>
  </section>
</main></body></html>"""
    (output / "report.html").write_text(report, encoding="utf-8")


def validate_real_nlp_signal(
    *,
    input_records: str | Path = DEFAULT_INPUT,
    start_date="2020-01-01",
    end_date="2026-06-21",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    multi_source_output_dir: str | Path | None = None,
    rbi_manifest_path: str | Path | None = None,
    decision_lag_days: int | None = None,
    min_coverage_ratio: float | None = None,
) -> dict[str, object]:
    """Build a conservative empirical-validation or insufficiency report."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    multi_output = (
        Path(multi_source_output_dir)
        if multi_source_output_dir is not None
        else DEFAULT_MULTI_SOURCE_OUTPUT_DIR
        if output.resolve() == DEFAULT_OUTPUT_DIR.resolve()
        else output.parent / "phase_4a8_multisource_nlp_monitoring"
    )
    multi_output.mkdir(parents=True, exist_ok=True)
    config = load_provider_config()
    intake = validate_nlp_corpus_intake()
    validation_config = dict(config.get("validation", {}))
    lag_days = int(
        decision_lag_days
        if decision_lag_days is not None
        else validation_config.get("decision_lag_days", 1)
    )
    minimum_coverage = float(
        min_coverage_ratio
        if min_coverage_ratio is not None
        else validation_config.get("min_coverage_ratio", 0.20)
    )
    min_records = int(validation_config.get("min_records", 50))
    min_dates = int(validation_config.get("min_distinct_dates", 20))
    max_reaction_rate = float(
        validation_config.get("max_reaction_warning_rate", 0.25)
    )

    records = score_source_quality(_read_records(input_records))
    real_records = records.loc[
        records.get(
            "is_real_provider_data",
            pd.Series(False, index=records.index),
        ).fillna(False)
    ].copy()
    ex_ante = validate_ex_ante_records(real_records)
    ex_ante = flag_reaction_data_leakage(ex_ante)
    ex_ante = apply_publication_lag(ex_ante, lag_days=max(1, lag_days))
    source_quality = score_source_quality(ex_ante)
    reaction_warnings = source_quality.loc[
        source_quality.get(
            "possible_reaction_data",
            pd.Series(False, index=source_quality.index),
        ).fillna(False)
    ].copy()
    eligible = source_quality.loc[
        source_quality.get(
            "is_ex_ante_valid",
            pd.Series(False, index=source_quality.index),
        ).fillna(False)
        & ~source_quality.get(
            "possible_reaction_data",
            pd.Series(False, index=source_quality.index),
        ).fillna(False)
    ].copy()
    scored, finbert_status = _score_records(
        eligible, dict(config.get("scoring", {}))
    )

    returns = _build_market_returns(start_date, end_date)
    market_index = returns.index
    provider = scored.get("provider", pd.Series("", index=scored.index)).astype(
        str
    )
    document_type = scored.get(
        "document_type", pd.Series("", index=scored.index)
    ).astype(str)
    earnings = scored.loc[
        provider.isin({"earnings", "earnings_calls"})
        | document_type.eq("earnings_call")
    ].copy()
    news = scored.loc[
        provider.isin({"gdelt", "alpha_vantage", "alpha_vantage_news"})
        | document_type.isin({"financial_news", "news"})
    ].copy()
    daily_nlp_signal = build_daily_nlp_signal(
        news,
        market_index,
        decision_lag=max(1, lag_days),
    )
    configured_rbi_manifest = (
        rbi_manifest_path
        or config.get("rbi", {}).get("local_manifest_path")
        or DEFAULT_RBI_MANIFEST
    )
    configured_rbi_manifest = Path(configured_rbi_manifest)
    if not configured_rbi_manifest.is_absolute():
        configured_rbi_manifest = (REPO_ROOT / configured_rbi_manifest).resolve()
    rbi_payload = _build_real_rbi_macro_payload(
        manifest_path=configured_rbi_manifest,
        market_index=market_index,
        decision_lag=max(1, lag_days),
    )
    rbi_macro_index = (
        rbi_payload["macro_index"]
        if bool(rbi_payload.get("available"))
        else _rbi_component(scored, market_index)
    )
    composite = build_composite_nlp_risk_index(
        rbi_macro_index=rbi_macro_index,
        earnings_sentiment=earnings,
        news_sentiment=news,
        market_index=market_index,
        decision_lag=max(1, lag_days),
    )
    source_mix_diagnostics = _source_mix_diagnostics(composite)
    rbi_status, rbi_status_diagnostics = build_rbi_corpus_status(
        configured_rbi_manifest
    )

    enabled_rows = pd.DataFrame(config["_validation"]["providers"])
    provider_diag = enabled_rows.rename(columns={"enabled": "configured_enabled"})
    coverage = calculate_nlp_coverage(
        source_quality,
        composite_index=composite,
        provider_diagnostics=provider_diag,
        start_date=start_date,
        end_date=end_date,
        min_coverage_ratio=minimum_coverage,
        min_records=min_records,
        min_distinct_dates=min_dates,
    )
    reaction_rate = (
        float(len(reaction_warnings) / len(source_quality))
        if len(source_quality)
        else 0.0
    )
    decision_labels = composite["decision_composite_nlp_label"].fillna(
        "insufficient_nlp_data"
    ).astype(str)
    valid_decision_mask = decision_labels.isin(VALID_COMPOSITE_NLP_LABELS)
    valid_decision_label_dates = int(valid_decision_mask.sum())
    comparison_allowed = (
        valid_decision_label_dates > 0
        and reaction_rate <= max_reaction_rate
    )
    comparison: dict[str, object]
    if comparison_allowed:
        features = calculate_regime_features(returns)
        rule = lag_regime_labels(
            classify_rule_based_regime(features), lag=1
        )
        try:
            hmm_result = fit_hmm_walk_forward(
                features,
                n_states=2,
                min_train_size=252,
                refit_frequency=126,
                covariance_type="diag",
                decision_lag=1,
            )
            hmm = hmm_result["decision_regimes"].reindex(market_index)
        except Exception:
            hmm = pd.Series("Unknown", index=market_index)
        comparison = compare_composite_nlp_to_regimes(
            composite, rule, hmm
        )
        comparison["exploratory_monitoring_only"] = (
            coverage["coverage_quality"] != "sufficient"
            or bool(coverage.get("source_diversity_limited", True))
        )
    else:
        comparison_table = pd.DataFrame(
            columns=[
                "composite_nlp_label",
                "composite_nlp_risk_score",
                "coverage_score",
                "rule_based_regime",
                "hmm_walk_forward_regime",
                "agreement_rule_based",
                "agreement_hmm",
                "comparison_note",
            ],
        )
        comparison_table.index.name = "date"
        comparison_table["comparison_note"] = (
            "No empirical comparison: zero valid decision-lagged NLP labels."
        )
        comparison = {
            "comparison_table": comparison_table,
            "agreement_with_rule_based": np.nan,
            "agreement_with_hmm_walk_forward": np.nan,
            "pre_stress_warning_count": 0,
            "coverage_ratio": coverage["decision_label_coverage"],
            "predictiveness_claim": False,
            "exploratory_monitoring_only": False,
        }

    quality_distribution = (
        source_quality["source_quality_label"]
        .value_counts()
        .reindex(["high", "medium", "low", "unknown"], fill_value=0)
        .to_dict()
        if "source_quality_label" in source_quality
        else {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    )
    acceptable_quality_share = (
        float(
            source_quality["source_quality_label"].isin(
                {"high", "medium"}
            ).mean()
        )
        if len(source_quality)
        else 0.0
    )
    source_mix_distribution = (
        composite.get("decision_source_mix", pd.Series(dtype="string"))
        .fillna("none")
        .astype(str)
        .value_counts()
        .to_dict()
    )
    multi_source_monitoring = bool(
        source_mix_distribution.get("rbi_and_news", 0)
    )
    records_pass = int(coverage["record_count"]) >= int(min_records)
    dates_pass = int(coverage["distinct_publication_dates"]) >= int(min_dates)
    decision_coverage = float(coverage["decision_label_coverage"])
    decision_ratio_pass = decision_coverage >= float(minimum_coverage)
    decision_positive = decision_coverage > 0
    reaction_ok = reaction_rate <= max_reaction_rate
    if records_pass and dates_pass and decision_positive and reaction_ok:
        verdict = VERDICT_B
    else:
        verdict = VERDICT_C
    caveat = (
        MONITORING_CAVEAT
        if verdict == VERDICT_C
        else (
            "RBI + news NLP is useful for multi-source monitoring only. "
            "Allocation impact: None."
        )
        if multi_source_monitoring
        else MONITORING_ONLY_CAVEAT
    )

    diagnostics = _diagnostic_rows(
        coverage,
        reaction_rate=reaction_rate,
        max_reaction_rate=max_reaction_rate,
        valid_decision_label_dates=valid_decision_label_dates,
    )
    signal_diagnostics = _signal_construction_diagnostics(
        eligible_records=eligible,
        scored_records=scored,
        composite_records=pd.concat(
            [earnings, news],
            ignore_index=True,
            sort=False,
        )
        if not earnings.empty or not news.empty
        else pd.DataFrame(),
        daily_signal=daily_nlp_signal,
    )
    chart_path = output / "charts" / "coverage_threshold_attainment.png"
    _render_coverage_chart(diagnostics, chart_path)
    scored.to_csv(output / "scored_records.csv", index=False)
    daily_nlp_signal.to_csv(output / "daily_nlp_signal.csv", index=False)
    signal_diagnostics.to_csv(
        output / "signal_construction_diagnostics.csv",
        index=False,
    )
    composite.to_csv(output / "composite_nlp_risk_index.csv", index=True)
    comparison["comparison_table"].to_csv(
        output / "nlp_regime_comparison.csv", index=True
    )
    diagnostics.to_csv(output / "coverage_diagnostics.csv", index=False)
    reaction_warnings.to_csv(
        output / "reaction_data_warnings.csv", index=False
    )
    source_quality.to_csv(output / "source_quality.csv", index=False)

    publication = pd.to_datetime(
        source_quality.get(
            "publication_time",
            pd.Series(pd.NaT, index=source_quality.index),
        ),
        errors="coerce",
        utc=True,
    ).dropna()
    record_date_range = (
        f"{publication.min().date().isoformat()} to "
        f"{publication.max().date().isoformat()}"
        if not publication.empty
        else "Unavailable"
    )
    enabled_providers = config["_validation"]["enabled_providers"]
    returned_providers = sorted(
        source_quality.get(
            "provider", pd.Series(dtype="string")
        ).dropna().astype(str).unique().tolist()
    )
    label_distribution = (
        composite["decision_composite_nlp_label"]
        .value_counts()
        .to_dict()
    )
    daily_insufficient_reason_counts = (
        daily_nlp_signal.get(
            "insufficient_reason",
            pd.Series(dtype="string"),
        )
        .replace("", pd.NA)
        .dropna()
        .astype(str)
        .value_counts()
        .to_dict()
    )
    rbi_news_rows = source_mix_diagnostics.loc[
        source_mix_diagnostics.get(
            "source_mix",
            pd.Series(dtype="string"),
        )
        .astype(str)
        .eq("rbi_and_news")
    ].copy()
    rbi_news_agreement = (
        float(rbi_news_rows["rbi_news_agree"].dropna().astype(bool).mean())
        if not rbi_news_rows.empty
        and not rbi_news_rows["rbi_news_agree"].dropna().empty
        else np.nan
    )
    ex_ante_share = (
        float(source_quality["is_ex_ante_valid"].mean())
        if len(source_quality)
        else 0.0
    )
    question_rows = [
        ("1. Which real providers were enabled?", ", ".join(enabled_providers) or "None"),
        ("2. Which providers returned real data?", ", ".join(returned_providers) or "None"),
        (
            "3. How many records were collected?",
            f"{len(records)} input record(s); {len(source_quality)} classified "
            "as real provider data",
        ),
        ("4. What is the date range?", record_date_range),
        ("5. What is the source-quality distribution?", json.dumps(quality_distribution, sort_keys=True)),
        ("6. What share of records passed ex-ante validation?", _percentage(ex_ante_share)),
        ("7. What share were flagged as possible reaction data?", _percentage(reaction_rate)),
        ("8. Was FinBERT used or did scoring fall back?", finbert_status),
        ("9. What is the coverage ratio?", _percentage(coverage["decision_label_coverage"])),
        ("10. What is the composite NLP risk label distribution?", json.dumps(label_distribution, sort_keys=True)),
        ("11. What is the source mix?", json.dumps(source_mix_distribution, sort_keys=True)),
        ("12. How many valid decision-label dates exist?", str(valid_decision_label_dates)),
        ("13. What are the insufficient reasons?", json.dumps(daily_insufficient_reason_counts, sort_keys=True)),
        ("14. What is agreement with HMM regimes?", _percentage(comparison["agreement_with_hmm_walk_forward"])),
        ("15. What is agreement with rule-based regimes?", _percentage(comparison["agreement_with_rule_based"])),
        ("16. Did NLP produce pre-stress warnings?", f"{comparison['pre_stress_warning_count']} warning(s); exploratory monitoring only"),
        ("17. Is evidence sufficient for future allocation testing?", verdict),
    ]
    summary = {
        "generated": date.today().isoformat(),
        "verdict": verdict,
        "caveat": caveat,
        "real_record_count": int(len(source_quality)),
        "coverage_quality": coverage["coverage_quality"],
        "coverage_ratio": coverage["decision_label_coverage"],
        "valid_decision_label_dates": valid_decision_label_dates,
        "source_mix": source_mix_distribution,
        "source_families": coverage.get("source_families", []),
        "source_family_count": coverage.get("source_family_count", 0),
        "source_diversity_limited": coverage.get(
            "source_diversity_limited", True
        ),
        "multi_source_monitoring": multi_source_monitoring,
        "real_rbi_document_count": int(rbi_status["valid_document_count"]),
        "real_news_record_count": int(len(news)),
        "rbi_manual_action_required": bool(
            rbi_status["manual_action_required"]
        ),
        "rbi_news_agreement": rbi_news_agreement,
        "insufficient_reason_counts": daily_insufficient_reason_counts,
        "source_quality_distribution": quality_distribution,
        "reaction_warning_rate": reaction_rate,
        "finbert_status": finbert_status,
        "providers_enabled": enabled_providers,
        "providers_returning_real_data": returned_providers,
        "date_range": record_date_range,
        "agreement_with_hmm_walk_forward": comparison[
            "agreement_with_hmm_walk_forward"
        ],
        "agreement_with_rule_based": comparison[
            "agreement_with_rule_based"
        ],
        "pre_stress_warning_count": comparison[
            "pre_stress_warning_count"
        ],
        "predictiveness_claim": False,
        "allocation_impact": False,
        "exploratory_monitoring_only": bool(
            comparison.get("exploratory_monitoring_only", False)
        ),
        "intake_manual_action_required": bool(
            intake["manual_action_required"]
        ),
        "valid_real_records_by_corpus": (
            intake["valid_real_records_by_corpus"]
        ),
        "corpus_sufficiency_status": (
            "ready"
            if intake["all_corpora_ready"]
            else "manual_action_required"
        ),
    }
    availability_statement = (
        "Real provider data unavailable; validation harness produced an "
        "insufficiency report without empirical or predictive claims."
        if not len(source_quality)
        else (
            "Real provider records were evaluated under ex-ante, reaction, "
            "quality, freshness, and coverage controls. Passing any threshold "
            "does not establish predictiveness."
        )
    )
    summary_md = f"""# Phase 4A.6 Real NLP Signal Validation

Generated: {summary["generated"]}

## Technical summary

**{verdict}**

{caveat}

- Real provider records: {summary["real_record_count"]}.
- Real RBI documents: {summary["real_rbi_document_count"]}.
- Real GDELT/news records: {summary["real_news_record_count"]}.
- Coverage quality: {summary["coverage_quality"]}.
- Decision-label coverage: {_percentage(summary["coverage_ratio"])}.
- Valid decision-label dates: {summary["valid_decision_label_dates"]}.
- Source mix: `{json.dumps(summary["source_mix"], sort_keys=True)}`.
- Source families: `{json.dumps(summary["source_families"], sort_keys=True)}`.
- RBI/news agreement: {_percentage(summary["rbi_news_agreement"])}.
- Multi-source monitoring active: {'Yes' if summary["multi_source_monitoring"] else 'No'}.
- RBI manual action required: {'Yes' if summary["rbi_manual_action_required"] else 'No'}.
- Insufficient reasons: `{json.dumps(summary["insufficient_reason_counts"], sort_keys=True)}`.
- Source-quality distribution: `{json.dumps(quality_distribution, sort_keys=True)}`.
- Reaction-warning rate: {_percentage(reaction_rate)}.
- Scoring status: {finbert_status}.
- HMM walk-forward agreement: {_percentage(summary["agreement_with_hmm_walk_forward"])}.
- Rule-based agreement: {_percentage(summary["agreement_with_rule_based"])}.
- Corpus intake status: {summary["corpus_sufficiency_status"]}.
- Manual intake action required: {'Yes' if summary["intake_manual_action_required"] else 'No'}.

{availability_statement}

GDELT-only NLP is a real news monitoring signal, not an allocation signal.
"""
    (output / "summary.md").write_text(summary_md, encoding="utf-8")
    omitted_visuals = (
        "Source-quality distribution and regime-agreement charts were omitted "
        "because zero real records would make them analytically empty."
        if not len(source_quality)
        else (
            "Additional distribution charts were omitted to keep the report "
            "focused on the configured evidence gate and exact audit tables."
        )
    )
    source_notes = f"""# Phase 4A.6 Source Notes

Generated: {summary["generated"]}

## Report contract

- Delivery mode: portable technical HTML.
- Evidence scope: real provider records only; explicit fixtures and placeholders excluded.
- Scoring: {finbert_status}.
- Market comparison history: deterministic synthetic data used only to exercise lagged interfaces.
- Full-sample HMM: prohibited.
- Market reaction inputs: prohibited; warning phrases are filtered before composite scoring.
- Allocation, scoring, gates, confidence, and backtest impact: none.
- Corpus intake manual action required: {summary["intake_manual_action_required"]}.
- Daily NLP signal: `daily_nlp_signal.csv`.
- Scored records: `scored_records.csv`.
- Construction diagnostics: `signal_construction_diagnostics.csv`.
- GDELT/news-only interpretation: real monitoring signal only; not allocation-ready evidence.

## Chart map

- `coverage_threshold_attainment.png`: ranked horizontal bars comparing real evidence with configured minimum thresholds; supports the insufficiency verdict.

## Required structure mapping

- Technical summary: verdict, caveat, real record count, coverage, and reaction warnings.
- Key findings: threshold-attainment evidence.
- Scope/data/definitions: real-data provenance, coverage, and source quality.
- Methodology: ex-ante validation, lagging, scoring, composite construction, and safe regime comparison.
- Limitations/robustness: fixtures excluded, synthetic comparator history, no predictive claims.
- Recommended next steps: governed collection and shadow monitoring.
- Further questions: source mix, quality thresholds, and out-of-sample horizon.

## Omitted visuals

- {omitted_visuals}
"""
    (output / "source_notes.md").write_text(source_notes, encoding="utf-8")
    _build_report(
        output,
        summary=summary,
        question_rows=question_rows,
        diagnostics=diagnostics,
    )
    rbi_status_diagnostics.to_csv(
        multi_output / "rbi_corpus_status.csv",
        index=False,
    )
    rbi_payload["scored_sentences"].to_csv(
        multi_output / "rbi_sentence_scores.csv",
        index=False,
    )
    rbi_payload["macro_index"].to_csv(
        multi_output / "rbi_macro_index.csv",
        index=True,
    )
    news.to_csv(multi_output / "scored_news_records.csv", index=False)
    daily_nlp_signal.to_csv(multi_output / "daily_nlp_signal.csv", index=False)
    source_mix_diagnostics.to_csv(
        multi_output / "source_mix_diagnostics.csv",
        index=False,
    )
    comparison["comparison_table"].to_csv(
        multi_output / "multi_source_nlp_comparison.csv",
        index=True,
    )
    phase48_questions = [
        ("1. How many real RBI documents are available?", str(summary["real_rbi_document_count"])),
        ("2. How many real GDELT/news records are available?", str(summary["real_news_record_count"])),
        ("3. What source mix is active?", json.dumps(summary["source_mix"], sort_keys=True)),
        (
            "4. What is the RBI macro stance signal?",
            "Available in rbi_macro_index.csv"
            if summary["real_rbi_document_count"]
            else "Unavailable: no valid real RBI documents",
        ),
        (
            "5. What is the news/geopolitical signal?",
            "Available in daily_nlp_signal.csv",
        ),
        ("6. How often do RBI and news agree?", _percentage(summary["rbi_news_agreement"])),
        ("7. How much decision-label coverage exists?", _percentage(summary["coverage_ratio"])),
        ("8. Is the NLP layer monitoring-only or allocation-ready?", "Monitoring-only; allocation impact is None."),
    ]
    phase48_table = pd.DataFrame(
        phase48_questions,
        columns=["question", "answer"],
    ).to_html(index=False, border=0, escape=True)
    phase48_summary = f"""# Phase 4A.8 Real RBI + News Multi-Source NLP Monitoring

Generated: {summary["generated"]}

## Technical summary

**{summary["verdict"]}**

{summary["caveat"]}

- Real RBI documents: {summary["real_rbi_document_count"]}.
- Real GDELT/news records: {summary["real_news_record_count"]}.
- Source mix: `{json.dumps(summary["source_mix"], sort_keys=True)}`.
- Decision-label coverage: {_percentage(summary["coverage_ratio"])}.
- Valid decision-label dates: {summary["valid_decision_label_dates"]}.
- RBI/news agreement: {_percentage(summary["rbi_news_agreement"])}.
- Multi-source monitoring active: {'Yes' if summary["multi_source_monitoring"] else 'No'}.
- Allocation impact: None.

The NLP layer is useful for multi-source monitoring if RBI and news are both
present, but it still does not affect allocation.
"""
    (multi_output / "summary.md").write_text(phase48_summary, encoding="utf-8")
    phase48_report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Phase 4A.8 Multi-Source NLP Monitoring</title>
  <style>
    body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 40px auto; max-width: 960px; color: #1f2430; }}
    .status {{ border-left: 4px solid #2e4780; background: #eaf1fe; padding: 14px 18px; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; }}
    th, td {{ border-bottom: 1px solid #e6e8f0; padding: 9px 11px; text-align: left; }}
    th {{ background: #f4f7fd; color: #2e4780; }}
  </style>
</head>
<body>
  <h1>Phase 4A.8 Real RBI + News Multi-Source NLP Monitoring</h1>
  <div class="status"><strong>{escape(str(summary["verdict"]))}</strong><br>{escape(str(summary["caveat"]))}</div>
  <p>The NLP layer is useful for multi-source monitoring if RBI and news are both present, but it still does not affect allocation.</p>
  {phase48_table}
  <h2>Artifacts</h2>
  <ul>
    <li>rbi_corpus_status.csv</li>
    <li>rbi_sentence_scores.csv</li>
    <li>rbi_macro_index.csv</li>
    <li>scored_news_records.csv</li>
    <li>daily_nlp_signal.csv</li>
    <li>source_mix_diagnostics.csv</li>
    <li>multi_source_nlp_comparison.csv</li>
  </ul>
</body>
</html>"""
    (multi_output / "report.html").write_text(phase48_report, encoding="utf-8")
    phase48_notes = f"""# Phase 4A.8 Source Notes

Generated: {summary["generated"]}

- RBI corpus manifest: `{configured_rbi_manifest}`.
- RBI manual action required: {summary["rbi_manual_action_required"]}.
- Real GDELT/news input: `{input_records}`.
- NLP remains monitoring/commentary only.
- Allocation, strategy scoring, evidence gates, confidence, and backtests are unchanged.
- No web content is fetched by the RBI bootstrap/import/status tooling.
"""
    (multi_output / "source_notes.md").write_text(
        phase48_notes,
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "coverage": coverage,
        "coverage_diagnostics": diagnostics,
        "signal_construction_diagnostics": signal_diagnostics,
        "rbi_corpus_status": rbi_status,
        "rbi_corpus_status_diagnostics": rbi_status_diagnostics,
        "rbi_macro_payload": rbi_payload,
        "source_quality": source_quality,
        "scored_records": scored,
        "daily_nlp_signal": daily_nlp_signal,
        "source_mix_diagnostics": source_mix_diagnostics,
        "composite_index": composite,
        "comparison": comparison,
        "reaction_data_warnings": reaction_warnings,
        "report_path": output / "report.html",
        "multi_source_report_path": multi_output / "report.html",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate collected real NLP signal coverage and quality."
    )
    parser.add_argument("--input-records", default=str(DEFAULT_INPUT))
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-06-21")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--multi-source-output-dir", default=None)
    parser.add_argument("--rbi-manifest-path", default=None)
    parser.add_argument("--decision-lag-days", type=int, default=None)
    parser.add_argument("--min-coverage-ratio", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_real_nlp_signal(
        input_records=args.input_records,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        multi_source_output_dir=args.multi_source_output_dir,
        rbi_manifest_path=args.rbi_manifest_path,
        decision_lag_days=args.decision_lag_days,
        min_coverage_ratio=args.min_coverage_ratio,
    )
    summary = result["summary"]
    print(f"Validation verdict: {summary['verdict']}")
    print(f"Real records: {summary['real_record_count']}")
    print(f"Coverage quality: {summary['coverage_quality']}")
    print(f"Decision-label coverage: {summary['coverage_ratio']:.1%}")
    print(f"Valid decision-label dates: {summary['valid_decision_label_dates']}")
    print(f"Report: {result['report_path'].resolve()}")
    print(f"Phase 4A.8 report: {result['multi_source_report_path'].resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
