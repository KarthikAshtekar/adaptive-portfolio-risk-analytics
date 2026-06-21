"""Empirical validation runner for a locally stored real-RBI corpus."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .macro_index import build_macro_stance_index
from .macro_regime_comparison import compare_macro_to_regimes
from .rbi_corpus_builder import load_real_rbi_corpus
from .rbi_processing import split_rbi_documents_into_sentences
from .rbi_scoring import score_rbi_sentences


EMPIRICAL_OUTPUT_FILES = (
    "rbi_documents.csv",
    "rbi_sentence_scores.csv",
    "macro_stance_index.csv",
    "macro_regime_comparison.csv",
    "disagreement_dates.csv",
    "coverage_diagnostics.csv",
    "corpus_diagnostics.csv",
)


def _market_index(market_returns: object) -> pd.DatetimeIndex:
    if isinstance(market_returns, pd.DatetimeIndex):
        index = market_returns
    elif isinstance(market_returns, (pd.Series, pd.DataFrame)):
        index = market_returns.index
    else:
        raise TypeError(
            "market_returns must be a Series, DataFrame, or DatetimeIndex"
        )
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError("market_returns must have a non-empty DatetimeIndex")
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(index).sort_values().drop_duplicates()


def _add_metadata(
    frame: pd.DataFrame,
    *,
    scoring_method: str,
    decision_lag: int,
    lookback_window: int,
) -> pd.DataFrame:
    result = frame.copy()
    result["corpus_type"] = "real_rbi"
    result["scoring_method"] = scoring_method
    result["decision_lag"] = int(decision_lag)
    result["lookback_window"] = int(lookback_window)
    return result


def _coverage_diagnostics(
    macro_index: pd.DataFrame,
    documents: pd.DataFrame,
) -> pd.DataFrame:
    covered = (
        macro_index["decision_macro_label"]
        .astype(str)
        .ne("insufficient_macro_data")
    )
    publication_days = (
        macro_index.get(
            "document_count",
            pd.Series(0, index=macro_index.index),
        )
        .gt(0)
    )
    dates = pd.to_datetime(
        documents.get("publication_date", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
    ).dropna()
    rows = [
        {"metric": "market_session_count", "value": int(len(macro_index))},
        {
            "metric": "covered_decision_session_count",
            "value": int(covered.sum()),
        },
        {
            "metric": "decision_coverage_ratio",
            "value": float(covered.mean()) if len(covered) else 0.0,
        },
        {
            "metric": "publication_window_session_count",
            "value": int(publication_days.sum()),
        },
        {
            "metric": "valid_document_count",
            "value": int(len(documents)),
        },
        {
            "metric": "date_start",
            "value": dates.min().date().isoformat() if not dates.empty else "",
        },
        {
            "metric": "date_end",
            "value": dates.max().date().isoformat() if not dates.empty else "",
        },
    ]
    return pd.DataFrame(rows)


def run_rbi_empirical_validation(
    manifest_path,
    market_returns,
    rule_based_regimes,
    hmm_regimes,
    output_dir,
    scoring_method: str = "lexicon",
    lookback_window: int = 63,
    decision_lag: int = 1,
) -> dict[str, object]:
    """Run the real-RBI confirmation pipeline and persist audit-ready CSVs."""
    if int(decision_lag) < 1:
        raise ValueError("decision_lag must be at least 1")
    index = _market_index(market_returns)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    documents = load_real_rbi_corpus(manifest_path)
    validation = documents.attrs.get("manifest_validation", {})
    sentences = split_rbi_documents_into_sentences(documents)
    scores = score_rbi_sentences(sentences, method=scoring_method)
    macro_index = build_macro_stance_index(
        scores,
        index,
        lookback_window=int(lookback_window),
        decision_lag=int(decision_lag),
    )
    comparison = compare_macro_to_regimes(
        macro_index,
        rule_based_regimes,
        hmm_regimes,
    )
    coverage = _coverage_diagnostics(macro_index, documents)
    corpus_diagnostics = validation.get(
        "diagnostics",
        pd.DataFrame(columns=["metric", "category", "value"]),
    )

    outputs = {
        "rbi_documents": _add_metadata(
            documents,
            scoring_method=scoring_method,
            decision_lag=decision_lag,
            lookback_window=lookback_window,
        ),
        "rbi_sentence_scores": _add_metadata(
            scores,
            scoring_method=scoring_method,
            decision_lag=decision_lag,
            lookback_window=lookback_window,
        ),
        "macro_stance_index": _add_metadata(
            macro_index,
            scoring_method=scoring_method,
            decision_lag=decision_lag,
            lookback_window=lookback_window,
        ),
        "macro_regime_comparison": _add_metadata(
            comparison["comparison_table"],
            scoring_method=scoring_method,
            decision_lag=decision_lag,
            lookback_window=lookback_window,
        ),
        "disagreement_dates": _add_metadata(
            comparison["dates_of_major_disagreement"],
            scoring_method=scoring_method,
            decision_lag=decision_lag,
            lookback_window=lookback_window,
        ),
        "coverage_diagnostics": _add_metadata(
            coverage,
            scoring_method=scoring_method,
            decision_lag=decision_lag,
            lookback_window=lookback_window,
        ),
        "corpus_diagnostics": _add_metadata(
            corpus_diagnostics,
            scoring_method=scoring_method,
            decision_lag=decision_lag,
            lookback_window=lookback_window,
        ),
    }
    for key, frame in outputs.items():
        frame.to_csv(output / f"{key}.csv", index=True)

    return {
        **outputs,
        "comparison": comparison,
        "manifest_validation": validation,
        "invalid_documents": validation.get(
            "invalid_documents",
            pd.DataFrame(),
        ),
        "output_dir": output,
        "corpus_type": "real_rbi",
        "scoring_method": scoring_method,
        "decision_lag": int(decision_lag),
        "lookback_window": int(lookback_window),
        "corpus_sufficiency_status": (
            "ready" if not documents.empty else "manual_action_required"
        ),
        "manual_action_required": bool(documents.empty),
    }
