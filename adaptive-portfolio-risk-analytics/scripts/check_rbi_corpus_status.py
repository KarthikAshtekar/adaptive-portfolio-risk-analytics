"""Report governed RBI corpus readiness for multi-source NLP monitoring."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sentiment import validate_rbi_manifest  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "data" / "sentiment" / "rbi_real" / "manifest.csv"
DEFAULT_FETCH_DIAGNOSTICS = (
    REPO_ROOT / "outputs" / "reports" / "rbi_official_fetcher" / "fetch_diagnostics.csv"
)
MIN_VALID_DOCUMENTS = 10
MIN_DISTINCT_PUBLICATION_DATES = 6
MIN_DOCUMENT_TYPES = 2
MIN_POLICY_CORE_DOCUMENTS = 4


def _fetch_index_page_diagnostics(path: Path = DEFAULT_FETCH_DIAGNOSTICS) -> dict[str, object]:
    if not path.is_file():
        return {
            "skipped_index_pages": 0,
            "index_page_reason_counts": {},
        }
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {
            "skipped_index_pages": 0,
            "index_page_reason_counts": {},
        }
    if frame.empty or "is_index_page" not in frame:
        return {
            "skipped_index_pages": 0,
            "index_page_reason_counts": {},
        }
    index_mask = frame["is_index_page"].astype(str).str.lower().isin({"true", "1", "yes"})
    skipped = frame.loc[index_mask].copy()
    reason_counts = (
        skipped.get("index_page_reason", pd.Series(dtype="string"))
        .fillna("")
        .astype(str)
        .replace("", "unspecified")
        .value_counts()
        .to_dict()
        if not skipped.empty
        else {}
    )
    return {
        "skipped_index_pages": int(len(skipped)),
        "index_page_reason_counts": reason_counts,
    }


def build_rbi_corpus_status(
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Return status metrics and a long-form diagnostics table."""
    validation = validate_rbi_manifest(manifest_path)
    valid = validation["valid_documents"].copy()
    invalid = validation["invalid_documents"].copy()
    publication_dates = pd.to_datetime(
        valid.get("publication_date", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
    ).dropna()
    document_type_counts = (
        valid.get("document_type", pd.Series(dtype="string"))
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .to_dict()
        if not valid.empty
        else {}
    )
    index_diagnostics = _fetch_index_page_diagnostics()
    document_type_count = len(document_type_counts)
    mpc_minutes_count = int(document_type_counts.get("mpc_minutes", 0))
    monetary_policy_statement_count = int(document_type_counts.get("monetary_policy_statement", 0))
    governor_speech_count = int(document_type_counts.get("governor_speech", 0))
    financial_stability_report_count = int(
        document_type_counts.get("financial_stability_report", 0)
    )
    policy_core_documents = int(mpc_minutes_count + monetary_policy_statement_count)
    status = {
        "manifest_path": str(Path(manifest_path)),
        "valid_document_count": int(len(valid)),
        "invalid_document_count": int(len(invalid)),
        "distinct_publication_dates": int(publication_dates.dt.normalize().nunique()),
        "document_type_count": int(document_type_count),
        "document_type_counts": document_type_counts,
        "mpc_minutes_count": mpc_minutes_count,
        "monetary_policy_statement_count": monetary_policy_statement_count,
        "governor_speech_count": governor_speech_count,
        "financial_stability_report_count": financial_stability_report_count,
        "policy_core_documents": policy_core_documents,
        "date_start": publication_dates.min().date().isoformat()
        if not publication_dates.empty
        else "",
        "date_end": publication_dates.max().date().isoformat()
        if not publication_dates.empty
        else "",
        "word_count": int(validation["summary"].get("total_word_count", 0)),
        "sentence_count": int(validation["summary"].get("total_sentence_count", 0)),
        "skipped_index_pages": int(index_diagnostics["skipped_index_pages"]),
        "index_page_reason_counts": index_diagnostics["index_page_reason_counts"],
    }
    status["minimum_requirements_passed"] = bool(
        status["valid_document_count"] >= MIN_VALID_DOCUMENTS
        and status["distinct_publication_dates"] >= MIN_DISTINCT_PUBLICATION_DATES
        and status["document_type_count"] >= MIN_DOCUMENT_TYPES
        and status["policy_core_documents"] >= MIN_POLICY_CORE_DOCUMENTS
    )
    status["manual_action_required"] = not bool(status["minimum_requirements_passed"])
    rows = [
        {
            "metric": "valid_document_count",
            "actual": status["valid_document_count"],
            "threshold": MIN_VALID_DOCUMENTS,
            "passes": status["valid_document_count"] >= MIN_VALID_DOCUMENTS,
        },
        {
            "metric": "invalid_document_count",
            "actual": status["invalid_document_count"],
            "threshold": 0,
            "passes": status["invalid_document_count"] == 0,
        },
        {
            "metric": "distinct_publication_dates",
            "actual": status["distinct_publication_dates"],
            "threshold": MIN_DISTINCT_PUBLICATION_DATES,
            "passes": status["distinct_publication_dates"] >= MIN_DISTINCT_PUBLICATION_DATES,
        },
        {
            "metric": "document_type_count",
            "actual": status["document_type_count"],
            "threshold": MIN_DOCUMENT_TYPES,
            "passes": status["document_type_count"] >= MIN_DOCUMENT_TYPES,
        },
        {
            "metric": "policy_core_documents",
            "actual": status["policy_core_documents"],
            "threshold": MIN_POLICY_CORE_DOCUMENTS,
            "passes": status["policy_core_documents"] >= MIN_POLICY_CORE_DOCUMENTS,
        },
        {
            "metric": "mpc_minutes_count",
            "actual": status["mpc_minutes_count"],
            "threshold": "",
            "passes": True,
        },
        {
            "metric": "monetary_policy_statement_count",
            "actual": status["monetary_policy_statement_count"],
            "threshold": "",
            "passes": True,
        },
        {
            "metric": "governor_speech_count",
            "actual": status["governor_speech_count"],
            "threshold": "",
            "passes": True,
        },
        {
            "metric": "financial_stability_report_count",
            "actual": status["financial_stability_report_count"],
            "threshold": "",
            "passes": True,
        },
        {
            "metric": "word_count",
            "actual": status["word_count"],
            "threshold": 1,
            "passes": status["word_count"] > 0,
        },
        {
            "metric": "sentence_count",
            "actual": status["sentence_count"],
            "threshold": 1,
            "passes": status["sentence_count"] > 0,
        },
        {
            "metric": "skipped_index_pages",
            "actual": status["skipped_index_pages"],
            "threshold": "",
            "passes": True,
        },
        {
            "metric": "minimum_requirements_passed",
            "actual": status["minimum_requirements_passed"],
            "threshold": True,
            "passes": status["minimum_requirements_passed"],
        },
        {
            "metric": "manual_action_required",
            "actual": status["manual_action_required"],
            "threshold": False,
            "passes": not status["manual_action_required"],
        },
    ]
    for document_type, count in document_type_counts.items():
        rows.append(
            {
                "metric": "document_type_count_detail",
                "category": document_type,
                "actual": int(count),
                "threshold": "",
                "passes": True,
            }
        )
    for reason, count in status["index_page_reason_counts"].items():
        rows.append(
            {
                "metric": "index_page_reason_count",
                "category": reason,
                "actual": int(count),
                "threshold": "",
                "passes": True,
            }
        )
    diagnostics = pd.DataFrame(rows)
    for column in ("metric", "category", "actual", "threshold", "passes"):
        if column not in diagnostics:
            diagnostics[column] = ""
    return status, diagnostics.loc[:, ["metric", "category", "actual", "threshold", "passes"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check real RBI corpus readiness for multi-source NLP monitoring."
    )
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-csv", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status, diagnostics = build_rbi_corpus_status(args.manifest_path)
    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(output, index=False)
        print(f"Status CSV: {output.resolve()}")
    print(f"RBI manifest: {Path(args.manifest_path).resolve()}")
    print(f"Valid document count: {status['valid_document_count']}")
    print(f"Invalid document count: {status['invalid_document_count']}")
    print(f"Distinct publication dates: {status['distinct_publication_dates']}")
    print(f"Document type counts: {status['document_type_counts']}")
    print(f"MPC minutes count: {status['mpc_minutes_count']}")
    print(f"Monetary policy statement count: {status['monetary_policy_statement_count']}")
    print(f"Governor speech count: {status['governor_speech_count']}")
    print(f"Financial stability report count: {status['financial_stability_report_count']}")
    print(f"Policy core documents: {status['policy_core_documents']}")
    print(
        "Date range: "
        + (
            f"{status['date_start']} to {status['date_end']}"
            if status["date_start"]
            else "unavailable"
        )
    )
    print(f"Word count: {status['word_count']}")
    print(f"Sentence count: {status['sentence_count']}")
    print(f"Skipped index pages: {status['skipped_index_pages']}")
    print(f"Index page reason counts: {status['index_page_reason_counts']}")
    print(
        "Minimum requirements passed: " + ("yes" if status["minimum_requirements_passed"] else "no")
    )
    print("Manual action required: " + ("yes" if status["manual_action_required"] else "no"))
    if status["manual_action_required"]:
        print("RBI manual action required; current signal remains news-only monitoring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
