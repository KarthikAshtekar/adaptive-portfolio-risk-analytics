"""Validation contracts for governed real NLP corpus intake."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER_MARKER = "DO_NOT_USE_PLACEHOLDER"
NON_REAL_MARKERS = (
    PLACEHOLDER_MARKER.lower(),
    "synthetic fixture",
    "fixture for offline",
    "example.com",
    "test only",
)
RBI_MANIFEST_COLUMNS = (
    "document_id",
    "publication_date",
    "document_type",
    "title",
    "local_path",
    "source_url",
    "retrieval_date",
    "language",
    "notes",
)
EARNINGS_MANIFEST_COLUMNS = (
    "document_id",
    "company",
    "ticker",
    "sector",
    "quarter",
    "publication_date",
    "title",
    "local_path",
    "source_url",
    "retrieval_date",
    "language",
    "notes",
)
NEWS_MANIFEST_COLUMNS = (
    "record_id",
    "publication_time",
    "source",
    "provider",
    "document_type",
    "entity",
    "ticker",
    "sector",
    "country",
    "title",
    "text",
    "url",
    "language",
    "retrieval_time",
    "notes",
)
RBI_DOCUMENT_TYPES = {
    "mpc_minutes",
    "monetary_policy_statement",
    "governor_speech",
    "press_release",
    "financial_stability_report",
    "annual_report",
    "unknown",
}
NEWS_DOCUMENT_TYPES = {
    "financial_news",
    "geopolitical_news",
    "news",
    "article",
    "analysis",
    "press_release",
    "unknown",
}
DEFAULT_MANIFESTS = {
    "rbi": REPO_ROOT / "data" / "sentiment" / "rbi_real" / "manifest.csv",
    "earnings": (
        REPO_ROOT / "data" / "sentiment" / "earnings_calls" / "manifest.csv"
    ),
    "news": REPO_ROOT / "data" / "sentiment" / "news_real" / "manifest.csv",
}


def placeholder_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify explicit placeholders and known bundled fixtures."""
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    combined = frame.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    result = pd.Series(False, index=frame.index)
    for marker in NON_REAL_MARKERS:
        result |= combined.str.contains(marker, regex=False)
    return result


def is_explicit_placeholder(record: dict[str, object]) -> bool:
    """Return True only for rows carrying the required placeholder marker."""
    return PLACEHOLDER_MARKER.lower() in " ".join(
        str(value or "") for value in record.values()
    ).lower()


def _read_text(path: Path) -> str:
    if path.suffix.lower() not in {".txt", ".md", ".csv"}:
        raise ValueError("unsupported local text format")
    return path.read_text(encoding="utf-8").strip()


def _empty_status(
    corpus: str,
    manifest: Path,
    error: str,
    required_columns: tuple[str, ...],
) -> dict[str, object]:
    rows = pd.DataFrame(
        [
            {
                "corpus": corpus,
                "manifest_path": str(manifest),
                "manifest_exists": False,
                "row_number": pd.NA,
                "record_id": "",
                "is_placeholder": False,
                "validation_status": "manual_action_required",
                "valid_real_record": False,
                "validation_errors": error,
                "resolved_path": "",
            }
        ]
    )
    return {
        "corpus": corpus,
        "manifest_path": manifest,
        "manifest_exists": False,
        "required_columns": required_columns,
        "missing_required_columns": list(required_columns),
        "rows": rows,
        "valid_records": pd.DataFrame(columns=required_columns),
        "summary": {
            "corpus": corpus,
            "manifest_path": str(manifest),
            "manifest_exists": False,
            "manifest_row_count": 0,
            "valid_record_count": 0,
            "invalid_record_count": 0,
            "placeholder_record_count": 0,
            "corpus_status": "manual_action_required",
            "manual_action_required": True,
            "status_message": error,
        },
    }


def validate_corpus_manifest(
    corpus: str,
    manifest_path: str | Path,
) -> dict[str, object]:
    """Validate one RBI, earnings, or news intake manifest."""
    manifest = Path(manifest_path)
    schemas = {
        "rbi": RBI_MANIFEST_COLUMNS,
        "earnings": EARNINGS_MANIFEST_COLUMNS,
        "news": NEWS_MANIFEST_COLUMNS,
    }
    if corpus not in schemas:
        raise ValueError(f"unsupported corpus: {corpus}")
    required = schemas[corpus]
    if not manifest.is_file():
        return _empty_status(
            corpus,
            manifest,
            f"manifest not found: {manifest}; manual action required",
            required,
        )
    try:
        frame = pd.read_csv(manifest, dtype="string").fillna("")
    except Exception as exc:
        return _empty_status(
            corpus,
            manifest,
            f"manifest unreadable: {exc}; manual action required",
            required,
        )
    frame.columns = [str(column).strip() for column in frame.columns]
    missing_columns = [column for column in required if column not in frame]
    for column in missing_columns:
        frame[column] = ""
    frame = frame.loc[:, list(required)].copy()
    placeholders = placeholder_mask(frame)
    id_column = "record_id" if corpus == "news" else "document_id"
    date_column = "publication_time" if corpus == "news" else "publication_date"
    source_column = "url" if corpus == "news" else "source_url"
    retrieval_column = "retrieval_time" if corpus == "news" else "retrieval_date"
    duplicate_ids = (
        frame[id_column].str.strip().str.lower().duplicated(keep=False)
    )
    duplicate_identity = (
        frame[[date_column, "title", source_column]]
        .apply(lambda series: series.str.strip().str.lower())
        .duplicated(keep=False)
    )

    rows: list[dict[str, object]] = []
    valid_indices: list[int] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        values = {column: str(row[column]).strip() for column in required}
        errors: list[str] = []
        is_placeholder = bool(placeholders.iloc[position])
        publication = pd.to_datetime(values[date_column], errors="coerce", utc=True)
        retrieval = pd.to_datetime(
            values[retrieval_column], errors="coerce", utc=True
        )
        if missing_columns:
            errors.append(
                f"missing required columns: {', '.join(missing_columns)}"
            )
        if not values[id_column]:
            errors.append(f"missing {id_column}")
        if values[date_column] and pd.isna(publication):
            errors.append(f"invalid {date_column}")
        elif not values[date_column]:
            errors.append(f"missing {date_column}")
        if values[retrieval_column] and pd.isna(retrieval):
            errors.append(f"invalid {retrieval_column}")
        elif not values[retrieval_column]:
            errors.append(f"missing {retrieval_column}")
        if (
            corpus == "news"
            and pd.notna(publication)
            and pd.notna(retrieval)
            and publication > retrieval
        ):
            errors.append("publication_time after retrieval_time")
        if duplicate_ids.iloc[position]:
            errors.append(f"duplicate {id_column}")
        if duplicate_identity.iloc[position]:
            errors.append("duplicate title/date/source row")
        if not values[source_column]:
            errors.append(f"missing {source_column}")
        if not values["language"]:
            errors.append("missing language")
        if not values["title"]:
            errors.append("missing title")
        if corpus == "rbi" and values["document_type"] not in RBI_DOCUMENT_TYPES:
            errors.append("invalid document_type")
        if corpus == "news" and values["document_type"] not in NEWS_DOCUMENT_TYPES:
            errors.append("invalid document_type")
        if corpus == "earnings" and not values["sector"]:
            errors.append("missing sector")
        if corpus == "news":
            if not values["provider"]:
                errors.append("missing provider")
            if not values["source"]:
                errors.append("missing source")
            if not values["text"]:
                errors.append("empty text")

        resolved = ""
        if corpus in {"rbi", "earnings"}:
            local_path = values["local_path"]
            if not local_path:
                errors.append("missing local_path")
            else:
                resolved_path = (manifest.parent / local_path).resolve()
                resolved = str(resolved_path)
                if not resolved_path.is_file():
                    errors.append("local file not found")
                else:
                    try:
                        if not _read_text(resolved_path):
                            errors.append("empty text file")
                    except Exception as exc:
                        errors.append(str(exc))

        if is_placeholder:
            status = "placeholder_excluded"
        elif errors:
            status = "invalid"
        else:
            status = "valid"
            valid_indices.append(position)
        rows.append(
            {
                "corpus": corpus,
                "manifest_path": str(manifest),
                "manifest_exists": True,
                "row_number": position,
                "record_id": values[id_column],
                "is_placeholder": is_placeholder,
                "validation_status": status,
                "valid_real_record": status == "valid",
                "validation_errors": (
                    "placeholder or synthetic fixture excluded"
                    if is_placeholder
                    else "; ".join(dict.fromkeys(errors))
                ),
                "resolved_path": resolved,
            }
        )
    row_status = pd.DataFrame(
        rows,
        columns=[
            "corpus",
            "manifest_path",
            "manifest_exists",
            "row_number",
            "record_id",
            "is_placeholder",
            "validation_status",
            "valid_real_record",
            "validation_errors",
            "resolved_path",
        ],
    )
    valid_records = frame.iloc[valid_indices].reset_index(drop=True)
    invalid_count = int(row_status["validation_status"].eq("invalid").sum())
    placeholder_count = int(
        row_status["validation_status"].eq("placeholder_excluded").sum()
    )
    valid_count = int(len(valid_records))
    manual_action = bool(valid_count == 0 or invalid_count > 0 or missing_columns)
    corpus_status = "ready" if not manual_action else "manual_action_required"
    message = (
        f"{valid_count} valid real record(s) ready"
        if corpus_status == "ready"
        else (
            f"manual action required: {valid_count} valid, {invalid_count} "
            f"invalid, {placeholder_count} placeholder/fixture record(s)"
        )
    )
    return {
        "corpus": corpus,
        "manifest_path": manifest,
        "manifest_exists": True,
        "required_columns": required,
        "missing_required_columns": missing_columns,
        "rows": row_status,
        "valid_records": valid_records,
        "summary": {
            "corpus": corpus,
            "manifest_path": str(manifest),
            "manifest_exists": True,
            "manifest_row_count": int(len(frame)),
            "valid_record_count": valid_count,
            "invalid_record_count": invalid_count,
            "placeholder_record_count": placeholder_count,
            "corpus_status": corpus_status,
            "manual_action_required": manual_action,
            "status_message": message,
        },
    }


def validate_nlp_corpus_intake(
    *,
    rbi_manifest: str | Path | None = None,
    earnings_manifest: str | Path | None = None,
    news_manifest: str | Path | None = None,
) -> dict[str, object]:
    """Validate all governed real-text intake surfaces."""
    paths = {
        "rbi": Path(rbi_manifest) if rbi_manifest else DEFAULT_MANIFESTS["rbi"],
        "earnings": (
            Path(earnings_manifest)
            if earnings_manifest
            else DEFAULT_MANIFESTS["earnings"]
        ),
        "news": (
            Path(news_manifest) if news_manifest else DEFAULT_MANIFESTS["news"]
        ),
    }
    results = {
        corpus: validate_corpus_manifest(corpus, path)
        for corpus, path in paths.items()
    }
    status = pd.DataFrame(
        [results[corpus]["summary"] for corpus in ("rbi", "earnings", "news")]
    )
    return {
        "corpora": results,
        "intake_status": status,
        "valid_records": {
            corpus: results[corpus]["valid_records"]
            for corpus in results
        },
        "row_diagnostics": {
            corpus: results[corpus]["rows"] for corpus in results
        },
        "manual_action_required": bool(
            status["manual_action_required"].fillna(True).any()
        ),
        "all_corpora_ready": bool(
            status["corpus_status"].eq("ready").all()
        ),
        "valid_real_records_by_corpus": {
            row.corpus: int(row.valid_record_count)
            for row in status.itertuples(index=False)
        },
    }
