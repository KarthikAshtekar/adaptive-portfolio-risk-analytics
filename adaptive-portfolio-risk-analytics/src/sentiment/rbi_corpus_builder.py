"""Build, validate, and load a reproducible local real-RBI corpus."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re

import pandas as pd

from .corpus_intake import is_explicit_placeholder
from .rbi_processing import split_rbi_documents_into_sentences


REAL_RBI_MANIFEST_COLUMNS = (
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
REAL_RBI_DOCUMENT_TYPES = (
    "mpc_minutes",
    "monetary_policy_statement",
    "governor_speech",
    "press_release",
    "financial_stability_report",
    "annual_report",
    "unknown",
)
DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DATE_IN_FILENAME = re.compile(
    r"(?P<year>20\d{2})[-_]?(?P<month>0[1-9]|1[0-2])[-_]?"
    r"(?P<day>0[1-9]|[12]\d|3[01])"
)


def _document_type_from_name(name: str) -> str:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    mapping = (
        ("mpc", "mpc_minutes"),
        ("monetary_policy", "monetary_policy_statement"),
        ("policy_statement", "monetary_policy_statement"),
        ("governor", "governor_speech"),
        ("speech", "governor_speech"),
        ("financial_stability", "financial_stability_report"),
        ("fsr", "financial_stability_report"),
        ("annual_report", "annual_report"),
        ("press_release", "press_release"),
    )
    return next(
        (document_type for token, document_type in mapping if token in normalized),
        "unknown",
    )


def _date_from_name(name: str) -> str:
    match = DATE_IN_FILENAME.search(name)
    if match is None:
        return ""
    return f"{match.group('year')}-{match.group('month')}-{match.group('day')}"


def _document_id(path: Path) -> str:
    identifier = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return identifier.lower() or "unknown_document"


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8").strip()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF extraction unavailable; convert the PDF to reviewed "
                "UTF-8 text as documented in docs/rbi_corpus_manual_download.md"
            ) from exc
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages).strip()
    raise ValueError(f"unsupported real-RBI file type: {suffix or '<none>'}")


def build_rbi_manifest_from_directory(
    raw_dir: str | Path,
    output_manifest_path: str | Path,
) -> pd.DataFrame:
    """Build an exact-format draft manifest from locally downloaded files."""
    source_dir = Path(raw_dir)
    output_path = Path(output_manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}
    )
    rows: list[dict[str, object]] = []
    for path in candidates:
        resolved_path = path.resolve()
        manifest_root = output_path.parent.resolve()
        relative_path = (
            resolved_path.relative_to(manifest_root)
            if resolved_path.is_relative_to(manifest_root)
            else resolved_path
        )
        rows.append(
            {
                "document_id": _document_id(path),
                "publication_date": _date_from_name(path.stem),
                "document_type": _document_type_from_name(path.stem),
                "title": path.stem.replace("_", " ").replace("-", " ").strip(),
                "local_path": str(relative_path).replace("\\", "/"),
                "source_url": "",
                "retrieval_date": date.today().isoformat(),
                "language": "en",
                "notes": "Draft row; review publication date, title, and source URL.",
            }
        )
    manifest = pd.DataFrame(rows, columns=REAL_RBI_MANIFEST_COLUMNS)
    manifest.to_csv(output_path, index=False)
    return manifest


def _diagnostic_frame(summary: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, value in summary.items():
        if isinstance(value, dict):
            for key, count in value.items():
                rows.append(
                    {
                        "metric": metric,
                        "category": key,
                        "value": count,
                    }
                )
        else:
            rows.append({"metric": metric, "category": "", "value": value})
    return pd.DataFrame(rows, columns=["metric", "category", "value"])


def validate_rbi_manifest(
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
) -> dict[str, object]:
    """Validate a real-RBI manifest and retain row-level failure diagnostics."""
    manifest = Path(manifest_path)
    root = Path(base_dir) if base_dir is not None else manifest.parent
    if not manifest.is_file():
        summary = {
            "manifest_row_count": 0,
            "valid_document_count": 0,
            "invalid_document_count": 0,
            "date_start": "",
            "date_end": "",
            "document_type_counts": {},
            "missing_field_count": 0,
            "invalid_file_count": 0,
            "duplicate_record_count": 0,
            "total_word_count": 0,
            "total_sentence_count": 0,
            "manifest_error": f"manifest not found: {manifest}",
        }
        return {
            "is_valid": False,
            "manifest": pd.DataFrame(columns=REAL_RBI_MANIFEST_COLUMNS),
            "valid_documents": pd.DataFrame(),
            "invalid_documents": pd.DataFrame(),
            "duplicate_records": pd.DataFrame(),
            "missing_required_columns": list(REAL_RBI_MANIFEST_COLUMNS),
            "unexpected_columns": [],
            "summary": summary,
            "diagnostics": _diagnostic_frame(summary),
        }

    frame = pd.read_csv(manifest, dtype="string")
    frame.columns = [str(column).strip() for column in frame.columns]
    missing_columns = [column for column in REAL_RBI_MANIFEST_COLUMNS if column not in frame]
    unexpected_columns = [column for column in frame if column not in REAL_RBI_MANIFEST_COLUMNS]
    for column in missing_columns:
        frame[column] = pd.NA
    frame = frame.loc[:, list(REAL_RBI_MANIFEST_COLUMNS)].copy()
    frame = frame.fillna("")

    duplicate_id = frame["document_id"].str.strip().duplicated(keep=False)
    duplicate_identity = (
        frame[["title", "publication_date", "source_url"]]
        .apply(lambda series: series.str.strip().str.lower())
        .duplicated(keep=False)
    )
    records: list[dict[str, object]] = []
    required_nonempty = (
        "document_id",
        "publication_date",
        "document_type",
        "title",
        "local_path",
        "source_url",
        "retrieval_date",
        "language",
    )
    for row_number, row in frame.iterrows():
        values = {column: str(row[column]).strip() for column in frame.columns}
        errors: list[str] = []
        if is_explicit_placeholder(values):
            errors.append("placeholder excluded")
        missing_fields = [column for column in required_nonempty if not values[column]]
        if missing_fields:
            errors.append(f"missing fields: {', '.join(missing_fields)}")
        if values["document_id"] and not DOCUMENT_ID_PATTERN.fullmatch(values["document_id"]):
            errors.append("invalid document_id")
        if duplicate_id.iloc[row_number]:
            errors.append("duplicate document_id")
        if duplicate_identity.iloc[row_number]:
            errors.append("duplicate title/date/source combination")

        publication_date = pd.to_datetime(values["publication_date"], errors="coerce")
        retrieval_date = pd.to_datetime(values["retrieval_date"], errors="coerce")
        if values["publication_date"] and pd.isna(publication_date):
            errors.append("invalid publication_date")
        if values["retrieval_date"] and pd.isna(retrieval_date):
            errors.append("invalid retrieval_date")
        if values["document_type"] and values["document_type"] not in REAL_RBI_DOCUMENT_TYPES:
            errors.append("invalid document_type")

        resolved_path = (
            (root / values["local_path"]).resolve() if values["local_path"] else root.resolve()
        )
        text = ""
        if values["local_path"]:
            if not resolved_path.is_file():
                errors.append("local file not found")
            else:
                try:
                    text = _read_text(resolved_path)
                    if not text:
                        errors.append("document text is empty")
                except Exception as exc:
                    errors.append(str(exc))
        records.append(
            {
                **values,
                "publication_date": publication_date,
                "retrieval_date": retrieval_date,
                "resolved_path": str(resolved_path),
                "text": text,
                "word_count": len(text.split()),
                "validation_status": "invalid" if errors else "valid",
                "validation_errors": "; ".join(dict.fromkeys(errors)),
                "manifest_row": int(row_number),
                "load_status": "error" if errors else "loaded",
                "error": "; ".join(dict.fromkeys(errors)) or None,
                "source": "Reserve Bank of India",
                "file_path": values["local_path"],
                "url": values["source_url"],
                "manifest_order": int(row_number),
            }
        )

    validated_columns = [
        *REAL_RBI_MANIFEST_COLUMNS,
        "resolved_path",
        "text",
        "word_count",
        "validation_status",
        "validation_errors",
        "manifest_row",
        "load_status",
        "error",
        "source",
        "file_path",
        "url",
        "manifest_order",
    ]
    validated = pd.DataFrame(records, columns=validated_columns)
    valid = validated.loc[validated["validation_status"].eq("valid")].copy()
    invalid = validated.loc[validated["validation_status"].eq("invalid")].copy()
    duplicate_records = validated.loc[
        duplicate_id.to_numpy() | duplicate_identity.to_numpy()
    ].copy()
    sentences = split_rbi_documents_into_sentences(valid) if not valid.empty else pd.DataFrame()
    dates = pd.to_datetime(valid.get("publication_date"), errors="coerce").dropna()
    summary = {
        "manifest_row_count": int(len(frame)),
        "valid_document_count": int(len(valid)),
        "invalid_document_count": int(len(invalid)),
        "date_start": dates.min().date().isoformat() if not dates.empty else "",
        "date_end": dates.max().date().isoformat() if not dates.empty else "",
        "document_type_counts": valid["document_type"].value_counts().to_dict(),
        "missing_field_count": int(
            validated["validation_errors"].str.contains("missing fields", na=False).sum()
        ),
        "invalid_file_count": int(
            validated["validation_errors"]
            .str.contains(
                "file not found|text is empty|PDF extraction|unsupported",
                regex=True,
                na=False,
            )
            .sum()
        ),
        "duplicate_record_count": int(len(duplicate_records)),
        "total_word_count": int(valid["word_count"].sum()) if not valid.empty else 0,
        "total_sentence_count": int(len(sentences)),
    }
    return {
        "is_valid": bool(not missing_columns and not unexpected_columns and not valid.empty),
        "manifest": frame,
        "valid_documents": valid.reset_index(drop=True),
        "invalid_documents": invalid.reset_index(drop=True),
        "duplicate_records": duplicate_records.reset_index(drop=True),
        "missing_required_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "summary": summary,
        "diagnostics": _diagnostic_frame(summary),
    }


def load_real_rbi_corpus(
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load only valid real-RBI documents and attach validation diagnostics."""
    validation = validate_rbi_manifest(manifest_path, base_dir=base_dir)
    documents = validation["valid_documents"].copy()
    if "corpus_type" not in documents:
        documents["corpus_type"] = "real_rbi"
    documents.attrs["manifest_validation"] = validation
    documents.attrs["diagnostics"] = validation["summary"]
    return documents
