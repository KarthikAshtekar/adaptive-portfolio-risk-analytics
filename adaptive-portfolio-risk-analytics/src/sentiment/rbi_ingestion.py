"""Manifest-driven ingestion for locally stored RBI documents."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .schema import RBI_DOCUMENT_TYPES


REQUIRED_MANIFEST_COLUMNS = (
    "document_id",
    "publication_date",
    "title",
    "document_type",
)
PATH_COLUMNS = ("local_path", "file_path")
URL_COLUMNS = ("source_url", "url")
OPTIONAL_MANIFEST_COLUMNS = ("source", "language")
DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _read_csv_text(path: Path) -> str:
    frame = pd.read_csv(path)
    if frame.empty:
        return ""
    preferred = next(
        (column for column in ("text", "content", "sentence") if column in frame),
        None,
    )
    if preferred is not None:
        return "\n".join(frame[preferred].dropna().astype(str))
    return "\n".join(
        " ".join(row.dropna().astype(str))
        for _, row in frame.iterrows()
    )


def _read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF skipped because optional dependency 'pypdf' is unavailable"
        ) from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".csv":
        return _read_csv_text(path)
    if suffix == ".pdf":
        return _read_pdf_text(path)
    raise ValueError(f"unsupported document extension: {suffix or '<none>'}")


def load_rbi_documents(
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load RBI documents while retaining row-level errors for auditability."""
    manifest = Path(manifest_path)
    frame = pd.read_csv(manifest)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")
    if not any(column in frame for column in PATH_COLUMNS):
        raise ValueError(
            "manifest is missing required path column: local_path "
            "(legacy alias file_path is also accepted)"
        )
    if "local_path" not in frame:
        frame["local_path"] = frame["file_path"]
    if "source_url" not in frame:
        frame["source_url"] = (
            frame["url"] if "url" in frame else pd.NA
        )
    for column in OPTIONAL_MANIFEST_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA

    root = Path(base_dir) if base_dir is not None else manifest.parent
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for manifest_order, row in frame.iterrows():
        document_id = str(row.get("document_id", "")).strip()
        publication_date = pd.to_datetime(
            row.get("publication_date"),
            errors="coerce",
            utc=True,
        )
        if not pd.isna(publication_date):
            publication_date = publication_date.tz_convert(None).normalize()
        raw_type = str(row.get("document_type", "unknown")).strip().lower()
        document_type = (
            raw_type if raw_type in RBI_DOCUMENT_TYPES else "unknown"
        )
        raw_path = row.get("local_path", "")
        relative_path = (
            ""
            if pd.isna(raw_path)
            else str(raw_path).strip()
        )
        resolved_path = (root / relative_path).resolve()
        error: str | None = None
        text = ""

        if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
            error = "invalid document_id"
        elif document_id in seen_ids:
            error = "duplicate document_id"
        elif pd.isna(publication_date):
            error = "invalid publication_date"
        elif not relative_path:
            error = "missing local_path"
        elif not resolved_path.is_file():
            error = f"document file not found: {relative_path}"
        else:
            try:
                text = _read_document_text(resolved_path).strip()
                if not text:
                    error = "document text is empty"
            except Exception as exc:
                error = str(exc)

        if document_id:
            seen_ids.add(document_id)
        raw_source = row.get("source", "RBI")
        source = (
            "RBI"
            if pd.isna(raw_source) or not str(raw_source).strip()
            else str(raw_source).strip()
        )
        raw_language = row.get("language", "en")
        language = (
            "en"
            if pd.isna(raw_language) or not str(raw_language).strip()
            else str(raw_language).strip()
        )
        rows.append(
            {
                "document_id": document_id,
                "publication_date": publication_date,
                "title": str(row.get("title", "")).strip(),
                "document_type": document_type,
                "source": source,
                "language": language,
                "local_path": relative_path,
                "source_url": row.get("source_url"),
                "file_path": relative_path,
                "resolved_path": str(resolved_path),
                "url": row.get("source_url"),
                "text": text,
                "load_status": "error" if error else "loaded",
                "error": error,
                "manifest_order": int(manifest_order),
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["diagnostics"] = {
        "manifest_path": str(manifest.resolve()),
        "base_dir": str(root.resolve()),
        "manifest_row_count": int(len(frame)),
        "loaded_document_count": int(result["load_status"].eq("loaded").sum()),
        "error_document_count": int(result["load_status"].eq("error").sum()),
        "empty_document_count": int(result["error"].eq("document text is empty").sum()),
    }
    return result
