"""Import a manually extracted public RBI text document into the local corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sentiment import (  # noqa: E402
    REAL_RBI_DOCUMENT_TYPES,
    REAL_RBI_MANIFEST_COLUMNS,
    validate_rbi_manifest,
)


DEFAULT_MANIFEST = REPO_ROOT / "data" / "sentiment" / "rbi_real" / "manifest.csv"
DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _ensure_manifest(path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "raw").mkdir(parents=True, exist_ok=True)
    (path.parent / "processed").mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=REAL_RBI_MANIFEST_COLUMNS).to_csv(path, index=False)
    frame = pd.read_csv(path, dtype="string").fillna("")
    missing = [column for column in REAL_RBI_MANIFEST_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    return frame.loc[:, list(REAL_RBI_MANIFEST_COLUMNS)].copy()


def _read_nonempty_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"input text file not found: {path}")
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
    if not text:
        raise ValueError("input text file is empty")
    return text


def _validate_import_args(
    *,
    document_id: str,
    publication_date: str,
    document_type: str,
    title: str,
    source_url: str,
    retrieval_date: str,
) -> None:
    if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise ValueError("document_id must contain only letters, numbers, '.', '_' or '-'")
    if document_type not in REAL_RBI_DOCUMENT_TYPES:
        raise ValueError(
            "invalid document_type; supported values are: " + ", ".join(REAL_RBI_DOCUMENT_TYPES)
        )
    for field_name, value in {
        "publication_date": publication_date,
        "title": title,
        "source_url": source_url,
        "retrieval_date": retrieval_date,
    }.items():
        if not str(value or "").strip():
            raise ValueError(f"{field_name} is required")
    publication = pd.to_datetime(publication_date, errors="coerce")
    retrieval = pd.to_datetime(retrieval_date, errors="coerce")
    if pd.isna(publication):
        raise ValueError("publication_date must be a valid date")
    if pd.isna(retrieval):
        raise ValueError("retrieval_date must be a valid date")
    if publication > retrieval:
        raise ValueError("publication_date must be on or before retrieval_date")


def import_rbi_text_document(
    *,
    document_id: str,
    publication_date: str,
    document_type: str,
    title: str,
    source_url: str,
    input_text_file: str | Path,
    retrieval_date: str,
    language: str = "en",
    notes: str = "Public RBI communication; manually imported.",
    manifest_path: str | Path = DEFAULT_MANIFEST,
    overwrite: bool = False,
) -> dict[str, object]:
    """Copy text into the corpus and add or update the manifest row."""
    document_id = str(document_id).strip()
    document_type = str(document_type).strip()
    title = str(title).strip()
    source_url = str(source_url).strip()
    publication_date = pd.Timestamp(publication_date).date().isoformat()
    retrieval_date = pd.Timestamp(retrieval_date).date().isoformat()
    _validate_import_args(
        document_id=document_id,
        publication_date=publication_date,
        document_type=document_type,
        title=title,
        source_url=source_url,
        retrieval_date=retrieval_date,
    )
    source_path = Path(input_text_file)
    text = _read_nonempty_text(source_path)
    manifest = Path(manifest_path)
    frame = _ensure_manifest(manifest)
    duplicate_mask = frame["document_id"].astype(str).str.strip().eq(document_id)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count and not overwrite:
        raise ValueError(
            f"document_id already exists in manifest: {document_id}; pass --overwrite to replace it"
        )

    destination = manifest.parent / "raw" / f"{document_id}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text + "\n", encoding="utf-8", newline="\n")
    shutil.copystat(source_path, destination, follow_symlinks=True)
    relative_path = destination.relative_to(manifest.parent).as_posix()
    new_row = {
        "document_id": document_id,
        "publication_date": publication_date,
        "document_type": document_type,
        "title": title,
        "local_path": relative_path,
        "source_url": source_url,
        "retrieval_date": retrieval_date,
        "language": str(language or "en").strip() or "en",
        "notes": str(notes or "").strip(),
    }
    if duplicate_count:
        frame = frame.loc[~duplicate_mask].copy()
    frame = pd.concat(
        [frame, pd.DataFrame([new_row], columns=REAL_RBI_MANIFEST_COLUMNS)],
        ignore_index=True,
    )
    frame = frame.loc[:, list(REAL_RBI_MANIFEST_COLUMNS)]
    frame.to_csv(manifest, index=False)
    validation = validate_rbi_manifest(manifest)
    return {
        "manifest_path": manifest,
        "copied_path": destination,
        "row": new_row,
        "validation": validation,
        "overwrote_existing": bool(duplicate_count),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import one manually extracted RBI text file into the real corpus."
    )
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--publication-date", required=True)
    parser.add_argument("--document-type", required=True, choices=REAL_RBI_DOCUMENT_TYPES)
    parser.add_argument("--title", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--input-text-file", required=True)
    parser.add_argument("--retrieval-date", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--notes", default="Public RBI communication; manually imported.")
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = import_rbi_text_document(
            document_id=args.document_id,
            publication_date=args.publication_date,
            document_type=args.document_type,
            title=args.title,
            source_url=args.source_url,
            input_text_file=args.input_text_file,
            retrieval_date=args.retrieval_date,
            language=args.language,
            notes=args.notes,
            manifest_path=args.manifest_path,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"RBI import failed: {exc}", file=sys.stderr)
        return 1
    summary = result["validation"]["summary"]
    print("RBI document imported.")
    print(f"Copied text: {Path(result['copied_path']).resolve()}")
    print(f"Manifest: {Path(result['manifest_path']).resolve()}")
    print(f"Valid documents: {summary['valid_document_count']}")
    print(f"Invalid documents: {summary['invalid_document_count']}")
    print("Overwrote existing row: " + ("yes" if result["overwrote_existing"] else "no"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
