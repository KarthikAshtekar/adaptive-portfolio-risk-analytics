"""Local-first earnings-call transcript provider."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.sentiment.corpus_intake import is_explicit_placeholder

from .base import SentimentProvider, normalized_frame, stable_record_id


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


class EarningsCallProvider(SentimentProvider):
    """Load reviewed local earnings-call transcripts without paid APIs."""

    provider_name = "earnings_calls"

    def __init__(self, manifest_path: str | Path) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)

    def fetch(
        self,
        start_date,
        end_date,
        query=None,
        symbols=None,
        sectors=None,
        limit=None,
    ) -> list[dict[str, object]]:
        failures: list[str] = []
        if not self.manifest_path.is_file():
            self.last_diagnostics = {
                "provider": self.provider_name,
                "status": "empty",
                "fetched_record_count": 0,
                "failures": f"manifest not found: {self.manifest_path}",
            }
            return []
        manifest = pd.read_csv(self.manifest_path, dtype="string").fillna("")
        missing = [
            column for column in EARNINGS_MANIFEST_COLUMNS if column not in manifest
        ]
        if missing:
            self.last_diagnostics = {
                "provider": self.provider_name,
                "status": "invalid_manifest",
                "fetched_record_count": 0,
                "failures": f"missing columns: {', '.join(missing)}",
            }
            return []
        records: list[dict[str, object]] = []
        for row in manifest.to_dict("records"):
            if is_explicit_placeholder(row):
                failures.append(
                    f"{row.get('document_id', 'unknown')}: placeholder excluded"
                )
                continue
            local_path = (self.manifest_path.parent / row["local_path"]).resolve()
            try:
                text = local_path.read_text(encoding="utf-8").strip()
                if not text:
                    raise ValueError("transcript is empty")
            except Exception as exc:
                failures.append(f"{row['document_id']}: {exc}")
                continue
            if symbols and row["ticker"] not in symbols:
                continue
            if sectors and row["sector"] not in sectors:
                continue
            publication = pd.to_datetime(row["publication_date"], errors="coerce")
            if pd.isna(publication):
                failures.append(f"{row['document_id']}: invalid publication_date")
                continue
            if publication.date() < pd.Timestamp(start_date).date():
                continue
            if publication.date() > pd.Timestamp(end_date).date():
                continue
            records.append({**row, "text": text})
        if limit is not None:
            records = records[: int(limit)]
        self.last_diagnostics = {
            "provider": self.provider_name,
            "status": "success" if records else "empty",
            "fetched_record_count": int(len(records)),
            "failure_count": int(len(failures)),
            "failures": " | ".join(failures),
            "source_kind": "local_transcript_manifest",
        }
        return records

    def normalize(self, raw_records) -> pd.DataFrame:
        now = pd.Timestamp(datetime.now(timezone.utc))
        rows: list[dict[str, object]] = []
        for raw in raw_records or []:
            publication = pd.to_datetime(
                raw.get("publication_date"), errors="coerce", utc=True
            )
            retrieval = pd.to_datetime(
                raw.get("retrieval_date"), errors="coerce", utc=True
            )
            if pd.isna(retrieval):
                retrieval = now
            rows.append(
                {
                    "record_id": raw.get("document_id")
                    or stable_record_id(
                        self.provider_name,
                        raw.get("ticker"),
                        publication,
                        raw.get("title"),
                    ),
                    "timestamp": publication,
                    "publication_time": publication,
                    "retrieval_time": retrieval,
                    "source": raw.get("company") or "Company transcript",
                    "provider": self.provider_name,
                    "document_type": "earnings_call",
                    "entity": raw.get("company", ""),
                    "ticker": raw.get("ticker", ""),
                    "sector": raw.get("sector", ""),
                    "country": "IN",
                    "title": raw.get("title", ""),
                    "text": raw.get("text", ""),
                    "url": raw.get("source_url", ""),
                    "language": raw.get("language", "en"),
                    "raw_metadata": raw,
                    "quarter": raw.get("quarter", ""),
                }
            )
        return normalized_frame(rows)
