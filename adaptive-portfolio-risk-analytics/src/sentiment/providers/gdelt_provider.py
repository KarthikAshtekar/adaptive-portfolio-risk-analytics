"""Optional GDELT article provider with injectable offline responses."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from .base import SentimentProvider, normalized_frame, stable_record_id


DEFAULT_GDELT_QUERIES = (
    "India inflation",
    "RBI rate hike",
    "geopolitical risk India",
    "oil price shock",
    "banking stress",
    "currency crisis",
    "war escalation",
    "supply chain disruption",
)


class GDELTProvider(SentimentProvider):
    """Fetch GDELT DOC API records only when explicitly enabled."""

    provider_name = "gdelt"
    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(
        self,
        *,
        enabled: bool = False,
        response_loader: Callable[[dict[str, object]], object] | None = None,
        fixture_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.response_loader = response_loader
        self.fixture_path = Path(fixture_path) if fixture_path else None

    def _response(self, params: dict[str, object]) -> object:
        if self.response_loader is not None:
            return self.response_loader(params)
        if self.fixture_path is not None:
            return json.loads(self.fixture_path.read_text(encoding="utf-8"))
        url = f"{self.endpoint}?{urlencode(params)}"
        with urlopen(url, timeout=20) as response:  # nosec B310 - opt-in API
            return json.loads(response.read().decode("utf-8"))

    def fetch(
        self,
        start_date,
        end_date,
        query=None,
        symbols=None,
        sectors=None,
        limit=None,
    ) -> list[dict[str, object]]:
        if not self.enabled:
            self.last_diagnostics = {
                "provider": self.provider_name,
                "status": "disabled",
                "fetched_record_count": 0,
            }
            return []
        queries = (
            list(query)
            if isinstance(query, (list, tuple))
            else [query or DEFAULT_GDELT_QUERIES[0]]
        )
        records: list[dict[str, object]] = []
        failures: list[str] = []
        for query_text in queries:
            params = {
                "query": query_text,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": int(limit or 50),
                "startdatetime": pd.Timestamp(start_date).strftime("%Y%m%d000000"),
                "enddatetime": pd.Timestamp(end_date).strftime("%Y%m%d235959"),
            }
            try:
                response = self._response(params)
                articles = (
                    response.get("articles", [])
                    if isinstance(response, dict)
                    else response
                )
                for article in articles or []:
                    records.append({**article, "query": query_text})
            except Exception as exc:
                failures.append(f"{query_text}: {exc}")
        if limit is not None:
            records = records[: int(limit)]
        self.last_diagnostics = {
            "provider": self.provider_name,
            "status": "success" if records else "empty",
            "fetched_record_count": int(len(records)),
            "query_count": int(len(queries)),
            "failure_count": int(len(failures)),
            "failures": " | ".join(failures),
            "source_kind": "fixture"
            if self.fixture_path is not None
            else "api",
        }
        return records

    def normalize(self, raw_records) -> pd.DataFrame:
        retrieval = pd.Timestamp(datetime.now(timezone.utc))
        rows: list[dict[str, object]] = []
        for raw in raw_records or []:
            publication = pd.to_datetime(
                raw.get("seendate")
                or raw.get("publication_time")
                or raw.get("date"),
                errors="coerce",
                utc=True,
            )
            url = str(raw.get("url") or "").strip()
            title = str(raw.get("title") or "").strip()
            rows.append(
                {
                    "record_id": stable_record_id(
                        self.provider_name, url, publication, title
                    ),
                    "timestamp": publication,
                    "publication_time": publication,
                    "retrieval_time": retrieval,
                    "source": raw.get("domain")
                    or raw.get("source")
                    or "GDELT",
                    "provider": self.provider_name,
                    "document_type": "financial_news",
                    "entity": raw.get("entity", ""),
                    "ticker": raw.get("ticker", ""),
                    "sector": raw.get("sector", ""),
                    "country": raw.get("sourcecountry")
                    or raw.get("country")
                    or "",
                    "title": title,
                    "text": raw.get("text")
                    or raw.get("summary")
                    or title,
                    "url": url,
                    "language": raw.get("language", "English"),
                    "raw_metadata": raw,
                    "query": raw.get("query", ""),
                }
            )
        return normalized_frame(rows)
