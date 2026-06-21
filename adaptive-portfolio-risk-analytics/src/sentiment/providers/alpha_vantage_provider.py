"""Optional Alpha Vantage news provider."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Callable
from urllib.parse import urlencode
from urllib.request import urlopen
import json

import pandas as pd

from .base import SentimentProvider, normalized_frame, stable_record_id


class AlphaVantageNewsProvider(SentimentProvider):
    """Use Alpha Vantage NEWS_SENTIMENT only when enabled with a key."""

    provider_name = "alpha_vantage_news"
    endpoint = "https://www.alphavantage.co/query"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        enabled: bool | None = None,
        response_loader: Callable[[dict[str, object]], object] | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY", "")
        env_enabled = os.getenv(
            "ALPHAVANTAGE_NEWS_ENABLED", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.enabled = env_enabled if enabled is None else bool(enabled)
        self.response_loader = response_loader

    def _response(self, params: dict[str, object]) -> object:
        if self.response_loader is not None:
            return self.response_loader(params)
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
            status = "disabled"
        elif not self.api_key:
            status = "missing_api_key"
        else:
            status = "enabled"
        if status != "enabled":
            self.last_diagnostics = {
                "provider": self.provider_name,
                "status": status,
                "fetched_record_count": 0,
            }
            return []
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "limit": int(limit or 50),
            "time_from": pd.Timestamp(start_date).strftime("%Y%m%dT0000"),
            "time_to": pd.Timestamp(end_date).strftime("%Y%m%dT2359"),
        }
        if symbols:
            params["tickers"] = ",".join(symbols)
        if query:
            params["topics"] = (
                ",".join(query)
                if isinstance(query, (list, tuple))
                else str(query)
            )
        try:
            response = self._response(params)
            records = (
                response.get("feed", [])
                if isinstance(response, dict)
                else response or []
            )
            self.last_diagnostics = {
                "provider": self.provider_name,
                "status": "success" if records else "empty",
                "fetched_record_count": int(len(records)),
            }
            return list(records)
        except Exception as exc:
            self.last_diagnostics = {
                "provider": self.provider_name,
                "status": "error",
                "fetched_record_count": 0,
                "failures": str(exc),
            }
            return []

    def normalize(self, raw_records) -> pd.DataFrame:
        retrieval = pd.Timestamp(datetime.now(timezone.utc))
        rows: list[dict[str, object]] = []
        for raw in raw_records or []:
            publication = pd.to_datetime(
                raw.get("time_published"),
                format="%Y%m%dT%H%M%S",
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
                    "source": raw.get("source", "Alpha Vantage"),
                    "provider": self.provider_name,
                    "document_type": "financial_news",
                    "entity": "",
                    "ticker": "",
                    "sector": "",
                    "country": "",
                    "title": title,
                    "text": raw.get("summary") or title,
                    "url": url,
                    "language": "en",
                    "raw_metadata": raw,
                    "provider_sentiment_score": pd.to_numeric(
                        raw.get("overall_sentiment_score"), errors="coerce"
                    ),
                    "provider_sentiment_label": raw.get(
                        "overall_sentiment_label", ""
                    ),
                }
            )
        return normalized_frame(rows)
