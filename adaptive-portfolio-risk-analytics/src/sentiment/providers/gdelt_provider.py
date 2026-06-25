"""Optional, rate-limit-aware GDELT DOC API article provider."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Callable
from urllib.error import HTTPError
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
GDELT_QUERY_DIAGNOSTIC_COLUMNS = (
    "provider",
    "query",
    "request_url",
    "http_status",
    "response_bytes",
    "parsed_article_count",
    "normalized_record_count",
    "rate_limited",
    "retry_count",
    "success",
    "error",
    "warning",
)


class GDELTResponseError(RuntimeError):
    """Carry request metadata for HTTP and payload failures."""

    def __init__(
        self,
        message: str,
        *,
        request_url: str,
        http_status: int = 0,
        response_bytes: int = 0,
    ) -> None:
        super().__init__(message)
        self.request_url = request_url
        self.http_status = int(http_status)
        self.response_bytes = int(response_bytes)


class GDELTProvider(SentimentProvider):
    """Fetch GDELT articles with bounded retries and query diagnostics."""

    provider_name = "gdelt"
    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(
        self,
        *,
        enabled: bool = False,
        response_loader: Callable[[dict[str, object]], object] | None = None,
        fixture_path: str | Path | None = None,
        request_delay_seconds: float = 6,
        retry_delay_seconds: float = 10,
        max_retries: int = 3,
        timeout_seconds: float = 30,
        sleep_func: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.response_loader = response_loader
        self.fixture_path = Path(fixture_path) if fixture_path else None
        self.request_delay_seconds = max(0.0, float(request_delay_seconds))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self.max_retries = max(0, int(max_retries))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.sleep_func = sleep_func or time.sleep

    def _request_url(self, params: dict[str, object]) -> str:
        return f"{self.endpoint}?{urlencode(params)}"

    @staticmethod
    def _decode_payload(
        value: object,
        *,
        request_url: str,
        http_status: int = 200,
    ) -> tuple[dict[str, object], int]:
        if isinstance(value, dict):
            encoded = json.dumps(value, ensure_ascii=False, default=str).encode(
                "utf-8"
            )
            return value, len(encoded)
        if isinstance(value, bytes):
            raw = value
        elif isinstance(value, str):
            raw = value.encode("utf-8")
        else:
            raise GDELTResponseError(
                f"unexpected response type: {type(value).__name__}",
                request_url=request_url,
                http_status=http_status,
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GDELTResponseError(
                "non-JSON response from GDELT",
                request_url=request_url,
                http_status=http_status,
                response_bytes=len(raw),
            ) from exc
        if not isinstance(payload, dict):
            raise GDELTResponseError(
                "GDELT JSON response must be an object",
                request_url=request_url,
                http_status=http_status,
                response_bytes=len(raw),
            )
        return payload, len(raw)

    def _response(
        self, params: dict[str, object]
    ) -> tuple[dict[str, object], dict[str, object]]:
        request_url = self._request_url(params)
        if self.response_loader is not None:
            loaded = self.response_loader(params)
            metadata: dict[str, object] = {}
            if (
                isinstance(loaded, tuple)
                and len(loaded) == 2
                and isinstance(loaded[1], dict)
            ):
                loaded, metadata = loaded
            status = int(metadata.get("http_status", 200))
            payload, response_bytes = self._decode_payload(
                loaded,
                request_url=request_url,
                http_status=status,
            )
            if status >= 400:
                raise GDELTResponseError(
                    f"HTTP Error {status}",
                    request_url=str(metadata.get("request_url", request_url)),
                    http_status=status,
                    response_bytes=int(
                        metadata.get("response_bytes", response_bytes)
                    ),
                )
            return payload, {
                "request_url": str(metadata.get("request_url", request_url)),
                "http_status": status,
                "response_bytes": int(
                    metadata.get("response_bytes", response_bytes)
                ),
            }
        if self.fixture_path is not None:
            raw = self.fixture_path.read_bytes()
            payload, response_bytes = self._decode_payload(
                raw,
                request_url=str(self.fixture_path),
            )
            return payload, {
                "request_url": str(self.fixture_path),
                "http_status": 200,
                "response_bytes": response_bytes,
            }
        try:
            with urlopen(  # nosec B310 - fixed opt-in GDELT endpoint
                request_url,
                timeout=self.timeout_seconds,
            ) as response:
                raw = response.read()
                status = int(getattr(response, "status", 200))
        except HTTPError as exc:
            try:
                body = exc.read()
            except Exception:
                body = b""
            raise GDELTResponseError(
                f"HTTP Error {exc.code}: {exc.reason}",
                request_url=request_url,
                http_status=int(exc.code),
                response_bytes=len(body),
            ) from exc
        payload, response_bytes = self._decode_payload(
            raw,
            request_url=request_url,
            http_status=status,
        )
        if status >= 400:
            raise GDELTResponseError(
                f"HTTP Error {status}",
                request_url=request_url,
                http_status=status,
                response_bytes=response_bytes,
            )
        return payload, {
            "request_url": request_url,
            "http_status": status,
            "response_bytes": response_bytes,
        }

    @staticmethod
    def _deduplicate_articles(
        records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[dict[str, object]] = []
        for record in records:
            key = (
                str(record.get("url") or "").strip().lower(),
                str(record.get("title") or "").strip().lower(),
                str(
                    record.get("seendate")
                    or record.get("publication_time")
                    or record.get("date")
                    or ""
                ).strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

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
                "cache_safe": False,
                "query_diagnostics": [],
            }
            return []
        queries = (
            list(query)
            if isinstance(query, (list, tuple))
            else [query or DEFAULT_GDELT_QUERIES[0]]
        )
        queries = [str(item).strip() for item in queries if str(item).strip()]
        if not queries:
            queries = [DEFAULT_GDELT_QUERIES[0]]
        records: list[dict[str, object]] = []
        failures: list[str] = []
        query_diagnostics: list[dict[str, object]] = []
        successful_queries = 0
        failed_queries = 0

        for query_index, query_text in enumerate(queries):
            if (
                query_index > 0
                and self.fixture_path is None
                and self.request_delay_seconds > 0
            ):
                self.sleep_func(self.request_delay_seconds)
            params = {
                "query": query_text,
                "mode": "artlist",
                "format": "json",
                "maxrecords": int(limit or 50),
                "startdatetime": pd.Timestamp(start_date).strftime(
                    "%Y%m%d000000"
                ),
                "enddatetime": pd.Timestamp(end_date).strftime(
                    "%Y%m%d235959"
                ),
            }
            request_url = self._request_url(params)
            diagnostic = {
                "provider": self.provider_name,
                "query": query_text,
                "request_url": request_url,
                "http_status": 0,
                "response_bytes": 0,
                "parsed_article_count": 0,
                "normalized_record_count": 0,
                "rate_limited": False,
                "retry_count": 0,
                "success": False,
                "error": "",
                "warning": "",
            }
            for attempt in range(self.max_retries + 1):
                try:
                    response, metadata = self._response(params)
                    diagnostic["request_url"] = metadata["request_url"]
                    diagnostic["http_status"] = metadata["http_status"]
                    diagnostic["response_bytes"] = metadata["response_bytes"]
                    articles = response.get("articles", [])
                    if not isinstance(articles, list):
                        raise GDELTResponseError(
                            "GDELT articles field must be a list",
                            request_url=str(metadata["request_url"]),
                            http_status=int(metadata["http_status"]),
                            response_bytes=int(metadata["response_bytes"]),
                        )
                    diagnostic["parsed_article_count"] = len(articles)
                    diagnostic["success"] = True
                    diagnostic["error"] = ""
                    successful_queries += 1
                    if diagnostic["rate_limited"]:
                        diagnostic["warning"] = (
                            "request succeeded after rate-limit retry"
                        )
                    elif not articles:
                        diagnostic["warning"] = (
                            "successful GDELT response contained no articles"
                        )
                    records.extend(
                        {**article, "query": query_text}
                        for article in articles
                        if isinstance(article, dict)
                    )
                    break
                except GDELTResponseError as exc:
                    diagnostic["request_url"] = exc.request_url
                    diagnostic["http_status"] = exc.http_status
                    diagnostic["response_bytes"] = exc.response_bytes
                    diagnostic["error"] = str(exc)
                    if exc.http_status == 429:
                        diagnostic["rate_limited"] = True
                        if attempt < self.max_retries:
                            diagnostic["retry_count"] = int(
                                diagnostic["retry_count"]
                            ) + 1
                            self.sleep_func(self.retry_delay_seconds)
                            continue
                    break
                except HTTPError as exc:
                    diagnostic["http_status"] = int(exc.code)
                    diagnostic["error"] = (
                        f"HTTP Error {exc.code}: {exc.reason}"
                    )
                    if exc.code == 429:
                        diagnostic["rate_limited"] = True
                        if attempt < self.max_retries:
                            diagnostic["retry_count"] = int(
                                diagnostic["retry_count"]
                            ) + 1
                            self.sleep_func(self.retry_delay_seconds)
                            continue
                    break
                except Exception as exc:
                    diagnostic["error"] = str(exc)
                    break
            if not diagnostic["success"]:
                failed_queries += 1
                failures.append(f"{query_text}: {diagnostic['error']}")
            query_diagnostics.append(diagnostic)

        records = self._deduplicate_articles(records)
        if records:
            status = "partial_success" if failed_queries else "success"
        elif successful_queries and not failed_queries:
            status = "empty"
        else:
            status = "error"
        cache_safe = (
            successful_queries == len(queries) and failed_queries == 0
        )
        self.last_diagnostics = {
            "provider": self.provider_name,
            "status": status,
            "fetched_record_count": int(len(records)),
            "query_count": int(len(queries)),
            "successful_query_count": int(successful_queries),
            "failure_count": int(failed_queries),
            "failures": " | ".join(failures),
            "source_kind": (
                "fixture" if self.fixture_path is not None else "api"
            ),
            "rate_limited": any(
                bool(row["rate_limited"]) for row in query_diagnostics
            ),
            "retry_count": int(
                sum(int(row["retry_count"]) for row in query_diagnostics)
            ),
            "all_queries_failed": bool(
                failed_queries == len(queries)
            ),
            "cache_safe": cache_safe,
            "query_diagnostics": query_diagnostics,
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
            body = str(
                raw.get("text")
                or raw.get("summary")
                or raw.get("snippet")
                or ""
            ).strip()
            title = str(raw.get("title") or "").strip()
            if not title:
                title = body[:240].strip() or url or "Untitled GDELT article"
            rows.append(
                {
                    "record_id": stable_record_id(
                        self.provider_name, url, title, publication
                    ),
                    "timestamp": publication,
                    "publication_time": publication,
                    "retrieval_time": retrieval,
                    "source": raw.get("domain")
                    or raw.get("source")
                    or "GDELT",
                    "provider": self.provider_name,
                    "document_type": "news",
                    "entity": raw.get("entity", ""),
                    "ticker": raw.get("ticker", ""),
                    "sector": raw.get("sector", ""),
                    "country": raw.get("sourcecountry")
                    or raw.get("country")
                    or "",
                    "title": title,
                    "text": body or title,
                    "url": url,
                    "language": raw.get("language") or "unknown",
                    "raw_metadata": raw,
                    "query": raw.get("query", ""),
                }
            )
        frame = normalized_frame(rows)
        counts = (
            frame.get("query", pd.Series(dtype="string"))
            .fillna("")
            .astype(str)
            .value_counts()
        )
        for diagnostic in self.last_diagnostics.get(
            "query_diagnostics", []
        ):
            diagnostic["normalized_record_count"] = int(
                counts.get(str(diagnostic.get("query", "")), 0)
            )
        return frame
