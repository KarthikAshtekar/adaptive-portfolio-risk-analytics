"""Optional RBI feed provider with local-manifest fallback."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Callable
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import pandas as pd

from src.sentiment.rbi_corpus_builder import load_real_rbi_corpus
from src.sentiment.rbi_ingestion import load_rbi_documents

from .base import (
    SentimentProvider,
    normalized_frame,
    stable_record_id,
)


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _document_type(title: object) -> str:
    text = str(title or "").lower()
    if "minute" in text or "mpc" in text:
        return "mpc_minutes"
    if "monetary policy" in text or "policy statement" in text:
        return "monetary_policy_statement"
    if "speech" in text or "governor" in text:
        return "governor_speech"
    if "financial stability" in text:
        return "financial_stability_report"
    if "annual report" in text:
        return "annual_report"
    return "press_release"


class RBIProvider(SentimentProvider):
    """Read configured RBI RSS feeds and fall back to reviewed local files."""

    provider_name = "rbi"

    def __init__(
        self,
        *,
        feeds_enabled: bool | None = None,
        feed_urls: list[str] | None = None,
        local_manifest_path: str | Path | None = None,
        local_corpus_path: str | Path | None = None,
        feed_loader: Callable[[str], str | bytes] | None = None,
    ) -> None:
        super().__init__()
        self.feeds_enabled = (
            _env_enabled("RBI_FEEDS_ENABLED", False)
            if feeds_enabled is None
            else bool(feeds_enabled)
        )
        configured_urls = os.getenv("RBI_FEED_URLS", "")
        self.feed_urls = feed_urls or [
            item.strip() for item in configured_urls.split(",") if item.strip()
        ]
        self.local_manifest_path = (
            Path(
                local_manifest_path
                or os.getenv(
                    "RBI_LOCAL_MANIFEST_PATH",
                    "data/sentiment/rbi_real/manifest.csv",
                )
            )
        )
        self.local_corpus_path = (
            Path(local_corpus_path) if local_corpus_path is not None else None
        )
        self.feed_loader = feed_loader

    def _read_feed(self, url: str) -> str:
        if self.feed_loader is not None:
            payload = self.feed_loader(url)
            return payload.decode("utf-8") if isinstance(payload, bytes) else payload
        local = Path(url)
        if local.is_file():
            return local.read_text(encoding="utf-8")
        with urlopen(url, timeout=15) as response:  # nosec B310 - opt-in URLs
            return response.read().decode("utf-8", errors="replace")

    @staticmethod
    def _parse_feed(xml_text: str, feed_url: str) -> list[dict[str, object]]:
        root = ET.fromstring(xml_text)
        records: list[dict[str, object]] = []
        for item in root.findall(".//item") + root.findall(
            ".//{http://www.w3.org/2005/Atom}entry"
        ):
            def first_text(*names: str) -> str:
                for name in names:
                    node = item.find(name)
                    if node is not None:
                        if node.text:
                            return node.text.strip()
                        href = node.attrib.get("href")
                        if href:
                            return href.strip()
                return ""

            records.append(
                {
                    "record_kind": "feed",
                    "title": first_text(
                        "title",
                        "{http://www.w3.org/2005/Atom}title",
                    ),
                    "text": first_text(
                        "description",
                        "summary",
                        "{http://www.w3.org/2005/Atom}summary",
                        "{http://purl.org/rss/1.0/modules/content/}encoded",
                    ),
                    "url": first_text(
                        "link",
                        "{http://www.w3.org/2005/Atom}link",
                    ),
                    "publication_time": first_text(
                        "pubDate",
                        "published",
                        "updated",
                        "{http://www.w3.org/2005/Atom}published",
                        "{http://www.w3.org/2005/Atom}updated",
                    ),
                    "feed_url": feed_url,
                }
            )
        return records

    def _local_records(self) -> list[dict[str, object]]:
        documents = load_real_rbi_corpus(self.local_manifest_path)
        source_kind = "real_rbi"
        if documents.empty and self.local_corpus_path is not None:
            documents = load_rbi_documents(self.local_corpus_path)
            source_kind = "local_fixture"
        records: list[dict[str, object]] = []
        for row in documents.to_dict("records"):
            records.append(
                {
                    "record_kind": "local",
                    "document_id": row.get("document_id"),
                    "publication_time": row.get("publication_date"),
                    "document_type": row.get("document_type"),
                    "title": row.get("title"),
                    "text": row.get("text"),
                    "url": row.get("source_url") or row.get("url"),
                    "language": row.get("language", "en"),
                    "source_kind": source_kind,
                    "local_path": row.get("local_path") or row.get("file_path"),
                }
            )
        return records

    def fetch(
        self,
        start_date,
        end_date,
        query=None,
        symbols=None,
        sectors=None,
        limit=None,
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        failures: list[str] = []
        if self.feeds_enabled:
            for feed_url in self.feed_urls:
                try:
                    records.extend(
                        self._parse_feed(self._read_feed(feed_url), feed_url)
                    )
                except Exception as exc:
                    failures.append(f"{feed_url}: {exc}")
        live_count = len(records)
        if not records:
            try:
                records.extend(self._local_records())
            except Exception as exc:
                failures.append(f"local fallback: {exc}")
        if limit is not None:
            records = records[: int(limit)]
        self.last_diagnostics = {
            "provider": self.provider_name,
            "status": "success" if records else "empty",
            "fetched_record_count": int(len(records)),
            "live_record_count": int(live_count),
            "fallback_used": bool(live_count == 0),
            "failure_count": int(len(failures)),
            "failures": " | ".join(failures),
        }
        return records

    def normalize(self, raw_records) -> pd.DataFrame:
        retrieval = pd.Timestamp(datetime.now(timezone.utc))
        rows: list[dict[str, object]] = []
        for raw in raw_records or []:
            publication = pd.to_datetime(
                raw.get("publication_time"), errors="coerce", utc=True
            )
            title = str(raw.get("title") or "").strip()
            url = str(raw.get("url") or "").strip()
            rows.append(
                {
                    "record_id": raw.get("document_id")
                    or stable_record_id(self.provider_name, url, publication, title),
                    "timestamp": publication,
                    "publication_time": publication,
                    "retrieval_time": retrieval,
                    "source": "Reserve Bank of India",
                    "provider": self.provider_name,
                    "document_type": raw.get("document_type")
                    or _document_type(title),
                    "entity": "Reserve Bank of India",
                    "ticker": "",
                    "sector": "central_bank",
                    "country": "IN",
                    "title": title,
                    "text": raw.get("text", ""),
                    "url": url,
                    "language": raw.get("language", "en"),
                    "raw_metadata": raw,
                }
            )
        return normalized_frame(rows)
