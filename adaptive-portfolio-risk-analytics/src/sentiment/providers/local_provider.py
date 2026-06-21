"""Simple local-record provider used for fixtures and reviewed exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import SentimentProvider, normalized_frame


class LocalProvider(SentimentProvider):
    """Load already normalized records from a DataFrame or CSV file."""

    provider_name = "local"

    def __init__(
        self,
        source: str | Path | pd.DataFrame,
        *,
        provider_name: str = "local",
    ) -> None:
        super().__init__()
        self.source = source
        self.provider_name = provider_name

    def fetch(
        self,
        start_date,
        end_date,
        query=None,
        symbols=None,
        sectors=None,
        limit=None,
    ) -> list[dict[str, object]]:
        frame = (
            self.source.copy()
            if isinstance(self.source, pd.DataFrame)
            else pd.read_csv(self.source)
        )
        if limit is not None:
            frame = frame.head(int(limit))
        self.last_diagnostics = {
            "provider": self.provider_name,
            "status": "success",
            "fetched_record_count": int(len(frame)),
            "source_kind": "local",
        }
        return frame.to_dict("records")

    def normalize(self, raw_records) -> pd.DataFrame:
        frame = normalized_frame(raw_records)
        frame["provider"] = frame["provider"].fillna(self.provider_name)
        return frame
