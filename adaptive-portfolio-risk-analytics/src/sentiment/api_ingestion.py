"""Unified, cache-aware ingestion for optional sentiment providers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .providers import NORMALIZED_SENTIMENT_COLUMNS, SentimentProvider, normalized_frame


INGESTION_OUTPUT_FILES = (
    "raw_provider_records.jsonl",
    "normalized_sentiment_records.csv",
    "provider_diagnostics.csv",
    "deduped_sentiment_records.csv",
)


def _json_safe(records: object) -> list[dict[str, object]]:
    if isinstance(records, pd.DataFrame):
        values = records.to_dict("records")
    elif isinstance(records, list):
        values = records
    elif records is None:
        values = []
    else:
        values = [records]
    return [
        json.loads(json.dumps(record, default=str, ensure_ascii=False))
        for record in values
    ]


def _provider_options(
    provider: SentimentProvider,
    query_config: dict[str, object] | None,
) -> dict[str, object]:
    config = dict(query_config or {})
    provider_config = config.get(provider.provider_name, config.get("default", {}))
    return dict(provider_config) if isinstance(provider_config, dict) else {}


def _cache_path(
    cache_dir: Path,
    provider_name: str,
    start_date,
    end_date,
) -> Path:
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in provider_name
    )
    return cache_dir / (
        f"{safe_name}_{pd.Timestamp(start_date).date().isoformat()}_"
        f"{pd.Timestamp(end_date).date().isoformat()}.json"
    )


def _deduplicate(records: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if records.empty:
        return records.copy(), 0
    frame = records.copy()
    publication = pd.to_datetime(
        frame["publication_time"], errors="coerce", utc=True
    ).dt.floor("min")
    url_key = frame["url"].fillna("").astype(str).str.strip().str.lower()
    title_key = frame["title"].fillna("").astype(str).str.strip().str.lower()
    frame["_dedupe_key"] = (
        url_key.where(url_key.ne(""), title_key)
        + "|"
        + publication.astype(str)
    )
    before = len(frame)
    frame = frame.drop_duplicates("_dedupe_key", keep="first").drop(
        columns="_dedupe_key"
    )
    return frame.reset_index(drop=True), int(before - len(frame))


def run_sentiment_provider_ingestion(
    providers: Iterable[SentimentProvider],
    start_date,
    end_date,
    output_dir,
    query_config=None,
    use_cache: bool = True,
) -> dict[str, object]:
    """Fetch, normalize, validate, deduplicate, and persist provider records."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    normalized_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []

    for provider in list(providers):
        options = _provider_options(provider, query_config)
        cache_path = _cache_path(
            cache_dir,
            provider.provider_name,
            start_date,
            end_date,
        )
        cache_hit = False
        fetch_error = ""
        raw_records: list[dict[str, object]] = []
        try:
            if use_cache and cache_path.is_file():
                raw_records = json.loads(cache_path.read_text(encoding="utf-8"))
                cache_hit = True
            else:
                raw_records = _json_safe(
                    provider.fetch(
                        start_date,
                        end_date,
                        query=options.get("query"),
                        symbols=options.get("symbols"),
                        sectors=options.get("sectors"),
                        limit=options.get("limit"),
                    )
                )
                if use_cache:
                    cache_path.write_text(
                        json.dumps(
                            raw_records,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        ),
                        encoding="utf-8",
                    )
            normalized = provider.normalize(raw_records)
            validation = provider.validate(normalized)
            validated = validation.records.copy()
            normalized_parts.append(validated)
            valid_parts.append(validation.valid_records.copy())
            invalid_count = int(len(validation.invalid_records))
            valid_count = int(len(validation.valid_records))
        except Exception as exc:
            fetch_error = str(exc)
            invalid_count = 0
            valid_count = 0
        raw_rows.extend(
            {"provider": provider.provider_name, "raw_record": row}
            for row in raw_records
        )
        provider_diagnostics = dict(provider.last_diagnostics)
        diagnostics.append(
            {
                "provider": provider.provider_name,
                "status": provider_diagnostics.get(
                    "status", "error" if fetch_error else "success"
                ),
                "cache_hit": cache_hit,
                "raw_record_count": int(len(raw_records)),
                "valid_record_count": valid_count,
                "invalid_record_count": invalid_count,
                "fetch_error": fetch_error,
                "provider_failures": provider_diagnostics.get("failures", ""),
                "fallback_used": provider_diagnostics.get(
                    "fallback_used", False
                ),
                "source_kind": provider_diagnostics.get("source_kind", ""),
            }
        )

    normalized = (
        pd.concat(normalized_parts, ignore_index=True, sort=False)
        if normalized_parts
        else normalized_frame()
    )
    valid_normalized = (
        pd.concat(valid_parts, ignore_index=True, sort=False)
        if valid_parts
        else normalized_frame()
    )
    deduped, duplicate_count = _deduplicate(valid_normalized)
    diagnostics_frame = pd.DataFrame(diagnostics)
    if diagnostics_frame.empty:
        diagnostics_frame = pd.DataFrame(
            columns=[
                "provider",
                "status",
                "cache_hit",
                "raw_record_count",
                "valid_record_count",
                "invalid_record_count",
                "fetch_error",
                "provider_failures",
                "fallback_used",
                "source_kind",
            ]
        )
    diagnostics_frame["deduplicated_record_count"] = 0
    if not diagnostics_frame.empty:
        diagnostics_frame.loc[
            diagnostics_frame.index[0], "deduplicated_record_count"
        ] = duplicate_count
    unique_counts = (
        deduped["provider"].astype(str).value_counts()
        if not deduped.empty
        else pd.Series(dtype="int64")
    )
    diagnostics_frame["deduped_valid_record_count"] = (
        diagnostics_frame["provider"].astype(str).map(unique_counts).fillna(0).astype(int)
    )
    diagnostics_frame["provider_duplicates_removed"] = (
        pd.to_numeric(
            diagnostics_frame["valid_record_count"], errors="coerce"
        ).fillna(0).astype(int)
        - diagnostics_frame["deduped_valid_record_count"]
    )

    with (output / "raw_provider_records.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    normalized.to_csv(output / "normalized_sentiment_records.csv", index=False)
    diagnostics_frame.to_csv(output / "provider_diagnostics.csv", index=False)
    deduped.to_csv(output / "deduped_sentiment_records.csv", index=False)

    return {
        "raw_provider_records": raw_rows,
        "normalized_sentiment_records": normalized,
        "provider_diagnostics": diagnostics_frame,
        "deduped_sentiment_records": deduped,
        "duplicate_record_count": duplicate_count,
        "output_dir": output,
    }
