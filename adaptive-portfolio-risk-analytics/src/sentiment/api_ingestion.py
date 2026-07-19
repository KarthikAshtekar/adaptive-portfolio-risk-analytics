"""Unified, cache-aware ingestion for optional sentiment providers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .providers import SentimentProvider, normalized_frame


INGESTION_OUTPUT_FILES = (
    "raw_provider_records.jsonl",
    "normalized_sentiment_records.csv",
    "provider_diagnostics.csv",
    "deduped_sentiment_records.csv",
    "provider_query_diagnostics.csv",
)
QUERY_DIAGNOSTIC_COLUMNS = (
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


def _json_safe(records: object) -> list[dict[str, object]]:
    if isinstance(records, pd.DataFrame):
        values = records.to_dict("records")
    elif isinstance(records, list):
        values = records
    elif records is None:
        values = []
    else:
        values = [records]
    return [json.loads(json.dumps(record, default=str, ensure_ascii=False)) for record in values]


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
    options: dict[str, object] | None = None,
) -> Path:
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in provider_name
    )
    options_hash = hashlib.sha256(
        json.dumps(
            options or {},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    return cache_dir / (
        f"{safe_name}_{pd.Timestamp(start_date).date().isoformat()}_"
        f"{pd.Timestamp(end_date).date().isoformat()}_{options_hash}.json"
    )


def _read_cache(
    cache_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return _json_safe(payload), {
            "status": "success" if payload else "empty",
            "source_kind": "legacy_cache",
            "cache_safe": True,
            "query_diagnostics": [],
        }
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError("unsupported provider cache payload")
    records = _json_safe(payload.get("records"))
    diagnostics = payload.get("provider_diagnostics", {})
    return records, dict(diagnostics) if isinstance(diagnostics, dict) else {}


def _write_cache(
    cache_path: Path,
    *,
    provider_name: str,
    records: list[dict[str, object]],
    provider_diagnostics: dict[str, object],
) -> None:
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": provider_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "records": records,
                "provider_diagnostics": provider_diagnostics,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def _deduplicate(records: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if records.empty:
        return records.copy(), 0
    frame = records.copy()
    publication = pd.to_datetime(frame["publication_time"], errors="coerce", utc=True).dt.floor(
        "min"
    )
    url_key = frame["url"].fillna("").astype(str).str.strip().str.lower()
    title_key = frame["title"].fillna("").astype(str).str.strip().str.lower()
    frame["_dedupe_key"] = url_key.where(url_key.ne(""), title_key) + "|" + publication.astype(str)
    before = len(frame)
    frame = frame.drop_duplicates("_dedupe_key", keep="first").drop(columns="_dedupe_key")
    return frame.reset_index(drop=True), int(before - len(frame))


def run_sentiment_provider_ingestion(
    providers: Iterable[SentimentProvider],
    start_date,
    end_date,
    output_dir,
    query_config=None,
    use_cache: bool = True,
    cache_dir=None,
    ignore_cache: bool = False,
) -> dict[str, object]:
    """Fetch, normalize, validate, deduplicate, and persist provider records."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_dir) if cache_dir is not None else output / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    normalized_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    query_diagnostics: list[dict[str, object]] = []

    for provider in list(providers):
        options = _provider_options(provider, query_config)
        cache_path = _cache_path(
            cache_root,
            provider.provider_name,
            start_date,
            end_date,
            options,
        )
        cache_hit = False
        cache_written = False
        cache_ignored = bool(ignore_cache)
        fetch_error = ""
        raw_records: list[dict[str, object]] = []
        try:
            if use_cache and not ignore_cache and cache_path.is_file():
                raw_records, cached_diagnostics = _read_cache(cache_path)
                provider.last_diagnostics = cached_diagnostics
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
            normalized = provider.normalize(raw_records)
            validation = provider.validate(normalized)
            validated = validation.records.copy()
            normalized_parts.append(validated)
            valid_parts.append(validation.valid_records.copy())
            invalid_count = int(len(validation.invalid_records))
            valid_count = int(len(validation.valid_records))
            provider_diagnostics = dict(provider.last_diagnostics)
            cache_safe = provider_diagnostics.get("cache_safe")
            if cache_safe is None:
                cache_safe = (
                    not fetch_error
                    and provider_diagnostics.get("status")
                    in {"success", "partial_success", "empty"}
                    and not provider_diagnostics.get("failures")
                )
            if use_cache and not cache_hit and bool(cache_safe):
                _write_cache(
                    cache_path,
                    provider_name=provider.provider_name,
                    records=raw_records,
                    provider_diagnostics=provider_diagnostics,
                )
                cache_written = True
        except Exception as exc:
            fetch_error = str(exc)
            invalid_count = 0
            valid_count = 0
        raw_rows.extend(
            {"provider": provider.provider_name, "raw_record": row} for row in raw_records
        )
        provider_diagnostics = dict(provider.last_diagnostics)
        for row in provider_diagnostics.get("query_diagnostics", []) or []:
            if isinstance(row, dict):
                query_diagnostics.append(
                    {column: row.get(column, "") for column in QUERY_DIAGNOSTIC_COLUMNS}
                )
        diagnostics.append(
            {
                "provider": provider.provider_name,
                "status": provider_diagnostics.get("status", "error" if fetch_error else "success"),
                "cache_hit": cache_hit,
                "cache_written": cache_written,
                "cache_ignored": cache_ignored,
                "raw_record_count": int(len(raw_records)),
                "valid_record_count": valid_count,
                "invalid_record_count": invalid_count,
                "fetch_error": fetch_error,
                "provider_failures": provider_diagnostics.get("failures", ""),
                "fallback_used": provider_diagnostics.get("fallback_used", False),
                "source_kind": provider_diagnostics.get("source_kind", ""),
                "cache_path": str(cache_path),
                "rate_limited": provider_diagnostics.get("rate_limited", False),
                "retry_count": provider_diagnostics.get("retry_count", 0),
            }
        )

    normalized = (
        pd.concat(normalized_parts, ignore_index=True, sort=False)
        if normalized_parts
        else normalized_frame()
    )
    valid_normalized = (
        pd.concat(valid_parts, ignore_index=True, sort=False) if valid_parts else normalized_frame()
    )
    deduped, duplicate_count = _deduplicate(valid_normalized)
    diagnostics_frame = pd.DataFrame(diagnostics)
    if diagnostics_frame.empty:
        diagnostics_frame = pd.DataFrame(
            columns=[
                "provider",
                "status",
                "cache_hit",
                "cache_written",
                "cache_ignored",
                "raw_record_count",
                "valid_record_count",
                "invalid_record_count",
                "fetch_error",
                "provider_failures",
                "fallback_used",
                "source_kind",
                "cache_path",
                "rate_limited",
                "retry_count",
            ]
        )
    query_diagnostics_frame = pd.DataFrame(
        query_diagnostics,
        columns=QUERY_DIAGNOSTIC_COLUMNS,
    )
    diagnostics_frame["deduplicated_record_count"] = 0
    if not diagnostics_frame.empty:
        diagnostics_frame.loc[diagnostics_frame.index[0], "deduplicated_record_count"] = (
            duplicate_count
        )
    unique_counts = (
        deduped["provider"].astype(str).value_counts()
        if not deduped.empty
        else pd.Series(dtype="int64")
    )
    diagnostics_frame["deduped_valid_record_count"] = (
        diagnostics_frame["provider"].astype(str).map(unique_counts).fillna(0).astype(int)
    )
    diagnostics_frame["provider_duplicates_removed"] = (
        pd.to_numeric(diagnostics_frame["valid_record_count"], errors="coerce")
        .fillna(0)
        .astype(int)
        - diagnostics_frame["deduped_valid_record_count"]
    )

    with (output / "raw_provider_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    normalized.to_csv(output / "normalized_sentiment_records.csv", index=False)
    diagnostics_frame.to_csv(output / "provider_diagnostics.csv", index=False)
    deduped.to_csv(output / "deduped_sentiment_records.csv", index=False)
    query_diagnostics_frame.to_csv(output / "provider_query_diagnostics.csv", index=False)

    return {
        "raw_provider_records": raw_rows,
        "normalized_sentiment_records": normalized,
        "provider_diagnostics": diagnostics_frame,
        "deduped_sentiment_records": deduped,
        "provider_query_diagnostics": query_diagnostics_frame,
        "duplicate_record_count": duplicate_count,
        "output_dir": output,
    }
