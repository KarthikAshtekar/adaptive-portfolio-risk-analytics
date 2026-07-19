"""Fetch official RBI documents into the governed real-RBI local cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bootstrap_rbi_real_corpus import bootstrap_rbi_real_corpus  # noqa: E402
from scripts.check_rbi_corpus_status import build_rbi_corpus_status  # noqa: E402
from src.sentiment import REAL_RBI_DOCUMENT_TYPES  # noqa: E402
from src.sentiment import rbi_official_fetcher as fetcher  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "sentiment" / "rbi_real" / "raw"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "sentiment" / "rbi_real" / "manifest.csv"
DEFAULT_DIAGNOSTICS_DIR = REPO_ROOT / "outputs" / "reports" / "rbi_official_fetcher"
DEFAULT_SOURCES = "press_releases,publications,speeches"
DEFAULT_KEYWORDS = (
    "monetary policy,mpc minutes,financial stability,governor speech,inflation,liquidity"
)
DEFAULT_EXCLUDE_KEYWORDS = ",".join(fetcher.DEFAULT_RBI_EXCLUDE_KEYWORDS)


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_existing_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.is_file():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(manifest_path, dtype="string").fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    return frame


def _notes(
    *,
    source_channel: str,
    extraction_method: str,
    warning: str = "",
) -> str:
    parts = [
        "fetched_by=rbi_official_fetcher",
        f"source_channel={source_channel}",
        f"extraction_method={extraction_method}",
    ]
    if warning:
        parts.append(f"warning={warning}")
    return "; ".join(parts)


def _remove_manifest_matches(
    manifest_path: Path,
    *,
    document_ids: set[str] | None = None,
    source_urls: set[str] | None = None,
) -> int:
    """Remove known skipped/non-corpus rows from the manifest only."""
    if not manifest_path.is_file():
        return 0
    frame = _read_existing_manifest(manifest_path)
    if frame.empty:
        return 0
    for column in fetcher.REAL_RBI_MANIFEST_COLUMNS:
        if column not in frame:
            frame[column] = ""
    mask = pd.Series(False, index=frame.index)
    if document_ids and "document_id" in frame:
        mask |= frame["document_id"].astype(str).str.strip().isin(document_ids)
    if source_urls and "source_url" in frame:
        mask |= frame["source_url"].astype(str).str.strip().isin(source_urls)
    removed = int(mask.sum())
    if removed:
        frame = frame.loc[~mask, list(fetcher.REAL_RBI_MANIFEST_COLUMNS)]
        frame.to_csv(manifest_path, index=False)
    return removed


def _diagnostic_row(
    entry: dict[str, object],
    *,
    document_type: str,
    download_status: str,
    http_status: int = 0,
    content_type: str = "",
    local_path: str = "",
    text_char_count: int = 0,
    substantive_sentence_count: int = 0,
    boilerplate_ratio_estimate: float = 0.0,
    is_index_page: bool = False,
    index_page_reason: str = "",
    excluded_by_keyword: bool = False,
    exclude_keyword_matched: str = "",
    target_policy_docs_mode: bool = False,
    target_document_type_match: bool = False,
    target_priority_rank: int = 999,
    matched_policy_phrase: str = "",
    candidate_source_page: str = "",
    included_in_manifest: bool = False,
    skip_reason: str = "",
    extraction_method: str = "",
    warning: str = "",
    error: str = "",
) -> dict[str, object]:
    return {
        "publication_date": entry.get("publication_date", ""),
        "title": entry.get("title", ""),
        "document_type": document_type,
        "source_url": entry.get("source_url", ""),
        "source_channel": entry.get("source_channel", ""),
        "download_status": download_status,
        "http_status": int(http_status or 0),
        "content_type": content_type,
        "local_path": local_path,
        "text_char_count": int(text_char_count or 0),
        "substantive_sentence_count": int(substantive_sentence_count or 0),
        "boilerplate_ratio_estimate": float(boilerplate_ratio_estimate or 0.0),
        "is_index_page": bool(is_index_page),
        "index_page_reason": index_page_reason,
        "excluded_by_keyword": bool(excluded_by_keyword),
        "exclude_keyword_matched": exclude_keyword_matched,
        "target_policy_docs_mode": bool(target_policy_docs_mode),
        "target_document_type_match": bool(target_document_type_match),
        "target_priority_rank": int(target_priority_rank or 999),
        "matched_policy_phrase": matched_policy_phrase,
        "candidate_source_page": candidate_source_page
        or str(entry.get("candidate_source_page", "") or entry.get("feed_url", "")),
        "included_in_manifest": bool(included_in_manifest),
        "skip_reason": skip_reason,
        "extraction_method": extraction_method,
        "warning": warning,
        "error": error,
    }


def _target_fields(
    entry: dict[str, object],
    *,
    document_type: str,
    target_policy_docs_mode: bool,
    target_document_types: set[str],
    text: str = "",
) -> dict[str, object]:
    metadata = fetcher.rbi_policy_target_metadata(
        entry.get("title", ""),
        entry.get("source_url", ""),
        text or entry.get("summary", ""),
        document_type=document_type,
    )
    resolved_type = str(metadata["document_type"])
    type_match = (
        resolved_type in target_document_types
        if target_document_types
        else bool(metadata["is_policy_target"])
    )
    return {
        "target_policy_docs_mode": bool(target_policy_docs_mode),
        "target_document_type_match": bool(type_match),
        "target_priority_rank": int(metadata["target_priority_rank"]),
        "matched_policy_phrase": str(metadata["matched_policy_phrase"]),
        "candidate_source_page": str(
            entry.get("candidate_source_page", "") or entry.get("feed_url", "")
        ),
    }


def _target_summary_counts(
    diagnostics: list[dict[str, object]],
) -> dict[str, int]:
    target_rows = [row for row in diagnostics if bool(row.get("target_policy_docs_mode", False))]
    found_statuses = {"downloaded", "skipped_existing"}
    targeted_found = [
        row
        for row in target_rows
        if str(row.get("download_status", "")) in found_statuses
        and bool(row.get("target_document_type_match", False))
    ]
    return {
        "targeted_mpc_minutes_found": sum(
            1 for row in targeted_found if row.get("document_type") == "mpc_minutes"
        ),
        "targeted_monetary_policy_statements_found": sum(
            1 for row in targeted_found if row.get("document_type") == "monetary_policy_statement"
        ),
        "targeted_documents_downloaded": sum(
            1 for row in target_rows if row.get("download_status") == "downloaded"
        ),
        "targeted_documents_skipped": sum(
            1 for row in target_rows if row.get("download_status") != "downloaded"
        ),
    }


def _write_diagnostics(
    *,
    diagnostics_dir: Path,
    diagnostics: list[dict[str, object]],
    summary: dict[str, object],
    downloaded_documents: list[dict[str, object]],
) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_frame = pd.DataFrame(
        diagnostics,
        columns=fetcher.FETCH_DIAGNOSTIC_COLUMNS,
    )
    diagnostics_frame.to_csv(diagnostics_dir / "fetch_diagnostics.csv", index=False)
    manual = diagnostics_frame.loc[
        diagnostics_frame["download_status"].isin({"manual_required", "failed"})
        | diagnostics_frame["warning"].astype(str).ne("")
    ].copy()
    manual.to_csv(diagnostics_dir / "manual_fallback_required.csv", index=False)
    pd.DataFrame(
        downloaded_documents,
        columns=[
            "document_id",
            "publication_date",
            "document_type",
            "title",
            "local_path",
            "source_url",
            "source_channel",
            "extraction_method",
        ],
    ).to_csv(diagnostics_dir / "downloaded_documents.csv", index=False)
    lines = [
        "# RBI Official Fetcher Summary",
        "",
        "RBI official-source document fetching is a corpus-population workflow only.",
        "It does not alter allocation, strategy scoring, evidence gates, confidence, or backtests.",
        "",
        f"- From date: {summary['from_date']}",
        f"- To date: {summary['to_date']}",
        f"- Fetched index entries: {summary['fetched_index_entries']}",
        f"- Relevant entries: {summary['relevant_entries']}",
        f"- Excluded irrelevant press releases: {summary['excluded_irrelevant_press_releases']}",
        f"- Target policy docs mode: {'yes' if summary['target_policy_docs_mode'] else 'no'}",
        f"- Target document types: `{summary['target_document_types']}`",
        f"- Targeted MPC minutes found: {summary['targeted_mpc_minutes_found']}",
        f"- Targeted monetary policy statements found: {summary['targeted_monetary_policy_statements_found']}",
        f"- Targeted documents downloaded: {summary['targeted_documents_downloaded']}",
        f"- Targeted documents skipped: {summary['targeted_documents_skipped']}",
        f"- Downloaded documents: {summary['downloaded_documents']}",
        f"- Skipped index/navigation pages: {summary['skipped_index_pages']}",
        f"- Skipped existing documents: {summary['skipped_existing_documents']}",
        f"- Failed documents: {summary['failed_documents']}",
        f"- Manual fallback required: {summary['manual_fallback_required']}",
        f"- Dry run: {'yes' if summary['dry_run'] else 'no'}",
        f"- Refresh: {'yes' if summary['refresh'] else 'no'}",
        f"- Manifest path: `{summary['manifest_path']}`",
        f"- Diagnostics path: `{summary['diagnostics_path']}`",
    ]
    if "validation" in summary:
        validation = summary["validation"]
        lines.extend(
            [
                "",
                "## Corpus status",
                "",
                f"- Valid document count: {validation['valid_document_count']}",
                f"- Distinct publication dates: {validation['distinct_publication_dates']}",
                f"- Document type counts: {validation['document_type_counts']}",
                f"- Manual action required: {'yes' if validation['manual_action_required'] else 'no'}",
            ]
        )
    (diagnostics_dir / "fetch_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_fetch(
    *,
    from_date: str,
    to_date: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    sources: str = DEFAULT_SOURCES,
    keywords: str = DEFAULT_KEYWORDS,
    exclude_keywords: str | None = DEFAULT_EXCLUDE_KEYWORDS,
    min_policy_relevance: str = "none",
    target_policy_docs: bool = False,
    target_document_types: str | None = None,
    max_documents: int | None = 50,
    request_delay_seconds: float = 5,
    refresh: bool = False,
    dry_run: bool = False,
    validate_after: bool = False,
    diagnostics_dir: str | Path = DEFAULT_DIAGNOSTICS_DIR,
) -> dict[str, object]:
    """Run the official RBI fetch workflow and return summary diagnostics."""
    manifest = Path(manifest_path)
    raw_dir = Path(output_dir)
    diagnostics_root = Path(diagnostics_dir)
    if not manifest.is_absolute():
        manifest = (REPO_ROOT / manifest).resolve()
    if not raw_dir.is_absolute():
        raw_dir = (REPO_ROOT / raw_dir).resolve()
    if not diagnostics_root.is_absolute():
        diagnostics_root = (REPO_ROOT / diagnostics_root).resolve()

    bootstrap_rbi_real_corpus(corpus_dir=manifest.parent)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_type_filter = fetcher.parse_rbi_target_document_types(
        target_document_types,
    )
    source_spec = (
        fetcher.resolve_rbi_target_policy_sources(sources) if target_policy_docs else sources
    )

    fetched_entries = fetcher.fetch_rbi_document_index(
        from_date,
        to_date,
        sources=source_spec,
        keywords=None,
    )
    fetched_without_index_errors = [
        entry for entry in fetched_entries if not entry.get("index_error")
    ]
    diagnostics: list[dict[str, object]] = []
    relevance_candidates: list[dict[str, object]] = []
    excluded_irrelevant = 0
    manifest_rows_removed = 0
    for entry in fetched_without_index_errors:
        matched_keyword = fetcher.rbi_entry_exclude_keyword_match(
            entry,
            exclude_keywords,
        )
        if not matched_keyword:
            relevance_candidates.append(entry)
            continue
        excluded_irrelevant += 1
        title = str(entry.get("title", "") or "").strip()
        source_url = str(entry.get("source_url", "") or "").strip()
        publication_date = str(entry.get("publication_date", "") or "").strip()
        document_type = fetcher.classify_rbi_document_type(title, source_url)
        if document_type not in REAL_RBI_DOCUMENT_TYPES:
            document_type = "unknown"
        document_id = fetcher.build_rbi_document_id(
            publication_date,
            document_type,
            title,
        )
        target_fields = _target_fields(
            entry,
            document_type=document_type,
            target_policy_docs_mode=target_policy_docs,
            target_document_types=target_type_filter,
        )
        destination = raw_dir / f"{document_id}.txt"
        manifest_rows_removed += _remove_manifest_matches(
            manifest,
            document_ids={document_id},
            source_urls={source_url},
        )
        diagnostics.append(
            _diagnostic_row(
                entry,
                document_type=document_type,
                download_status="skipped",
                local_path=_repo_relative(destination, manifest.parent),
                excluded_by_keyword=True,
                exclude_keyword_matched=matched_keyword,
                **target_fields,
                included_in_manifest=False,
                skip_reason="excluded_irrelevant_press_release",
                extraction_method="not_downloaded",
            )
        )
    relevant_entries = fetcher.filter_rbi_entries_by_keywords(
        relevance_candidates,
        keywords,
    )
    relevant_entries = fetcher.filter_rbi_entries_by_policy_relevance(
        relevant_entries,
        min_policy_relevance,
    )
    if target_policy_docs:
        relevant_entries = fetcher.sort_rbi_entries_for_policy_targets(
            relevant_entries,
        )
    download_candidates: list[dict[str, object]] = list(relevant_entries)
    target_skipped_candidates = 0
    if target_type_filter:
        filtered_candidates: list[dict[str, object]] = []
        for entry in download_candidates:
            title = str(entry.get("title", "") or "").strip()
            source_url = str(entry.get("source_url", "") or "").strip()
            publication_date = str(entry.get("publication_date", "") or "").strip()
            document_type = fetcher.classify_rbi_document_type(title, source_url)
            if document_type not in REAL_RBI_DOCUMENT_TYPES:
                document_type = "unknown"
            if document_type in target_type_filter:
                filtered_candidates.append(entry)
                continue
            target_skipped_candidates += 1
            document_id = fetcher.build_rbi_document_id(
                publication_date,
                document_type,
                title,
            )
            destination = raw_dir / f"{document_id}.txt"
            diagnostics.append(
                _diagnostic_row(
                    entry,
                    document_type=document_type,
                    download_status="skipped",
                    local_path=_repo_relative(destination, manifest.parent),
                    **_target_fields(
                        entry,
                        document_type=document_type,
                        target_policy_docs_mode=target_policy_docs,
                        target_document_types=target_type_filter,
                    ),
                    included_in_manifest=False,
                    skip_reason="non_target_document_type",
                    extraction_method="not_downloaded",
                )
            )
        download_candidates = filtered_candidates
    if max_documents is not None:
        download_candidates = download_candidates[: max(0, int(max_documents))]

    existing_manifest = _read_existing_manifest(manifest)
    existing_ids = (
        set(existing_manifest["document_id"].astype(str).str.strip()) - {""}
        if "document_id" in existing_manifest
        else set()
    )
    existing_by_url = {}
    if "source_url" in existing_manifest:
        for row in existing_manifest.to_dict("records"):
            source_url = str(row.get("source_url", "") or "").strip()
            if source_url:
                existing_by_url[source_url] = row
    manifest_records: list[dict[str, str]] = []
    downloaded_documents: list[dict[str, object]] = []
    skipped_existing = 0
    skipped_index_pages = 0
    failed_documents = 0
    manual_fallback_required = 0
    downloaded_count = 0
    retrieval_date = datetime.now(timezone.utc).date().isoformat()

    for position, entry in enumerate(download_candidates):
        title = str(entry.get("title", "") or "").strip()
        source_url = str(entry.get("source_url", "") or "").strip()
        publication_date = str(entry.get("publication_date", "") or "").strip()
        source_channel = str(entry.get("source_channel", "") or "").strip()
        document_type = fetcher.classify_rbi_document_type(title, source_url)
        if document_type not in REAL_RBI_DOCUMENT_TYPES:
            document_type = "unknown"
        entry_target_fields = _target_fields(
            entry,
            document_type=document_type,
            target_policy_docs_mode=target_policy_docs,
            target_document_types=target_type_filter,
        )
        document_id = fetcher.build_rbi_document_id(
            publication_date,
            document_type,
            title,
        )
        destination = raw_dir / f"{document_id}.txt"
        local_path = _repo_relative(destination, manifest.parent)

        existing_row = existing_by_url.get(source_url)
        existing_local_path = (
            manifest.parent / str(existing_row.get("local_path", ""))
            if existing_row
            else destination
        )
        if not refresh and (
            (document_id in existing_ids and destination.is_file())
            or (existing_row is not None and existing_local_path.is_file())
        ):
            skip_path = existing_local_path if existing_local_path.is_file() else destination
            cached_text = skip_path.read_text(encoding="utf-8")
            final_document_type = fetcher.classify_rbi_document_type(
                title,
                source_url,
                cached_text,
            )
            quality = fetcher.rbi_text_quality_metrics(
                title,
                cached_text,
                source_url,
            )
            is_index_page, index_reason = fetcher.is_rbi_index_or_navigation_page(
                title,
                cached_text,
                source_url,
            )
            if is_index_page:
                skipped_index_pages += 1
                manifest_rows_removed += _remove_manifest_matches(
                    manifest,
                    document_ids={
                        document_id,
                        str(existing_row.get("document_id", "")) if existing_row else "",
                    },
                    source_urls={source_url},
                )
                diagnostics.append(
                    _diagnostic_row(
                        entry,
                        document_type=final_document_type,
                        download_status="skipped",
                        local_path=_repo_relative(skip_path, manifest.parent),
                        text_char_count=len(cached_text),
                        substantive_sentence_count=int(quality["substantive_sentence_count"]),
                        boilerplate_ratio_estimate=float(quality["boilerplate_ratio_estimate"]),
                        is_index_page=True,
                        index_page_reason=index_reason,
                        **_target_fields(
                            entry,
                            document_type=final_document_type,
                            target_policy_docs_mode=target_policy_docs,
                            target_document_types=target_type_filter,
                            text=cached_text,
                        ),
                        included_in_manifest=False,
                        skip_reason="index_or_navigation_page",
                        extraction_method="cache",
                    )
                )
                continue
            skipped_existing += 1
            diagnostics.append(
                _diagnostic_row(
                    entry,
                    document_type=final_document_type,
                    download_status="skipped_existing",
                    local_path=_repo_relative(skip_path, manifest.parent),
                    text_char_count=len(cached_text),
                    substantive_sentence_count=int(quality["substantive_sentence_count"]),
                    boilerplate_ratio_estimate=float(quality["boilerplate_ratio_estimate"]),
                    **_target_fields(
                        entry,
                        document_type=final_document_type,
                        target_policy_docs_mode=target_policy_docs,
                        target_document_types=target_type_filter,
                        text=cached_text,
                    ),
                    included_in_manifest=True,
                    extraction_method="cache",
                )
            )
            continue

        if dry_run:
            diagnostics.append(
                _diagnostic_row(
                    entry,
                    document_type=document_type,
                    download_status="dry_run",
                    local_path=local_path,
                    **entry_target_fields,
                    included_in_manifest=False,
                    extraction_method="not_downloaded",
                )
            )
            continue

        result = fetcher.download_rbi_document(source_url)
        if not bool(result.get("ok")):
            failed_documents += 1
            warning = str(result.get("error", "download failed"))
            manual_fallback_required += 1
            diagnostics.append(
                _diagnostic_row(
                    entry,
                    document_type=document_type,
                    download_status="failed",
                    http_status=int(result.get("http_status", 0) or 0),
                    content_type=str(result.get("content_type", "")),
                    local_path=local_path,
                    **entry_target_fields,
                    included_in_manifest=False,
                    extraction_method="manual_required",
                    warning=warning,
                    error=warning,
                )
            )
            continue

        extraction = fetcher.extract_rbi_text_with_diagnostics(
            result,
            content_type=str(result.get("content_type", "")),
        )
        text = str(extraction.get("text", "") or "").strip()
        extraction_method = str(extraction.get("extraction_method", ""))
        warning = str(extraction.get("warning", "") or "")
        document_type = fetcher.classify_rbi_document_type(title, source_url, text)
        if document_type not in REAL_RBI_DOCUMENT_TYPES:
            document_type = "unknown"
        entry_target_fields = _target_fields(
            entry,
            document_type=document_type,
            target_policy_docs_mode=target_policy_docs,
            target_document_types=target_type_filter,
            text=text,
        )
        document_id = fetcher.build_rbi_document_id(
            publication_date,
            document_type,
            title,
        )
        destination = raw_dir / f"{document_id}.txt"
        local_path = _repo_relative(destination, manifest.parent)
        quality = fetcher.rbi_text_quality_metrics(title, text, source_url)
        is_index_page, index_reason = fetcher.is_rbi_index_or_navigation_page(
            title,
            text,
            source_url,
        )
        if not text:
            manual_fallback_required += 1
            diagnostics.append(
                _diagnostic_row(
                    entry,
                    document_type=document_type,
                    download_status="manual_required",
                    http_status=int(result.get("http_status", 0) or 0),
                    content_type=str(result.get("content_type", "")),
                    local_path=local_path,
                    **entry_target_fields,
                    included_in_manifest=False,
                    extraction_method="manual_required",
                    warning=warning,
                    error=str(extraction.get("error", "") or warning),
                )
            )
            continue
        if target_type_filter and document_type not in target_type_filter:
            diagnostics.append(
                _diagnostic_row(
                    entry,
                    document_type=document_type,
                    download_status="skipped",
                    http_status=int(result.get("http_status", 0) or 0),
                    content_type=str(result.get("content_type", "")),
                    local_path=local_path,
                    text_char_count=len(text),
                    substantive_sentence_count=int(quality["substantive_sentence_count"]),
                    boilerplate_ratio_estimate=float(quality["boilerplate_ratio_estimate"]),
                    **entry_target_fields,
                    included_in_manifest=False,
                    skip_reason="non_target_document_type",
                    extraction_method=extraction_method,
                )
            )
            continue
        if is_index_page:
            skipped_index_pages += 1
            manifest_rows_removed += _remove_manifest_matches(
                manifest,
                document_ids={document_id},
                source_urls={source_url},
            )
            diagnostics.append(
                _diagnostic_row(
                    entry,
                    document_type=document_type,
                    download_status="skipped",
                    http_status=int(result.get("http_status", 0) or 0),
                    content_type=str(result.get("content_type", "")),
                    local_path=local_path,
                    text_char_count=len(text),
                    substantive_sentence_count=int(quality["substantive_sentence_count"]),
                    boilerplate_ratio_estimate=float(quality["boilerplate_ratio_estimate"]),
                    is_index_page=True,
                    index_page_reason=index_reason,
                    **entry_target_fields,
                    included_in_manifest=False,
                    skip_reason="index_or_navigation_page",
                    extraction_method=extraction_method,
                )
            )
            continue

        destination.write_text(text + "\n", encoding="utf-8", newline="\n")
        downloaded_count += 1
        row = fetcher.build_manifest_record(
            document_id=document_id,
            publication_date=publication_date,
            document_type=document_type,
            title=title,
            local_path=local_path,
            source_url=source_url,
            retrieval_date=retrieval_date,
            notes=_notes(
                source_channel=source_channel,
                extraction_method=extraction_method,
                warning=warning,
            ),
        )
        manifest_records.append(row)
        downloaded_documents.append(
            {
                "document_id": document_id,
                "publication_date": publication_date,
                "document_type": document_type,
                "title": title,
                "local_path": local_path,
                "source_url": source_url,
                "source_channel": source_channel,
                "extraction_method": extraction_method,
            }
        )
        diagnostics.append(
            _diagnostic_row(
                entry,
                document_type=document_type,
                download_status="downloaded",
                http_status=int(result.get("http_status", 0) or 0),
                content_type=str(result.get("content_type", "")),
                local_path=local_path,
                text_char_count=len(text),
                substantive_sentence_count=int(quality["substantive_sentence_count"]),
                boilerplate_ratio_estimate=float(quality["boilerplate_ratio_estimate"]),
                **entry_target_fields,
                included_in_manifest=True,
                extraction_method=extraction_method,
                warning=warning,
            )
        )
        if position < len(download_candidates) - 1:
            fetcher.sleep_respecting_rate_limit(request_delay_seconds)

    manifest_update = (
        fetcher.update_rbi_manifest(manifest_records, manifest)
        if manifest_records and not dry_run
        else {"added_count": 0, "updated_count": 0}
    )
    target_counts = _target_summary_counts(diagnostics)
    summary: dict[str, object] = {
        "from_date": from_date,
        "to_date": to_date,
        "fetched_index_entries": int(len(fetched_without_index_errors)),
        "relevant_entries": int(len(relevant_entries)),
        "excluded_irrelevant_press_releases": int(excluded_irrelevant),
        "target_policy_docs_mode": bool(target_policy_docs),
        "target_document_types": ",".join(sorted(target_type_filter)),
        "target_skipped_candidates": int(target_skipped_candidates),
        **target_counts,
        "download_candidate_entries": int(len(download_candidates)),
        "downloaded_documents": int(downloaded_count),
        "skipped_index_pages": int(skipped_index_pages),
        "skipped_existing_documents": int(skipped_existing),
        "failed_documents": int(failed_documents),
        "manual_fallback_required": int(manual_fallback_required),
        "manifest_rows_removed": int(manifest_rows_removed),
        "manifest_added_count": int(manifest_update.get("added_count", 0)),
        "manifest_updated_count": int(manifest_update.get("updated_count", 0)),
        "manifest_path": str(manifest),
        "diagnostics_path": str(diagnostics_root),
        "dry_run": bool(dry_run),
        "refresh": bool(refresh),
    }
    if validate_after:
        status, _ = build_rbi_corpus_status(manifest)
        summary["validation"] = status
    _write_diagnostics(
        diagnostics_dir=diagnostics_root,
        diagnostics=diagnostics,
        summary=summary,
        downloaded_documents=downloaded_documents,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch official RBI documents into the local real-RBI corpus."
    )
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--to-date", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--keywords", default=DEFAULT_KEYWORDS)
    parser.add_argument(
        "--exclude-keywords",
        default=DEFAULT_EXCLUDE_KEYWORDS,
        help=(
            "Comma-separated title keywords for irrelevant RBI releases to skip with diagnostics."
        ),
    )
    parser.add_argument(
        "--min-policy-relevance",
        default="none",
        choices=["none", "low", "medium", "high"],
        help="Minimum pre-download RBI title/summary policy relevance.",
    )
    parser.add_argument(
        "--target-policy-docs",
        action="store_true",
        help=(
            "Use conservative official RBI archive discovery and prioritize "
            "MPC minutes / monetary policy statement candidates."
        ),
    )
    parser.add_argument(
        "--target-document-types",
        default=None,
        help=(
            "Comma-separated allowed RBI document types for this fetch; skipped "
            "non-target candidates are written to diagnostics."
        ),
    )
    parser.add_argument("--max-documents", type=int, default=50)
    parser.add_argument("--request-delay-seconds", type=float, default=5)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-after", action="store_true")
    parser.add_argument("--diagnostics-dir", default=str(DEFAULT_DIAGNOSTICS_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_fetch(
            from_date=args.from_date,
            to_date=args.to_date,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
            sources=args.sources,
            keywords=args.keywords,
            exclude_keywords=args.exclude_keywords,
            min_policy_relevance=args.min_policy_relevance,
            target_policy_docs=args.target_policy_docs,
            target_document_types=args.target_document_types,
            max_documents=args.max_documents,
            request_delay_seconds=args.request_delay_seconds,
            refresh=args.refresh,
            dry_run=args.dry_run,
            validate_after=args.validate_after,
            diagnostics_dir=args.diagnostics_dir,
        )
    except Exception as exc:
        print(f"RBI official fetch failed: {exc}", file=sys.stderr)
        return 1

    print("RBI official fetch complete.")
    print(f"Fetched index entries: {summary['fetched_index_entries']}")
    print(f"Relevant entries: {summary['relevant_entries']}")
    print(f"Excluded irrelevant press releases: {summary['excluded_irrelevant_press_releases']}")
    print("Target policy docs mode: " + ("yes" if summary["target_policy_docs_mode"] else "no"))
    print(f"Target document types: {summary['target_document_types']}")
    print(f"Targeted MPC minutes found: {summary['targeted_mpc_minutes_found']}")
    print(
        "Targeted monetary policy statements found: "
        f"{summary['targeted_monetary_policy_statements_found']}"
    )
    print(f"Targeted documents downloaded: {summary['targeted_documents_downloaded']}")
    print(f"Targeted documents skipped: {summary['targeted_documents_skipped']}")
    print(f"Downloaded documents: {summary['downloaded_documents']}")
    print(f"Skipped index/navigation pages: {summary['skipped_index_pages']}")
    print(f"Skipped existing documents: {summary['skipped_existing_documents']}")
    print(f"Failed documents: {summary['failed_documents']}")
    print(f"Manual fallback required: {summary['manual_fallback_required']}")
    print(f"Manifest: {Path(str(summary['manifest_path'])).resolve()}")
    print(f"Diagnostics: {Path(str(summary['diagnostics_path'])).resolve()}")
    if "validation" in summary:
        validation = summary["validation"]
        print(f"Valid document count: {validation['valid_document_count']}")
        print(f"Distinct publication dates: {validation['distinct_publication_dates']}")
        print(f"Document type counts: {validation['document_type_counts']}")
        if "policy_core_documents" in validation:
            print(f"Policy core documents: {validation['policy_core_documents']}")
        print(
            "Manual action required: " + ("yes" if validation["manual_action_required"] else "no")
        )
    print("Allocation impact: None. NLP remains monitoring-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
