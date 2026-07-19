"""CLI-level tests for RBI official document fetching without internet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import fetch_rbi_documents
from src.sentiment.rbi_official_fetcher import build_rbi_document_id


def _entry(title: str = "Minutes of the Monetary Policy Committee Meeting") -> dict:
    return {
        "publication_date": "2026-06-06",
        "title": title,
        "summary": "Inflation and liquidity discussion.",
        "source_url": "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=1",
        "source_channel": "press_releases",
        "feed_url": "https://www.rbi.org.in/pressreleases_rss.xml",
    }


def test_cli_dry_run_does_not_write_raw_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "fetch_rbi_document_index",
        lambda *args, **kwargs: [_entry()],
    )
    raw_dir = tmp_path / "rbi_real" / "raw"
    manifest = tmp_path / "rbi_real" / "manifest.csv"
    diagnostics = tmp_path / "diagnostics"

    exit_code = fetch_rbi_documents.main(
        [
            "--from-date",
            "2026-06-01",
            "--to-date",
            "2026-06-30",
            "--output-dir",
            str(raw_dir),
            "--manifest-path",
            str(manifest),
            "--diagnostics-dir",
            str(diagnostics),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not list(raw_dir.glob("*.txt"))
    assert (diagnostics / "fetch_diagnostics.csv").is_file()
    frame = pd.read_csv(diagnostics / "fetch_diagnostics.csv")
    assert frame.loc[0, "download_status"] == "dry_run"


def test_existing_document_is_skipped_unless_refresh(
    tmp_path: Path,
    monkeypatch,
) -> None:
    entry = _entry()
    document_type = "mpc_minutes"
    document_id = build_rbi_document_id(
        entry["publication_date"],
        document_type,
        entry["title"],
    )
    raw_dir = tmp_path / "rbi_real" / "raw"
    raw_dir.mkdir(parents=True)
    cached = raw_dir / f"{document_id}.txt"
    cached.write_text("Cached policy text.", encoding="utf-8")
    manifest = tmp_path / "rbi_real" / "manifest.csv"
    pd.DataFrame(
        [
            {
                "document_id": document_id,
                "publication_date": entry["publication_date"],
                "document_type": document_type,
                "title": entry["title"],
                "local_path": f"raw/{document_id}.txt",
                "source_url": entry["source_url"],
                "retrieval_date": "2026-06-25",
                "language": "en",
                "notes": "existing",
            }
        ]
    ).to_csv(manifest, index=False)
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "fetch_rbi_document_index",
        lambda *args, **kwargs: [entry],
    )
    calls = {"downloads": 0}

    def fake_download(url: str) -> dict[str, object]:
        calls["downloads"] += 1
        return {
            "url": url,
            "ok": True,
            "content": b"<html><body><p>Refreshed policy text.</p></body></html>",
            "http_status": 200,
            "content_type": "text/html",
            "error": "",
        }

    monkeypatch.setattr(fetch_rbi_documents.fetcher, "download_rbi_document", fake_download)

    skipped = fetch_rbi_documents.run_fetch(
        from_date="2026-06-01",
        to_date="2026-06-30",
        output_dir=raw_dir,
        manifest_path=manifest,
        diagnostics_dir=tmp_path / "diagnostics_skip",
        request_delay_seconds=0,
    )

    assert skipped["skipped_existing_documents"] == 1
    assert skipped["downloaded_documents"] == 0
    assert calls["downloads"] == 0
    assert cached.read_text(encoding="utf-8") == "Cached policy text."

    refreshed = fetch_rbi_documents.run_fetch(
        from_date="2026-06-01",
        to_date="2026-06-30",
        output_dir=raw_dir,
        manifest_path=manifest,
        diagnostics_dir=tmp_path / "diagnostics_refresh",
        request_delay_seconds=0,
        refresh=True,
    )

    assert refreshed["skipped_existing_documents"] == 0
    assert refreshed["downloaded_documents"] == 1
    assert calls["downloads"] == 1
    assert "Refreshed policy text" in cached.read_text(encoding="utf-8")


def test_validate_after_reports_corpus_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "fetch_rbi_document_index",
        lambda *args, **kwargs: [_entry("Monetary Policy Statement, June 2026")],
    )
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "download_rbi_document",
        lambda url: {
            "url": url,
            "ok": True,
            "content": b"<html><body><p>Monetary Policy Statement text.</p></body></html>",
            "http_status": 200,
            "content_type": "text/html",
            "error": "",
        },
    )

    summary = fetch_rbi_documents.run_fetch(
        from_date="2026-06-01",
        to_date="2026-06-30",
        output_dir=tmp_path / "rbi_real" / "raw",
        manifest_path=tmp_path / "rbi_real" / "manifest.csv",
        diagnostics_dir=tmp_path / "diagnostics",
        request_delay_seconds=0,
        validate_after=True,
    )

    assert summary["downloaded_documents"] == 1
    assert summary["validation"]["valid_document_count"] == 1
    assert summary["validation"]["manual_action_required"] is True
