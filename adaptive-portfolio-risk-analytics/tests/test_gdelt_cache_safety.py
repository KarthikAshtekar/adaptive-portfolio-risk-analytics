"""GDELT cache bypass and failed-response safety tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

from scripts.collect_real_nlp_data import build_parser
from src.sentiment.api_ingestion import run_sentiment_provider_ingestion
from src.sentiment.providers.gdelt_provider import GDELTProvider


def _article(title: str) -> dict[str, object]:
    return {
        "url": f"https://news.example/{title.lower().replace(' ', '-')}",
        "title": title,
        "seendate": "20260620T103000Z",
        "domain": "news.example",
        "language": "English",
    }


def _options() -> dict[str, object]:
    return {"gdelt": {"query": ["India inflation"], "limit": 25}}


def test_no_cache_cli_flag_requests_cache_bypass() -> None:
    args = build_parser().parse_args(["--no-cache"])

    assert args.no_cache is True
    assert args.use_cache is True


def test_rate_limited_empty_response_is_not_cached(tmp_path) -> None:
    def loader(_params):
        raise HTTPError(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            429,
            "Too Many Requests",
            {},
            BytesIO(b"rate limited"),
        )

    provider = GDELTProvider(
        enabled=True,
        response_loader=loader,
        max_retries=1,
        sleep_func=lambda _seconds: None,
    )
    result = run_sentiment_provider_ingestion(
        [provider],
        "2026-04-01",
        "2026-06-21",
        tmp_path / "out",
        query_config=_options(),
        cache_dir=tmp_path / "cache",
    )

    diagnostic = result["provider_diagnostics"].iloc[0]
    assert diagnostic["cache_hit"] == False  # noqa: E712
    assert diagnostic["cache_written"] == False  # noqa: E712
    assert diagnostic["rate_limited"] == True  # noqa: E712
    assert diagnostic["retry_count"] == 1
    assert list((tmp_path / "cache").glob("*.json")) == []


def test_no_cache_ignores_existing_cache_and_replaces_success(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    first = GDELTProvider(
        enabled=True,
        response_loader=lambda _params: {"articles": [_article("Old article")]},
        sleep_func=lambda _seconds: None,
    )
    initial = run_sentiment_provider_ingestion(
        [first],
        "2026-04-01",
        "2026-06-21",
        tmp_path / "first",
        query_config=_options(),
        cache_dir=cache_dir,
    )
    assert initial["provider_diagnostics"].loc[0, "cache_written"]

    second = GDELTProvider(
        enabled=True,
        response_loader=lambda _params: {"articles": [_article("New article")]},
        sleep_func=lambda _seconds: None,
    )
    refreshed = run_sentiment_provider_ingestion(
        [second],
        "2026-04-01",
        "2026-06-21",
        tmp_path / "second",
        query_config=_options(),
        cache_dir=cache_dir,
        ignore_cache=True,
    )

    diagnostic = refreshed["provider_diagnostics"].iloc[0]
    records = refreshed["deduped_sentiment_records"]
    assert records.loc[0, "title"] == "New article"
    assert diagnostic["cache_ignored"] == True  # noqa: E712
    assert diagnostic["cache_hit"] == False  # noqa: E712
    assert diagnostic["cache_written"] == True  # noqa: E712
    assert list(refreshed["provider_query_diagnostics"].columns) == [
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
    ]


def test_non_json_provider_error_does_not_overwrite_good_cache(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    good = GDELTProvider(
        enabled=True,
        response_loader=lambda _params: {"articles": [_article("Good article")]},
        sleep_func=lambda _seconds: None,
    )
    initial = run_sentiment_provider_ingestion(
        [good],
        "2026-04-01",
        "2026-06-21",
        tmp_path / "good",
        query_config=_options(),
        cache_dir=cache_dir,
    )
    cache_path = initial["provider_diagnostics"].loc[0, "cache_path"]
    before = Path(cache_path).read_text(encoding="utf-8")

    bad = GDELTProvider(
        enabled=True,
        response_loader=lambda _params: "<html>not JSON</html>",
        sleep_func=lambda _seconds: None,
    )
    failed = run_sentiment_provider_ingestion(
        [bad],
        "2026-04-01",
        "2026-06-21",
        tmp_path / "bad",
        query_config=_options(),
        cache_dir=cache_dir,
        ignore_cache=True,
    )

    assert failed["provider_diagnostics"].loc[0, "cache_written"] == False  # noqa: E712
    assert Path(cache_path).read_text(encoding="utf-8") == before
