"""GDELT query pacing, retry, partial-success, and normalization tests."""

from __future__ import annotations

from io import BytesIO
from urllib.error import HTTPError

from src.sentiment.providers.gdelt_provider import GDELTProvider


def _article(
    title: str = "India inflation outlook",
    *,
    url: str = "https://news.example/india-inflation",
) -> dict[str, object]:
    return {
        "url": url,
        "title": title,
        "seendate": "20260620T103000Z",
        "domain": "news.example",
        "sourcecountry": "India",
        "language": "English",
    }


def _rate_limit_error() -> HTTPError:
    return HTTPError(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        429,
        "Too Many Requests",
        {},
        BytesIO(b"rate limited"),
    )


def test_gdelt_retries_http_429_with_bounded_delay() -> None:
    calls = 0
    sleeps: list[float] = []

    def loader(_params):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _rate_limit_error()
        return {"articles": [_article()]}

    provider = GDELTProvider(
        enabled=True,
        response_loader=loader,
        retry_delay_seconds=10,
        max_retries=3,
        sleep_func=sleeps.append,
    )
    records = provider.fetch(
        "2026-04-01",
        "2026-06-21",
        query="India inflation",
        limit=25,
    )

    diagnostic = provider.last_diagnostics["query_diagnostics"][0]
    assert len(records) == 1
    assert calls == 2
    assert sleeps == [10.0]
    assert diagnostic["rate_limited"] is True
    assert diagnostic["retry_count"] == 1
    assert diagnostic["success"] is True
    assert diagnostic["error"] == ""


def test_gdelt_retains_partial_success_and_paces_queries() -> None:
    sleeps: list[float] = []

    def loader(params):
        if params["query"] == "failed query":
            raise HTTPError(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                503,
                "Unavailable",
                {},
                BytesIO(b"unavailable"),
            )
        return {"articles": [_article("Successful query article")]}

    provider = GDELTProvider(
        enabled=True,
        response_loader=loader,
        request_delay_seconds=6,
        sleep_func=sleeps.append,
    )
    records = provider.fetch(
        "2026-04-01",
        "2026-06-21",
        query=["failed query", "successful query"],
        limit=25,
    )

    assert len(records) == 1
    assert provider.last_diagnostics["status"] == "partial_success"
    assert provider.last_diagnostics["cache_safe"] is False
    assert sleeps == [6.0]
    diagnostics = provider.last_diagnostics["query_diagnostics"]
    assert diagnostics[0]["success"] is False
    assert diagnostics[1]["success"] is True


def test_gdelt_handles_non_json_without_crashing() -> None:
    provider = GDELTProvider(
        enabled=True,
        response_loader=lambda _params: "<html>rate limit</html>",
        sleep_func=lambda _seconds: None,
    )

    records = provider.fetch(
        "2026-04-01",
        "2026-06-21",
        query="India inflation",
    )

    diagnostic = provider.last_diagnostics["query_diagnostics"][0]
    assert records == []
    assert provider.last_diagnostics["status"] == "error"
    assert provider.last_diagnostics["cache_safe"] is False
    assert "non-JSON" in diagnostic["error"]
    assert diagnostic["success"] is False


def test_gdelt_successful_empty_response_has_clear_status() -> None:
    provider = GDELTProvider(
        enabled=True,
        response_loader=lambda _params: {"articles": []},
        sleep_func=lambda _seconds: None,
    )

    records = provider.fetch(
        "2026-04-01",
        "2026-06-21",
        query="India inflation",
    )

    diagnostic = provider.last_diagnostics["query_diagnostics"][0]
    assert records == []
    assert provider.last_diagnostics["status"] == "empty"
    assert provider.last_diagnostics["cache_safe"] is True
    assert diagnostic["success"] is True
    assert "no articles" in diagnostic["warning"]


def test_gdelt_normalizes_json_articles_to_news_schema() -> None:
    article = _article(title="")
    article["snippet"] = "Inflation remained elevated in the latest release."
    article.pop("language")
    provider = GDELTProvider(
        enabled=True,
        response_loader=lambda _params: {"articles": [article, article]},
        sleep_func=lambda _seconds: None,
    )

    raw = provider.fetch(
        "2026-04-01",
        "2026-06-21",
        query="India inflation",
        limit=25,
    )
    normalized = provider.normalize(raw)
    diagnostic = provider.last_diagnostics["query_diagnostics"][0]

    assert len(raw) == 1
    assert len(normalized) == 1
    assert normalized.loc[0, "provider"] == "gdelt"
    assert normalized.loc[0, "document_type"] == "news"
    assert normalized.loc[0, "title"].startswith("Inflation remained")
    assert normalized.loc[0, "language"] == "unknown"
    assert "snippet" in normalized.loc[0, "raw_metadata"]
    assert diagnostic["parsed_article_count"] == 2
    assert diagnostic["normalized_record_count"] == 1
