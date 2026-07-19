"""Official-source RBI document fetcher for the governed real-RBI cache.

The fetcher is intentionally conservative:

* it reads configured official RBI RSS/feed or page URLs;
* it filters by publication date and keyword relevance;
* it writes only UTF-8 extracted text into the local real-RBI corpus;
* it records manual-fallback diagnostics when extraction is blocked or unsafe.

It does not score text, change portfolio allocation, or make predictive claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
import hashlib
import re
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd

from src.sentiment.rbi_corpus_builder import (
    REAL_RBI_DOCUMENT_TYPES,
    REAL_RBI_MANIFEST_COLUMNS,
    validate_rbi_manifest,
)


DEFAULT_RBI_SOURCE_URLS: dict[str, str] = {
    "press_releases": "https://www.rbi.org.in/pressreleases_rss.xml",
    "publications": "https://www.rbi.org.in/Publication_rss.xml",
    "speeches": "https://www.rbi.org.in/speeches_rss.xml",
}
DEFAULT_RBI_TARGET_POLICY_SOURCE_URLS: dict[str, str] = {
    "commonman_press_release_archive": (
        "https://www.rbi.org.in/commonman/English/Scripts/PressReleases.aspx"
    ),
}
DEFAULT_RBI_FETCH_KEYWORDS = (
    "monetary policy",
    "mpc minutes",
    "minutes of the monetary policy committee",
    "financial stability",
    "governor speech",
    "deputy governor",
    "inflation",
    "liquidity",
)
DEFAULT_RBI_EXCLUDE_KEYWORDS = (
    "auction result",
    "treasury bills",
    "state government securities",
    "penalty",
    "recruitment",
    "tender",
    "customer liability",
    "digital transactions",
    "amendment directions",
    "campus inauguration",
    "net open position",
    "nbfc directions",
)
RBI_TARGET_POLICY_PHRASES = (
    "Minutes of the Monetary Policy Committee Meeting",
    "MPC Minutes",
    "Under Section 45ZL",
    "Monetary Policy Statement",
    "Resolution of the Monetary Policy Committee",
    "Monetary Policy Resolution",
    "Monetary Policy Committee Meeting",
    "Governor's Statement",
    "Governor’s Statement",
    "Policy Repo Rate",
    "Liquidity Adjustment Facility",
    "Standing Deposit Facility",
    "Marginal Standing Facility",
    "Monetary Policy Decision",
)
RBI_TARGET_DOCUMENT_TYPE_PRIORITY = {
    "mpc_minutes": 1,
    "monetary_policy_statement": 2,
    "financial_stability_report": 3,
    "governor_speech": 4,
    "press_release": 5,
}
FETCH_DIAGNOSTIC_COLUMNS = (
    "publication_date",
    "title",
    "document_type",
    "source_url",
    "source_channel",
    "download_status",
    "http_status",
    "content_type",
    "local_path",
    "text_char_count",
    "substantive_sentence_count",
    "boilerplate_ratio_estimate",
    "is_index_page",
    "index_page_reason",
    "excluded_by_keyword",
    "exclude_keyword_matched",
    "target_policy_docs_mode",
    "target_document_type_match",
    "target_priority_rank",
    "matched_policy_phrase",
    "candidate_source_page",
    "included_in_manifest",
    "skip_reason",
    "extraction_method",
    "warning",
    "error",
)
USER_AGENT = "adaptive-portfolio-risk-analytics/1.2.8 (research corpus cache; contact: local-user)"
BLOCKED_MARKERS = (
    "captcha",
    "access denied",
    "enable javascript",
    "javascript is disabled",
    "verify you are human",
    "temporarily blocked",
)
DATE_PATTERN = re.compile(
    r"(?P<year>20\d{2})[-_/](?P<month>0?[1-9]|1[0-2])[-_/]"
    r"(?P<day>0?[1-9]|[12]\d|3[01])"
)
YEAR_TOKEN_PATTERN = re.compile(r"\b20\d{2}\b")
SUBSTANTIVE_SPLIT_PATTERN = re.compile(r"[.!?\n]+")
INDEX_NAVIGATION_MARKERS = (
    "skip to main content",
    "change language",
    "search the website",
    "back to previous page",
    "archives",
    "archive",
    "note: to read the chapter of your choice",
    "note : to read the chapter of your choice",
    "please click on the links below",
    "selected selected",
    "top back to previous page",
)
NAVIGATION_FRAGMENT_MARKERS = (
    "skip to main content",
    "change language",
    "search the website",
    "back to previous page",
    "archives",
    "archive",
    "selected",
    "top",
    "home",
    "sitemap",
    "contact us",
    "change font size",
)


@dataclass(frozen=True)
class DownloadResult:
    """Downloaded document payload plus request diagnostics."""

    url: str
    ok: bool
    content: bytes
    http_status: int
    content_type: str
    error: str = ""


class _HTMLTextExtractor(HTMLParser):
    """Small dependency-free HTML-to-text extractor."""

    _skip_tags = {"footer", "nav", "script", "style", "noscript", "svg"}
    _block_tags = {
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dt",
        "figcaption",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "p",
        "section",
        "td",
        "th",
        "title",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        tag = tag.lower()
        if tag in self._skip_tags:
            self._skip_depth += 1
        if tag in self._block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._skip_tags and self._skip_depth:
            self._skip_depth -= 1
        if tag in self._block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data:
            self.parts.append(data)

    def text(self) -> str:
        return _normalize_text(" ".join(self.parts))


class _HTMLAnchorExtractor(HTMLParser):
    """Extract candidate links from an official RBI HTML index page."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self._current_href = ""
        self._current_text: list[str] = []
        self.links: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag.lower() != "a":
            return
        attributes = {str(key).lower(): str(value) for key, value in attrs}
        href = attributes.get("href", "").strip()
        if href:
            self._current_href = urljoin(self.base_url, href)
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._current_href:
            return
        title = _normalize_text(" ".join(self._current_text))
        if title:
            self.links.append({"title": title, "url": self._current_href})
        self._current_href = ""
        self._current_text = []


def _normalize_text(value: object) -> str:
    text = unescape(str(value or ""))
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_html_fragment(value: object) -> str:
    return _normalize_text(re.sub(r"<[^>]+>", " ", str(value or "")))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _keyword_list(keywords: str | Iterable[str] | None) -> list[str]:
    if keywords is None:
        return []
    if isinstance(keywords, str):
        values = keywords.split(",")
    else:
        values = list(keywords)
    return [str(value).strip().lower() for value in values if str(value).strip()]


def _resolve_sources(
    sources: str | Iterable[str] | dict[str, str] | None,
) -> dict[str, str]:
    if sources is None:
        return dict(DEFAULT_RBI_SOURCE_URLS)
    if isinstance(sources, dict):
        return {
            str(channel).strip(): str(url).strip()
            for channel, url in sources.items()
            if str(channel).strip() and str(url).strip()
        }
    if isinstance(sources, str):
        values = [item.strip() for item in sources.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in sources if str(item).strip()]
    resolved: dict[str, str] = {}
    for index, value in enumerate(values):
        if "=" in value:
            channel, url = value.split("=", 1)
            resolved[channel.strip()] = url.strip()
        elif value in DEFAULT_RBI_SOURCE_URLS:
            resolved[value] = DEFAULT_RBI_SOURCE_URLS[value]
        elif value.startswith(("http://", "https://", "file://")) or Path(value).exists():
            channel = Path(urlparse(value).path).stem or f"source_{index + 1}"
            resolved[channel] = value
        else:
            raise ValueError(f"unknown RBI source: {value}")
    return resolved


def resolve_rbi_target_policy_sources(
    sources: str | Iterable[str] | dict[str, str] | None = None,
) -> dict[str, str]:
    """Return conservative official-RBI sources for targeted policy discovery."""
    resolved = _resolve_sources(sources)
    for channel, url in DEFAULT_RBI_TARGET_POLICY_SOURCE_URLS.items():
        resolved.setdefault(channel, url)
    return resolved


def _coerce_date(value: object) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return timestamp.tz_convert(None).normalize()


def _date_from_text(value: object) -> pd.Timestamp | None:
    text = str(value or "")
    match = DATE_PATTERN.search(text)
    if match:
        return _coerce_date(
            f"{match.group('year')}-{int(match.group('month')):02d}-{int(match.group('day')):02d}"
        )
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True, utc=True)
    if pd.isna(parsed):
        return None
    return parsed.tz_convert(None).normalize()


def _read_source_payload(source_url: str, timeout_seconds: float = 30) -> bytes:
    parsed = urlparse(source_url)
    if parsed.scheme == "file":
        return Path(parsed.path).read_bytes()
    local_path = Path(source_url)
    if local_path.is_file():
        return local_path.read_bytes()
    request = Request(
        source_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/rss+xml, application/xml, text/xml, text/html;q=0.9, */*;q=0.5",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
        return response.read()


def _first_child_text(item: ET.Element, *names: str) -> str:
    target_names = {name.lower() for name in names}
    for child in list(item):
        child_name = _local_name(child.tag)
        if child_name not in target_names:
            continue
        href = child.attrib.get("href", "").strip()
        if href:
            return href
        if child.text:
            return child.text.strip()
    return ""


def _parse_feed_entries(
    payload: bytes,
    *,
    source_url: str,
    source_channel: str,
) -> list[dict[str, object]]:
    text = payload.decode("utf-8", errors="replace")
    root = ET.fromstring(text)
    entries: list[dict[str, object]] = []
    for item in root.iter():
        if _local_name(item.tag) not in {"item", "entry"}:
            continue
        title = _normalize_text(_first_child_text(item, "title"))
        url = _first_child_text(item, "link")
        if not url:
            url = _first_child_text(item, "id", "guid")
        url = urljoin(source_url, url)
        summary = _normalize_text(
            _first_child_text(item, "description", "summary", "encoded", "content")
        )
        raw_date = _first_child_text(
            item,
            "pubDate",
            "published",
            "updated",
            "date",
            "lastBuildDate",
        )
        publication = _coerce_date(raw_date)
        entries.append(
            {
                "publication_date": publication.date().isoformat()
                if publication is not None
                else "",
                "title": title,
                "summary": summary,
                "source_url": url,
                "source_channel": source_channel,
                "feed_url": source_url,
                "candidate_source_page": source_url,
                "raw_publication_date": raw_date,
            }
        )
    return entries


def _parse_rbi_press_release_archive_entries(
    payload: bytes,
    *,
    source_url: str,
    source_channel: str,
) -> list[dict[str, object]]:
    """Parse dated official RBI Commonman press-release archive rows."""
    text = payload.decode("utf-8", errors="replace")
    token_pattern = re.compile(
        r"<td[^>]*class\s*=\s*['\"]textHead['\"][^>]*>"
        r"(?P<date>.*?)</td>"
        r"|<a[^>]*class\s*=\s*['\"]Indexlink['\"][^>]*"
        r"href\s*=\s*['\"]?(?P<href>[^'\"\s>]+)['\"]?[^>]*>"
        r"(?P<title>.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    current_date = ""
    entries: list[dict[str, object]] = []
    for match in token_pattern.finditer(text):
        raw_date = match.group("date")
        if raw_date is not None:
            parsed = _date_from_text(_strip_html_fragment(raw_date))
            current_date = parsed.date().isoformat() if parsed is not None else ""
            continue
        raw_href = match.group("href")
        raw_title = match.group("title")
        if not raw_href or raw_title is None:
            continue
        title = _strip_html_fragment(raw_title)
        if not title:
            continue
        entries.append(
            {
                "publication_date": current_date,
                "title": title,
                "summary": "",
                "source_url": urljoin(source_url, raw_href),
                "source_channel": source_channel,
                "feed_url": source_url,
                "candidate_source_page": source_url,
                "raw_publication_date": current_date,
            }
        )
    return entries


def _parse_html_index_entries(
    payload: bytes,
    *,
    source_url: str,
    source_channel: str,
) -> list[dict[str, object]]:
    archive_entries = _parse_rbi_press_release_archive_entries(
        payload,
        source_url=source_url,
        source_channel=source_channel,
    )
    if archive_entries:
        return archive_entries
    text = payload.decode("utf-8", errors="replace")
    parser = _HTMLAnchorExtractor(source_url)
    parser.feed(text)
    entries: list[dict[str, object]] = []
    for link in parser.links:
        publication = _date_from_text(f"{link['title']} {link['url']}")
        entries.append(
            {
                "publication_date": publication.date().isoformat()
                if publication is not None
                else "",
                "title": link["title"],
                "summary": "",
                "source_url": link["url"],
                "source_channel": source_channel,
                "feed_url": source_url,
                "candidate_source_page": source_url,
                "raw_publication_date": "",
            }
        )
    return entries


def _entry_matches_keywords(
    entry: dict[str, object],
    keywords: str | Iterable[str] | None,
) -> bool:
    normalized_keywords = _keyword_list(keywords)
    if not normalized_keywords:
        return True
    haystack = " ".join(
        str(entry.get(column, "")) for column in ("title", "summary", "source_url")
    ).lower()
    return any(keyword in haystack for keyword in normalized_keywords)


def filter_rbi_entries_by_keywords(
    entries: Iterable[dict[str, object]],
    keywords: str | Iterable[str] | None,
) -> list[dict[str, object]]:
    """Return entries whose title, summary, or URL matches any keyword."""
    return [entry for entry in entries if _entry_matches_keywords(entry, keywords)]


_POLICY_RELEVANCE_ORDER = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}
_CORE_POLICY_DOCUMENT_TYPES = {
    "mpc_minutes",
    "monetary_policy_statement",
    "financial_stability_report",
}
_MEDIUM_POLICY_RELEVANCE_MARKERS = (
    "monetary policy",
    "mpc minutes",
    "monetary policy committee",
    "financial stability",
    "governor",
    "deputy governor",
    "inflation",
    "liquidity",
    "credit",
    "banking",
    "macroeconomic",
    "macro-economic",
    "growth",
)


def rbi_entry_policy_relevance(entry: dict[str, object]) -> str:
    """Classify lightweight title/summary relevance before document download."""
    title = str(entry.get("title", "") or "")
    source_url = str(entry.get("source_url", "") or "")
    summary = str(entry.get("summary", "") or "")
    title_type = classify_rbi_document_type(title, source_url)
    if title_type in _CORE_POLICY_DOCUMENT_TYPES:
        return "high"
    haystack = f"{title} {summary} {source_url}".lower()
    if any(marker in haystack for marker in _MEDIUM_POLICY_RELEVANCE_MARKERS):
        return "medium"
    if haystack.strip():
        return "low"
    return "none"


def filter_rbi_entries_by_policy_relevance(
    entries: Iterable[dict[str, object]],
    min_policy_relevance: str | None,
) -> list[dict[str, object]]:
    """Return entries meeting a coarse pre-download policy-relevance threshold."""
    threshold = str(min_policy_relevance or "none").strip().lower()
    if threshold not in _POLICY_RELEVANCE_ORDER:
        raise ValueError(
            "min_policy_relevance must be one of: " + ", ".join(_POLICY_RELEVANCE_ORDER)
        )
    required = _POLICY_RELEVANCE_ORDER[threshold]
    if required <= 0:
        return list(entries)
    return [
        entry
        for entry in entries
        if _POLICY_RELEVANCE_ORDER[rbi_entry_policy_relevance(entry)] >= required
    ]


def rbi_entry_exclude_keyword_match(
    entry: dict[str, object],
    exclude_keywords: str | Iterable[str] | None = DEFAULT_RBI_EXCLUDE_KEYWORDS,
) -> str:
    """Return the title-matched irrelevant-document keyword, if any."""
    title = str(entry.get("title", "") or "").strip()
    if not title:
        return ""
    source_url = str(entry.get("source_url", "") or "")
    title_type = classify_rbi_document_type(title, source_url)
    if title_type in _CORE_POLICY_DOCUMENT_TYPES:
        return ""
    normalized_title = re.sub(r"\s+", " ", title.lower()).strip()
    for keyword in sorted(_keyword_list(exclude_keywords), key=len, reverse=True):
        if keyword in normalized_title:
            return keyword
    return ""


def _target_phrase_rules() -> tuple[tuple[str, str], ...]:
    return (
        ("Minutes of the Monetary Policy Committee Meeting", "mpc_minutes"),
        ("MPC Minutes", "mpc_minutes"),
        ("Under Section 45ZL", "mpc_minutes"),
        ("Monetary Policy Statement", "monetary_policy_statement"),
        (
            "Resolution of the Monetary Policy Committee",
            "monetary_policy_statement",
        ),
        ("Monetary Policy Resolution", "monetary_policy_statement"),
        ("Monetary Policy Committee Meeting", "mpc_minutes"),
        ("Governor's Statement", "governor_speech"),
        ("Governor’s Statement", "governor_speech"),
        ("Policy Repo Rate", "monetary_policy_statement"),
        ("Liquidity Adjustment Facility", "monetary_policy_statement"),
        ("Standing Deposit Facility", "monetary_policy_statement"),
        ("Marginal Standing Facility", "monetary_policy_statement"),
        ("Monetary Policy Decision", "monetary_policy_statement"),
    )


def rbi_policy_target_metadata(
    title: object,
    url: object = "",
    text: object = "",
    document_type: str | None = None,
) -> dict[str, object]:
    """Return targeted policy-document match metadata for one RBI candidate."""
    haystack = " ".join(str(value or "") for value in (title, url, text)).lower()
    matched_phrase = ""
    phrase_document_type = ""
    for phrase, phrase_type in _target_phrase_rules():
        if phrase.lower() in haystack:
            matched_phrase = phrase
            phrase_document_type = phrase_type
            break
    classified_type = document_type or classify_rbi_document_type(title, url, text)
    if classified_type not in REAL_RBI_DOCUMENT_TYPES:
        classified_type = "unknown"
    priority_type = (
        classified_type
        if classified_type in RBI_TARGET_DOCUMENT_TYPE_PRIORITY
        else phrase_document_type
    )
    priority_rank = int(RBI_TARGET_DOCUMENT_TYPE_PRIORITY.get(priority_type, 999))
    is_policy_target = bool(matched_phrase) or priority_rank < 999
    return {
        "matched_policy_phrase": matched_phrase,
        "phrase_document_type": phrase_document_type,
        "document_type": classified_type,
        "target_priority_rank": priority_rank,
        "is_policy_target": bool(is_policy_target),
    }


def rbi_entry_policy_target_metadata(entry: dict[str, object]) -> dict[str, object]:
    """Return targeted policy metadata using only pre-download entry fields."""
    return rbi_policy_target_metadata(
        entry.get("title", ""),
        entry.get("source_url", ""),
        entry.get("summary", ""),
    )


def sort_rbi_entries_for_policy_targets(
    entries: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Prioritize targeted policy documents while preserving recent coverage."""
    annotated: list[tuple[tuple[object, ...], dict[str, object]]] = []
    for index, entry in enumerate(entries):
        metadata = rbi_entry_policy_target_metadata(entry)
        publication = _coerce_date(entry.get("publication_date"))
        date_sort = -int(publication.timestamp()) if publication is not None else 0
        sort_key = (
            0 if metadata["is_policy_target"] else 1,
            date_sort,
            int(metadata["target_priority_rank"]),
            index,
        )
        annotated.append((sort_key, entry))
    return [entry for _, entry in sorted(annotated, key=lambda item: item[0])]


def parse_rbi_target_document_types(
    target_document_types: str | Iterable[str] | None,
) -> set[str]:
    """Parse and validate a target document-type filter."""
    values = set(_keyword_list(target_document_types))
    invalid = sorted(values - set(REAL_RBI_DOCUMENT_TYPES))
    if invalid:
        raise ValueError("invalid RBI target document type(s): " + ", ".join(invalid))
    return values


def fetch_rbi_document_index(
    from_date,
    to_date,
    sources: str | Iterable[str] | dict[str, str] | None = None,
    keywords: str | Iterable[str] | None = None,
) -> list[dict[str, object]]:
    """Fetch official RBI feed/index entries for an inclusive date range."""
    start = _coerce_date(from_date)
    end = _coerce_date(to_date)
    if start is None or end is None:
        raise ValueError("from_date and to_date must be valid dates")
    if start > end:
        raise ValueError("from_date must be on or before to_date")

    source_map = _resolve_sources(sources)
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for channel, source_url in source_map.items():
        try:
            payload = _read_source_payload(source_url)
            try:
                entries = _parse_feed_entries(
                    payload,
                    source_url=source_url,
                    source_channel=channel,
                )
            except ET.ParseError:
                entries = _parse_html_index_entries(
                    payload,
                    source_url=source_url,
                    source_channel=channel,
                )
        except (OSError, HTTPError, URLError, TimeoutError) as exc:
            records.append(
                {
                    "publication_date": "",
                    "title": "",
                    "summary": "",
                    "source_url": source_url,
                    "source_channel": channel,
                    "feed_url": source_url,
                    "candidate_source_page": source_url,
                    "index_error": str(exc),
                }
            )
            continue
        for entry in entries:
            publication = _coerce_date(entry.get("publication_date"))
            if publication is None or publication < start or publication > end:
                continue
            if not _entry_matches_keywords(entry, keywords):
                continue
            key = (
                str(entry.get("publication_date", "")),
                str(entry.get("title", "")),
                str(entry.get("source_url", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            records.append(entry)
    return records


def download_rbi_document(url: str, timeout_seconds: float = 30) -> dict[str, object]:
    """Download a single RBI document or local fixture path with diagnostics."""
    parsed = urlparse(str(url))
    try:
        if parsed.scheme == "file":
            path = Path(parsed.path)
            content = path.read_bytes()
            content_type = "application/pdf" if path.suffix.lower() == ".pdf" else "text/html"
            return DownloadResult(
                url=str(url),
                ok=True,
                content=content,
                http_status=200,
                content_type=content_type,
            ).__dict__
        local_path = Path(str(url))
        if local_path.is_file():
            content = local_path.read_bytes()
            content_type = "application/pdf" if local_path.suffix.lower() == ".pdf" else "text/html"
            return DownloadResult(
                url=str(url),
                ok=True,
                content=content,
                http_status=200,
                content_type=content_type,
            ).__dict__
        request = Request(
            str(url),
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.5",
            },
        )
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
            return DownloadResult(
                url=str(url),
                ok=True,
                content=response.read(),
                http_status=int(getattr(response, "status", 200)),
                content_type=response.headers.get("Content-Type", ""),
            ).__dict__
    except HTTPError as exc:
        body = exc.read() if hasattr(exc, "read") else b""
        return DownloadResult(
            url=str(url),
            ok=False,
            content=body,
            http_status=int(exc.code),
            content_type=getattr(exc, "headers", {}).get("Content-Type", ""),
            error=str(exc),
        ).__dict__
    except Exception as exc:  # pragma: no cover - network diagnostics path
        return DownloadResult(
            url=str(url),
            ok=False,
            content=b"",
            http_status=0,
            content_type="",
            error=str(exc),
        ).__dict__


def _payload_bytes_and_type(
    response_or_file,
    content_type: str | None = None,
) -> tuple[bytes, str]:
    if isinstance(response_or_file, dict):
        payload = response_or_file.get("content", b"")
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return bytes(payload), str(content_type or response_or_file.get("content_type", ""))
    if isinstance(response_or_file, bytes):
        return response_or_file, str(content_type or "")
    if isinstance(response_or_file, Path) or Path(str(response_or_file)).is_file():
        path = Path(str(response_or_file))
        inferred = "application/pdf" if path.suffix.lower() == ".pdf" else ""
        return path.read_bytes(), str(content_type or inferred)
    return str(response_or_file or "").encode("utf-8"), str(content_type or "text/html")


def _looks_blocked(raw_or_text: object) -> bool:
    value = str(raw_or_text or "").lower()
    return any(marker in value for marker in BLOCKED_MARKERS)


def _is_year_listing(fragment: str) -> bool:
    years = YEAR_TOKEN_PATTERN.findall(fragment)
    if len(years) < 4:
        return False
    stripped = re.sub(r"\b20\d{2}\b", " ", fragment)
    stripped = re.sub(r"[\s,|/\\:;._-]+", "", stripped)
    return not stripped


def _is_navigation_fragment(fragment: str) -> bool:
    text = re.sub(r"\s+", " ", fragment.lower()).strip()
    if not text:
        return True
    if _is_year_listing(text):
        return True
    if text.isdigit():
        return True
    words = re.findall(r"[a-z]+", text)
    if len(words) <= 2 and any(
        marker == text or marker in text for marker in ("top", "selected", "archives", "archive")
    ):
        return True
    return any(marker in text for marker in NAVIGATION_FRAGMENT_MARKERS)


def _substantive_sentence_count(text: object) -> int:
    count = 0
    for fragment in SUBSTANTIVE_SPLIT_PATTERN.split(str(text or "")):
        normalized = re.sub(r"\s+", " ", fragment).strip()
        if len(normalized) < 30:
            continue
        if _is_navigation_fragment(normalized):
            continue
        if len(re.findall(r"[A-Za-z]{3,}", normalized)) < 5:
            continue
        count += 1
    return int(count)


def _boilerplate_ratio_estimate(text: object) -> float:
    fragments = [
        re.sub(r"\s+", " ", fragment).strip()
        for fragment in SUBSTANTIVE_SPLIT_PATTERN.split(str(text or ""))
        if re.sub(r"\s+", " ", fragment).strip()
    ]
    if not fragments:
        return 1.0
    boilerplate = sum(1 for fragment in fragments if _is_navigation_fragment(fragment))
    return round(float(boilerplate) / float(len(fragments)), 4)


def rbi_text_quality_metrics(title, text, url=None) -> dict[str, object]:
    """Return lightweight quality metrics used by fetch diagnostics."""
    combined = "\n".join(str(value or "") for value in (title, text, url))
    return {
        "text_char_count": int(len(str(text or ""))),
        "substantive_sentence_count": _substantive_sentence_count(combined),
        "boilerplate_ratio_estimate": _boilerplate_ratio_estimate(combined),
    }


def is_rbi_index_or_navigation_page(title, text, url=None) -> tuple[bool, str]:
    """Detect RBI section index, archive, and navigation-only pages."""
    combined = "\n".join(str(value or "") for value in (title, text, url))
    lowered = re.sub(r"\s+", " ", combined.lower()).strip()
    metrics = rbi_text_quality_metrics(title, text, url)
    substantive_count = int(metrics["substantive_sentence_count"])
    boilerplate_ratio = float(metrics["boilerplate_ratio_estimate"])
    years = YEAR_TOKEN_PATTERN.findall(lowered)
    navigation_hits = sum(1 for marker in INDEX_NAVIGATION_MARKERS if marker in lowered)
    title_url = " ".join(str(value or "") for value in (title, url)).lower()
    annual_report_primary = "annual report" in title_url
    annual_report_context = "annual report" in lowered
    chapter_click_context = (
        "read the chapter of your choice" in lowered or "click on the links below" in lowered
    )
    archive_context = (
        "archive" in lowered
        or "archives" in lowered
        or "past reports" in lowered
        or "past speeches" in lowered
        or "press release archive" in lowered
    )

    if annual_report_primary and (
        chapter_click_context
        or (archive_context and len(years) >= 4)
        or (navigation_hits >= 3 and substantive_count <= 3)
    ):
        return True, "annual_report_index_page"
    if (
        annual_report_context
        and chapter_click_context
        and (substantive_count <= 3 and boilerplate_ratio >= 0.55)
    ):
        return True, "annual_report_index_page"
    if archive_context and (
        substantive_count <= 3
        and (len(years) >= 4 or navigation_hits >= 3 or boilerplate_ratio >= 0.60)
    ):
        return True, "archive_navigation_page"
    if len(years) >= 6 and substantive_count <= 2:
        return True, "archive_navigation_page"
    if navigation_hits >= 4 and substantive_count <= 1:
        return True, "low_substantive_text"
    if boilerplate_ratio >= 0.75 and substantive_count <= 1 and len(str(text or "")) < 2000:
        return True, "low_substantive_text"
    return False, ""


def _extract_html_text(payload: bytes) -> str:
    raw = payload.decode("utf-8", errors="replace")
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    text = parser.text()
    if not text:
        text = _normalize_text(re.sub(r"<[^>]+>", " ", raw))
    if _looks_blocked(raw) or _looks_blocked(text):
        raise RuntimeError(
            "RBI page appears blocked by CAPTCHA/JavaScript protection; manual fallback required"
        )
    return text


def _extract_pdf_text(payload: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF extraction requires optional dependency 'pypdf'; manual fallback required"
        ) from exc
    reader = PdfReader(BytesIO(payload))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return _normalize_text(text)


def extract_rbi_text(response_or_file, content_type: str | None = None) -> str:
    """Extract normalized text from an RBI HTML/PDF response or local file."""
    payload, detected_content_type = _payload_bytes_and_type(
        response_or_file,
        content_type=content_type,
    )
    content_type_lower = detected_content_type.lower()
    if "pdf" in content_type_lower or payload.lstrip().startswith(b"%PDF"):
        text = _extract_pdf_text(payload)
    elif "html" in content_type_lower or b"<" in payload[:500]:
        text = _extract_html_text(payload)
    else:
        text = _normalize_text(payload.decode("utf-8", errors="replace"))
    if not text:
        raise RuntimeError("document text is empty; manual fallback required")
    return text


def extract_rbi_text_with_diagnostics(
    response_or_file,
    content_type: str | None = None,
) -> dict[str, object]:
    """Extract text and return extraction method/warning diagnostics."""
    payload, detected_content_type = _payload_bytes_and_type(
        response_or_file,
        content_type=content_type,
    )
    content_type_lower = detected_content_type.lower()
    method = (
        "pdf" if ("pdf" in content_type_lower or payload.lstrip().startswith(b"%PDF")) else "html"
    )
    try:
        text = extract_rbi_text(
            {"content": payload, "content_type": detected_content_type},
            content_type=detected_content_type,
        )
        return {
            "text": text,
            "extraction_method": method,
            "warning": "",
            "error": "",
        }
    except Exception as exc:
        return {
            "text": "",
            "extraction_method": "manual_required",
            "warning": str(exc),
            "error": str(exc),
        }


def classify_rbi_document_type(title, url, text=None) -> str:
    """Classify an official RBI document into the accepted real-RBI types."""
    haystack = " ".join(str(value or "") for value in (title, url, text)).lower()
    is_index_page, _ = is_rbi_index_or_navigation_page(title, text or "", url)
    if (
        "minutes of the monetary policy committee" in haystack
        or "mpc minutes" in haystack
        or ("under section 45zl" in haystack and "monetary policy committee" in haystack)
    ):
        return "mpc_minutes"
    if (
        "monetary policy statement" in haystack
        or "resolution of the monetary policy committee" in haystack
        or "monetary policy resolution" in haystack
    ):
        return "monetary_policy_statement"
    if "financial stability report" in haystack:
        return "financial_stability_report"
    if "annual report" in haystack and not is_index_page:
        return "annual_report"
    if (
        "governor speech" in haystack
        or "governor's statement" in haystack
        or "governor’s statement" in haystack
        or "deputy governor" in haystack
        or "speech" in haystack
    ):
        return "governor_speech"
    if (
        "press release" in haystack
        or "press-release" in haystack
        or "pressrelease" in haystack
        or "pressreleases" in haystack
    ):
        return "press_release"
    return "unknown"


def _slug(value: object, max_chars: int = 64) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return slug[:max_chars].strip("_") or "untitled"


def build_rbi_document_id(publication_date, document_type, title) -> str:
    """Build a deterministic manifest-safe document ID."""
    publication = _coerce_date(publication_date)
    date_part = (
        publication.date().isoformat().replace("-", "") if publication is not None else "undated"
    )
    doc_type = str(document_type or "unknown").strip().lower()
    if doc_type not in REAL_RBI_DOCUMENT_TYPES:
        doc_type = "unknown"
    title_slug = _slug(title)
    digest = hashlib.sha1(f"{date_part}|{doc_type}|{title_slug}".encode("utf-8")).hexdigest()[:8]
    return f"rbi_{date_part}_{doc_type}_{title_slug}_{digest}"


def _ensure_manifest(path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "raw").mkdir(parents=True, exist_ok=True)
    (path.parent / "processed").mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=REAL_RBI_MANIFEST_COLUMNS).to_csv(path, index=False)
    frame = pd.read_csv(path, dtype="string").fillna("")
    for column in REAL_RBI_MANIFEST_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame.loc[:, list(REAL_RBI_MANIFEST_COLUMNS)].copy()


def update_rbi_manifest(records, manifest_path) -> dict[str, object]:
    """Add or replace manifest rows for successfully cached RBI documents."""
    manifest = Path(manifest_path)
    frame = _ensure_manifest(manifest)
    rows = []
    for record in records or []:
        row = {
            column: str(record.get(column, "") or "").strip()
            for column in REAL_RBI_MANIFEST_COLUMNS
        }
        if not row["document_id"]:
            row["document_id"] = build_rbi_document_id(
                row["publication_date"],
                row["document_type"],
                row["title"],
            )
        if row["document_type"] not in REAL_RBI_DOCUMENT_TYPES:
            raise ValueError(f"invalid RBI document_type: {row['document_type']}")
        rows.append(row)
    updated = 0
    added = 0
    for row in rows:
        duplicate_mask = frame["document_id"].astype(str).str.strip().eq(row["document_id"])
        if row["source_url"]:
            duplicate_mask |= frame["source_url"].astype(str).str.strip().eq(row["source_url"])
        if bool(duplicate_mask.any()):
            frame = frame.loc[~duplicate_mask].copy()
            updated += 1
        else:
            added += 1
        frame = pd.concat(
            [frame, pd.DataFrame([row], columns=REAL_RBI_MANIFEST_COLUMNS)],
            ignore_index=True,
        )
    if not frame.empty:
        frame = frame.sort_values(
            by=["publication_date", "document_id"],
            kind="stable",
        ).reset_index(drop=True)
    frame = frame.loc[:, list(REAL_RBI_MANIFEST_COLUMNS)]
    frame.to_csv(manifest, index=False)
    validation = validate_rbi_manifest(manifest)
    return {
        "manifest_path": manifest,
        "added_count": int(added),
        "updated_count": int(updated),
        "row_count": int(len(frame)),
        "validation": validation,
    }


def build_manifest_record(
    *,
    document_id: str,
    publication_date: str,
    document_type: str,
    title: str,
    local_path: str,
    source_url: str,
    retrieval_date: str | None = None,
    language: str = "en",
    notes: str = "",
) -> dict[str, str]:
    """Return one exact-column real-RBI manifest row."""
    retrieval = retrieval_date or datetime.now(timezone.utc).date().isoformat()
    return {
        "document_id": document_id,
        "publication_date": publication_date,
        "document_type": document_type,
        "title": title,
        "local_path": local_path,
        "source_url": source_url,
        "retrieval_date": retrieval,
        "language": language,
        "notes": notes,
    }


def sleep_respecting_rate_limit(delay_seconds: float) -> None:
    """Sleep only for positive delays; extracted for tests and clarity."""
    delay = max(0.0, float(delay_seconds))
    if delay:
        time.sleep(delay)
