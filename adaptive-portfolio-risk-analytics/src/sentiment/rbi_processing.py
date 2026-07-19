"""Sentence extraction and cleaning for RBI macro documents."""

from __future__ import annotations

import re

import pandas as pd


BOILERPLATE_PATTERNS = (
    re.compile(r"^\s*(reserve bank of india|press release|contents?)\s*$", re.I),
    re.compile(r"^\s*synthetic research fixture\b", re.I),
    re.compile(r"^\s*(page\s+\d+(\s+of\s+\d+)?|\d+)\s*$", re.I),
    re.compile(r"^\s*(copyright|all rights reserved)\b", re.I),
)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def clean_rbi_sentence(sentence: object) -> str:
    """Normalize one RBI sentence while preserving its substantive text."""
    value = str(sentence or "").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", value).strip(" -\t")


def _clean_text(text: object) -> str:
    value = str(text or "").replace("\u00a0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def _is_boilerplate(sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in BOILERPLATE_PATTERNS)


def split_rbi_documents_into_sentences(
    documents: pd.DataFrame,
    *,
    min_words: int = 4,
    min_characters: int = 20,
) -> pd.DataFrame:
    """Split loaded documents into stable, ordered, metadata-rich sentences."""
    if not isinstance(documents, pd.DataFrame):
        raise TypeError("documents must be a pandas DataFrame")
    required = {"document_id", "publication_date", "text"}
    missing = required.difference(documents.columns)
    if missing:
        raise ValueError(f"documents are missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    ordered = documents.copy()
    if "manifest_order" in ordered:
        ordered = ordered.sort_values("manifest_order", kind="mergesort")
    for document in ordered.itertuples(index=False):
        if getattr(document, "load_status", "loaded") != "loaded":
            continue
        text = _clean_text(getattr(document, "text", ""))
        blocks = [block.strip() for block in re.split(r"\n+", text) if block.strip()]
        candidates: list[str] = []
        for block in blocks:
            candidates.extend(
                sentence.strip() for sentence in SENTENCE_BOUNDARY.split(block) if sentence.strip()
            )
        sentence_order = 0
        for candidate in candidates:
            cleaned = clean_rbi_sentence(candidate)
            if (
                len(cleaned) < int(min_characters)
                or len(cleaned.split()) < int(min_words)
                or _is_boilerplate(cleaned)
            ):
                continue
            sentence_id = f"{getattr(document, 'document_id')}_s{sentence_order:04d}"
            rows.append(
                {
                    "sentence_id": sentence_id,
                    "document_id": getattr(document, "document_id"),
                    "publication_date": pd.Timestamp(getattr(document, "publication_date")),
                    "document_type": getattr(document, "document_type", "unknown"),
                    "document_title": getattr(document, "title", ""),
                    "source": getattr(document, "source", "RBI"),
                    "sentence_order": sentence_order,
                    "sentence": cleaned,
                    "sentence_text": cleaned,
                }
            )
            sentence_order += 1
    return pd.DataFrame(
        rows,
        columns=[
            "sentence_id",
            "document_id",
            "publication_date",
            "document_type",
            "document_title",
            "source",
            "sentence_order",
            "sentence",
            "sentence_text",
        ],
    )


split_rbi_sentences = split_rbi_documents_into_sentences
process_rbi_documents = split_rbi_documents_into_sentences
