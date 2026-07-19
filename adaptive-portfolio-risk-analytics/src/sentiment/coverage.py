"""Coverage and freshness diagnostics for real NLP evidence."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .composite_index import VALID_COMPOSITE_NLP_LABELS


def _source_family(provider: object) -> str:
    value = str(provider or "").strip().lower()
    if value == "rbi":
        return "rbi_macro"
    if value in {"earnings", "earnings_calls"}:
        return "earnings"
    if value in {"gdelt", "alpha_vantage", "alpha_vantage_news"}:
        return "news"
    return value or "unknown"


def calculate_nlp_coverage(
    records_df: pd.DataFrame,
    *,
    composite_index: pd.DataFrame | None = None,
    provider_diagnostics: pd.DataFrame | None = None,
    start_date=None,
    end_date=None,
    min_coverage_ratio: float = 0.20,
    min_records: int = 50,
    min_distinct_dates: int = 20,
    stale_after_days: int = 30,
) -> dict[str, object]:
    """Summarize evidence volume, trading-day coverage, mix, and freshness."""
    if not isinstance(records_df, pd.DataFrame):
        raise TypeError("records_df must be a pandas DataFrame")
    frame = records_df.copy()
    publication = pd.to_datetime(
        frame.get(
            "publication_time",
            pd.Series(pd.NaT, index=frame.index),
        ),
        errors="coerce",
        utc=True,
    )
    publication_dates = publication.dt.tz_convert(None).dt.normalize().dropna()
    record_count = int(len(frame))
    distinct_dates = int(publication_dates.nunique())

    if start_date is None:
        start = publication_dates.min() if not publication_dates.empty else pd.NaT
    else:
        start = pd.Timestamp(start_date).normalize()
    if end_date is None:
        end = publication_dates.max() if not publication_dates.empty else pd.NaT
    else:
        end = pd.Timestamp(end_date).normalize()
    business_days = (
        len(pd.bdate_range(start, end)) if pd.notna(start) and pd.notna(end) and start <= end else 0
    )
    article_day_coverage = float(distinct_dates / business_days) if business_days else 0.0

    decision_label_coverage = 0.0
    if isinstance(composite_index, pd.DataFrame) and not composite_index.empty:
        label_column = (
            "decision_composite_nlp_label"
            if "decision_composite_nlp_label" in composite_index
            else "composite_nlp_label"
        )
        if label_column in composite_index:
            labels = composite_index[label_column].fillna("insufficient_nlp_data").astype(str)
            decision_label_coverage = float(labels.isin(VALID_COMPOSITE_NLP_LABELS).mean())

    provider_values = frame.get("provider", pd.Series(dtype="string")).fillna("unknown").astype(str)
    source_mix_dict = provider_values.value_counts().to_dict()
    providers_with_data = int(provider_values[provider_values.ne("")].nunique())
    source_families = sorted(
        {
            _source_family(provider)
            for provider in provider_values[provider_values.ne("")]
            if _source_family(provider) != "unknown"
        }
    )
    if isinstance(composite_index, pd.DataFrame) and not composite_index.empty:
        mix_column = (
            "decision_source_mix" if "decision_source_mix" in composite_index else "source_mix"
        )
        if mix_column in composite_index:
            mixes = composite_index[mix_column].dropna().astype(str).str.lower().unique().tolist()
            families = set(source_families)
            for mix in mixes:
                if "news" in mix:
                    families.add("news")
                if "rbi" in mix:
                    families.add("rbi_macro")
            source_families = sorted(families)
    source_family_count = len(source_families)
    enabled_provider_count = providers_with_data
    if isinstance(provider_diagnostics, pd.DataFrame) and not provider_diagnostics.empty:
        if "configured_enabled" in provider_diagnostics:
            enabled_provider_count = int(
                provider_diagnostics["configured_enabled"].fillna(False).sum()
            )
        elif "enabled" in provider_diagnostics:
            enabled_provider_count = int(provider_diagnostics["enabled"].fillna(False).sum())
        else:
            enabled_provider_count = int(len(provider_diagnostics))
    provider_coverage = (
        float(providers_with_data / enabled_provider_count) if enabled_provider_count else 0.0
    )

    latest = publication_dates.max() if not publication_dates.empty else pd.NaT
    staleness_days = (
        int(max(0, (end - latest).days)) if pd.notna(end) and pd.notna(latest) else None
    )
    stale = staleness_days is not None and staleness_days > int(stale_after_days)
    meets_thresholds = (
        record_count >= int(min_records)
        and distinct_dates >= int(min_distinct_dates)
        and decision_label_coverage >= float(min_coverage_ratio)
        and providers_with_data > 0
    )
    partial_thresholds = (
        record_count > 0
        and distinct_dates > 0
        and (
            record_count >= max(1, int(min_records) // 2)
            or distinct_dates >= max(1, int(min_distinct_dates) // 2)
            or decision_label_coverage > 0
        )
    )
    if stale and record_count > 0:
        coverage_quality = "stale"
    elif meets_thresholds and source_family_count >= 2:
        coverage_quality = "sufficient"
    elif meets_thresholds or partial_thresholds:
        coverage_quality = "limited"
    else:
        coverage_quality = "insufficient"

    return {
        "record_count": record_count,
        "distinct_publication_dates": distinct_dates,
        "article_day_coverage": float(np.clip(article_day_coverage, 0.0, 1.0)),
        "decision_label_coverage": float(np.clip(decision_label_coverage, 0.0, 1.0)),
        "provider_coverage": float(np.clip(provider_coverage, 0.0, 1.0)),
        "source_mix": json.dumps(source_mix_dict, sort_keys=True),
        "source_mix_dict": source_mix_dict,
        "source_families": source_families,
        "source_family_count": int(source_family_count),
        "source_diversity_limited": bool(source_family_count < 2),
        "latest_record_date": (latest.date().isoformat() if pd.notna(latest) else None),
        "staleness_days": staleness_days,
        "coverage_quality": coverage_quality,
        "business_day_count": int(business_days),
        "threshold_min_coverage_ratio": float(min_coverage_ratio),
        "threshold_min_records": int(min_records),
        "threshold_min_distinct_dates": int(min_distinct_dates),
    }
