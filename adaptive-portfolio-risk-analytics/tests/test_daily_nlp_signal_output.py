"""Validation-script tests for daily NLP signal outputs and verdicts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.validate_real_nlp_signal as validator
from scripts.validate_real_nlp_signal import VERDICT_B, VERDICT_C
from src.sentiment import REAL_RBI_MANIFEST_COLUMNS


def _records(*, provider: str = "gdelt", document_type: str = "news") -> pd.DataFrame:
    dates = pd.bdate_range("2026-04-01", "2026-06-21")[::2][:25]
    rows: list[dict[str, object]] = []
    for idx, day in enumerate(dates):
        for copy_id in range(2):
            rows.append(
                {
                    "record_id": f"{provider}-{idx}-{copy_id}",
                    "timestamp": day.tz_localize("UTC").isoformat(),
                    "publication_time": day.tz_localize("UTC").isoformat(),
                    "retrieval_time": "2026-06-23T10:00:00Z",
                    "source": "economictimes.indiatimes.com",
                    "provider": provider,
                    "document_type": document_type,
                    "entity": "India",
                    "ticker": "",
                    "sector": "",
                    "country": "IN",
                    "title": "Inflation shock and rate hike concerns hit India outlook",
                    "text": "",
                    "url": f"https://economictimes.indiatimes.com/news/{provider}-{idx}-{copy_id}",
                    "language": "en",
                    "raw_metadata": "{}",
                    "query": "India inflation",
                }
            )
    return pd.DataFrame(rows)


def _patch_validation_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        validator,
        "load_provider_config",
        lambda: {
            "scoring": {"method": "lexicon", "finbert_enabled": False},
            "validation": {
                "decision_lag_days": 1,
                "min_coverage_ratio": 0.20,
                "min_records": 50,
                "min_distinct_dates": 20,
                "max_reaction_warning_rate": 0.25,
            },
            "_validation": {
                "providers": [
                    {"provider": "gdelt", "enabled": True},
                ],
                "enabled_providers": ["gdelt"],
            },
        },
    )
    monkeypatch.setattr(
        validator,
        "validate_nlp_corpus_intake",
        lambda: {
            "manual_action_required": False,
            "valid_real_records_by_corpus": {"news": 50},
            "all_corpora_ready": False,
        },
    )
    monkeypatch.setattr(
        validator,
        "fit_hmm_walk_forward",
        lambda features, *args, **kwargs: {
            "decision_regimes": pd.Series("Risk-Off", index=features.index)
        },
    )


def _empty_rbi_manifest(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "rbi_real"
    (corpus_dir / "raw").mkdir(parents=True)
    (corpus_dir / "processed").mkdir(parents=True)
    manifest = corpus_dir / "manifest.csv"
    pd.DataFrame(columns=REAL_RBI_MANIFEST_COLUMNS).to_csv(manifest, index=False)
    return manifest


def test_validation_writes_daily_nlp_signal_and_monitoring_verdict_b(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_validation_dependencies(monkeypatch)
    input_path = tmp_path / "gdelt_records.csv"
    output_dir = tmp_path / "report"
    rbi_manifest = _empty_rbi_manifest(tmp_path)
    _records().to_csv(input_path, index=False)

    result = validator.validate_real_nlp_signal(
        input_records=input_path,
        start_date="2026-04-01",
        end_date="2026-06-21",
        output_dir=output_dir,
        multi_source_output_dir=tmp_path / "multi_source_report",
        rbi_manifest_path=rbi_manifest,
    )

    summary = result["summary"]
    assert summary["verdict"] == VERDICT_B
    assert summary["allocation_impact"] is False
    assert summary["coverage_ratio"] > 0
    assert summary["valid_decision_label_dates"] > 0
    assert summary["source_family_count"] == 1
    assert summary["source_diversity_limited"] is True
    assert summary["source_mix"].get("news_only", 0) > 0

    daily_path = output_dir / "daily_nlp_signal.csv"
    scored_path = output_dir / "scored_records.csv"
    diagnostics_path = output_dir / "signal_construction_diagnostics.csv"
    assert daily_path.is_file()
    assert scored_path.is_file()
    assert diagnostics_path.is_file()

    daily = pd.read_csv(daily_path)
    required_columns = {
        "date",
        "raw_record_count",
        "valid_record_count",
        "english_record_count",
        "high_quality_record_count",
        "mean_sentiment_score",
        "mean_risk_score",
        "news_geopolitical_risk_score",
        "raw_nlp_label",
        "decision_nlp_label",
        "coverage_score",
        "source_mix",
        "insufficient_reason",
    }
    assert required_columns.issubset(set(daily.columns))
    assert daily["decision_nlp_label"].isin(
        {"nlp_risk_on", "nlp_neutral", "nlp_risk_off"}
    ).any()
    assert "decision_lag_no_prior_signal" in set(
        daily["insufficient_reason"].dropna()
    )

    scored = pd.read_csv(scored_path)
    assert {
        "sentiment_score",
        "sentiment_label",
        "risk_score",
        "risk_label",
        "scoring_method_used",
        "model_name",
        "model_version",
    }.issubset(scored.columns)


def test_validation_remains_c_when_decision_coverage_is_zero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_validation_dependencies(monkeypatch)
    input_path = tmp_path / "earnings_only_records.csv"
    output_dir = tmp_path / "report"
    rbi_manifest = _empty_rbi_manifest(tmp_path)
    _records(provider="earnings_calls", document_type="earnings_call").to_csv(
        input_path,
        index=False,
    )

    result = validator.validate_real_nlp_signal(
        input_records=input_path,
        start_date="2026-04-01",
        end_date="2026-06-21",
        output_dir=output_dir,
        multi_source_output_dir=tmp_path / "multi_source_report",
        rbi_manifest_path=rbi_manifest,
    )

    assert result["summary"]["verdict"] == VERDICT_C
    assert result["summary"]["coverage_ratio"] == 0
    assert result["summary"]["valid_decision_label_dates"] == 0
