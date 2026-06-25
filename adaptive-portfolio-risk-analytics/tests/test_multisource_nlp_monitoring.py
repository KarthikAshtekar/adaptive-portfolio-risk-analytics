"""Multi-source NLP monitoring validation tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.validate_real_nlp_signal as validator
from scripts.validate_real_nlp_signal import VERDICT_B
from scripts.bootstrap_rbi_real_corpus import bootstrap_rbi_real_corpus
from scripts.import_rbi_text_document import import_rbi_text_document


def _news_records() -> pd.DataFrame:
    dates = pd.bdate_range("2026-04-01", "2026-06-21")[::2][:25]
    rows: list[dict[str, object]] = []
    for idx, day in enumerate(dates):
        for copy_id in range(2):
            rows.append(
                {
                    "record_id": f"gdelt-{idx}-{copy_id}",
                    "timestamp": day.tz_localize("UTC").isoformat(),
                    "publication_time": day.tz_localize("UTC").isoformat(),
                    "retrieval_time": "2026-06-23T10:00:00Z",
                    "source": "economictimes.indiatimes.com",
                    "provider": "gdelt",
                    "document_type": "news",
                    "entity": "India",
                    "ticker": "",
                    "sector": "",
                    "country": "IN",
                    "title": "Inflation shock and rate hike concerns hit India outlook",
                    "text": "",
                    "url": f"https://economictimes.indiatimes.com/news/{idx}-{copy_id}",
                    "language": "en",
                    "raw_metadata": "{}",
                    "query": "India inflation",
                }
            )
    return pd.DataFrame(rows)


def _patch_validation_dependencies(monkeypatch, manifest_path: Path) -> None:
    monkeypatch.setattr(
        validator,
        "load_provider_config",
        lambda: {
            "rbi": {"local_manifest_path": str(manifest_path)},
            "scoring": {"method": "lexicon", "finbert_enabled": False},
            "validation": {
                "decision_lag_days": 1,
                "min_coverage_ratio": 0.20,
                "min_records": 50,
                "min_distinct_dates": 20,
                "max_reaction_warning_rate": 0.25,
            },
            "_validation": {
                "providers": [{"provider": "gdelt", "enabled": True}],
                "enabled_providers": ["gdelt"],
            },
        },
    )
    monkeypatch.setattr(
        validator,
        "validate_nlp_corpus_intake",
        lambda: {
            "manual_action_required": False,
            "valid_real_records_by_corpus": {"news": 50, "rbi": 1},
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


def test_multisource_validation_uses_rbi_and_news_monitoring_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus_dir = tmp_path / "rbi_real"
    bootstrap_rbi_real_corpus(corpus_dir=corpus_dir)
    source_text = tmp_path / "rbi.txt"
    source_text.write_text(
        (
            "Inflation remains elevated. Upside risks to inflation remain. "
            "Policy will remain vigilant and data dependent."
        ),
        encoding="utf-8",
    )
    import_rbi_text_document(
        document_id="RBI_MPC_2026_04",
        publication_date="2026-04-02",
        document_type="mpc_minutes",
        title="MPC Minutes April 2026",
        source_url="https://www.rbi.org.in/example",
        input_text_file=source_text,
        retrieval_date="2026-06-23",
        manifest_path=corpus_dir / "manifest.csv",
    )
    _patch_validation_dependencies(monkeypatch, corpus_dir / "manifest.csv")
    news_path = tmp_path / "news.csv"
    _news_records().to_csv(news_path, index=False)
    output_dir = tmp_path / "phase_4a6"
    multi_dir = tmp_path / "phase_4a8"

    result = validator.validate_real_nlp_signal(
        input_records=news_path,
        start_date="2026-04-01",
        end_date="2026-06-21",
        output_dir=output_dir,
        multi_source_output_dir=multi_dir,
        rbi_manifest_path=corpus_dir / "manifest.csv",
    )

    summary = result["summary"]
    assert summary["verdict"] == VERDICT_B
    assert summary["allocation_impact"] is False
    assert summary["multi_source_monitoring"] is True
    assert summary["real_rbi_document_count"] == 1
    assert summary["real_news_record_count"] == 50
    assert summary["source_mix"].get("rbi_and_news", 0) > 0
    assert summary["source_family_count"] == 2
    assert (multi_dir / "rbi_corpus_status.csv").is_file()
    assert (multi_dir / "source_mix_diagnostics.csv").is_file()
    assert (multi_dir / "multi_source_nlp_comparison.csv").is_file()
