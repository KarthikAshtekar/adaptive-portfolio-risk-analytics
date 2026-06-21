"""Real NLP collection command tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.collect_real_nlp_data import collect_real_nlp_data
from src.sentiment.providers import GDELTProvider


def _write_config(path: Path, *, gdelt_enabled: bool) -> Path:
    payload = {
        "rbi": {"enabled": False, "mode": "local_manifest"},
        "earnings": {"enabled": False, "mode": "local_manifest"},
        "gdelt": {
            "enabled": gdelt_enabled,
            "mode": "api",
            "queries": ["India inflation RBI"],
            "max_records_per_query": 5,
        },
        "alpha_vantage": {
            "enabled": False,
            "mode": "api",
            "api_key_env": "ALPHAVANTAGE_API_KEY",
        },
        "scoring": {
            "method": "lexicon",
            "finbert_enabled": False,
            "finbert_model": "ProsusAI/finbert",
        },
        "validation": {
            "decision_lag_days": 1,
            "min_coverage_ratio": 0.20,
            "min_records": 50,
            "min_distinct_dates": 20,
            "max_reaction_warning_rate": 0.25,
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_no_live_mode_avoids_network_calls(
    tmp_path: Path, monkeypatch
) -> None:
    config = _write_config(tmp_path / "providers.yaml", gdelt_enabled=True)

    def fail_network(*args, **kwargs):
        raise AssertionError("network path must not run in --no-live mode")

    monkeypatch.setattr(GDELTProvider, "_response", fail_network)
    result = collect_real_nlp_data(
        config_path=config,
        start_date="2026-01-01",
        end_date="2026-06-21",
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        no_live=True,
    )

    assert result["summary"]["status"] == "success"
    assert result["summary"]["network_calls_allowed"] is False
    assert result["summary"]["real_record_count"] == 0


def test_collection_succeeds_with_no_data_and_clear_diagnostics(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path / "providers.yaml", gdelt_enabled=False)
    output = tmp_path / "out"
    result = collect_real_nlp_data(
        config_path=config,
        start_date="2026-01-01",
        end_date="2026-06-21",
        output_dir=output,
        cache_dir=tmp_path / "cache",
        no_live=True,
    )

    assert "Real provider data unavailable" in result["summary"]["warning"]
    assert (output / "collection_summary.json").is_file()
    assert (output / "provider_diagnostics.csv").is_file()
    assert (output / "reaction_data_warnings.csv").is_file()
