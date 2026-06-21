"""Provider configuration validation tests."""

from __future__ import annotations

from copy import deepcopy

from src.sentiment import load_provider_config, validate_provider_config


def test_config_loader_reads_example_config() -> None:
    config = load_provider_config()

    assert config["earnings"]["enabled"] is True
    assert config["gdelt"]["max_records_per_query"] == 50
    assert config["validation"]["min_records"] == 50
    assert config["_validation"]["is_valid"] is True
    assert config["_validation"]["enabled_providers"] == ["earnings"]


def test_api_key_env_is_required_only_when_provider_enabled(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    config = load_provider_config()
    disabled = validate_provider_config(config)
    assert disabled["is_valid"] is True

    enabled_config = deepcopy(config)
    enabled_config["alpha_vantage"]["enabled"] = True
    enabled = validate_provider_config(enabled_config)

    assert enabled["is_valid"] is False
    assert any(
        "ALPHAVANTAGE_API_KEY" in error for error in enabled["errors"]
    )
    assert all("secret" not in str(row).lower() for row in enabled["providers"])
