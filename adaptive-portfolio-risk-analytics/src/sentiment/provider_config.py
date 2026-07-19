"""Configuration loading and validation for optional real NLP providers."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from src.paths import PROJECT_ROOT


REPO_ROOT = PROJECT_ROOT
DEFAULT_PROVIDER_CONFIG = REPO_ROOT / "config" / "nlp_providers.example.yaml"
PROVIDER_NAMES = ("rbi", "earnings", "gdelt", "alpha_vantage")
LOCAL_MODES = {"local", "local_manifest", "manifest"}
API_MODES = {"api", "feed", "feeds"}
GDELT_DEFAULTS = {
    "max_records_per_query": 50,
    "request_delay_seconds": 6,
    "retry_delay_seconds": 10,
    "max_retries": 3,
    "timeout_seconds": 30,
}


def _resolve_path(value: object, base_dir: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def validate_provider_config(
    config: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate provider settings without reading or exposing secret values."""
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")
    root = Path(base_dir).resolve() if base_dir else REPO_ROOT
    errors: list[str] = []
    warnings: list[str] = []
    provider_rows: list[dict[str, object]] = []

    for provider_name in PROVIDER_NAMES:
        settings = config.get(provider_name, {})
        if not isinstance(settings, dict):
            errors.append(f"{provider_name}: settings must be a mapping")
            settings = {}
        enabled = bool(settings.get("enabled", False))
        mode = str(settings.get("mode", "")).strip().lower()
        row_errors: list[str] = []
        row_warnings: list[str] = []

        if enabled and not mode:
            row_errors.append("enabled provider requires mode")
        if mode and mode not in LOCAL_MODES | API_MODES:
            row_errors.append(f"unsupported mode '{mode}'")

        manifest_value = settings.get("local_manifest_path")
        manifest_path = _resolve_path(manifest_value, root)
        if manifest_value and manifest_path is not None and not manifest_path.is_file():
            message = f"local manifest not found: {manifest_path}"
            if enabled and mode in LOCAL_MODES:
                row_errors.append(message)
            else:
                row_warnings.append(message)
        if enabled and mode in LOCAL_MODES and manifest_path is None:
            row_errors.append("enabled local provider requires local_manifest_path")

        key_env = str(settings.get("api_key_env", "")).strip()
        key_env_present: bool | None = None
        if enabled and provider_name == "alpha_vantage":
            if not key_env:
                row_errors.append("enabled API provider requires api_key_env")
                key_env_present = False
            else:
                key_env_present = bool(os.getenv(key_env))
                if not key_env_present:
                    row_errors.append(
                        f"required API key environment variable is not set: {key_env}"
                    )
        if provider_name == "gdelt":
            gdelt_numeric_rules = {
                "request_delay_seconds": (0.0, None),
                "retry_delay_seconds": (0.0, None),
                "max_retries": (0.0, None),
                "timeout_seconds": (1.0, None),
                "max_records_per_query": (1.0, None),
            }
            for key, (minimum, maximum) in gdelt_numeric_rules.items():
                try:
                    value = float(settings.get(key, GDELT_DEFAULTS[key]))
                except (TypeError, ValueError):
                    row_errors.append(f"{key} must be numeric")
                    continue
                if value < minimum or (maximum is not None and value > maximum):
                    row_errors.append(
                        f"{key} must be between {minimum} and "
                        f"{maximum if maximum is not None else 'infinity'}"
                    )

        errors.extend(f"{provider_name}: {message}" for message in row_errors)
        warnings.extend(f"{provider_name}: {message}" for message in row_warnings)
        provider_rows.append(
            {
                "provider": provider_name,
                "enabled": enabled,
                "mode": mode or "unspecified",
                "status": ("invalid" if row_errors else "enabled" if enabled else "disabled"),
                "manifest_path": str(manifest_path or ""),
                "manifest_exists": (manifest_path.is_file() if manifest_path is not None else None),
                "api_key_env": key_env,
                "api_key_env_present": key_env_present,
                "errors": " | ".join(row_errors),
                "warnings": " | ".join(row_warnings),
            }
        )

    scoring = config.get("scoring", {})
    if not isinstance(scoring, dict):
        errors.append("scoring: settings must be a mapping")
    else:
        method = str(scoring.get("method", "lexicon")).strip().lower()
        if method not in {"lexicon", "finbert"}:
            errors.append("scoring: method must be 'lexicon' or 'finbert'")

    validation = config.get("validation", {})
    if not isinstance(validation, dict):
        errors.append("validation: settings must be a mapping")
    else:
        numeric_rules = {
            "decision_lag_days": (1, None),
            "min_coverage_ratio": (0.0, 1.0),
            "min_records": (0, None),
            "min_distinct_dates": (0, None),
            "max_reaction_warning_rate": (0.0, 1.0),
        }
        for key, (minimum, maximum) in numeric_rules.items():
            try:
                value = float(validation.get(key))
            except (TypeError, ValueError):
                errors.append(f"validation: {key} must be numeric")
                continue
            if value < minimum or (maximum is not None and value > maximum):
                errors.append(
                    f"validation: {key} must be between {minimum} and "
                    f"{maximum if maximum is not None else 'infinity'}"
                )

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "providers": provider_rows,
        "enabled_providers": [
            row["provider"]
            for row in provider_rows
            if row["enabled"] and row["status"] != "invalid"
        ],
    }


def load_provider_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML provider configuration and attach non-secret diagnostics."""
    config_path = Path(path).expanduser() if path else DEFAULT_PROVIDER_CONFIG
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"NLP provider config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("NLP provider config root must be a mapping")
    result = copy.deepcopy(payload)
    gdelt = result.setdefault("gdelt", {})
    if not isinstance(gdelt, dict):
        gdelt = {}
        result["gdelt"] = gdelt
    for key, value in GDELT_DEFAULTS.items():
        gdelt.setdefault(key, value)
    result["_config_path"] = str(config_path)
    result["_validation"] = validate_provider_config(result, base_dir=REPO_ROOT)
    return result
