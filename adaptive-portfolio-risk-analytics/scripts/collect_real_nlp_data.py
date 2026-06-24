"""Collect optional real NLP provider data with offline-safe caching."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sentiment import (  # noqa: E402
    AlphaVantageNewsProvider,
    EarningsCallProvider,
    GDELTProvider,
    LocalProvider,
    RBIProvider,
    apply_publication_lag,
    flag_reaction_data_leakage,
    load_provider_config,
    run_sentiment_provider_ingestion,
    score_source_quality,
    validate_ex_ante_records,
    validate_nlp_corpus_intake,
    validate_provider_config,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "reports" / "phase_4a6_real_nlp_validation"
)
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "sentiment" / "cache"
RUNTIME_PROVIDER_NAMES = {
    "rbi": "rbi",
    "earnings": "earnings_calls",
    "gdelt": "gdelt",
    "alpha_vantage": "alpha_vantage_news",
}


def _repo_path(value: object) -> Path:
    path = Path(str(value or "")).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def initialize_providers(
    config: dict[str, object],
    *,
    no_live: bool,
    news_records: pd.DataFrame | None = None,
) -> tuple[list[object], dict[str, dict[str, object]]]:
    """Create configured providers while hard-disabling network in no-live mode."""
    providers: list[object] = []
    query_config: dict[str, dict[str, object]] = {}
    validation = config.get("_validation", {})
    valid_enabled = set(validation.get("enabled_providers", []))

    if "rbi" in valid_enabled:
        settings = dict(config.get("rbi", {}))
        feed_urls = list(settings.get("feed_urls", []) or [])
        providers.append(
            RBIProvider(
                feeds_enabled=(
                    not no_live
                    and str(settings.get("mode", "")).lower()
                    in {"api", "feed", "feeds"}
                    and bool(feed_urls)
                ),
                feed_urls=feed_urls,
                local_manifest_path=_repo_path(
                    settings.get("local_manifest_path")
                ),
            )
        )
    if "earnings" in valid_enabled:
        settings = dict(config.get("earnings", {}))
        providers.append(
            EarningsCallProvider(
                _repo_path(settings.get("local_manifest_path"))
            )
        )
    if "gdelt" in valid_enabled:
        settings = dict(config.get("gdelt", {}))
        providers.append(
            GDELTProvider(
                enabled=not no_live,
                request_delay_seconds=float(
                    settings.get("request_delay_seconds", 6)
                ),
                retry_delay_seconds=float(
                    settings.get("retry_delay_seconds", 10)
                ),
                max_retries=int(settings.get("max_retries", 3)),
                timeout_seconds=float(settings.get("timeout_seconds", 30)),
            )
        )
        queries = list(settings.get("queries", []) or [])
        per_query = int(settings.get("max_records_per_query", 50))
        query_config["gdelt"] = {
            "query": queries,
            "limit": max(1, per_query),
        }
    if "alpha_vantage" in valid_enabled:
        settings = dict(config.get("alpha_vantage", {}))
        key_env = str(settings.get("api_key_env", "")).strip()
        providers.append(
            AlphaVantageNewsProvider(
                api_key=os.getenv(key_env, "") if key_env else "",
                enabled=not no_live,
            )
        )
        query_config["alpha_vantage_news"] = {
            "symbols": list(settings.get("tickers", []) or []),
            "limit": int(settings.get("max_records", 100)),
        }
    if isinstance(news_records, pd.DataFrame) and not news_records.empty:
        providers.append(
            LocalProvider(news_records, provider_name="news_manifest")
        )
    return providers, query_config


def _merge_provider_diagnostics(
    config: dict[str, object],
    ingestion_diagnostics: pd.DataFrame,
    intake_status: pd.DataFrame,
) -> pd.DataFrame:
    config_rows = pd.DataFrame(config["_validation"]["providers"]).rename(
        columns={
            "enabled": "configured_enabled",
            "status": "config_status",
            "errors": "config_errors",
            "warnings": "config_warnings",
        }
    )
    config_rows["runtime_provider"] = config_rows["provider"].map(
        RUNTIME_PROVIDER_NAMES
    )
    news_status = intake_status.loc[
        intake_status["corpus"].eq("news")
    ]
    if not news_status.empty:
        news_row = news_status.iloc[0]
        config_rows = pd.concat(
            [
                config_rows,
                pd.DataFrame(
                    [
                        {
                            "provider": "news",
                            "configured_enabled": bool(
                                news_row["manifest_exists"]
                            ),
                            "mode": "local_manifest",
                            "config_status": news_row["corpus_status"],
                            "manifest_path": news_row["manifest_path"],
                            "manifest_exists": news_row["manifest_exists"],
                            "api_key_env": "",
                            "api_key_env_present": None,
                            "config_errors": "",
                            "config_warnings": news_row["status_message"],
                            "runtime_provider": "news_manifest",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    ingestion = ingestion_diagnostics.copy()
    if ingestion.empty:
        ingestion = pd.DataFrame(columns=["provider"])
    ingestion = ingestion.rename(columns={"provider": "runtime_provider"})
    merged = config_rows.merge(ingestion, on="runtime_provider", how="left")
    merged["provider"] = merged["provider"].astype(str)
    for column in (
        "raw_record_count",
        "valid_record_count",
        "invalid_record_count",
        "deduped_valid_record_count",
        "provider_duplicates_removed",
        "retry_count",
    ):
        if column not in merged:
            merged[column] = 0
        merged[column] = pd.to_numeric(
            merged[column], errors="coerce"
        ).fillna(0).astype(int)
    for column in (
        "cache_hit",
        "cache_written",
        "cache_ignored",
        "rate_limited",
    ):
        if column not in merged:
            merged[column] = False
        values = merged[column]
        merged[column] = values.where(values.notna(), False).astype(bool)
    merged["status"] = merged.get("status", pd.Series(index=merged.index)).fillna(
        merged["config_status"]
    )
    return merged


def collect_real_nlp_data(
    *,
    config_path: str | Path | None,
    start_date,
    end_date,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    use_cache: bool = True,
    no_live: bool = False,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ignore_cache: bool = False,
    provider_filter: str | None = None,
    sanity_query: str | None = None,
) -> dict[str, object]:
    """Run collection and persist diagnostics even when real data is absent."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(load_provider_config(config_path))
    if provider_filter:
        aliases = {
            "rbi": "rbi",
            "earnings": "earnings",
            "gdelt": "gdelt",
            "alpha_vantage": "alpha_vantage",
        }
        selected = aliases[provider_filter]
        for provider_name in aliases.values():
            settings = config.get(provider_name)
            if isinstance(settings, dict):
                settings["enabled"] = provider_name == selected
        config["_validation"] = validate_provider_config(config)
    if sanity_query:
        config.setdefault("gdelt", {})["queries"] = [sanity_query]
    rbi_path = str(config.get("rbi", {}).get("local_manifest_path", "")).strip()
    earnings_path = str(
        config.get("earnings", {}).get("local_manifest_path", "")
    ).strip()
    intake = validate_nlp_corpus_intake(
        rbi_manifest=_repo_path(rbi_path) if rbi_path else None,
        earnings_manifest=_repo_path(earnings_path) if earnings_path else None,
    )
    intake["intake_status"].to_csv(
        output / "corpus_intake_status.csv", index=False
    )
    news_records = intake["valid_records"]["news"].copy()
    if not news_records.empty:
        publication = pd.to_datetime(
            news_records["publication_time"], errors="coerce", utc=True
        )
        news_records = news_records.loc[
            publication.dt.date.ge(pd.Timestamp(start_date).date())
            & publication.dt.date.le(pd.Timestamp(end_date).date())
        ].copy()
    providers, query_config = initialize_providers(
        config,
        no_live=no_live,
        news_records=news_records,
    )
    ingestion = run_sentiment_provider_ingestion(
        providers,
        start_date,
        end_date,
        output,
        query_config=query_config,
        use_cache=bool(use_cache),
        cache_dir=cache_dir,
        ignore_cache=bool(ignore_cache),
    )
    query_diagnostics = ingestion["provider_query_diagnostics"].copy()
    gdelt_query_diagnostics = query_diagnostics.loc[
        query_diagnostics.get(
            "provider",
            pd.Series("", index=query_diagnostics.index),
        ).astype(str).eq("gdelt")
    ].copy()
    gdelt_query_diagnostics.to_csv(
        output / "gdelt_query_diagnostics.csv", index=False
    )

    normalized = score_source_quality(
        ingestion["normalized_sentiment_records"]
    )
    deduped = score_source_quality(ingestion["deduped_sentiment_records"])
    normalized.to_csv(output / "normalized_sentiment_records.csv", index=False)
    deduped.to_csv(output / "deduped_sentiment_records.csv", index=False)

    ex_ante = validate_ex_ante_records(deduped)
    ex_ante = flag_reaction_data_leakage(ex_ante)
    lag_days = int(config.get("validation", {}).get("decision_lag_days", 1))
    ex_ante = apply_publication_lag(ex_ante, lag_days=max(1, lag_days))
    ex_ante = score_source_quality(ex_ante)
    ex_ante.to_csv(output / "ex_ante_validation.csv", index=False)
    reaction_warnings = ex_ante.loc[
        ex_ante.get(
            "possible_reaction_data",
            pd.Series(False, index=ex_ante.index),
        ).fillna(False)
    ].copy()
    reaction_warnings.to_csv(
        output / "reaction_data_warnings.csv", index=False
    )

    provider_diagnostics = _merge_provider_diagnostics(
        config, ingestion["provider_diagnostics"], intake["intake_status"]
    )
    provider_diagnostics.to_csv(
        output / "provider_diagnostics.csv", index=False
    )
    real_records = ex_ante.loc[
        ex_ante.get(
            "is_real_provider_data",
            pd.Series(False, index=ex_ante.index),
        ).fillna(False)
    ].copy()
    warning = (
        ""
        if not real_records.empty
        else (
            "Real provider data unavailable; collection completed with "
            "configuration, cache, fixture, and provenance diagnostics only."
        )
    )
    summary = {
        "status": "success",
        "warning": warning,
        "config_path": config["_config_path"],
        "config_valid": bool(config["_validation"]["is_valid"]),
        "config_errors": config["_validation"]["errors"],
        "config_warnings": config["_validation"]["warnings"],
        "no_live": bool(no_live),
        "network_calls_allowed": not bool(no_live),
        "cache_enabled": bool(use_cache),
        "cache_ignored": bool(ignore_cache),
        "cache_dir": str(Path(cache_dir).resolve()),
        "start_date": pd.Timestamp(start_date).date().isoformat(),
        "end_date": pd.Timestamp(end_date).date().isoformat(),
        "providers_enabled": config["_validation"]["enabled_providers"],
        "providers_initialized": [
            getattr(provider, "provider_name", "unknown")
            for provider in providers
        ],
        "collected_record_count": int(len(deduped)),
        "real_record_count": int(len(real_records)),
        "fixture_or_placeholder_count": int(
            deduped.get(
                "data_provenance",
                pd.Series("", index=deduped.index),
            ).eq("fixture_or_placeholder").sum()
        ),
        "ex_ante_valid_count": int(
            ex_ante.get(
                "is_ex_ante_valid",
                pd.Series(False, index=ex_ante.index),
            ).fillna(False).sum()
        ),
        "reaction_warning_count": int(len(reaction_warnings)),
        "gdelt_query_count": int(len(gdelt_query_diagnostics)),
        "intake_manual_action_required": bool(
            intake["manual_action_required"]
        ),
        "valid_real_records_by_corpus": (
            intake["valid_real_records_by_corpus"]
        ),
        "corpus_sufficiency_status": (
            "ready"
            if intake["all_corpora_ready"]
            else "manual_action_required"
        ),
        "output_dir": str(output.resolve()),
    }
    (output / "collection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "config": config,
        "provider_diagnostics": provider_diagnostics,
        "normalized_records": normalized,
        "deduped_records": deduped,
        "ex_ante_validation": ex_ante,
        "reaction_data_warnings": reaction_warnings,
        "gdelt_query_diagnostics": gdelt_query_diagnostics,
        "intake": intake,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect real or cached NLP provider records."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Ignore an existing cache and force provider fetches; successful "
            "responses may replace the cache."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("rbi", "earnings", "gdelt", "alpha_vantage"),
        default=None,
    )
    parser.add_argument("--sanity-query", default=None)
    parser.add_argument("--no-live", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = collect_real_nlp_data(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        use_cache=args.use_cache,
        no_live=args.no_live,
        ignore_cache=args.no_cache,
        provider_filter=args.provider,
        sanity_query=args.sanity_query,
    )
    summary = result["summary"]
    print(
        "Collection completed: "
        f"{summary['real_record_count']} real record(s), "
        f"{summary['fixture_or_placeholder_count']} fixture/placeholder "
        "record(s)."
    )
    if summary["warning"]:
        print(f"WARNING: {summary['warning']}")
    if summary["config_errors"]:
        print(
            "CONFIG DIAGNOSTICS: "
            + " | ".join(summary["config_errors"])
        )
    print(f"Outputs: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
