"""Tests for the Phase 4A.13 NLP shadow-impact experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_nlp_shadow_impact_experiment as runner
from src.selection import select_strategy_for_profile


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(20260625)
    index = pd.bdate_range("2025-07-01", "2026-06-19")
    common = rng.normal(0.0002, 0.006, len(index))
    return pd.DataFrame(
        {
            "A": common + rng.normal(0.0, 0.005, len(index)),
            "B": common + rng.normal(0.0, 0.006, len(index)),
            "C": common + rng.normal(0.0, 0.007, len(index)),
            "D": common + rng.normal(0.0, 0.008, len(index)),
        },
        index=index,
    )


def _nlp_signal(index: pd.DatetimeIndex) -> pd.DataFrame:
    labels = pd.Series("nlp_neutral", index=index, dtype=object)
    labels.loc["2026-05-05":"2026-05-12"] = "nlp_risk_off"
    return pd.DataFrame(
        {
            "date": index,
            "decision_nlp_label": labels.to_numpy(),
            "source_mix": "news_only",
            "coverage_score": 1.0 / 3.0,
            "decision_source_date": [pd.NaT] + list(index[:-1]),
        }
    )


def test_shadow_impact_experiment_writes_required_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    returns = _returns()
    monkeypatch.setattr(runner, "HMM_AVAILABLE", False)

    result = runner.run_nlp_shadow_impact_experiment(
        start_date="2026-04-01",
        end_date="2026-06-19",
        include_transaction_costs=True,
        decision_lag_days=1,
        output_dir=tmp_path,
        returns_df=returns,
        nlp_signal=_nlp_signal(returns.index),
    )

    required = {
        "summary.md",
        "report.html",
        "strategy_metrics.csv",
        "pain_ratio_comparison.csv",
        "drawdown_comparison.csv",
        "overlay_decisions.csv",
        "nlp_signal_alignment.csv",
        "lookahead_diagnostics.csv",
        "limitations.md",
    }
    assert required.issubset({path.name for path in tmp_path.iterdir()})
    assert set(result.strategy_metrics["strategy"]) == {
        "Fixed HERC",
        "HMM Conservative",
        "Rule Conservative",
        "HMM + NLP Confirmation Overlay",
        "HMM + NLP Early-Warning Overlay",
    }
    assert {"pain_index", "pain_ratio"}.issubset(result.strategy_metrics.columns)
    assert result.strategy_metrics["transaction_costs_included"].all()
    assert result.strategy_metrics["total_transaction_cost"].notna().all()
    assert bool(result.lookahead_diagnostics["lookahead_check_passed"].all()) is True
    shadow = result.strategy_metrics.loc[result.strategy_metrics["shadow_experimental"]]
    assert shadow["production_allocation_active"].eq(False).all()


def test_shadow_experiment_does_not_alter_production_strategy_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    returns = _returns()
    monkeypatch.setattr(runner, "HMM_AVAILABLE", False)
    baseline = select_strategy_for_profile("Balanced")

    runner.run_nlp_shadow_impact_experiment(
        start_date="2026-04-01",
        end_date="2026-06-19",
        include_transaction_costs=True,
        decision_lag_days=1,
        output_dir=tmp_path,
        returns_df=returns,
        nlp_signal=_nlp_signal(returns.index),
    )
    after = select_strategy_for_profile("Balanced")

    assert after.main_strategy == baseline.main_strategy
    assert after.overlay_strategy == baseline.overlay_strategy
    assert after.confidence_score == baseline.confidence_score
