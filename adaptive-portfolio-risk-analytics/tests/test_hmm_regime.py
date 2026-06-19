"""Tests for optional Phase 3B.2 probabilistic HMM regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.regime.hmm_regime as hmm_module
from src.regime import (
    HMM_AVAILABLE,
    calculate_hmm_transition_matrix,
    compare_regime_methods,
    fit_hmm_full_sample,
    fit_hmm_walk_forward,
    map_hmm_states_to_regimes,
    prepare_hmm_features,
)


class _FakeMonitor:
    converged = True


class _FakeGaussianHMM:
    def __init__(self, n_components, **kwargs):
        self.n_components = int(n_components)
        self.monitor_ = _FakeMonitor()

    def fit(self, values):
        return self

    def predict(self, values):
        return np.arange(len(values), dtype=int) % self.n_components

    def predict_proba(self, values):
        probabilities = np.zeros((len(values), self.n_components), dtype=float)
        states = self.predict(values)
        probabilities[np.arange(len(values)), states] = 1.0
        return probabilities

    def score(self, values):
        return -float(len(values))


def _features(periods: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(20260618)
    index = pd.date_range("2022-01-03", periods=periods, freq="B")
    volatility = np.concatenate(
        [
            np.full(periods // 2, 0.10),
            np.full(periods - periods // 2, 0.30),
        ]
    )
    drawdown = np.concatenate(
        [
            np.linspace(0.0, -0.02, periods // 2),
            np.linspace(-0.05, -0.25, periods - periods // 2),
        ]
    )
    return pd.DataFrame(
        {
            "benchmark_return_21d": np.concatenate(
                [
                    np.full(periods // 2, 0.04),
                    np.full(periods - periods // 2, -0.08),
                ]
            ),
            "rolling_volatility": volatility + rng.normal(0.0, 0.005, size=periods),
            "rolling_drawdown": drawdown,
            "momentum_63d": np.concatenate(
                [
                    np.full(periods // 2, 0.08),
                    np.full(periods - periods // 2, -0.12),
                ]
            ),
            "average_correlation": np.concatenate(
                [
                    np.full(periods // 2, 0.30),
                    np.full(periods - periods // 2, 0.80),
                ]
            ),
        },
        index=index,
    )


def test_prepare_hmm_features_returns_aligned_output() -> None:
    features = _features()

    prepared = prepare_hmm_features(features)

    assert prepared["X"].shape == (len(features), 5)
    assert prepared["feature_index"].equals(features.index)
    assert len(prepared["used_columns"]) == 5
    assert prepared["scaler"] is not None


def test_prepare_hmm_features_handles_nans_and_fallback_columns() -> None:
    features = _features()
    features.loc[features.index[:5], "rolling_volatility"] = np.nan
    features["fallback_numeric"] = np.arange(len(features), dtype=float)

    prepared = prepare_hmm_features(
        features,
        feature_columns=["missing_column", "fallback_numeric"],
    )

    assert prepared["used_columns"] == ["fallback_numeric"]
    assert len(prepared["feature_index"]) == len(features)


def test_prepare_hmm_features_raises_when_no_numeric_features_exist() -> None:
    features = pd.DataFrame(
        {"label": ["a", "b"]},
        index=pd.date_range("2024-01-01", periods=2),
    )

    with pytest.raises(ValueError, match="no usable numeric"):
        prepare_hmm_features(features)


@pytest.mark.parametrize(
    ("n_states", "expected_labels"),
    [
        (2, {"Risk-On", "Risk-Off"}),
        (3, {"Calm", "Normal", "Stress"}),
        (4, {"Calm", "Normal", "Stress", "Crisis"}),
    ],
)
def test_state_mapping_returns_readable_regimes(
    n_states: int,
    expected_labels: set[str],
) -> None:
    features = _features(periods=120)
    repeated_states = np.resize(np.arange(n_states), len(features))
    states = pd.Series(repeated_states, index=features.index, name="state")

    mapped = map_hmm_states_to_regimes(
        states,
        features,
        n_states=n_states,
    )

    assert set(mapped["mapping"].values()) == expected_labels
    assert set(mapped["regimes"].unique()) == expected_labels
    assert set(mapped["state_summary"]["mapped_regime"]) == expected_labels


@pytest.mark.skipif(not HMM_AVAILABLE, reason="hmmlearn is not installed")
def test_full_sample_hmm_returns_aligned_states() -> None:
    features = _features(periods=240)

    result = fit_hmm_full_sample(
        features,
        n_states=2,
        covariance_type="diag",
        n_iter=50,
    )

    assert result["states"].index.equals(result["feature_index"])
    assert result["state_probabilities"].index.equals(result["feature_index"])
    assert result["state_probabilities"].shape[1] == 2


@pytest.mark.skipif(not HMM_AVAILABLE, reason="hmmlearn is not installed")
def test_walk_forward_hmm_has_unknown_warmup_and_lagged_decisions() -> None:
    features = _features(periods=180)

    result = fit_hmm_walk_forward(
        features,
        n_states=2,
        min_train_size=80,
        refit_frequency=20,
        covariance_type="diag",
        n_iter=50,
        decision_lag=1,
    )

    assert result["regimes"].iloc[:80].eq("Unknown").all()
    assert result["decision_regimes"].iloc[:81].eq("Unknown").all()
    first_state_date = result["states"].first_valid_index()
    assert result["decision_states"].loc[first_state_date] is pd.NA or pd.isna(
        result["decision_states"].loc[first_state_date]
    )


def test_hmm_transition_matrix_reuses_common_analytics() -> None:
    regimes = pd.Series(
        ["Calm", "Calm", "Stress", "Stress", "Normal"],
        index=pd.date_range("2024-01-01", periods=5),
    )

    result = calculate_hmm_transition_matrix(regimes)

    assert result["transition_count_matrix"].loc["Calm", "Calm"] == 1
    assert result["transition_count_matrix"].loc["Calm", "Stress"] == 1
    assert result["current_regime"] == "Normal"


def test_rule_based_vs_hmm_comparison_reports_agreement() -> None:
    index = pd.date_range("2024-01-01", periods=5)
    rule_based = pd.Series(
        ["Calm", "Normal", "Stress", "Crisis", "Unknown"],
        index=index,
    )
    hmm = pd.Series(
        ["Calm", "Stress", "Stress", "Crisis", "Calm"],
        index=index,
    )

    comparison = compare_regime_methods(rule_based, hmm)

    assert comparison["agreement_rate"] == pytest.approx(0.75)
    assert len(comparison["dates_of_disagreement"]) == 1
    assert comparison["crosstab"].loc["Stress", "Stress"] == 1


def test_hmm_fit_raises_clear_optional_dependency_error_when_unavailable() -> None:
    if HMM_AVAILABLE:
        pytest.skip("hmmlearn is installed")

    with pytest.raises(ImportError, match="optional dependency `hmmlearn`"):
        fit_hmm_full_sample(_features(), n_states=2)


def test_walk_forward_logic_is_lagged_and_time_ordered_with_fake_hmm(
    monkeypatch,
) -> None:
    monkeypatch.setattr(hmm_module, "GaussianHMM", _FakeGaussianHMM)
    features = _features(periods=120)

    result = fit_hmm_walk_forward(
        features,
        n_states=2,
        min_train_size=60,
        refit_frequency=15,
        covariance_type="diag",
        decision_lag=1,
    )

    assert result["regimes"].iloc[:60].eq("Unknown").all()
    assert result["states"].iloc[60:].notna().all()
    assert result["decision_regimes"].iloc[:61].eq("Unknown").all()
    assert result["decision_regimes"].iloc[61] == result["regimes"].iloc[60]
    probabilities = result["state_probabilities"].iloc[60:].dropna(how="all")
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (result["diagnostics"]["status"] == "success").all()
