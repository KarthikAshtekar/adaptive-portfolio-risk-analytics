"""Experimental Gaussian HMM market regime detection for Phase 3B.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .analytics import calculate_regime_transitions
from .rule_based import lag_regime_labels

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover - environment-dependent optional dependency
    GaussianHMM = None

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover - repository normally includes scikit-learn
    StandardScaler = None


HMM_AVAILABLE = GaussianHMM is not None

DEFAULT_HMM_FEATURE_COLUMNS = [
    "benchmark_return_21d",
    "rolling_volatility",
    "rolling_drawdown",
    "momentum_63d",
    "average_correlation",
]

STATE_SUMMARY_COLUMNS = [
    "rolling_volatility",
    "rolling_drawdown",
    "benchmark_return_21d",
    "momentum_63d",
    "average_correlation",
]


@dataclass
class ArrayStandardizer:
    """Small StandardScaler-compatible fallback using population moments."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "ArrayStandardizer":
        self.mean_ = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0, ddof=0)
        self.scale_ = np.where(scale > 0.0, scale, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("standardizer must be fitted before transform")
        return (values - self.mean_) / self.scale_

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)


def _new_scaler():
    return StandardScaler() if StandardScaler is not None else ArrayStandardizer()


def _require_hmmlearn() -> None:
    if GaussianHMM is None:
        raise ImportError("HMM regime detection requires the optional dependency `hmmlearn`.")


def _validate_features(features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if features.empty:
        raise ValueError("features must not be empty")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise ValueError("features index must be a DatetimeIndex")
    return features.sort_index()


def _select_feature_columns(
    features: pd.DataFrame,
    feature_columns: Iterable[str] | None,
) -> list[str]:
    requested = list(feature_columns or DEFAULT_HMM_FEATURE_COLUMNS)
    used_columns: list[str] = []
    for column in requested:
        if column not in features.columns:
            continue
        numeric = pd.to_numeric(features[column], errors="coerce")
        if numeric.notna().any():
            used_columns.append(column)

    if not used_columns:
        fallback_columns = [
            column
            for column in features.select_dtypes(include=[np.number]).columns
            if pd.to_numeric(features[column], errors="coerce").notna().any()
        ]
        used_columns = fallback_columns[: len(DEFAULT_HMM_FEATURE_COLUMNS)]

    if not used_columns:
        raise ValueError("no usable numeric HMM feature columns are available")
    return used_columns


def prepare_hmm_features(
    features,
    feature_columns=None,
    dropna: bool = True,
    standardize: bool = True,
) -> dict[str, object]:
    """Prepare an aligned numeric HMM design matrix.

    Missing requested columns fall back to other available numeric columns. With
    ``dropna=False``, feature medians are used for imputation. The returned scaler
    is fitted only on the observations supplied to this function.
    """
    frame = _validate_features(features)
    used_columns = _select_feature_columns(frame, feature_columns)
    selected = frame[used_columns].apply(pd.to_numeric, errors="coerce")

    if dropna:
        selected = selected.dropna(how="any")
    else:
        medians = selected.median(axis=0, skipna=True)
        selected = selected.fillna(medians).dropna(how="any")

    if selected.empty:
        raise ValueError("no complete HMM feature observations are available")

    raw_values = selected.to_numpy(dtype=float)
    scaler = None
    values = raw_values
    if standardize:
        scaler = _new_scaler()
        values = scaler.fit_transform(raw_values)

    return {
        "X": np.asarray(values, dtype=float),
        "feature_index": selected.index,
        "used_columns": used_columns,
        "scaler": scaler,
        "feature_frame": selected,
    }


def _new_hmm(
    n_states: int,
    covariance_type: str,
    n_iter: int,
    random_state: int,
):
    _require_hmmlearn()
    if int(n_states) < 2:
        raise ValueError("n_states must be at least 2")
    if covariance_type not in {"diag", "full", "tied", "spherical"}:
        raise ValueError("covariance_type must be one of: diag, full, tied, spherical")
    return GaussianHMM(
        n_components=int(n_states),
        covariance_type=covariance_type,
        n_iter=int(n_iter),
        random_state=int(random_state),
    )


def fit_hmm_full_sample(
    features,
    n_states: int = 4,
    feature_columns=None,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42,
) -> dict[str, object]:
    """Fit a full-sample Gaussian HMM for historical visualization only.

    The model uses the complete supplied feature history and is therefore not
    trading-safe. Use :func:`fit_hmm_walk_forward` for decision-facing output.
    """
    prepared = prepare_hmm_features(
        features,
        feature_columns=feature_columns,
        dropna=True,
        standardize=True,
    )
    if len(prepared["feature_index"]) < max(10, int(n_states) * 3):
        raise ValueError("insufficient complete observations to fit the HMM")

    model = _new_hmm(n_states, covariance_type, n_iter, random_state)
    model.fit(prepared["X"])
    predicted_states = model.predict(prepared["X"])
    probabilities = model.predict_proba(prepared["X"])
    feature_index = prepared["feature_index"]

    states = pd.Series(
        predicted_states,
        index=feature_index,
        name="hmm_state",
        dtype=int,
    )
    probability_frame = pd.DataFrame(
        probabilities,
        index=feature_index,
        columns=[f"state_{state}" for state in range(int(n_states))],
    )
    converged = bool(getattr(getattr(model, "monitor_", None), "converged", False))

    return {
        "model": model,
        "states": states,
        "state_probabilities": probability_frame,
        "feature_index": feature_index,
        "used_columns": prepared["used_columns"],
        "scaler": prepared["scaler"],
        "converged": converged,
        "log_likelihood": float(model.score(prepared["X"])),
        "historical_only": True,
    }


def _risk_labels(n_states: int) -> list[str]:
    if n_states == 1:
        return ["Normal"]
    if n_states == 2:
        return ["Risk-On", "Risk-Off"]
    if n_states == 3:
        return ["Calm", "Normal", "Stress"]
    if n_states == 4:
        return ["Calm", "Normal", "Stress", "Crisis"]
    return [f"Regime {position + 1}" for position in range(n_states)]


def _state_summary(
    states: pd.Series,
    features: pd.DataFrame,
) -> pd.DataFrame:
    aligned = pd.concat(
        [
            pd.to_numeric(states, errors="coerce").rename("state"),
            features.reindex(columns=STATE_SUMMARY_COLUMNS).apply(
                pd.to_numeric,
                errors="coerce",
            ),
        ],
        axis=1,
        join="inner",
    ).dropna(subset=["state"])
    if aligned.empty:
        return pd.DataFrame()

    summary = (
        aligned.groupby("state", sort=True)
        .agg(
            count=("state", "size"),
            avg_rolling_volatility=("rolling_volatility", "mean"),
            avg_rolling_drawdown=("rolling_drawdown", "mean"),
            avg_benchmark_return_21d=("benchmark_return_21d", "mean"),
            avg_momentum_63d=("momentum_63d", "mean"),
            avg_average_correlation=("average_correlation", "mean"),
        )
        .reset_index()
    )
    summary["state"] = summary["state"].astype(int)

    risk_inputs = pd.DataFrame(index=summary.index)
    risk_inputs["volatility"] = summary["avg_rolling_volatility"].rank(
        pct=True,
        method="average",
    )
    risk_inputs["drawdown"] = (-summary["avg_rolling_drawdown"]).rank(
        pct=True,
        method="average",
    )
    risk_inputs["weak_return"] = (-summary["avg_benchmark_return_21d"]).rank(
        pct=True,
        method="average",
    )
    risk_inputs["weak_momentum"] = (-summary["avg_momentum_63d"]).rank(
        pct=True,
        method="average",
    )
    risk_inputs["correlation"] = summary["avg_average_correlation"].rank(
        pct=True,
        method="average",
    )
    summary["risk_score"] = risk_inputs.mean(axis=1, skipna=True).fillna(0.5)
    return summary


def map_hmm_states_to_regimes(
    states,
    features,
    n_states: int | None = None,
) -> dict[str, object]:
    """Map arbitrary HMM state IDs to readable regimes using state risk statistics."""
    if not isinstance(states, pd.Series):
        states = pd.Series(states)
    frame = _validate_features(features)
    state_summary = _state_summary(states, frame)
    if state_summary.empty:
        return {
            "regimes": pd.Series(
                "Unknown",
                index=states.index,
                name="hmm_regime",
                dtype="object",
            ),
            "mapping": {},
            "state_summary": state_summary,
        }

    observed_state_count = len(state_summary)
    requested_state_count = int(n_states or observed_state_count)
    requested_labels = _risk_labels(requested_state_count)
    labels = (
        requested_labels[:observed_state_count]
        if observed_state_count <= len(requested_labels)
        else _risk_labels(observed_state_count)
    )

    risk_order = state_summary.sort_values(
        ["risk_score", "state"],
        kind="mergesort",
    )["state"].tolist()
    mapping = {int(state): label for state, label in zip(risk_order, labels)}
    state_summary["mapped_regime"] = state_summary["state"].map(mapping)
    ordered_columns = [
        "state",
        "mapped_regime",
        "count",
        "avg_rolling_volatility",
        "avg_rolling_drawdown",
        "avg_benchmark_return_21d",
        "avg_momentum_63d",
        "avg_average_correlation",
        "risk_score",
    ]
    regimes = pd.to_numeric(states, errors="coerce").map(mapping).fillna("Unknown")
    regimes.name = "hmm_regime"

    return {
        "regimes": regimes.astype("object"),
        "mapping": mapping,
        "state_summary": state_summary[ordered_columns],
    }


def _canonical_state_mapping(n_states: int) -> tuple[list[str], dict[str, int]]:
    labels = _risk_labels(n_states)
    return labels, {label: position for position, label in enumerate(labels)}


def fit_hmm_walk_forward(
    features,
    n_states: int = 4,
    feature_columns=None,
    min_train_size: int = 504,
    refit_frequency: int = 21,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42,
    decision_lag: int = 1,
) -> dict[str, object]:
    """Fit expanding-window HMMs and infer each date using past data only.

    At each refit, the scaler and HMM are trained using observations strictly
    before the out-of-sample segment. Each date's probability uses the fitted
    model and observations available only through that date.
    """
    _require_hmmlearn()
    if int(min_train_size) <= 0:
        raise ValueError("min_train_size must be positive")
    if int(refit_frequency) <= 0:
        raise ValueError("refit_frequency must be positive")
    if int(decision_lag) < 0:
        raise ValueError("decision_lag must be non-negative")

    prepared = prepare_hmm_features(
        features,
        feature_columns=feature_columns,
        dropna=True,
        standardize=False,
    )
    feature_frame = prepared["feature_frame"]
    feature_index = feature_frame.index
    n_observations = len(feature_frame)
    labels, regime_to_state = _canonical_state_mapping(int(n_states))

    states = pd.Series(
        pd.NA,
        index=feature_index,
        name="hmm_state",
        dtype="Int64",
    )
    regimes = pd.Series(
        "Unknown",
        index=feature_index,
        name="hmm_regime",
        dtype="object",
    )
    state_probabilities = pd.DataFrame(
        np.nan,
        index=feature_index,
        columns=[f"state_{state}" for state in range(int(n_states))],
        dtype=float,
    )
    diagnostics: list[dict[str, object]] = []
    refit_dates: list[pd.Timestamp] = []

    if n_observations <= int(min_train_size):
        decision_states = states.shift(int(decision_lag))
        decision_regimes = lag_regime_labels(regimes, lag=int(decision_lag))
        return {
            "states": states,
            "regimes": regimes,
            "decision_states": decision_states,
            "decision_regimes": decision_regimes,
            "state_probabilities": state_probabilities,
            "refit_dates": refit_dates,
            "used_columns": prepared["used_columns"],
            "diagnostics": pd.DataFrame(),
            "state_summary": pd.DataFrame(),
            "mapping": {state: label for state, label in enumerate(labels)},
        }

    for segment_start in range(
        int(min_train_size),
        n_observations,
        int(refit_frequency),
    ):
        segment_end = min(
            n_observations,
            segment_start + int(refit_frequency),
        )
        train_frame = feature_frame.iloc[:segment_start]
        test_frame = feature_frame.iloc[segment_start:segment_end]
        refit_date = train_frame.index[-1]
        refit_dates.append(refit_date)
        diagnostic = {
            "refit_date": refit_date,
            "train_start": train_frame.index[0],
            "train_end": train_frame.index[-1],
            "test_start": test_frame.index[0],
            "test_end": test_frame.index[-1],
            "n_train": len(train_frame),
            "n_test": len(test_frame),
            "status": "success",
            "converged": False,
            "log_likelihood": np.nan,
            "error": None,
        }

        try:
            scaler = _new_scaler()
            train_values = scaler.fit_transform(train_frame.to_numpy(dtype=float))
            model = _new_hmm(
                n_states,
                covariance_type,
                n_iter,
                random_state,
            )
            model.fit(train_values)
            training_states = pd.Series(
                model.predict(train_values),
                index=train_frame.index,
                dtype=int,
            )
            local_mapping_result = map_hmm_states_to_regimes(
                training_states,
                train_frame,
                n_states=int(n_states),
            )
            local_mapping = dict(local_mapping_result["mapping"])
            unused_labels = [label for label in labels if label not in local_mapping.values()]
            for raw_state in range(int(n_states)):
                if raw_state not in local_mapping and unused_labels:
                    local_mapping[raw_state] = unused_labels.pop(0)

            all_segment_values = pd.concat([train_frame, test_frame]).to_numpy(dtype=float)
            transformed_segment = scaler.transform(all_segment_values)
            train_length = len(train_frame)

            for local_position, current_date in enumerate(test_frame.index):
                sequence_end = train_length + local_position + 1
                current_probabilities = model.predict_proba(transformed_segment[:sequence_end])[-1]
                local_state = int(np.argmax(current_probabilities))
                regime = local_mapping.get(local_state, "Unknown")
                canonical_state = regime_to_state.get(regime)
                if canonical_state is None:
                    continue

                states.loc[current_date] = canonical_state
                regimes.loc[current_date] = regime
                state_probabilities.loc[current_date] = 0.0
                for raw_state, probability in enumerate(current_probabilities):
                    raw_regime = local_mapping.get(raw_state)
                    mapped_state = regime_to_state.get(raw_regime)
                    if mapped_state is not None:
                        state_probabilities.loc[
                            current_date,
                            f"state_{mapped_state}",
                        ] += float(probability)

            diagnostic["converged"] = bool(
                getattr(getattr(model, "monitor_", None), "converged", False)
            )
            diagnostic["log_likelihood"] = float(model.score(train_values))
        except Exception as exc:
            diagnostic["status"] = "failed"
            diagnostic["error"] = str(exc)
        diagnostics.append(diagnostic)

    decision_states = states.shift(int(decision_lag))
    decision_states.name = "decision_hmm_state"
    decision_regimes = lag_regime_labels(regimes, lag=int(decision_lag))
    decision_regimes.name = "decision_hmm_regime"

    state_summary = _state_summary(states.dropna().astype(int), feature_frame)
    canonical_mapping = {state: label for state, label in enumerate(labels)}
    if not state_summary.empty:
        state_summary["mapped_regime"] = state_summary["state"].map(canonical_mapping)
        state_summary = state_summary[
            [
                "state",
                "mapped_regime",
                "count",
                "avg_rolling_volatility",
                "avg_rolling_drawdown",
                "avg_benchmark_return_21d",
                "avg_momentum_63d",
                "avg_average_correlation",
                "risk_score",
            ]
        ]

    return {
        "states": states,
        "regimes": regimes,
        "decision_states": decision_states,
        "decision_regimes": decision_regimes,
        "state_probabilities": state_probabilities,
        "refit_dates": refit_dates,
        "used_columns": prepared["used_columns"],
        "diagnostics": pd.DataFrame(diagnostics),
        "state_summary": state_summary,
        "mapping": canonical_mapping,
    }


def calculate_hmm_transition_matrix(states_or_regimes) -> dict[str, object]:
    """Reuse the common transition and duration analytics for HMM output."""
    if not isinstance(states_or_regimes, pd.Series):
        states_or_regimes = pd.Series(states_or_regimes)
    if not isinstance(states_or_regimes.index, pd.DatetimeIndex):
        states_or_regimes.index = pd.date_range(
            "2000-01-01",
            periods=len(states_or_regimes),
            freq="D",
        )
    values = states_or_regimes.astype("object").where(
        states_or_regimes.notna(),
        "Unknown",
    )
    return calculate_regime_transitions(values)


def compare_regime_methods(
    rule_based_regimes,
    hmm_regimes,
) -> dict[str, object]:
    """Compare aligned rule-based and probabilistic HMM regime labels."""
    if not isinstance(rule_based_regimes, pd.Series):
        rule_based_regimes = pd.Series(rule_based_regimes)
    if not isinstance(hmm_regimes, pd.Series):
        hmm_regimes = pd.Series(hmm_regimes)

    comparison = pd.concat(
        [
            rule_based_regimes.rename("rule_based_regime"),
            hmm_regimes.rename("hmm_regime"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    valid = comparison[
        ~comparison["rule_based_regime"].astype(str).eq("Unknown")
        & ~comparison["hmm_regime"].astype(str).eq("Unknown")
    ].copy()
    agreement_rate = (
        float((valid["rule_based_regime"].astype(str) == valid["hmm_regime"].astype(str)).mean())
        if not valid.empty
        else np.nan
    )
    crosstab = (
        pd.crosstab(
            valid["rule_based_regime"],
            valid["hmm_regime"],
        )
        if not valid.empty
        else pd.DataFrame()
    )
    disagreement = valid[
        valid["rule_based_regime"].astype(str) != valid["hmm_regime"].astype(str)
    ].copy()
    disagreement.insert(0, "date", disagreement.index)
    counts = (
        pd.concat(
            [
                valid["rule_based_regime"].value_counts().rename("Rule-based"),
                valid["hmm_regime"].value_counts().rename("HMM"),
            ],
            axis=1,
        )
        .fillna(0)
        .astype(int)
    )
    counts.index.name = "regime"

    return {
        "agreement_rate": agreement_rate,
        "crosstab": crosstab,
        "dates_of_disagreement": disagreement.reset_index(drop=True),
        "regime_counts_by_method": counts,
        "comparison_table": comparison,
    }
