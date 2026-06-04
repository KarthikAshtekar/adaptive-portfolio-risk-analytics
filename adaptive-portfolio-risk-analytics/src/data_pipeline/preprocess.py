"""Data preprocessing for returns and risk analytics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252
MISSING_DATA_DROP_THRESHOLD = 0.05
PRICE_ANOMALY_THRESHOLD = 0.50
RETURN_OUTLIER_MODIFIED_Z_THRESHOLD = 5.0
RETURN_ZSCORE_THRESHOLD = 3.0
RETURN_WINSORIZE_LOWER_BOUND = -0.20
RETURN_WINSORIZE_UPPER_BOUND = 0.20
MAX_PRICE_REPAIR_ITERATIONS_MULTIPLIER = 5

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReturnsRiskOutputs:
    """Container for Stage 2 return and volatility outputs."""

    simple_returns_df: pd.DataFrame
    returns_df: pd.DataFrame
    log_returns_df: pd.DataFrame
    volatility_summary_df: pd.DataFrame
    rolling_volatility_df: pd.DataFrame
    return_comparison_df: pd.DataFrame
    quality_report_df: pd.DataFrame
    anomaly_report_df: pd.DataFrame
    repair_report_df: pd.DataFrame
    outlier_report_df: pd.DataFrame
    stabilization_report_df: pd.DataFrame


@dataclass(frozen=True)
class MissingDataSummary:
    """Summary of asset-level missing-data cleaning."""

    total_assets_requested: int
    assets_retained: int
    assets_dropped: int
    missing_before: int
    missing_after: int
    dropped_asset_names: tuple[str, ...]
    dropped_asset_missing_percentages: dict[str, float]
    cleaning_method: str


class DataQualityProcessor:
    """Detect, repair, and stabilize anomalous observations before analytics."""

    def __init__(
        self,
        price_anomaly_threshold: float = PRICE_ANOMALY_THRESHOLD,
        return_outlier_threshold: float = RETURN_OUTLIER_MODIFIED_Z_THRESHOLD,
        winsorize_bounds: tuple[float, float] = (
            RETURN_WINSORIZE_LOWER_BOUND,
            RETURN_WINSORIZE_UPPER_BOUND,
        ),
    ) -> None:
        lower_bound, upper_bound = winsorize_bounds
        if lower_bound >= upper_bound:
            raise ValueError("winsorize_bounds must be ordered as (lower_bound, upper_bound)")

        self.price_anomaly_threshold = float(price_anomaly_threshold)
        self.return_outlier_threshold = float(return_outlier_threshold)
        self.winsorize_bounds = (float(lower_bound), float(upper_bound))

        self._price_repair_methods: dict[
            str,
            Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]],
        ] = {
            "interpolate": self._repair_price_anomalies_interpolate,
        }
        self._return_outlier_methods: dict[str, Callable[[pd.DataFrame, float], pd.DataFrame]] = {
            "mad": self._detect_return_outliers_mad,
            "zscore": self._detect_return_outliers_zscore,
        }
        self._return_stabilization_methods: dict[
            str,
            Callable[[pd.DataFrame, float, float], tuple[pd.DataFrame, pd.DataFrame]],
        ] = {
            "winsorize": self._stabilize_returns_winsorize,
        }

        self._last_prices_shape: tuple[int, int] = (0, 0)
        self._last_returns_shape: tuple[int, int] = (0, 0)
        self._last_price_anomaly_report_df = self._empty_price_anomaly_report()
        self._last_repair_report_df = self._empty_repair_report()
        self._last_outlier_report_df = self._empty_outlier_report()
        self._last_stabilization_report_df = self._empty_stabilization_report()
        self._last_price_repair_method = "interpolate"
        self._last_return_outlier_method = "mad"
        self._last_return_stabilization_method = "winsorize"
        self._last_repair_iterations = 0

    def register_price_repair_method(
        self,
        method: str,
        handler: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]],
    ) -> None:
        """Register a new price-repair method without changing downstream code."""

        self._price_repair_methods[method] = handler

    def register_return_outlier_method(
        self,
        method: str,
        handler: Callable[[pd.DataFrame, float], pd.DataFrame],
    ) -> None:
        """Register a new return-outlier detector."""

        self._return_outlier_methods[method] = handler

    def register_return_stabilization_method(
        self,
        method: str,
        handler: Callable[[pd.DataFrame, float, float], tuple[pd.DataFrame, pd.DataFrame]],
    ) -> None:
        """Register a new return-stabilization method."""

        self._return_stabilization_methods[method] = handler

    def detect_price_anomalies(
        self,
        prices_df: pd.DataFrame,
        threshold: float = PRICE_ANOMALY_THRESHOLD,
    ) -> pd.DataFrame:
        """Flag observations with suspiciously large log returns."""

        report_df = self._detect_price_anomalies_frame(prices_df, threshold=threshold)
        self._last_prices_shape = prices_df.shape
        self._last_price_anomaly_report_df = report_df.copy()
        return report_df

    def repair_price_anomalies(
        self,
        prices_df: pd.DataFrame,
        method: str = "interpolate",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Repair flagged price anomalies using a pluggable repair strategy."""

        if prices_df.empty:
            raise ValueError("prices_df must not be empty")
        if method not in self._price_repair_methods:
            raise ValueError(f"unknown price repair method: {method}")

        clean_prices_df, repair_report_df, anomaly_report_df, iterations = self._price_repair_methods[method](
            prices_df.copy(),
        )

        self._last_prices_shape = clean_prices_df.shape
        self._last_price_anomaly_report_df = anomaly_report_df.copy()
        self._last_repair_report_df = repair_report_df.copy()
        self._last_price_repair_method = method
        self._last_repair_iterations = iterations
        return clean_prices_df, repair_report_df

    def detect_return_outliers(
        self,
        returns_df: pd.DataFrame,
        method: str = "mad",
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """Detect return outliers using a selectable robust score."""

        if returns_df.empty:
            raise ValueError("returns_df must not be empty")
        if method not in self._return_outlier_methods:
            raise ValueError(f"unknown return outlier method: {method}")

        effective_threshold = self._resolve_outlier_threshold(method, threshold)
        report_df = self._return_outlier_methods[method](returns_df, effective_threshold)

        self._last_returns_shape = returns_df.shape
        self._last_outlier_report_df = report_df.copy()
        self._last_return_outlier_method = method
        return report_df

    def stabilize_returns(
        self,
        returns_df: pd.DataFrame,
        method: str = "winsorize",
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Stabilize returns before risk estimation."""

        if returns_df.empty:
            raise ValueError("returns_df must not be empty")
        if method not in self._return_stabilization_methods:
            raise ValueError(f"unknown return stabilization method: {method}")

        effective_lower = self.winsorize_bounds[0] if lower_bound is None else float(lower_bound)
        effective_upper = self.winsorize_bounds[1] if upper_bound is None else float(upper_bound)
        if effective_lower >= effective_upper:
            raise ValueError("lower_bound must be less than upper_bound")

        stabilized_returns_df, stabilization_report_df = self._return_stabilization_methods[method](
            returns_df,
            effective_lower,
            effective_upper,
        )

        self._last_returns_shape = stabilized_returns_df.shape
        self._last_stabilization_report_df = stabilization_report_df.copy()
        self._last_return_stabilization_method = method
        return stabilized_returns_df, stabilization_report_df

    def generate_quality_report(self) -> pd.DataFrame:
        """Summarize the full data-quality pass for notebooks and dashboards."""

        anomaly_assets = self._extract_assets(self._last_price_anomaly_report_df)
        repair_assets = self._extract_assets(self._last_repair_report_df)
        outlier_assets = self._extract_assets(self._last_outlier_report_df)
        stabilization_assets = self._extract_affected_assets_from_stabilization()

        affected_assets = tuple(
            sorted(
                set(anomaly_assets)
                | set(repair_assets)
                | set(outlier_assets)
                | set(stabilization_assets)
            )
        )

        stabilization_summary = (
            self._last_stabilization_report_df.iloc[0].to_dict()
            if not self._last_stabilization_report_df.empty
            else {}
        )

        quality_report_df = pd.DataFrame(
            [
                {
                    "price_rows": self._last_prices_shape[0],
                    "price_assets": self._last_prices_shape[1],
                    "return_rows": self._last_returns_shape[0],
                    "return_assets": self._last_returns_shape[1],
                    "price_anomalies_detected": int(len(self._last_price_anomaly_report_df)),
                    "price_repairs_applied": int(len(self._last_repair_report_df)),
                    "price_repair_iterations": int(self._last_repair_iterations),
                    "return_outliers_detected": int(len(self._last_outlier_report_df)),
                    "returns_clipped": int(stabilization_summary.get("num_clipped", 0)),
                    "affected_assets": affected_assets,
                    "price_anomaly_assets": anomaly_assets,
                    "repair_assets": repair_assets,
                    "return_outlier_assets": outlier_assets,
                    "price_anomaly_threshold": self.price_anomaly_threshold,
                    "price_repair_method": self._last_price_repair_method,
                    "return_outlier_method": self._last_return_outlier_method,
                    "return_outlier_threshold": self.return_outlier_threshold,
                    "return_stabilization_method": self._last_return_stabilization_method,
                    "lower_bound": stabilization_summary.get("lower_bound", self.winsorize_bounds[0]),
                    "upper_bound": stabilization_summary.get("upper_bound", self.winsorize_bounds[1]),
                }
            ]
        )
        return quality_report_df

    def _repair_price_anomalies_interpolate(
        self,
        prices_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
        working_prices_df = prices_df.astype(float).copy()
        repair_records: list[dict[str, object]] = []
        initial_anomaly_report_df = self._detect_price_anomalies_frame(
            working_prices_df,
            threshold=self.price_anomaly_threshold,
        )
        current_anomaly_report_df = initial_anomaly_report_df.copy()
        iterations = 0
        max_iterations = max(1, working_prices_df.shape[0] * MAX_PRICE_REPAIR_ITERATIONS_MULTIPLIER)

        while not current_anomaly_report_df.empty and iterations < max_iterations:
            iterations += 1
            updated_prices_df = working_prices_df.copy()

            for asset, asset_anomalies in current_anomaly_report_df.groupby("asset"):
                anomalous_dates = asset_anomalies["date"].tolist()
                interpolated_series = working_prices_df[asset].copy()
                interpolated_series.loc[anomalous_dates] = np.nan
                interpolated_series = interpolated_series.interpolate(
                    method="linear",
                    limit_direction="both",
                )

                for anomaly_row in asset_anomalies.itertuples(index=False):
                    repaired_price = float(interpolated_series.loc[anomaly_row.date])
                    original_price = float(working_prices_df.loc[anomaly_row.date, asset])
                    updated_prices_df.loc[anomaly_row.date, asset] = repaired_price
                    repair_records.append(
                        {
                            "date": anomaly_row.date,
                            "asset": asset,
                            "original_price": original_price,
                            "repaired_price": repaired_price,
                            "repair_method": "interpolate",
                            "iteration": iterations,
                        }
                    )

            working_prices_df = updated_prices_df
            current_anomaly_report_df = self._detect_price_anomalies_frame(
                working_prices_df,
                threshold=self.price_anomaly_threshold,
            )

        if not current_anomaly_report_df.empty:
            remaining_assets = sorted(current_anomaly_report_df["asset"].unique().tolist())
            raise ValueError(
                f"price anomalies remain after repair attempts for assets: {remaining_assets}"
            )

        repair_report_df = pd.DataFrame(
            repair_records,
            columns=[
                "date",
                "asset",
                "original_price",
                "repaired_price",
                "repair_method",
                "iteration",
            ],
        )
        return working_prices_df, repair_report_df, initial_anomaly_report_df, iterations

    def _detect_price_anomalies_frame(
        self,
        prices_df: pd.DataFrame,
        threshold: float,
    ) -> pd.DataFrame:
        if prices_df.empty:
            raise ValueError("prices_df must not be empty")

        with np.errstate(divide="ignore", invalid="ignore"):
            log_returns_df = np.log(prices_df / prices_df.shift(1))

        nonfinite_mask = ~np.isfinite(log_returns_df)
        if not nonfinite_mask.empty:
            nonfinite_mask.iloc[0] = False

        anomaly_mask = log_returns_df.abs().gt(threshold) | nonfinite_mask
        if not anomaly_mask.any().any():
            return self._empty_price_anomaly_report()

        report_df = (
            pd.DataFrame(
                {
                    "price": prices_df.stack(future_stack=True),
                    "log_return": log_returns_df.stack(future_stack=True),
                    "is_anomaly": anomaly_mask.stack(future_stack=True).fillna(False),
                }
            )
            .loc[lambda frame: frame["is_anomaly"]]
            .drop(columns="is_anomaly")
            .reset_index()
        )
        report_df.columns = ["date", "asset", "price", "log_return"]
        return report_df.sort_values(["date", "asset"]).reset_index(drop=True)

    def _detect_return_outliers_mad(
        self,
        returns_df: pd.DataFrame,
        threshold: float,
    ) -> pd.DataFrame:
        median = returns_df.median()
        mad = (returns_df - median).abs().median().replace(0.0, np.nan)
        scores_df = 0.6745 * (returns_df - median) / mad
        scores_df = scores_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return self._build_return_outlier_report(returns_df, scores_df, threshold, "mad")

    def _detect_return_outliers_zscore(
        self,
        returns_df: pd.DataFrame,
        threshold: float,
    ) -> pd.DataFrame:
        mean = returns_df.mean()
        std = returns_df.std(ddof=0).replace(0.0, np.nan)
        scores_df = (returns_df - mean) / std
        scores_df = scores_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return self._build_return_outlier_report(returns_df, scores_df, threshold, "zscore")

    def _build_return_outlier_report(
        self,
        returns_df: pd.DataFrame,
        scores_df: pd.DataFrame,
        threshold: float,
        method: str,
    ) -> pd.DataFrame:
        mask_df = scores_df.abs().gt(threshold)
        if not mask_df.any().any():
            return self._empty_outlier_report()

        report_df = (
            pd.DataFrame(
                {
                    "return": returns_df.stack(future_stack=True),
                    "score": scores_df.stack(future_stack=True),
                    "is_outlier": mask_df.stack(future_stack=True).fillna(False),
                }
            )
            .loc[lambda frame: frame["is_outlier"]]
            .drop(columns="is_outlier")
            .reset_index()
        )
        report_df.columns = ["date", "asset", "return", "score"]
        report_df["method"] = method
        return report_df[["date", "asset", "return", "score", "method"]].sort_values(
            ["date", "asset"]
        ).reset_index(drop=True)

    def _stabilize_returns_winsorize(
        self,
        returns_df: pd.DataFrame,
        lower_bound: float,
        upper_bound: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        clipped_mask_df = returns_df.lt(lower_bound) | returns_df.gt(upper_bound)
        stabilized_returns_df = returns_df.clip(lower=lower_bound, upper=upper_bound)
        affected_assets = tuple(sorted(clipped_mask_df.columns[clipped_mask_df.any()].tolist()))
        num_clipped = int(clipped_mask_df.to_numpy().sum())

        stabilization_report_df = pd.DataFrame(
            [
                {
                    "method": "winsorize",
                    "num_clipped": num_clipped,
                    "affected_assets": affected_assets,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                }
            ]
        )
        return stabilized_returns_df, stabilization_report_df

    def _resolve_outlier_threshold(self, method: str, threshold: float | None) -> float:
        if threshold is not None:
            return float(threshold)
        if method == "mad":
            return self.return_outlier_threshold
        if method == "zscore":
            return RETURN_ZSCORE_THRESHOLD
        raise ValueError(f"unknown return outlier method: {method}")

    def _extract_assets(self, report_df: pd.DataFrame) -> tuple[str, ...]:
        if report_df.empty or "asset" not in report_df.columns:
            return tuple()
        return tuple(sorted(report_df["asset"].dropna().astype(str).unique().tolist()))

    def _extract_affected_assets_from_stabilization(self) -> tuple[str, ...]:
        if self._last_stabilization_report_df.empty:
            return tuple()
        affected_assets = self._last_stabilization_report_df.iloc[0].get("affected_assets", tuple())
        if isinstance(affected_assets, tuple):
            return affected_assets
        if pd.isna(affected_assets):
            return tuple()
        if isinstance(affected_assets, list):
            return tuple(affected_assets)
        return tuple(str(affected_assets).split(","))

    @staticmethod
    def _empty_price_anomaly_report() -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "asset", "price", "log_return"])

    @staticmethod
    def _empty_repair_report() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "date",
                "asset",
                "original_price",
                "repaired_price",
                "repair_method",
                "iteration",
            ]
        )

    @staticmethod
    def _empty_outlier_report() -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "asset", "return", "score", "method"])

    @staticmethod
    def _empty_stabilization_report() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["method", "num_clipped", "affected_assets", "lower_bound", "upper_bound"]
        )


class DataPreprocessor:
    """Preprocess market price data for portfolio analytics."""

    @staticmethod
    def handle_missing_values(
        data: pd.DataFrame,
        method: str | None = None,
    ) -> tuple[pd.DataFrame, MissingDataSummary]:
        if data.empty:
            raise ValueError("data must not be empty")

        _ = method

        missing_before = int(data.isna().sum().sum())
        missing_percentages = data.isna().mean()
        dropped_assets = missing_percentages[missing_percentages > MISSING_DATA_DROP_THRESHOLD]
        retained_assets = missing_percentages.index[missing_percentages <= MISSING_DATA_DROP_THRESHOLD]

        if retained_assets.empty:
            raise ValueError("no assets remain after applying the missing-data threshold")

        cleaned = data.loc[:, retained_assets].copy()
        cleaned = cleaned.ffill().bfill()

        if cleaned.isna().any().any():
            remaining_missing_assets = cleaned.columns[cleaned.isna().any()].tolist()
            raise ValueError(
                f"missing values remain after cleaning for assets: {remaining_missing_assets}"
            )

        missing_after = int(cleaned.isna().sum().sum())
        dropped_asset_missing_percentages = {
            asset: float(missing_percentages[asset]) for asset in dropped_assets.index
        }

        if dropped_asset_missing_percentages:
            dropped_asset_report = ", ".join(
                f"{asset} ({percentage:.2%})"
                for asset, percentage in dropped_asset_missing_percentages.items()
            )
            logger.warning(
                "Dropped assets with more than 5%% missing observations: %s",
                dropped_asset_report,
            )

        summary = MissingDataSummary(
            total_assets_requested=int(data.shape[1]),
            assets_retained=int(cleaned.shape[1]),
            assets_dropped=int(len(dropped_assets)),
            missing_before=missing_before,
            missing_after=missing_after,
            dropped_asset_names=tuple(dropped_assets.index.tolist()),
            dropped_asset_missing_percentages=dropped_asset_missing_percentages,
            cleaning_method="drop_over_5pct_then_forward_fill_back_fill",
        )
        return cleaned, summary

    @staticmethod
    def detect_outliers(
        data: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        if data.empty:
            raise ValueError("data must not be empty")

        if method == "iqr":
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            return (data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))
        if method == "zscore":
            z = np.abs(stats.zscore(data, nan_policy="omit"))
            return pd.DataFrame(z > threshold, index=data.index, columns=data.columns)
        raise ValueError(f"unknown outlier detection method: {method}")

    @staticmethod
    def calculate_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
        if prices.empty:
            raise ValueError("prices must not be empty")

        clean = prices.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if clean.empty:
            raise ValueError("prices has no valid rows after cleanup")

        if method == "log":
            returns = np.log(clean / clean.shift(1))
        elif method == "simple":
            returns = clean.pct_change()
        else:
            raise ValueError(f"unknown returns method: {method}")

        returns = returns.dropna(how="any")
        if returns.empty:
            raise ValueError("returns series is empty after calculation")
        return returns

    @staticmethod
    def compare_return_methods(
        simple_returns: pd.DataFrame,
        log_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        DataPreprocessor._validate_returns_frame(simple_returns, "simple_returns")
        DataPreprocessor._validate_returns_frame(log_returns, "log_returns")

        aligned_simple, aligned_log = simple_returns.align(log_returns, join="inner")
        if aligned_simple.empty:
            raise ValueError("simple_returns and log_returns have no overlapping observations")

        comparison = pd.DataFrame(index=aligned_simple.columns)
        comparison.index.name = "asset"
        comparison["mean_simple_return"] = aligned_simple.mean()
        comparison["mean_log_return"] = aligned_log.mean()
        comparison["mean_abs_difference"] = (aligned_simple - aligned_log).abs().mean()
        comparison["max_abs_difference"] = (aligned_simple - aligned_log).abs().max()
        comparison["return_correlation"] = aligned_simple.corrwith(aligned_log)
        return comparison

    @staticmethod
    def calculate_volatility(
        returns: pd.DataFrame,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.DataFrame:
        DataPreprocessor._validate_returns_frame(returns, "returns")
        _validate_periods_per_year(periods_per_year)

        daily_volatility = returns.std(ddof=1)
        annualized_volatility = daily_volatility * np.sqrt(periods_per_year)

        volatility_summary = pd.DataFrame(index=returns.columns)
        volatility_summary.index.name = "asset"
        volatility_summary["daily_volatility"] = daily_volatility
        volatility_summary["annualized_volatility"] = annualized_volatility
        volatility_summary["annualization_factor"] = periods_per_year
        return volatility_summary

    @staticmethod
    def calculate_rolling_volatility(
        returns: pd.DataFrame,
        windows: tuple[int, ...] = (30, 90),
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.DataFrame:
        DataPreprocessor._validate_returns_frame(returns, "returns")
        _validate_periods_per_year(periods_per_year)

        invalid_windows = [window for window in windows if window <= 1]
        if invalid_windows:
            raise ValueError(f"rolling windows must be greater than 1: {invalid_windows}")

        rolling_frames: list[pd.DataFrame] = []
        rolling_scale = np.sqrt(periods_per_year)

        for window in windows:
            rolling_volatility = returns.rolling(window=window).std(ddof=1) * rolling_scale
            rolling_volatility.columns = pd.MultiIndex.from_product(
                [[f"{window}d"], rolling_volatility.columns],
                names=["window", "asset"],
            )
            rolling_frames.append(rolling_volatility)

        return pd.concat(rolling_frames, axis=1).sort_index(axis=1)

    @staticmethod
    def build_returns_risk_outputs(
        prices: pd.DataFrame,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
        rolling_windows: tuple[int, ...] = (30, 90),
    ) -> ReturnsRiskOutputs:
        quality_processor = DataQualityProcessor()

        clean_prices_df, repair_report_df = quality_processor.repair_price_anomalies(
            prices,
            method="interpolate",
        )

        simple_returns_df = DataPreprocessor.calculate_returns(clean_prices_df, method="simple")
        log_returns_df = DataPreprocessor.calculate_returns(clean_prices_df, method="log")

        outlier_report_df = quality_processor.detect_return_outliers(
            log_returns_df,
            method="mad",
        )
        stabilized_returns_df, stabilization_report_df = quality_processor.stabilize_returns(
            log_returns_df,
            method="winsorize",
        )
        volatility_summary_df = DataPreprocessor.calculate_volatility(
            stabilized_returns_df,
            periods_per_year=periods_per_year,
        )
        rolling_volatility_df = DataPreprocessor.calculate_rolling_volatility(
            stabilized_returns_df,
            windows=rolling_windows,
            periods_per_year=periods_per_year,
        )
        return_comparison_df = DataPreprocessor.compare_return_methods(
            simple_returns_df,
            log_returns_df,
        )
        quality_report_df = quality_processor.generate_quality_report()

        return ReturnsRiskOutputs(
            simple_returns_df=simple_returns_df,
            returns_df=stabilized_returns_df.copy(),
            log_returns_df=log_returns_df,
            volatility_summary_df=volatility_summary_df,
            rolling_volatility_df=rolling_volatility_df,
            return_comparison_df=return_comparison_df,
            quality_report_df=quality_report_df,
            anomaly_report_df=quality_processor._last_price_anomaly_report_df.copy(),
            repair_report_df=repair_report_df,
            outlier_report_df=outlier_report_df,
            stabilization_report_df=stabilization_report_df,
        )

    @staticmethod
    def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            raise ValueError("prices must not be empty")
        base = prices.iloc[0]
        return prices / base

    @staticmethod
    def _validate_returns_frame(data: pd.DataFrame, name: str) -> None:
        if data.empty:
            raise ValueError(f"{name} must not be empty")
        if data.isna().any().any():
            raise ValueError(f"{name} must not contain missing values")


class DataValidator:
    """Validate data quality and statistical properties."""

    @staticmethod
    def check_completeness(data: pd.DataFrame, min_coverage: float = 0.95) -> bool:
        if data.empty:
            return False
        coverage = 1.0 - (data.isna().sum().sum() / data.size)
        return bool(coverage >= min_coverage)

    @staticmethod
    def check_stationarity(data: pd.Series, test: str = "adf") -> Tuple[float, bool]:
        if test != "adf":
            raise ValueError("only adf test is supported in Phase 1")

        from statsmodels.tsa.stattools import adfuller

        result = adfuller(data.dropna())
        p_value = float(result[1])
        return p_value, p_value < 0.05


def _validate_periods_per_year(periods_per_year: int) -> None:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
