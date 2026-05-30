"""Main entry point for the Phase 1 portfolio optimization pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtesting import RollingBacktester
from src.clustering import HierarchicalClusterer
from src.config import get_config
from src.covariance import SampleCovarianceEstimator
from src.data_pipeline import DataPreprocessor, YFinanceIngester
from src.logging_config import get_logger
from src.optimization import (
    EqualWeightAllocator,
    HRPAllocator,
    InverseVolatilityAllocator,
    MeanVarianceAllocator,
)

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    returns: pd.DataFrame
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    linkage_matrix: object
    strategy_results: dict[str, dict]


def run_phase1_pipeline(
    symbols: list[str],
    start_date: str,
    end_date: str,
    rebalance_frequency: str = "M",
    train_window: int = 252,
) -> PipelineResult:
    ingester = YFinanceIngester()
    prices = ingester.fetch(symbols, start_date, end_date)
    prices = DataPreprocessor.handle_missing_values(prices)
    returns = DataPreprocessor.calculate_returns(prices, method="simple")

    covariance_estimator = SampleCovarianceEstimator().fit(returns)
    covariance = pd.DataFrame(covariance_estimator.get_covariance(), index=returns.columns, columns=returns.columns)
    correlation = returns.corr()

    clusterer = HierarchicalClusterer(linkage_method="single").fit(returns)

    strategies = {
        "Equal Weight": EqualWeightAllocator(),
        "Mean Variance": MeanVarianceAllocator(),
        "Inverse Volatility": InverseVolatilityAllocator(),
        "HRP": HRPAllocator(),
    }

    strategy_results: dict[str, dict] = {}
    for name, allocator in strategies.items():
        logger.info("Running strategy: %s", name)
        bt = RollingBacktester(
            allocator=allocator,
            train_window=train_window,
            rebalance_frequency=rebalance_frequency,
        )
        strategy_results[name] = bt.run(returns)

    return PipelineResult(
        returns=returns,
        covariance=covariance,
        correlation=correlation,
        linkage_matrix=clusterer.linkage_matrix,
        strategy_results=strategy_results,
    )


def main() -> None:
    config = get_config()

    symbols = config.get("data.symbols", ["SPY", "QQQ", "TLT", "GLD", "IEF"])
    start_date = config.get("backtesting.start_date", "2018-01-01")
    end_date = config.get("backtesting.end_date", "2025-01-01")
    rebalance_frequency = config.get("backtesting.rebalance_frequency", "M")
    train_window = int(config.get("portfolio.lookback_window", 252))

    logger.info("Starting Phase 1 pipeline")
    result = run_phase1_pipeline(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        rebalance_frequency=rebalance_frequency,
        train_window=train_window,
    )

    logger.info("Returns shape: %s", result.returns.shape)
    logger.info("Covariance shape: %s", result.covariance.shape)

    for name, bt_result in result.strategy_results.items():
        perf = bt_result["performance_metrics"]
        logger.info(
            "%s | Sharpe %.3f | Sortino %.3f | CAGR %.3f | Vol %.3f | MaxDD %.3f",
            name,
            perf["sharpe"],
            perf["sortino"],
            perf["cagr"],
            perf["volatility"],
            perf["max_drawdown"],
        )

    logger.info("Phase 1 pipeline completed")


if __name__ == "__main__":
    main()
