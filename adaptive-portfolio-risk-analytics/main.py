"""Main entry point for portfolio optimization pipeline."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.logging_config import get_logger
from src.config import get_config

logger = get_logger(__name__)


def main():
    """
    Main entry point for the adaptive portfolio optimization platform.

    TODO: Implement full pipeline orchestration
    """
    logger.info("Starting Adaptive Portfolio Optimization Platform")

    # Load configuration
    config = get_config()
    logger.info(f"Configuration loaded: {config.to_dict()}")

    # TODO: Implement pipeline
    # 1. Load data
    # 2. Preprocess data
    # 3. Calculate covariance
    # 4. Detect regimes
    # 5. Optimize portfolio
    # 6. Backtest
    # 7. Generate reports

    logger.info("Pipeline execution complete")


if __name__ == "__main__":
    main()
