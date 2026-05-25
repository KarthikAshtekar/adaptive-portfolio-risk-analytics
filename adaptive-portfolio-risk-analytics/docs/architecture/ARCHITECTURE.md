"""Architecture overview and design patterns."""

# Platform Architecture

## High-Level Components

### 1. Data Pipeline (`src/data_pipeline/`)
- **ingest.py**: Multi-source data fetching
  - YFinance integration
  - Alpha Vantage integration
  - Custom data sources
- **preprocess.py**: Data cleaning and validation
  - Missing value handling
  - Outlier detection
  - Normalization
- **feature_engineering.py**: Feature generation
  - Technical indicators
  - Volatility measures
  - Macro features

### 2. Covariance Estimation (`src/covariance/`)
- **Ledoit-Wolf Shrinkage**: Regularized covariance with optimal shrinkage
- **Gerber Covariance**: Robust rank-sign correlation estimation
- **Rolling Covariance**: Time-series covariance updates
- **Custom Estimators**: Extensible framework for custom methods

### 3. Hierarchical Clustering (`src/clustering/`)
- **Distance Metrics**: Correlation, Euclidean, KL divergence
- **HRP Algorithm**: Hierarchical Risk Parity construction
- **HERC Algorithm**: Hierarchical Equal Risk Contribution
- **Dendrogram Analysis**: Visualization and optimization

### 4. Regime Detection (`src/regime_detection/`)
- **Markov-Switching Models**: MSAR for bull/bear and volatility regimes
- **Volatility Targeting**: Dynamic risk adjustment
- **Defensive Risk Scaling**: Crisis mode reduction

### 5. NLP/Sentiment (`src/nlp/`)
- **RBI Sentiment**: Monetary policy sentiment extraction
- **Earnings Calls**: Management sentiment and guidance
- **Uncertainty Scoring**: Macro uncertainty quantification
- **Sentiment Pipeline**: Aggregated macro intelligence

### 6. Portfolio Optimization (`src/optimization/`)
- **Equal Weight**: Naive 1/N baseline
- **Mean-Variance**: Markowitz efficient frontier
- **Inverse Volatility**: Risk parity
- **Dynamic Allocation**: Regime-aware optimization

### 7. Backtesting (`src/backtesting/`)
- **Rolling Backtest**: Walk-forward validation
- **CPCV**: Combinatorial purged cross-validation
- **Transaction Costs**: Impact and slippage modeling
- **Portfolio Simulation**: Full trading simulation

### 8. Analytics (`src/analytics/`)
- **Risk Metrics**: VaR, CVaR, Maximum Drawdown
- **Performance Metrics**: Sharpe, Sortino, Calmar
- **Stress Testing**: Historical scenarios, reverse stress tests
- **Attribution**: Risk and performance attribution

### 9. Dashboard (`src/dashboard/`)
- **Streamlit Application**: Interactive web interface
- **Visualizations**: Plotly charts and interactive plots
- **Real-time Monitoring**: Live portfolio metrics

## Design Patterns

### 1. Strategy Pattern
- Multiple optimization methods (Equal Weight, Mean-Variance, HRP, HERC)
- Multiple regime detection strategies
- Pluggable covariance estimators

### 2. Factory Pattern
- Optimizer factory for creating allocation methods
- Data ingestion factory for multiple sources

### 3. Pipeline Pattern
- Data ingestion → Preprocessing → Feature Engineering → Analysis
- Sequential processing with clear interfaces

### 4. Singleton Pattern
- Global configuration manager
- Global logger instance

### 5. Template Method Pattern
- Base classes for optimizers, regime detectors, sentiment analyzers
- Concrete implementations follow template

## Key Abstractions

### PortfolioOptimizer
```python
class PortfolioOptimizer(ABC):
    @abstractmethod
    def optimize(returns, cov_matrix) -> np.ndarray:
        pass
```

### RegimeDetector
```python
class RegimeDetector(ABC):
    @abstractmethod
    def detect(returns) -> np.ndarray:
        pass
```

### CovarianceEstimator
```python
class CovarianceEstimator(ABC):
    @abstractmethod
    def estimate(returns) -> np.ndarray:
        pass
```

## Configuration Management

- YAML-based configuration (`config/portfolio_config.yaml`)
- Environment variable support
- Runtime configuration overrides
- Hierarchical dot-notation access

## Logging Strategy

- Centralized logging configuration
- Structured logging with loguru
- File rotation and retention policies
- Multiple output targets (console, file)

## Testing Strategy

- Unit tests for each module
- Pytest fixtures for common test data
- Coverage reporting
- Continuous integration ready

## Future Extensibility

- Plugin system for custom optimizers
- Multi-asset class support
- Real-time data streaming
- Distributed backtesting
- Machine learning model integration
