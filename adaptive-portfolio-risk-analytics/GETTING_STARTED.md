"""
Getting Started Guide

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/adaptive-portfolio-risk-analytics.git
cd adaptive-portfolio-risk-analytics
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
make install
# or manually:
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.template .env
# Edit .env with your API keys and paths
```

5. **Verify installation**
```bash
pytest tests/ -v
```

## Running the Platform

### Dashboard
```bash
make run-dashboard
# or directly:
streamlit run src/dashboard/app.py
```

### Main Pipeline
```bash
python main.py
```

### Running Tests
```bash
make test          # Run all tests
make test-cov      # With coverage report
```

### Code Quality
```bash
make lint          # Check code quality
make format        # Auto-format code
```

## Project Structure Quick Reference

```
src/
├── data_pipeline/      # Data ingestion and preprocessing
├── covariance/         # Covariance estimation methods
├── clustering/         # HRP/HERC clustering
├── regime_detection/   # Market regime detection
├── nlp/               # Sentiment analysis
├── optimization/      # Portfolio allocation
├── backtesting/       # Backtesting framework
├── analytics/         # Risk and performance metrics
├── dashboard/         # Streamlit dashboard
├── config.py          # Configuration management
├── logging_config.py  # Logging setup
├── types.py          # Type definitions
└── utils.py          # Utility functions

config/
└── portfolio_config.yaml  # Main configuration file

tests/
├── test_hrp.py
├── test_herc.py
├── test_covariance.py
├── test_regime_detection.py
└── test_optimization.py
```

## Key Classes and Functions

### Portfolio Optimization
```python
from src.optimization import (
    EqualWeightOptimizer,
    MeanVarianceOptimizer,
    HierarchicalRiskParity,
    HierarchicalEqualRiskContribution,
)

# Create optimizer
optimizer = HierarchicalRiskParity()
weights = optimizer.fit(returns).get_weights()
```

### Covariance Estimation
```python
from src.covariance import (
    LedoitWolfEstimator,
    GerberCovarianceEstimator,
    RollingCovarianceEstimator,
)

# Estimate covariance
estimator = LedoitWolfEstimator()
cov_matrix = estimator.estimate(returns)
```

### Backtesting
```python
from src.backtesting import RollingBacktest, CPCVValidator

# Rolling backtest
backtest = RollingBacktest(train_window=252, test_window=63)
results = backtest.run(returns)

# CPCV validation
cpcv = CPCVValidator(n_splits=5)
```

### Analytics
```python
from src.analytics import RiskAnalytics, PerformanceAnalytics

# Risk metrics
var = RiskAnalytics.value_at_risk(portfolio_returns)
cvar = RiskAnalytics.conditional_value_at_risk(portfolio_returns)
max_dd = RiskAnalytics.maximum_drawdown(portfolio_returns)

# Performance metrics
sharpe = PerformanceAnalytics.sharpe_ratio(portfolio_returns)
sortino = PerformanceAnalytics.sortino_ratio(portfolio_returns)
calmar = PerformanceAnalytics.calmar_ratio(portfolio_returns)
```

## Configuration

Edit `config/portfolio_config.yaml` to customize:
- Portfolio allocation parameters
- Covariance estimation method
- Clustering algorithm
- Regime detection settings
- Backtesting parameters
- Risk metrics

## Troubleshooting

### Import Errors
```bash
# Ensure src is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Missing Dependencies
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
```

### Test Failures
```bash
# Run specific test with verbose output
pytest tests/test_hrp.py::TestHierarchicalRiskParity -vv
```

## Contributing

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and add tests
3. Format code: `make format`
4. Lint code: `make lint`
5. Run tests: `make test`
6. Commit: `git commit -m "Add new feature"`
7. Push: `git push origin feature/new-feature`

## Next Steps

1. **Implement core algorithms** in each module
2. **Complete unit tests** for all functionality
3. **Add integration tests** for end-to-end workflows
4. **Enhance dashboard** with real-time data
5. **Add documentation** for complex algorithms
6. **Performance optimization** for large portfolios

## Resources

- [Documentation](docs/)
- [Architecture Overview](docs/architecture/ARCHITECTURE.md)
- [Research References](docs/architecture/REFERENCES.md)
- [Research Notes](docs/research_notes/)
"""
