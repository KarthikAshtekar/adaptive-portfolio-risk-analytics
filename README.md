# Adaptive Portfolio Risk Analytics Platform

Advanced Framework for Adaptive Portfolio Allocation and Risk Analytics: Integrating Hierarchical Risk Parity, Regime Switching, and Macro-Sentiment Dynamics

## 🎯 Project Overview

This repository contains an institutional-grade quantitative portfolio optimization and risk analytics platform designed for sophisticated portfolio management, research, and backtesting. The platform integrates multiple cutting-edge techniques including hierarchical clustering, regime detection, NLP sentiment analysis, and advanced risk metrics.

### Key Features

- **Adaptive Portfolio Allocation**: Multiple allocation methods (Equal Weight, Mean-Variance, HRP, HERC, Dynamic)
- **Robust Covariance Estimation**: Ledoit-Wolf, Gerber covariance, rolling window approaches
- **Dynamic Regime Detection**: Markov-switching models with volatility targeting
- **Macro Sentiment Analysis**: RBI communications, earnings calls, uncertainty quantification
- **Comprehensive Backtesting**: Rolling-window validation, CPCV framework
- **Advanced Risk Analytics**: VaR, CVaR, Sharpe ratio, maximum drawdown, stress testing
- **Interactive Dashboard**: Streamlit-based visualization and analytics interface

## 📁 Repository Structure

```
.
├── adaptive-portfolio-risk-analytics/    # Main project directory
│   ├── src/                              # Source code (9 modules)
│   │   ├── data_pipeline/                # Data ingestion and preprocessing
│   │   ├── covariance/                   # Covariance estimation methods
│   │   ├── clustering/                   # Hierarchical clustering (HRP, HERC)
│   │   ├── regime_detection/             # Regime detection & volatility models
│   │   ├── nlp/                          # NLP and sentiment analysis
│   │   ├── optimization/                 # Portfolio allocation methods
│   │   ├── backtesting/                  # Backtesting and validation frameworks
│   │   ├── analytics/                    # Risk metrics and performance analysis
│   │   ├── dashboard/                    # Streamlit dashboard
│   │   └── config.py, logging_config.py, types.py, utils.py
│   ├── tests/                            # Unit and integration tests
│   ├── config/                           # YAML configuration files
│   ├── data/                             # Data directories (raw, processed, interim)
│   ├── notebooks/                        # Exploratory analysis notebooks
│   ├── outputs/                          # Results, weights, reports
│   ├── docs/                             # Architecture, methodology, research
│   ├── requirements.txt                  # Python dependencies (50+ packages)
│   ├── setup.py                          # Package setup configuration
│   ├── README.md                         # Detailed project documentation
│   ├── CONTRIBUTING.md                   # Contribution guidelines
│   └── Makefile                          # Automation commands
│
├── Proposed_Files/                       # Placeholder for additional files
├── .github/                              # GitHub Actions, issue templates, PR templates
├── pyproject.toml                        # Project metadata
└── TODO.md                               # Project tasks and roadmap

```

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

```bash
# 1. Navigate to the project directory
cd adaptive-portfolio-risk-analytics

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python verify_setup.py
```

### Running the Platform

```bash
# View the interactive dashboard
streamlit run src/dashboard/app.py

# Or run the main application
python main.py

# Run tests
pytest tests/ -v --cov=src

# Check code quality
make lint

# Format code
make format
```

## 📚 Documentation

Comprehensive documentation is available in the following files:

| Document | Purpose |
|----------|---------|
| [README.md](adaptive-portfolio-risk-analytics/README.md) | Detailed project documentation |
| [GETTING_STARTED.md](adaptive-portfolio-risk-analytics/GETTING_STARTED.md) | Setup and quick reference guide |
| [CONTRIBUTING.md](adaptive-portfolio-risk-analytics/CONTRIBUTING.md) | Contribution guidelines and workflow |
| [GITHUB_READINESS.md](adaptive-portfolio-risk-analytics/GITHUB_READINESS.md) | Team onboarding guide |
| [docs/architecture/ARCHITECTURE.md](adaptive-portfolio-risk-analytics/docs/architecture/ARCHITECTURE.md) | Design patterns and architecture |
| [docs/methodology/METHODOLOGY.md](adaptive-portfolio-risk-analytics/docs/methodology/METHODOLOGY.md) | Algorithm specifications and theory |
| [docs/ROADMAP.md](adaptive-portfolio-risk-analytics/docs/ROADMAP.md) | Development roadmap (8 phases) |

## 🏗️ Architecture

The platform is built using several architectural patterns:

- **Strategy Pattern**: Multiple covariance estimators and portfolio optimizers
- **Factory Pattern**: Dynamic optimizer and estimator creation
- **Singleton Pattern**: Configuration management and logging
- **Template Method**: Base classes for extensibility
- **Configuration-Driven Design**: YAML-based configuration with environment overrides

## 🔧 Tech Stack

- **Core**: Python 3.10+, NumPy, SciPy, Pandas
- **ML/Statistics**: scikit-learn, statsmodels, arch
- **Portfolio Optimization**: scikit-portfolio, riskfolio-lib, cvxpy
- **NLP**: Transformers, torch, sentencepiece
- **Visualization**: Plotly, Matplotlib, Seaborn, Streamlit
- **Testing**: pytest, pytest-cov, black, flake8, mypy
- **CI/CD**: GitHub Actions

## 📊 Key Components

### 1. Data Pipeline
- Multi-source data ingestion (yfinance, Alpha Vantage)
- Data preprocessing and feature engineering
- Quality validation and error handling

### 2. Covariance Estimation
- Ledoit-Wolf shrinkage with optimal intensity
- Gerber robust covariance using rank correlations
- Rolling window estimation for dynamic analysis

### 3. Portfolio Optimization
- Equal-weight allocation (baseline)
- Mean-Variance optimization (Markowitz)
- Hierarchical Risk Parity (López de Prado)
- Hierarchical Equal Risk Contribution
- Regime-aware dynamic allocation

### 4. Regime Detection
- Markov-switching autoregression (MSAR)
- Volatility targeting and scaling
- Bull/bear market classification

### 5. NLP Sentiment Analysis
- RBI policy communication sentiment
- Earnings call transcript analysis
- Uncertainty quantification
- Macro sentiment aggregation

### 6. Backtesting
- Rolling-window validation
- Combinatorial Purged Cross-Validation (CPCV)
- Transaction cost modeling
- Comprehensive performance reporting

### 7. Risk Analytics
- Value-at-Risk (VaR) estimation
- Conditional Value-at-Risk (CVaR)
- Maximum drawdown and volatility
- Stress testing and scenario analysis
- Sharpe, Sortino, and Calmar ratios

## 🧪 Testing

The project includes a comprehensive test suite:

```bash
# Run all tests
make test

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_optimization.py -v

# Run linting and type checking
make lint

# Format code automatically
make format
```

**Testing Standards**:
- Minimum 80% code coverage
- Unit tests for all functions
- Integration tests for module interactions
- Edge case testing for robustness

## 🤝 Contributing

We welcome contributions from the community! Please read [CONTRIBUTING.md](adaptive-portfolio-risk-analytics/CONTRIBUTING.md) for:

- Branching strategy (git flow)
- Code style guidelines (PEP8, type hints, docstrings)
- Testing requirements
- Pull request process
- Commit message conventions

### Quick Contribution Steps

```bash
# 1. Create feature branch
git checkout -b feature/your-feature-name

# 2. Make changes and add tests
# 3. Format and lint
make format && make lint

# 4. Run tests
make test

# 5. Commit with clear message
git commit -m "[module] Brief description"

# 6. Push and create pull request
git push origin feature/your-feature-name
```

## 📈 Development Status

**Current Version**: 0.1.0

**Implementation Progress**:
- ✅ Core infrastructure (config, logging, types, utils)
- ✅ Framework for all 9 modules with abstract base classes
- ✅ Test framework with pytest and fixtures
- ✅ Complete documentation and architecture guides
- 🔨 Algorithm implementations (see ROADMAP for phase tracking)
- 🔨 Dashboard page implementations
- 🔨 NLP and sentiment analysis models

See [docs/ROADMAP.md](adaptive-portfolio-risk-analytics/docs/ROADMAP.md) for the 8-phase development plan and success metrics.

## 📖 Research References

This platform is grounded in academic and industry research:

- López de Prado et al. (2013) - Hierarchical Risk Parity
- Ledoit & Wolf (2004) - Covariance Matrix Shrinkage
- Markowitz (1952) - Modern Portfolio Theory
- Hamilton (1989) - Regime-Switching Models
- Baker, Bloom & Davis (2016) - Macro-uncertainty
- And 15+ additional peer-reviewed references

Full citation list: [docs/architecture/REFERENCES.md](adaptive-portfolio-risk-analytics/docs/architecture/REFERENCES.md)

## 📝 License

This project is licensed under the MIT License - see [LICENSE](adaptive-portfolio-risk-analytics/LICENSE) for details.

## 👥 Team

**Developed by**: Quantitative Finance Team

**Contributors**: See [CODEOWNERS](adaptive-portfolio-risk-analytics/CODEOWNERS) for module ownership

## 📞 Support & Questions

- **Setup Issues**: See [GETTING_STARTED.md](adaptive-portfolio-risk-analytics/GETTING_STARTED.md)
- **Development Help**: See [CONTRIBUTING.md](adaptive-portfolio-risk-analytics/CONTRIBUTING.md)
- **Architecture Questions**: See [docs/architecture/ARCHITECTURE.md](adaptive-portfolio-risk-analytics/docs/architecture/ARCHITECTURE.md)
- **Algorithm Details**: See [docs/methodology/METHODOLOGY.md](adaptive-portfolio-risk-analytics/docs/methodology/METHODOLOGY.md)

## 🔗 Related Resources

- [Project Status](adaptive-portfolio-risk-analytics/PROJECT_STATUS.md)
- [Implementation Summary](adaptive-portfolio-risk-analytics/IMPLEMENTATION_SUMMARY.md)
- [GitHub Readiness Assessment](adaptive-portfolio-risk-analytics/GITHUB_READINESS.md)
- [File Tree Documentation](adaptive-portfolio-risk-analytics/FILE_TREE.md)

---

**Repository**: https://github.com/KarthikAshtekar/adaptive-portfolio-risk-analytics  
**Status**: 🟢 Ready for Team Collaboration  
**Last Updated**: May 2026
