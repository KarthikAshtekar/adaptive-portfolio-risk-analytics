# Adaptive Portfolio Allocation and Risk Analytics under Dynamic Correlation & Sentiment Regimes

Institutional-grade quantitative portfolio optimization and risk analytics platform built for sophisticated portfolio management, research, and backtesting.

## 📋 Overview

This platform provides:

- **Adaptive Portfolio Allocation**: Multiple allocation methods (Equal Weight, Mean-Variance, Hierarchical Risk Parity, Hierarchical Equal Risk Contribution)
- **Robust Covariance Estimation**: Ledoit-Wolf shrinkage, Gerber covariance, rolling window estimation
- **Dynamic Clustering**: Ward linkage, dendrogram analysis, hierarchical portfolio construction
- **Regime Detection**: Markov-switching models, volatility targeting, dynamic risk scaling
- **NLP Sentiment Analysis**: RBI communications, earnings call transcripts, macro sentiment extraction
- **Backtesting Framework**: Rolling-window validation, Combinatorial Purged Cross-Validation (CPCV)
- **Risk Analytics**: VaR, CVaR, Sharpe ratio, maximum drawdown, stress testing
- **Interactive Dashboard**: Streamlit-based visualization and analytics interface

## 🏗️ Architecture

```
adaptive-portfolio-risk-analytics/
├── src/
│   ├── data_pipeline/          # Data ingestion and preprocessing
│   ├── covariance/             # Covariance estimation methods
│   ├── clustering/             # Hierarchical clustering algorithms
│   ├── regime_detection/       # Regime detection and volatility models
│   ├── nlp/                    # NLP and sentiment analysis
│   ├── optimization/           # Portfolio allocation methods
│   ├── backtesting/            # Backtesting and validation frameworks
│   ├── analytics/              # Risk metrics and performance analysis
│   ├── dashboard/              # Streamlit dashboard
│   └── utils/                  # Common utilities and helpers
├── data/                       # Raw, processed, and interim data
├── notebooks/                  # Exploratory analysis notebooks
├── tests/                      # Unit and integration tests
├── config/                     # Configuration management
├── outputs/                    # Weights, reports, metrics, figures
└── docs/                       # Architecture, methodology, research notes
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adaptive-portfolio-risk-analytics.git
cd adaptive-portfolio-risk-analytics
```

2. Create and activate Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

1. Create a `.env` file in the project root:
```
DATA_PATH=./data
RESULTS_PATH=./outputs
LOG_LEVEL=INFO
```

2. Configure allocation parameters in `config/portfolio_config.yaml`

### Running the Dashboard

```bash
streamlit run src/dashboard/app.py
```

### Running Tests

```bash
pytest tests/ -v --cov=src
```

## 📦 Core Components

### Data Pipeline
- Market data ingestion from multiple sources (yfinance, Alpha Vantage)
- Preprocessing and feature engineering
- Data validation and quality checks

### Covariance Estimation
- **Ledoit-Wolf Shrinkage**: Regularized covariance with optimal shrinkage intensity
- **Gerber Covariance**: Robust estimation using rank and sign correlation
- **Rolling Window**: Time-series covariance updates

### Portfolio Optimization
- **Equal Weight**: Naive equal-weighted allocation
- **Mean-Variance**: Markowitz efficient frontier optimization
- **HRP**: Hierarchical Risk Parity with optimal tree structure
- **HERC**: Hierarchical Equal Risk Contribution
- **Dynamic Allocation**: Regime-aware adaptive allocation

### Regime Detection
- Markov-switching autoregression (MSAR) models
- Volatility regimes and bull/bear classification
- Macro-driven regime transitions

### NLP Sentiment Analysis
- RBI monetary policy sentiment extraction
- Earnings call transcript analysis
- Uncertainty quantification
- Macro sentiment aggregation

### Backtesting
- Rolling-window in-sample validation
- Combinatorial Purged Cross-Validation (CPCV)
- Transaction cost modeling
- Performance metrics and stress testing

## 📊 Key Metrics

- **Risk Metrics**: VaR, CVaR, Volatility, Maximum Drawdown, Calmar Ratio
- **Performance Metrics**: Sharpe Ratio, Sortino Ratio, Information Ratio, CAGR
- **Portfolio Metrics**: Concentration, Turnover, Diversification Ratio

## 🔧 Technologies

- **Python 3.10+**: Core language
- **NumPy & SciPy**: Numerical computing
- **Pandas**: Data manipulation
- **scikit-learn**: Machine learning
- **Statsmodels**: Statistical modeling
- **scikit-portfolio & Riskfolio-lib**: Portfolio optimization
- **Transformers**: NLP models
- **Streamlit**: Dashboard framework
- **Plotly**: Interactive visualizations

## 📚 Documentation

- [Architecture Overview](docs/architecture/README.md)
- [Methodology](docs/methodology/README.md)
- [Research Notes](docs/research_notes/README.md)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please follow PEP8 standards and include unit tests.

## 📧 Contact

For questions or suggestions, contact: team@example.com

---

**Built for institutional-grade quantitative finance research and portfolio management**
