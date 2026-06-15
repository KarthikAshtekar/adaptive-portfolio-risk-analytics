"""
Integration Example: FinBERT Sentiment Analysis + Portfolio Volatility Targeting

This notebook demonstrates how to:
1. Load and analyze financial sentiment documents with FinBERT
2. Generate lagged sentiment signals
3. Map sentiment to adaptive volatility targets
4. Integrate with portfolio backtesting system
5. Compare sentiment-aware vs. baseline portfolio performance

Usage:
    python sentiment_integration_example.py

or in Jupyter:
    %run sentiment_integration_example.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import sentiment module
from finbert_sentiment import (
    FinBERTSentimentAnalyzer,
    SentimentSignalConfig,
    SentimentPipeline,
    UncertaintyScorer,
    load_sentiment_documents,
    score_documents_batch,
    aggregate_daily_sentiment,
    build_lagged_sentiment_signal,
    map_sentiment_to_volatility,
    analyze_sentiment,
)


def example_1_basic_sentiment_analysis():
    """
    Example 1: Basic sentiment analysis of financial documents.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Sentiment Analysis")
    print("="*80)
    
    # Create sample documents
    documents = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"],
        "source": ["news", "rbi", "news", "earnings"],
        "text": [
            "Bank earnings show strong growth momentum",
            "RBI warns about inflation pressure",
            "IT sector demand remains stable",
            "Oil demand outlook improves",
        ],
    })
    
    print("\nSample documents:")
    print(documents)
    
    # Mock backend for demonstration (no model download required)
    class MockBackend:
        def __call__(self, texts, **kwargs):
            results = []
            for text in texts:
                if any(w in text.lower() for w in ["growth", "demand", "improves", "momentum"]):
                    results.append({"label": "positive", "score": 0.92})
                elif any(w in text.lower() for w in ["pressure", "warns", "downgrade", "risk"]):
                    results.append({"label": "negative", "score": 0.88})
                else:
                    results.append({"label": "neutral", "score": 0.80})
            return results
    
    # Analyze sentiment
    analyzer = FinBERTSentimentAnalyzer(backend=MockBackend())
    scored = score_documents_batch(documents, analyzer=analyzer)
    
    print("\nScored documents:")
    print(scored[["date", "text", "label", "sentiment_score", "confidence"]].to_string())
    
    return scored


def example_2_daily_aggregation():
    """
    Example 2: Aggregate document-level sentiment to daily signals.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Daily Sentiment Aggregation")
    print("="*80)
    
    # Get scored documents from previous example
    scored = example_1_basic_sentiment_analysis()
    
    # Aggregate to daily level
    daily = aggregate_daily_sentiment(scored)
    
    print("\nDaily sentiment summary:")
    print(daily)
    
    print("\nDaily statistics:")
    print(f"  Mean sentiment: {daily['sentiment_mean'].mean():.3f}")
    print(f"  Sentiment range: [{daily['sentiment_mean'].min():.3f}, "
          f"{daily['sentiment_mean'].max():.3f}]")
    print(f"  Positive days (>0.2): {(daily['sentiment_mean'] > 0.2).sum()}")
    print(f"  Negative days (<-0.2): {(daily['sentiment_mean'] < -0.2).sum()}")
    
    return daily


def example_3_lagged_signal_construction():
    """
    Example 3: Build lagged sentiment signal (no look-ahead bias).
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Lagged Sentiment Signal Construction")
    print("="*80)
    
    # Create sample daily sentiment
    daily = pd.DataFrame(
        {
            "sentiment_mean": [0.7, -0.5, 0.2, 0.1, -0.3],
        },
        index=pd.date_range("2024-01-02", periods=5, freq="B"),
    )
    
    print("\nDaily sentiment (unlagged):")
    print(daily)
    
    # Build lagged signal with different lag periods
    for lag in [0, 1, 2]:
        config = SentimentSignalConfig(lag_days=lag, smoothing_window=1)
        signal = build_lagged_sentiment_signal(daily, config=config)
        print(f"\nLagged signal (lag={lag} days):")
        print(signal)
        
        if lag > 0:
            print(f"  → First {lag} value(s) are NaN (no prior data) - NO LOOK-AHEAD")


def example_4_sentiment_to_volatility():
    """
    Example 4: Map sentiment scores to adaptive volatility targets.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Sentiment to Volatility Mapping")
    print("="*80)
    
    # Create sample sentiment signal
    sentiment_values = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
    sentiment_signal = pd.Series(
        sentiment_values,
        index=pd.date_range("2024-01-02", periods=5, freq="B"),
        name="sentiment_score"
    )
    
    print("\nSentiment signal:")
    print(sentiment_signal)
    
    # Map to volatility
    config = SentimentSignalConfig()
    target_vol = map_sentiment_to_volatility(sentiment_signal, config=config)
    
    # Display mapping
    print("\nSentiment to Volatility Mapping:")
    print("-" * 60)
    for sent, vol in zip(sentiment_values, target_vol.values):
        print(f"  Sentiment: {sent:+.2f} → Target Vol: {vol:.2%}")
    
    # Show configuration
    print("\nDefault Configuration:")
    print(f"  Very positive (>{config.very_positive_threshold:+.2f}): "
          f"{config.very_positive_target_vol:.2%} volatility")
    print(f"  Positive (>{config.positive_threshold:+.2f}): "
          f"{config.positive_target_vol:.2%} volatility")
    print(f"  Neutral: "
          f"{config.neutral_target_vol:.2%} volatility")
    print(f"  Negative (<{config.negative_threshold:+.2f}): "
          f"{config.negative_target_vol:.2%} volatility")
    print(f"  Very negative (<{config.very_negative_threshold:+.2f}): "
          f"{config.very_negative_target_vol:.2%} volatility")


def example_5_end_to_end_pipeline():
    """
    Example 5: End-to-end sentiment analysis pipeline.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: End-to-End Sentiment Pipeline")
    print("="*80)
    
    # Create sample documents
    documents = pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=20, freq="B"),
        "source": np.random.choice(["news", "rbi", "earnings"], 20),
        "text": np.random.choice([
            "Growth momentum continues with strong earnings",
            "Inflation pressure concerns emerge",
            "Stable market conditions ahead",
            "Economic recovery shows promise",
            "Risk of downturn increases",
            "Demand outlook remains robust",
            "Uncertainty clouds the market",
            "Policy support for growth continues",
        ], 20),
    })
    
    # Mock backend
    class MockBackend:
        def __call__(self, texts, **kwargs):
            results = []
            for text in texts:
                if any(w in text.lower() for w in ["growth", "strong", "continues", "robust", "recovery"]):
                    results.append({"label": "positive", "score": 0.90})
                elif any(w in text.lower() for w in ["pressure", "risk", "uncertainty", "downturn"]):
                    results.append({"label": "negative", "score": 0.85})
                else:
                    results.append({"label": "neutral", "score": 0.75})
            return results
    
    # Run pipeline
    analyzer = FinBERTSentimentAnalyzer(backend=MockBackend())
    uncertainty_scorer = UncertaintyScorer()
    pipeline = SentimentPipeline(
        analyzer=analyzer,
        uncertainty_scorer=uncertainty_scorer,
    )
    
    result = pipeline.run(documents)
    
    print("\nPipeline Results:")
    print(f"  Documents analyzed: {result.summary_stats['total_documents']}")
    print(f"  Date range: {result.summary_stats['date_range_days']} days")
    print(f"  Mean sentiment: {result.summary_stats['mean_sentiment']:.3f}")
    print(f"  Positive documents: {result.summary_stats['positive_count']}")
    print(f"  Negative documents: {result.summary_stats['negative_count']}")
    print(f"  Mean target volatility: {result.summary_stats['mean_target_volatility']:.2%}")
    
    print("\nDaily Sentiment Aggregates:")
    print(result.daily_sentiment[["sentiment_mean", "document_count"]].head(10))
    
    print("\nTarget Volatility Series (first 10 days):")
    print(result.target_volatility.head(10))
    
    return result


def example_6_portfolio_integration():
    """
    Example 6: Integrate sentiment signal with portfolio volatility targeting.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Portfolio Integration - Sentiment-Aware Volatility Targeting")
    print("="*80)
    
    # Generate synthetic portfolio returns
    np.random.seed(42)
    dates = pd.date_range("2024-01-02", periods=60, freq="B")
    portfolio_returns = pd.Series(
        np.random.normal(0.001, 0.012, 60),
        index=dates,
        name="portfolio_returns"
    )
    
    # Generate sentiment signal
    sentiment_signal = pd.Series(
        np.sin(np.linspace(0, 4*np.pi, 60)) * 0.4,  # Cyclical sentiment
        index=dates,
        name="sentiment_signal"
    )
    
    print("\nPortfolio returns (first 10 days):")
    print(portfolio_returns.head(10))
    
    print("\nSentiment signal (first 10 days):")
    print(sentiment_signal.head(10))
    
    # Map sentiment to volatility targets
    config = SentimentSignalConfig()
    target_vol_from_sentiment = map_sentiment_to_volatility(sentiment_signal, config=config)
    
    # Compute realized volatility
    realized_vol = portfolio_returns.rolling(20).std() * np.sqrt(252)
    
    # Compute exposure adjustment
    exposure = (target_vol_from_sentiment / realized_vol.fillna(0.1)).clip(0.2, 1.0)
    
    # Apply exposure adjustment
    adjusted_returns = exposure * portfolio_returns
    
    print("\nPortfolio adjustments (first 20 days):")
    summary_df = pd.DataFrame({
        "Sentiment": sentiment_signal.head(20),
        "Target Vol": target_vol_from_sentiment.head(20),
        "Realized Vol": realized_vol.head(20),
        "Exposure": exposure.head(20),
    })
    print(summary_df)
    
    # Performance comparison
    baseline_cumulative = (1 + portfolio_returns).cumprod()
    adjusted_cumulative = (1 + adjusted_returns).cumprod()
    
    print("\nPerformance Comparison:")
    print(f"  Baseline final value: {baseline_cumulative.iloc[-1]:.4f}")
    print(f"  Adjusted final value: {adjusted_cumulative.iloc[-1]:.4f}")
    print(f"  Baseline volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
    print(f"  Adjusted volatility: {adjusted_returns.std() * np.sqrt(252):.2%}")
    
    # Plot comparison
    if __name__ == "__main__":
        plt.figure(figsize=(14, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(dates, sentiment_signal.values, label="Sentiment Signal", marker="o")
        plt.axhline(0, color="black", linestyle="--", alpha=0.3)
        plt.legend()
        plt.title("Sentiment Signal Over Time")
        plt.ylabel("Sentiment Score")
        
        plt.subplot(3, 1, 2)
        plt.plot(dates, target_vol_from_sentiment.values, label="Target Volatility", marker="s")
        plt.plot(dates, realized_vol.values, label="Realized Volatility", marker="^", alpha=0.7)
        plt.legend()
        plt.title("Volatility Targeting")
        plt.ylabel("Volatility (annualized)")
        
        plt.subplot(3, 1, 3)
        plt.plot(dates, baseline_cumulative.values, label="Baseline", marker="o")
        plt.plot(dates, adjusted_cumulative.values, label="Sentiment-Adjusted", marker="s")
        plt.legend()
        plt.title("Cumulative Returns: Baseline vs. Sentiment-Adjusted")
        plt.ylabel("Cumulative Return")
        plt.xlabel("Date")
        
        plt.tight_layout()
        plt.savefig("sentiment_portfolio_integration.png", dpi=100)
        print("\n✓ Chart saved as 'sentiment_portfolio_integration.png'")
        plt.show()


def example_7_configuration_customization():
    """
    Example 7: Customize sentiment analysis configuration.
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Custom Configuration Examples")
    print("="*80)
    
    # Configuration 1: Conservative (defensive)
    print("\nConfiguration 1: CONSERVATIVE")
    conservative = SentimentSignalConfig(
        lag_days=2,  # Longer lag for caution
        smoothing_window=10,  # Heavy smoothing
        very_negative_target_vol=0.02,  # Very low risk when negative
        very_positive_target_vol=0.10,  # Capped upside risk
    )
    print(f"  Lag: {conservative.lag_days} days")
    print(f"  Smoothing: {conservative.smoothing_window} days")
    print(f"  Volatility range: [{conservative.very_negative_target_vol:.2%}, "
          f"{conservative.very_positive_target_vol:.2%}]")
    
    # Configuration 2: Aggressive (growth)
    print("\nConfiguration 2: AGGRESSIVE")
    aggressive = SentimentSignalConfig(
        lag_days=0,  # Quick reaction
        smoothing_window=2,  # Minimal smoothing
        very_negative_target_vol=0.06,  # Still takes risks
        very_positive_target_vol=0.18,  # High growth target
    )
    print(f"  Lag: {aggressive.lag_days} days")
    print(f"  Smoothing: {aggressive.smoothing_window} days")
    print(f"  Volatility range: [{aggressive.very_negative_target_vol:.2%}, "
          f"{aggressive.very_positive_target_vol:.2%}]")
    
    # Configuration 3: Balanced
    print("\nConfiguration 3: BALANCED")
    balanced = SentimentSignalConfig()  # Defaults
    print(f"  Lag: {balanced.lag_days} days")
    print(f"  Smoothing: {balanced.smoothing_window} days")
    print(f"  Volatility range: [{balanced.very_negative_target_vol:.2%}, "
          f"{balanced.very_positive_target_vol:.2%}]")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("FinBERT SENTIMENT ANALYSIS - INTEGRATION EXAMPLES")
    print("="*80)
    
    try:
        # Run examples
        example_1_basic_sentiment_analysis()
        example_2_daily_aggregation()
        example_3_lagged_signal_construction()
        example_4_sentiment_to_volatility()
        example_5_end_to_end_pipeline()
        example_6_portfolio_integration()
        example_7_configuration_customization()
        
        print("\n" + "="*80)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        
        print("\n📚 Next Steps:")
        print("  1. Load your actual financial documents (CSV with date + text columns)")
        print("  2. Create FinBERTSentimentAnalyzer with real model or mock backend")
        print("  3. Run SentimentPipeline to get sentiment signals")
        print("  4. Integrate signals with portfolio backtesting system")
        print("  5. Compare sentiment-aware performance vs. baseline")
        print("  6. Fine-tune configuration based on your results")
        
    except Exception as e:
        logger.error(f"Error in examples: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
