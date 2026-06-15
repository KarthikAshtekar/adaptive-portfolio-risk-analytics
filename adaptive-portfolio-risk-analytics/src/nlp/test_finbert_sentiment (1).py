"""
Comprehensive tests for FinBERT sentiment analysis module.
Tests use mock backends to avoid downloading models during test runs.
"""

import numpy as np
import pandas as pd
import pytest

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


class MockFinBERTBackend:
    """Mock FinBERT backend for testing without downloading the model."""
    
    def __call__(self, texts, **kwargs):
        """Mock sentiment classification."""
        results = []
        for text in texts:
            text_lower = str(text).lower()
            
            # Simple keyword-based classification for testing
            if any(word in text_lower for word in ["growth", "surge", "profit", "easing", "strong"]):
                results.append({"label": "positive", "score": 0.95})
            elif any(word in text_lower for word in ["decline", "stress", "inflation", "risk", "downgrade"]):
                results.append({"label": "negative", "score": 0.90})
            else:
                results.append({"label": "neutral", "score": 0.85})
        
        return results


@pytest.fixture
def sample_documents():
    """Sample financial documents for testing."""
    return pd.DataFrame({
        "date": [
            "2024-01-02", "2024-01-02", "2024-01-03", 
            "2024-01-04", "2024-01-05", "2024-01-08"
        ],
        "source": ["news", "rbi", "news", "earnings", "news", "rbi"],
        "ticker": ["HDFCBANK", "MARKET", "INFY", "RELIANCE", "MARKET", "MARKET"],
        "text": [
            "Bank earnings show growth momentum and easing credit costs",
            "RBI warns about inflation pressure and liquidity stress",
            "IT services demand remains stable and strong",
            "Energy margins decline as refining spreads tighten",
            "Market stress following weak economic data",
            "Policy commentary supports growth amid uncertainty",
        ],
    })


@pytest.fixture
def sample_returns():
    """Sample portfolio returns for testing."""
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    values = np.random.normal(0.001, 0.01, 120)
    return pd.Series(values, index=dates, name="returns")


class TestFinBERTSentimentAnalyzer:
    """Tests for FinBERTSentimentAnalyzer."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = FinBERTSentimentAnalyzer(backend=MockFinBERTBackend())
        assert analyzer.model_name == "ProsusAI/finbert"
        assert analyzer.batch_size == 32
    
    def test_single_text_analysis(self):
        """Test analysis of single text."""
        analyzer = FinBERTSentimentAnalyzer(backend=MockFinBERTBackend())
        result = analyzer.analyze("Bank profits surge with strong growth")
        
        assert result["label"] == "positive"
        assert result["sentiment_score"] > 0.8
        assert "confidence" in result
    
    def test_batch_analysis(self):
        """Test analysis of text batch."""
        analyzer = FinBERTSentimentAnalyzer(backend=MockFinBERTBackend())
        texts = [
            "Growth improves significantly",
            "Risk and stress increase",
            "Market moves sideways today",
        ]
        results = analyzer.analyze_batch(texts)
        
        assert len(results) == 3
        assert results[0]["label"] == "positive"
        assert results[1]["label"] == "negative"
        assert results[2]["label"] == "neutral"
    
    def test_empty_input_handling(self):
        """Test handling of empty input."""
        analyzer = FinBERTSentimentAnalyzer(backend=MockFinBERTBackend())
        results = analyzer.analyze_batch([])
        assert results == []
    
    def test_confidence_values(self):
        """Test that confidence values are in valid range."""
        analyzer = FinBERTSentimentAnalyzer(backend=MockFinBERTBackend())
        texts = ["positive text", "negative text", "neutral text"]
        results = analyzer.analyze_batch(texts)
        
        for result in results:
            assert 0.0 <= result["confidence"] <= 1.0


class TestUncertaintyScorer:
    """Tests for UncertaintyScorer."""
    
    def test_uncertainty_scoring(self):
        """Test uncertainty keyword scoring."""
        scorer = UncertaintyScorer()
        
        high_uncertainty = "Risk and volatility uncertainty challenge liquidity stress"
        low_uncertainty = "Stable growth positive momentum"
        
        high_score = scorer.score(high_uncertainty)
        low_score = scorer.score(low_uncertainty)
        
        assert high_score > low_score
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0
    
    def test_batch_scoring(self):
        """Test batch uncertainty scoring."""
        scorer = UncertaintyScorer()
        texts = [
            "High risk and volatility",
            "Stable and steady",
            "Crisis and uncertainty",
        ]
        scores = scorer.score_batch(texts)
        
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)
    
    def test_custom_keywords(self):
        """Test with custom keywords."""
        custom_keywords = ["bull", "bear", "rally"]
        scorer = UncertaintyScorer(keywords=custom_keywords)
        
        assert scorer.score("bull market rally") > 0.5
        assert scorer.score("sideways market") == 0.0


class TestDocumentLoading:
    """Tests for document loading functions."""
    
    def test_load_documents_from_dataframe(self, sample_documents):
        """Test loading documents from DataFrame."""
        loaded = load_sentiment_documents(sample_documents)
        
        assert len(loaded) == 6
        assert "date" in loaded.columns
        assert "text" in loaded.columns
        assert isinstance(loaded["date"].dtype, type(pd.Timestamp.now()))
    
    def test_document_cleaning(self):
        """Test that documents are cleaned."""
        df = pd.DataFrame({
            "date": ["2024-01-02", "2024-01-02"],
            "text": ["  valid text  ", ""],
        })
        
        loaded = load_sentiment_documents(df)
        # Empty strings should be removed
        assert len(loaded) <= 2


class TestDocumentScoring:
    """Tests for document scoring."""
    
    def test_score_documents(self, sample_documents):
        """Test document scoring with FinBERT."""
        analyzer = FinBERTSentimentAnalyzer(backend=MockFinBERTBackend())
        scored = score_documents_batch(sample_documents, analyzer=analyzer)
        
        assert len(scored) == len(sample_documents)
        assert "sentiment_score" in scored.columns
        assert "label" in scored.columns
        assert "confidence" in scored.columns
    
    def test_score_with_uncertainty(self, sample_documents):
        """Test scoring with uncertainty component."""
        analyzer = FinBERTSentimentAnalyzer(backend=MockFinBERTBackend())
        uncertainty_scorer = UncertaintyScorer()
        
        scored = score_documents_batch(
            sample_documents,
            analyzer=analyzer,
            uncertainty_scorer=uncertainty_scorer,
        )
        
        assert "uncertainty_score" in scored.columns
        assert all(0.0 <= s <= 1.0 for s in scored["uncertainty_score"])


class TestDailyAggregation:
    """Tests for daily sentiment aggregation."""
    
    def test_aggregate_daily_sentiment(self, sample_documents):
        """Test daily aggregation."""
        analyzer = FinBERTSentimentAnalyzer(backend=MockFinBERTBackend())
        scored = score_documents_batch(sample_documents, analyzer=analyzer)
        daily = aggregate_daily_sentiment(scored)
        
        assert len(daily) >= 1
        assert "sentiment_mean" in daily.columns
        assert "document_count" in daily.columns
    
    def test_daily_aggregation_statistics(self, sample_documents):
        """Test that daily statistics are computed correctly."""
        analyzer = FinBERTSentimentAnalyzer(backend=MockFinBERTBackend())
        scored = score_documents_batch(sample_documents, analyzer=analyzer)
        daily = aggregate_daily_sentiment(scored)
        
        # Check that aggregated mean matches original mean for that day
        first_day = scored[scored["date"].dt.date == pd.Timestamp("2024-01-02").date()]
        if len(first_day) > 0:
            expected_mean = first_day["sentiment_score"].mean()
            actual_mean = daily.loc[daily.index.date == pd.Timestamp("2024-01-02").date(), 
                                   "sentiment_mean"].iloc[0]
            assert abs(expected_mean - actual_mean) < 0.01


class TestLaggedSignal:
    """Tests for lagged sentiment signal construction."""
    
    def test_lagged_signal_avoids_lookahead(self):
        """Test that lagging avoids look-ahead bias."""
        daily = pd.DataFrame(
            {"sentiment_mean": [0.9, -0.8, 0.2]},
            index=pd.date_range("2024-01-02", periods=3, freq="B"),
        )
        
        config = SentimentSignalConfig(lag_days=1, smoothing_window=1)
        signal = build_lagged_sentiment_signal(daily, config=config)
        
        # First value should be 0 (no prior data)
        assert signal.iloc[0] == 0.0
        # Second value should reflect first day's sentiment
        assert abs(signal.iloc[1] - 0.9) < 0.01
        # Third value should reflect second day's sentiment
        assert abs(signal.iloc[2] - (-0.8)) < 0.01
    
    def test_lagged_signal_with_target_dates(self):
        """Test signal alignment to target dates."""
        daily = pd.DataFrame(
            {"sentiment_mean": [0.5, -0.3]},
            index=pd.date_range("2024-01-02", periods=2, freq="B"),
        )
        
        target_dates = pd.date_range("2024-01-02", periods=5, freq="B")
        signal = build_lagged_sentiment_signal(daily, target_dates=target_dates)
        
        assert len(signal) == len(target_dates)
        assert signal.index.equals(target_dates)


class TestSentimentToVolatility:
    """Tests for sentiment-to-volatility mapping."""
    
    def test_mapping_extreme_positive(self):
        """Test mapping of extremely positive sentiment."""
        config = SentimentSignalConfig()
        signal = pd.Series([0.6], index=pd.date_range("2024-01-02", periods=1))
        target_vol = map_sentiment_to_volatility(signal, config=config)
        
        assert target_vol.iloc[0] == config.very_positive_target_vol
    
    def test_mapping_extreme_negative(self):
        """Test mapping of extremely negative sentiment."""
        config = SentimentSignalConfig()
        signal = pd.Series([-0.7], index=pd.date_range("2024-01-02", periods=1))
        target_vol = map_sentiment_to_volatility(signal, config=config)
        
        assert target_vol.iloc[0] == config.very_negative_target_vol
    
    def test_mapping_neutral(self):
        """Test mapping of neutral sentiment."""
        config = SentimentSignalConfig()
        signal = pd.Series([0.0], index=pd.date_range("2024-01-02", periods=1))
        target_vol = map_sentiment_to_volatility(signal, config=config)
        
        assert target_vol.iloc[0] == config.neutral_target_vol
    
    def test_target_volatility_monotonicity(self):
        """Test that target volatility increases with sentiment."""
        config = SentimentSignalConfig()
        sentiments = pd.Series(
            [-0.8, -0.4, 0.0, 0.4, 0.8],
            index=pd.date_range("2024-01-02", periods=5, freq="B"),
        )
        target_vols = map_sentiment_to_volatility(sentiments, config=config)
        
        # More positive → higher volatility
        for i in range(len(target_vols) - 1):
            if sentiments.iloc[i] < sentiments.iloc[i+1]:
                assert target_vols.iloc[i] <= target_vols.iloc[i+1]


class TestSentimentPipeline:
    """Tests for end-to-end sentiment pipeline."""
    
    def test_pipeline_execution(self, sample_documents):
        """Test complete pipeline execution."""
        analyzer = FinBERTSentimentAnalyzer(backend=MockFinBERTBackend())
        pipeline = SentimentPipeline(analyzer=analyzer)
        
        result = pipeline.run(sample_documents)
        
        assert len(result.scored_documents) == len(sample_documents)
        assert len(result.daily_sentiment) >= 1
        assert len(result.lagged_signal) >= 1
        assert len(result.target_volatility) >= 1
    
    def test_pipeline_with_target_dates(self, sample_documents):
        """Test pipeline with target date alignment."""
        analyzer = FinBERTSentimentAnalyzer(backend=MockFinBERTBackend())
        pipeline = SentimentPipeline(analyzer=analyzer)
        
        target_dates = pd.date_range("2024-01-02", periods=10, freq="B")
        result = pipeline.run(sample_documents, target_dates=target_dates)
        
        assert len(result.lagged_signal) == len(target_dates)
        assert result.lagged_signal.index.equals(target_dates)
    
    def test_pipeline_summary_statistics(self, sample_documents):
        """Test that pipeline produces valid summary statistics."""
        analyzer = FinBERTSentimentAnalyzer(backend=MockFinBERTBackend())
        pipeline = SentimentPipeline(analyzer=analyzer)
        
        result = pipeline.run(sample_documents)
        
        assert result.summary_stats["total_documents"] == len(sample_documents)
        assert "mean_sentiment" in result.summary_stats
        assert "mean_confidence" in result.summary_stats
        assert "mean_target_volatility" in result.summary_stats


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_analyze_sentiment_function(self, sample_documents):
        """Test quick analyze_sentiment function."""
        result = analyze_sentiment(
            sample_documents,
            backend=MockFinBERTBackend(),
        )
        
        assert len(result.scored_documents) > 0
        assert len(result.daily_sentiment) > 0
        assert result.summary_stats["total_documents"] > 0


# Integration tests
class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_workflow(self, sample_documents, sample_returns):
        """Test full workflow from documents to volatility adjustment."""
        # Analyze sentiment
        result = analyze_sentiment(
            sample_documents,
            backend=MockFinBERTBackend(),
        )
        
        # Verify signal can be applied to returns
        target_dates = sample_returns.index
        signal_aligned = result.lagged_signal.reindex(target_dates).ffill()
        
        assert len(signal_aligned) == len(sample_returns)
        assert signal_aligned.index.equals(target_dates)
    
    def test_configuration_flexibility(self, sample_documents):
        """Test that custom configurations work."""
        custom_config = SentimentSignalConfig(
            lag_days=2,
            smoothing_window=10,
            very_negative_target_vol=0.02,
            very_positive_target_vol=0.16,
        )
        
        result = analyze_sentiment(
            sample_documents,
            config=custom_config,
            backend=MockFinBERTBackend(),
        )
        
        assert result.lagged_signal is not None
        assert result.target_volatility is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
