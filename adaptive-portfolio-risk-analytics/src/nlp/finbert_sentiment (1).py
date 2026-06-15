"""
FinBERT Sentiment Analysis Module for Financial Risk Management
Converts dated financial text into lagged sentiment signals for adaptive portfolio risk control.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "FinBERTSentimentAnalyzer",
    "SentimentSignalConfig",
    "SentimentPipeline",
    "UncertaintyScorer",
    "load_sentiment_documents",
    "score_documents_batch",
    "aggregate_daily_sentiment",
    "build_lagged_sentiment_signal",
    "map_sentiment_to_volatility",
    "SentimentAnalysisResult",
]


@dataclass(frozen=True)
class SentimentSignalConfig:
    """Configuration for sentiment signal construction and risk mapping."""
    
    # Signal construction
    lag_days: int = 1
    smoothing_window: int = 5
    min_documents_per_day: int = 1
    
    # Sentiment thresholds
    very_positive_threshold: float = 0.55
    positive_threshold: float = 0.20
    negative_threshold: float = -0.20
    very_negative_threshold: float = -0.55
    
    # Target volatility mapping (annualized)
    very_positive_target_vol: float = 0.14
    positive_target_vol: float = 0.12
    neutral_target_vol: float = 0.10
    negative_target_vol: float = 0.06
    very_negative_target_vol: float = 0.03
    
    # Confidence weighting
    min_confidence: float = 0.5
    use_confidence_weights: bool = True


@dataclass
class SentimentAnalysisResult:
    """Container for sentiment analysis results."""
    
    scored_documents: pd.DataFrame
    daily_sentiment: pd.DataFrame
    lagged_signal: pd.Series
    target_volatility: pd.Series
    summary_stats: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FinBERTSentimentAnalyzer:
    """
    FinBERT-backed financial sentiment classifier.
    
    Uses Hugging Face's ProsusAI/finbert model for financial text sentiment analysis.
    Supports both online (download) and offline (local cache) modes.
    """
    
    LABEL_MAP = {
        "positive": 1.0,
        "neutral": 0.0,
        "negative": -1.0,
    }
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: int = -1,  # -1 for CPU, 0+ for GPU
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        batch_size: int = 32,
        max_length: int = 512,
        backend: Optional[Any] = None,  # For testing with mock backend
    ):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use (-1 for CPU, 0+ for GPU)
            cache_dir: Directory to cache downloaded models
            local_files_only: If True, only use locally cached models
            batch_size: Batch size for inference
            max_length: Maximum token length
            backend: Mock backend for testing (overrides real model)
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.batch_size = batch_size
        self.max_length = max_length
        self.backend = backend
        self._pipeline = None
        
        logger.info(
            f"Initialized FinBERTSentimentAnalyzer: model={model_name}, "
            f"device={device}, batch_size={batch_size}"
        )
    
    def analyze(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment of a single text."""
        return self.analyze_batch([text])[0]
    
    def analyze_batch(self, texts: Sequence[str]) -> List[Dict[str, Union[str, float]]]:
        """Analyze sentiment of multiple texts."""
        if not texts:
            return []
        
        # Clean texts
        clean_texts = [str(t).strip() for t in texts if str(t).strip()]
        if not clean_texts:
            logger.warning("No valid texts to analyze after cleaning")
            return [self._default_result("neutral") for _ in texts]
        
        try:
            # Use backend if provided (for testing)
            if self.backend is not None:
                outputs = self.backend(clean_texts, max_length=self.max_length)
            else:
                # Use real FinBERT pipeline
                pipeline = self._get_pipeline()
                outputs = pipeline(
                    clean_texts,
                    batch_size=self.batch_size,
                    truncation=True,
                    max_length=self.max_length,
                )
            
            # Normalize outputs
            results = [self._normalize_output(output) for output in outputs]
            
            # Pad results to match input count (in case of filtering)
            while len(results) < len(texts):
                results.append(self._default_result("neutral"))
            
            return results[:len(texts)]
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return [self._default_result("neutral") for _ in texts]
    
    def _get_pipeline(self):
        """Lazy load FinBERT pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                
                logger.info(f"Loading FinBERT model: {self.model_name}")
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=self.device,
                    model_kwargs={
                        "cache_dir": self.cache_dir,
                        "local_files_only": self.local_files_only,
                    },
                )
            except ImportError:
                raise ImportError(
                    "transformers library required. Install: pip install transformers torch"
                )
        
        return self._pipeline
    
    def _normalize_output(self, output: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        """Normalize model output to standard format."""
        label = str(output.get("label", "neutral")).lower()
        confidence = float(output.get("score", 0.0))
        score = self.LABEL_MAP.get(label, 0.0) * confidence
        
        return {
            "label": label,
            "confidence": float(confidence),
            "sentiment_score": float(score),
            "model": self.model_name,
        }
    
    @staticmethod
    def _default_result(label: str = "neutral") -> Dict[str, Union[str, float]]:
        """Return default sentiment result."""
        return {
            "label": label,
            "confidence": 0.0,
            "sentiment_score": 0.0,
            "model": "default",
        }


class UncertaintyScorer:
    """
    Keyword-based uncertainty/risk quantifier.
    Complements FinBERT with explicit uncertainty indicators.
    """
    
    DEFAULT_KEYWORDS = [
        "uncertainty", "uncertain", "risk", "risky", "volatile", "volatility",
        "downside risk", "tail risk", "inflation", "deflation", "recession",
        "liquidity", "stress", "crisis", "downgrade", "default", "bankruptcy",
        "warning", "caution", "concern", "challenge", "difficulty",
    ]
    
    def __init__(self, keywords: Optional[List[str]] = None):
        """Initialize uncertainty scorer."""
        self.keywords = keywords or self.DEFAULT_KEYWORDS
    
    def score(self, text: str) -> float:
        """
        Calculate uncertainty score (0-1).
        Higher score = more uncertainty/risk keywords present.
        """
        if not text:
            return 0.0
        
        text_lower = str(text).lower()
        hits = sum(1 for kw in self.keywords if kw.lower() in text_lower)
        
        if not self.keywords:
            return 0.0
        
        # Normalize to 0-1 range
        score = min(1.0, hits / len(self.keywords))
        return float(score)
    
    def score_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Score multiple texts."""
        return np.array([self.score(t) for t in texts])


def load_sentiment_documents(
    path: Union[str, Path],
    date_column: str = "date",
    text_column: str = "text",
) -> pd.DataFrame:
    """
    Load sentiment documents from CSV.
    
    Required columns: date, text
    Optional columns: source, ticker, url (preserved for audit trail)
    
    Args:
        path: Path to CSV file
        date_column: Name of date column
        text_column: Name of text column
    
    Returns:
        DataFrame with datetime index and validated text column
    """
    logger.info(f"Loading sentiment documents from {path}")
    
    df = pd.read_csv(path)
    
    # Validate required columns
    required = {date_column, text_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Clean and convert
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df[text_column] = df[text_column].astype(str).str.strip()
    
    # Remove empty texts
    df = df[df[text_column].str.len() > 0]
    
    # Sort by date
    df = df.sort_values(date_column).reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} documents from {df[date_column].min().date()} "
                f"to {df[date_column].max().date()}")
    
    return df


def score_documents_batch(
    documents: pd.DataFrame,
    analyzer: Optional[FinBERTSentimentAnalyzer] = None,
    uncertainty_scorer: Optional[UncertaintyScorer] = None,
    text_column: str = "text",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Score documents using FinBERT and optional uncertainty scoring.
    
    Args:
        documents: DataFrame with text column
        analyzer: FinBERTSentimentAnalyzer instance
        uncertainty_scorer: Optional UncertaintyScorer instance
        text_column: Name of text column
        batch_size: Batch size for processing
    
    Returns:
        Original DataFrame + sentiment columns
    """
    if text_column not in documents.columns:
        raise ValueError(f"Text column '{text_column}' not found in documents")
    
    analyzer = analyzer or FinBERTSentimentAnalyzer()
    
    logger.info(f"Scoring {len(documents)} documents with FinBERT")
    
    # Score in batches
    texts = documents[text_column].tolist()
    all_results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = analyzer.analyze_batch(batch)
        all_results.extend(batch_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Add uncertainty score if scorer provided
    if uncertainty_scorer:
        logger.info("Scoring uncertainty...")
        results_df["uncertainty_score"] = uncertainty_scorer.score_batch(texts)
    
    # Combine with original
    scored = pd.concat([documents.reset_index(drop=True), 
                        results_df.reset_index(drop=True)], axis=1)
    
    logger.info(f"Scoring complete. Mean sentiment: {scored['sentiment_score'].mean():.3f}")
    
    return scored


def aggregate_daily_sentiment(
    scored_documents: pd.DataFrame,
    date_column: str = "date",
    sentiment_column: str = "sentiment_score",
    confidence_column: str = "confidence",
    min_docs: int = 1,
) -> pd.DataFrame:
    """
    Aggregate document-level sentiment to daily level.
    
    Args:
        scored_documents: DataFrame with sentiment scores
        date_column: Name of date column
        sentiment_column: Name of sentiment score column
        confidence_column: Name of confidence column
        min_docs: Minimum documents per day for valid aggregation
    
    Returns:
        Daily sentiment DataFrame
    """
    logger.info("Aggregating sentiment to daily level")
    
    df = scored_documents.copy()
    df[date_column] = pd.to_datetime(df[date_column]).dt.normalize()
    df[sentiment_column] = pd.to_numeric(df[sentiment_column], errors="coerce")
    
    # Confidence weighting
    if confidence_column in df.columns:
        df[confidence_column] = pd.to_numeric(df[confidence_column], errors="coerce").fillna(0.5)
        df["weighted_sentiment"] = df[sentiment_column] * df[confidence_column]
    else:
        df["weighted_sentiment"] = df[sentiment_column]
    
    # Daily aggregation
    daily = df.groupby(date_column).agg({
        sentiment_column: ["mean", "std", "min", "max", "count"],
        "weighted_sentiment": "mean",
    }).round(4)
    
    daily.columns = ["sentiment_mean", "sentiment_std", "sentiment_min", 
                     "sentiment_max", "document_count", "weighted_sentiment_mean"]
    
    # Filter by minimum documents
    daily = daily[daily["document_count"] >= min_docs]
    
    logger.info(f"Created {len(daily)} daily sentiment observations")
    
    return daily


def build_lagged_sentiment_signal(
    daily_sentiment: Union[pd.DataFrame, pd.Series],
    target_dates: Optional[pd.DatetimeIndex] = None,
    config: Optional[SentimentSignalConfig] = None,
) -> pd.Series:
    """
    Build lagged sentiment signal to avoid look-ahead bias.
    
    The signal at time t reflects sentiment from time t-lag_days.
    This ensures no forward-looking information leakage.
    
    Args:
        daily_sentiment: Daily sentiment DataFrame or Series
        target_dates: Target date index for signal alignment
        config: SentimentSignalConfig with lag and smoothing parameters
    
    Returns:
        Lagged sentiment signal
    """
    config = config or SentimentSignalConfig()
    
    # Extract sentiment series
    if isinstance(daily_sentiment, pd.DataFrame):
        if "sentiment_mean" in daily_sentiment.columns:
            sentiment = daily_sentiment["sentiment_mean"].copy()
        elif "weighted_sentiment_mean" in daily_sentiment.columns:
            sentiment = daily_sentiment["weighted_sentiment_mean"].copy()
        else:
            sentiment = daily_sentiment.iloc[:, 0].copy()
    else:
        sentiment = daily_sentiment.copy()
    
    # Ensure datetime index
    if not isinstance(sentiment.index, pd.DatetimeIndex):
        raise ValueError("Sentiment index must be DatetimeIndex")
    
    sentiment = sentiment.sort_index().astype(float)
    
    # Apply smoothing
    smoothed = sentiment.rolling(
        window=config.smoothing_window,
        min_periods=1,
        center=False
    ).mean()
    
    # Apply lag to avoid look-ahead bias
    lagged = smoothed.shift(config.lag_days)
    
    # Align to target dates if provided
    if target_dates is not None:
        lagged = lagged.reindex(target_dates).ffill()
    
    # Fill remaining NaN with neutral
    lagged = lagged.fillna(0.0)
    lagged.name = "lagged_sentiment_score"
    
    logger.info(f"Created lagged sentiment signal: lag={config.lag_days}d, "
                f"smoothing={config.smoothing_window}d, mean={lagged.mean():.3f}")
    
    return lagged


def map_sentiment_to_volatility(
    sentiment_signal: pd.Series,
    config: Optional[SentimentSignalConfig] = None,
) -> pd.Series:
    """
    Map sentiment scores to target volatility levels.
    
    Mapping logic:
    - Very negative sentiment → most defensive (3% volatility)
    - Negative sentiment → defensive (6% volatility)
    - Neutral sentiment → standard (10% volatility)
    - Positive sentiment → modest risk (12% volatility)
    - Very positive sentiment → higher risk (14% volatility)
    
    Args:
        sentiment_signal: Lagged sentiment signal
        config: SentimentSignalConfig with mapping parameters
    
    Returns:
        Target volatility series
    """
    config = config or SentimentSignalConfig()
    
    signal = sentiment_signal.astype(float).copy()
    target_vol = pd.Series(config.neutral_target_vol, index=signal.index, dtype=float)
    
    # Apply mapping logic
    target_vol.loc[signal >= config.very_positive_threshold] = config.very_positive_target_vol
    target_vol.loc[(signal >= config.positive_threshold) & 
                   (signal < config.very_positive_threshold)] = config.positive_target_vol
    target_vol.loc[(signal >= config.negative_threshold) & 
                   (signal < config.positive_threshold)] = config.neutral_target_vol
    target_vol.loc[(signal >= config.very_negative_threshold) & 
                   (signal < config.negative_threshold)] = config.negative_target_vol
    target_vol.loc[signal < config.very_negative_threshold] = config.very_negative_target_vol
    
    target_vol.name = "sentiment_target_volatility"
    
    logger.info(f"Mapped sentiment to target volatility: mean={target_vol.mean():.4f}, "
                f"range=[{target_vol.min():.4f}, {target_vol.max():.4f}]")
    
    return target_vol


class SentimentPipeline:
    """
    End-to-end sentiment analysis pipeline.
    
    Converts dated financial documents → FinBERT scores → daily aggregation → 
    lagged signal → target volatility adjustments.
    """
    
    def __init__(
        self,
        analyzer: Optional[FinBERTSentimentAnalyzer] = None,
        uncertainty_scorer: Optional[UncertaintyScorer] = None,
        config: Optional[SentimentSignalConfig] = None,
    ):
        """Initialize sentiment pipeline."""
        self.analyzer = analyzer or FinBERTSentimentAnalyzer()
        self.uncertainty_scorer = uncertainty_scorer
        self.config = config or SentimentSignalConfig()
        
        logger.info("Initialized SentimentPipeline")
    
    def run(
        self,
        documents: Union[str, Path, pd.DataFrame],
        target_dates: Optional[pd.DatetimeIndex] = None,
    ) -> SentimentAnalysisResult:
        """
        Run complete sentiment analysis pipeline.
        
        Args:
            documents: Path to CSV or DataFrame of documents
            target_dates: Target date index for signal alignment
        
        Returns:
            SentimentAnalysisResult with all intermediate and final results
        """
        logger.info("Starting sentiment analysis pipeline")
        
        # Load documents
        if isinstance(documents, (str, Path)):
            docs = load_sentiment_documents(documents)
        else:
            docs = documents.copy()
        
        # Score documents
        scored = score_documents_batch(
            docs,
            analyzer=self.analyzer,
            uncertainty_scorer=self.uncertainty_scorer,
        )
        
        # Aggregate to daily
        daily = aggregate_daily_sentiment(
            scored,
            min_docs=self.config.min_documents_per_day,
        )
        
        # Build lagged signal
        signal = build_lagged_sentiment_signal(
            daily,
            target_dates=target_dates,
            config=self.config,
        )
        
        # Map to target volatility
        target_vol = map_sentiment_to_volatility(signal, config=self.config)
        
        # Summary statistics
        summary = {
            "total_documents": len(scored),
            "date_range_days": (scored["date"].max() - scored["date"].min()).days,
            "mean_sentiment": float(scored["sentiment_score"].mean()),
            "std_sentiment": float(scored["sentiment_score"].std()),
            "positive_count": int((scored["sentiment_score"] > self.config.positive_threshold).sum()),
            "negative_count": int((scored["sentiment_score"] < self.config.negative_threshold).sum()),
            "mean_confidence": float(scored.get("confidence", pd.Series([0.0])).mean()),
            "mean_target_volatility": float(target_vol.mean()),
            "signal_range": (float(signal.min()), float(signal.max())),
        }
        
        logger.info(f"Pipeline complete: {summary['total_documents']} documents, "
                    f"{len(daily)} daily observations, mean_sentiment={summary['mean_sentiment']:.3f}")
        
        return SentimentAnalysisResult(
            scored_documents=scored,
            daily_sentiment=daily,
            lagged_signal=signal,
            target_volatility=target_vol,
            summary_stats=summary,
        )


# Convenience function for quick analysis
def analyze_sentiment(
    documents: Union[str, Path, pd.DataFrame],
    config: Optional[SentimentSignalConfig] = None,
    target_dates: Optional[pd.DatetimeIndex] = None,
    backend: Optional[Any] = None,
) -> SentimentAnalysisResult:
    """
    Quick sentiment analysis with default configuration.
    
    Args:
        documents: Documents to analyze
        config: Optional sentiment configuration
        target_dates: Optional target date index
        backend: Optional mock backend for testing
    
    Returns:
        SentimentAnalysisResult
    """
    analyzer = FinBERTSentimentAnalyzer(backend=backend)
    pipeline = SentimentPipeline(analyzer=analyzer, config=config)
    return pipeline.run(documents, target_dates=target_dates)
