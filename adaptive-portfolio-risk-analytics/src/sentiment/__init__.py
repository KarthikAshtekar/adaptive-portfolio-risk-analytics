"""Phase 4A through 4A.6 sentiment confirmation public API."""

from src.sentiment.api_ingestion import (
    INGESTION_OUTPUT_FILES,
    run_sentiment_provider_ingestion,
)
from src.sentiment.composite_index import (
    COMPOSITE_NLP_LABELS,
    VALID_COMPOSITE_NLP_LABELS,
    build_composite_nlp_risk_index,
    build_daily_nlp_signal,
)
from src.sentiment.corpus_intake import (
    EARNINGS_MANIFEST_COLUMNS as INTAKE_EARNINGS_MANIFEST_COLUMNS,
    NEWS_MANIFEST_COLUMNS,
    PLACEHOLDER_MARKER,
    RBI_MANIFEST_COLUMNS as INTAKE_RBI_MANIFEST_COLUMNS,
    is_explicit_placeholder,
    placeholder_mask,
    validate_corpus_manifest,
    validate_nlp_corpus_intake,
)
from src.sentiment.coverage import calculate_nlp_coverage
from src.sentiment.ex_ante_filters import (
    REACTION_DATA_PATTERNS,
    apply_publication_lag,
    flag_reaction_data_leakage,
    validate_ex_ante_records,
)
from src.sentiment.finbert_scoring import score_with_finbert
from src.sentiment.nlp_regime_comparison import (
    compare_composite_nlp_to_regimes,
    nlp_agrees_with_regime,
)
from src.sentiment.provider_config import (
    load_provider_config,
    validate_provider_config,
)
from src.sentiment.source_quality import (
    classify_data_provenance,
    score_source_quality,
)
from src.sentiment.providers import (
    AlphaVantageNewsProvider,
    DEFAULT_GDELT_QUERIES,
    EARNINGS_MANIFEST_COLUMNS,
    EarningsCallProvider,
    GDELTProvider,
    LocalProvider,
    NORMALIZED_SENTIMENT_COLUMNS,
    NewsProvider,
    RBIProvider,
    SentimentProvider,
)

from src.sentiment.alignment import (
    assign_records_to_market_dates,
    build_alignment_checks,
    validate_market_index,
)
from src.sentiment.analytics import (
    calculate_sentiment_confirmation_score,
    compare_sentiment_to_regimes,
    sentiment_agrees_with_regime,
)
from src.sentiment.ingestion import load_local_sentiment_csv
from src.sentiment.macro_index import (
    MACRO_LABELS,
    build_current_macro_summary,
    build_macro_stance_index,
    macro_confirmation_status,
    plot_macro_regime_timeline,
    plot_macro_stance_shares,
    plot_macro_stance_index,
    plot_macro_uncertainty_share,
)
from src.sentiment.macro_regime_comparison import (
    compare_macro_to_regimes,
    macro_agrees_with_regime,
)
from src.sentiment.rbi_ingestion import load_rbi_documents
from src.sentiment.rbi_empirical_validation import (
    EMPIRICAL_OUTPUT_FILES,
    run_rbi_empirical_validation,
)
from src.sentiment.rbi_corpus_builder import (
    REAL_RBI_DOCUMENT_TYPES,
    REAL_RBI_MANIFEST_COLUMNS,
    build_rbi_manifest_from_directory,
    load_real_rbi_corpus,
    validate_rbi_manifest,
)
from src.sentiment.rbi_processing import (
    clean_rbi_sentence,
    process_rbi_documents,
    split_rbi_documents_into_sentences,
    split_rbi_sentences,
)
from src.sentiment.rbi_scoring import (
    RBI_HF_MODELS,
    RBI_LEXICON_MODEL_NAME,
    RBITransformerAdapter,
    score_rbi_sentences,
)
from src.sentiment.reporting import (
    build_current_sentiment_summary,
    plot_daily_sentiment_signal,
    plot_sentiment_regime_timeline,
    sentiment_commentary,
)
from src.sentiment.schema import (
    RBI_CERTAINTY_LABELS,
    RBI_DOCUMENT_TYPES,
    RBI_STANCE_LABELS,
    RBI_TIME_LABELS,
    SENTIMENT_LABELS,
    RBIDocument,
    RBISentenceScore,
    SentimentRecord,
)
from src.sentiment.scoring import (
    LEXICON_MODEL_NAME,
    LEXICON_MODEL_VERSION,
    RISK_OFF_TERMS,
    RISK_ON_TERMS,
    score_sentiment_records,
)
from src.sentiment.signals import build_daily_sentiment_signal

__all__ = [
    "LEXICON_MODEL_NAME",
    "LEXICON_MODEL_VERSION",
    "RISK_OFF_TERMS",
    "RISK_ON_TERMS",
    "MACRO_LABELS",
    "RBI_CERTAINTY_LABELS",
    "RBI_DOCUMENT_TYPES",
    "RBI_HF_MODELS",
    "RBI_LEXICON_MODEL_NAME",
    "RBI_STANCE_LABELS",
    "RBI_TIME_LABELS",
    "RBIDocument",
    "RBISentenceScore",
    "RBITransformerAdapter",
    "EMPIRICAL_OUTPUT_FILES",
    "REAL_RBI_DOCUMENT_TYPES",
    "REAL_RBI_MANIFEST_COLUMNS",
    "SENTIMENT_LABELS",
    "SentimentRecord",
    "AlphaVantageNewsProvider",
    "COMPOSITE_NLP_LABELS",
    "VALID_COMPOSITE_NLP_LABELS",
    "DEFAULT_GDELT_QUERIES",
    "EARNINGS_MANIFEST_COLUMNS",
    "EarningsCallProvider",
    "GDELTProvider",
    "INGESTION_OUTPUT_FILES",
    "LocalProvider",
    "NORMALIZED_SENTIMENT_COLUMNS",
    "NewsProvider",
    "RBIProvider",
    "REACTION_DATA_PATTERNS",
    "SentimentProvider",
    "apply_publication_lag",
    "assign_records_to_market_dates",
    "build_alignment_checks",
    "build_current_sentiment_summary",
    "build_current_macro_summary",
    "build_daily_sentiment_signal",
    "build_daily_nlp_signal",
    "build_macro_stance_index",
    "build_composite_nlp_risk_index",
    "build_rbi_manifest_from_directory",
    "calculate_sentiment_confirmation_score",
    "calculate_nlp_coverage",
    "classify_data_provenance",
    "INTAKE_EARNINGS_MANIFEST_COLUMNS",
    "INTAKE_RBI_MANIFEST_COLUMNS",
    "NEWS_MANIFEST_COLUMNS",
    "PLACEHOLDER_MARKER",
    "is_explicit_placeholder",
    "placeholder_mask",
    "compare_sentiment_to_regimes",
    "compare_macro_to_regimes",
    "compare_composite_nlp_to_regimes",
    "clean_rbi_sentence",
    "load_local_sentiment_csv",
    "load_provider_config",
    "load_rbi_documents",
    "load_real_rbi_corpus",
    "flag_reaction_data_leakage",
    "macro_agrees_with_regime",
    "macro_confirmation_status",
    "plot_daily_sentiment_signal",
    "plot_sentiment_regime_timeline",
    "plot_macro_stance_index",
    "plot_macro_stance_shares",
    "plot_macro_uncertainty_share",
    "plot_macro_regime_timeline",
    "process_rbi_documents",
    "score_rbi_sentences",
    "run_rbi_empirical_validation",
    "run_sentiment_provider_ingestion",
    "score_with_finbert",
    "score_sentiment_records",
    "score_source_quality",
    "split_rbi_documents_into_sentences",
    "split_rbi_sentences",
    "sentiment_agrees_with_regime",
    "sentiment_commentary",
    "nlp_agrees_with_regime",
    "validate_market_index",
    "validate_rbi_manifest",
    "validate_ex_ante_records",
    "validate_corpus_manifest",
    "validate_nlp_corpus_intake",
    "validate_provider_config",
]
