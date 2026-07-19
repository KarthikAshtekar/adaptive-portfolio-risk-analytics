"""Tests for deterministic and fallback RBI sentence scoring."""

from __future__ import annotations

import pandas as pd

from src.sentiment import (
    RBISentenceScore,
    RBITransformerAdapter,
    score_rbi_sentences,
)


class FailingAdapter:
    model_names = {"stance": "x", "certainty": "y", "time": "z"}

    def score(self, text: str) -> dict[str, str]:
        raise RuntimeError("model unavailable")


def _sentences() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sentence_id": "doc_s0000",
                "document_id": "doc",
                "publication_date": pd.Timestamp("2024-01-05"),
                "sentence_order": 0,
                "sentence_text": ("Persistent inflation and upside risk will require tightening."),
            },
            {
                "sentence_id": "doc_s0001",
                "document_id": "doc",
                "publication_date": pd.Timestamp("2024-01-05"),
                "sentence_order": 1,
                "sentence_text": (
                    "Benign inflation may permit easing to support growth going forward."
                ),
            },
        ]
    )


def test_lexicon_scores_stance_certainty_and_time() -> None:
    scored = score_rbi_sentences(_sentences(), method="lexicon")

    assert scored.iloc[0]["stance_label"] == "hawkish"
    assert scored.iloc[0]["certainty_label"] == "certain"
    assert scored.iloc[1]["stance_label"] == "dovish"
    assert scored.iloc[1]["certainty_label"] == "uncertain"
    assert scored.iloc[1]["time_label"] == "forward_looking"
    assert {"stance_score", "certainty_score", "time_score"}.issubset(scored.columns)


def test_transformer_failure_falls_back_per_sentence() -> None:
    scored = score_rbi_sentences(
        _sentences(),
        method="transformer",
        transformer_adapter=FailingAdapter(),
    )

    assert scored["scoring_method"].eq("lexicon").all()
    assert scored["fallback_used"].all()
    assert scored["fallback_reason"].str.contains("model unavailable").all()


def test_transformer_adapter_maps_documented_label_ids() -> None:
    def factory(task: str, *, model: str, **kwargs):
        label = "LABEL_1" if "stance" in model else "LABEL_0" if "certain" in model else "LABEL_0"

        return lambda text, **call_kwargs: [{"label": label, "score": 0.9}]

    adapter = RBITransformerAdapter(pipeline_factory=factory)

    assert adapter.score("Policy will remain vigilant.") == {
        "stance_label": "hawkish",
        "stance_score": 0.9,
        "certainty_label": "certain",
        "certainty_score": 0.9,
        "time_label": "forward_looking",
        "time_score": 0.9,
    }


def test_prompt_phrase_dictionaries_cover_uncertainty_and_forward_language() -> None:
    sentences = pd.DataFrame(
        [
            {
                "document_id": "doc",
                "publication_date": "2024-01-05",
                "sentence_id": "doc_s0000",
                "sentence": (
                    "Geopolitical tensions and external headwinds clouded the "
                    "outlook, which is expected to remain volatile."
                ),
            }
        ]
    )

    scored = score_rbi_sentences(sentences)

    assert scored.iloc[0]["certainty_label"] == "uncertain"
    assert scored.iloc[0]["time_label"] == "forward_looking"


def test_model_config_can_supply_optional_transformer_adapter() -> None:
    scored = score_rbi_sentences(
        _sentences(),
        method="transformer",
        model_config={"adapter": FailingAdapter()},
    )

    assert scored["fallback_used"].all()


def test_sentence_score_schema_exposes_requested_fields() -> None:
    score = RBISentenceScore(
        document_id="doc",
        publication_date=pd.Timestamp("2024-01-05"),
        sentence_id="doc_s0000",
        sentence="Inflation remains elevated.",
        stance_label="hawkish",
        stance_score=1.0,
        certainty_label="certain",
        certainty_score=1.0,
        time_label="current",
        time_score=1.0,
        model_name="fixture",
        model_version="1",
    )

    assert score.sentence_text == score.sentence
