"""Deterministic and optional transformer scoring for RBI macro language."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable

import pandas as pd


RBI_LEXICON_MODEL_NAME = "rbi_macro_lexicon"
RBI_LEXICON_MODEL_VERSION = "1.0"
RBI_HF_MODELS = {
    "stance": "gtfintechlab/model_reserve_bank_of_india_stance_label",
    "certainty": "gtfintechlab/model_reserve_bank_of_india_certain_label",
    "time": "gtfintechlab/model_reserve_bank_of_india_time_label",
}

HAWKISH_TERMS = (
    "inflation remains elevated",
    "inflationary pressure",
    "upside risk",
    "upside risks to inflation",
    "price stability",
    "withdrawal of accommodation",
    "tightening",
    "rate hike",
    "policy rate increase",
    "liquidity absorption",
    "liquidity withdrawal",
    "persistent inflation",
    "restrictive",
    "restrictive stance",
)
DOVISH_TERMS = (
    "rate cut",
    "easing",
    "accommodative",
    "growth support",
    "support growth",
    "liquidity injection",
    "liquidity support",
    "supportive policy",
    "revive demand",
    "growth concerns",
    "weak demand",
    "growth slowdown",
    "disinflation",
    "benign inflation",
    "lower policy rate",
)
CERTAIN_TERMS = (
    "will",
    "must",
    "remains",
    "is committed",
    "necessary",
    "clearly",
)
UNCERTAIN_TERMS = (
    "may",
    "might",
    "could",
    "uncertain",
    "uncertainty",
    "volatility",
    "risks remain",
    "geopolitical tensions",
    "external headwinds",
    "clouded outlook",
    "downside risks",
    "upside risks",
    "subject to",
    "data dependent",
    "evolving",
)
FORWARD_TERMS = (
    "will",
    "expected",
    "expected to",
    "outlook",
    "going forward",
    "projected",
    "projected to",
    "future",
    "likely to",
    "forecast",
    "anticipated",
    "over the coming months",
)
BACKWARD_TERMS = (
    "was",
    "were",
    "previous",
    "earlier",
    "since",
    "had",
    "declined",
    "increased",
)
CURRENT_TERMS = ("currently", "now", "at present", "remains", "is", "are")


def _normalize(value: object) -> str:
    text = str(value or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%.\s-]", " ", text)).strip()


def _count_terms(text: str, terms: tuple[str, ...]) -> int:
    return sum(
        len(re.findall(rf"(?<!\w){re.escape(term)}(?!\w)", text))
        for term in terms
    )


def _lexicon_row(text: object) -> dict[str, object]:
    normalized = _normalize(text)
    hawkish = _count_terms(normalized, HAWKISH_TERMS)
    dovish = _count_terms(normalized, DOVISH_TERMS)
    certain = _count_terms(normalized, CERTAIN_TERMS)
    uncertain = _count_terms(normalized, UNCERTAIN_TERMS)
    forward = _count_terms(normalized, FORWARD_TERMS)
    backward = _count_terms(normalized, BACKWARD_TERMS)
    current = _count_terms(normalized, CURRENT_TERMS)

    stance = "hawkish" if hawkish > dovish else "dovish" if dovish > hawkish else "neutral"
    certainty = (
        "uncertain"
        if uncertain > certain
        else "certain"
        if certain > uncertain
        else "neutral"
    )
    time_label = (
        "forward_looking"
        if forward > max(backward, current)
        else "backward_looking"
        if backward > max(forward, current)
        else "current"
        if current > 0
        else "unknown"
    )
    stance_total = hawkish + dovish
    certainty_total = certain + uncertain
    time_total = forward + backward + current
    return {
        "stance_label": stance,
        "stance_score": (
            hawkish / stance_total
            if stance == "hawkish" and stance_total
            else dovish / stance_total
            if stance == "dovish" and stance_total
            else 1.0
            if stance == "neutral"
            else 0.0
        ),
        "certainty_label": certainty,
        "certainty_score": (
            uncertain / certainty_total
            if certainty == "uncertain" and certainty_total
            else certain / certainty_total
            if certainty == "certain" and certainty_total
            else 1.0
            if certainty == "neutral"
            else 0.0
        ),
        "time_label": time_label,
        "time_score": (
            forward / time_total
            if time_label == "forward_looking" and time_total
            else backward / time_total
            if time_label == "backward_looking" and time_total
            else current / time_total
            if time_label == "current" and time_total
            else 0.0
        ),
        "hawkish_score": hawkish / stance_total if stance_total else 0.0,
        "dovish_score": dovish / stance_total if stance_total else 0.0,
        "uncertainty_score": uncertain / certainty_total if certainty_total else 0.0,
        "forward_looking_score": forward / time_total if time_total else 0.0,
    }


def _canonical_label(task: str, label: object) -> str:
    value = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "stance": {
            "label_0": "neutral",
            "label_1": "hawkish",
            "label_2": "dovish",
            "label_3": "neutral",
            "hawk": "hawkish",
            "hawkish": "hawkish",
            "neutral": "neutral",
            "dove": "dovish",
            "dovish": "dovish",
        },
        "certainty": {
            "label_0": "certain",
            "label_1": "uncertain",
            "certain": "certain",
            "uncertain": "uncertain",
            "neutral": "neutral",
        },
        "time": {
            "label_0": "forward_looking",
            "label_1": "unknown",
            "forward": "forward_looking",
            "forward_looking": "forward_looking",
            "backward": "backward_looking",
            "backward_looking": "backward_looking",
            "current": "current",
            "unknown": "unknown",
        },
    }
    return aliases[task].get(value, "neutral" if task != "time" else "unknown")


@dataclass
class RBITransformerAdapter:
    """Lazy Hugging Face adapter with explicit local-only default behavior."""

    model_names: dict[str, str] = field(default_factory=lambda: dict(RBI_HF_MODELS))
    local_files_only: bool = True
    pipeline_factory: Callable[..., object] | None = None
    _pipelines: dict[str, object] = field(default_factory=dict, init=False)

    def _get_pipeline(self, task: str) -> object:
        if task not in self._pipelines:
            factory = self.pipeline_factory
            if factory is None:
                from transformers import pipeline

                factory = pipeline
            self._pipelines[task] = factory(
                "text-classification",
                model=self.model_names[task],
                tokenizer=self.model_names[task],
                model_kwargs={"local_files_only": self.local_files_only},
            )
        return self._pipelines[task]

    def score(self, text: str) -> dict[str, object]:
        results: dict[str, object] = {}
        for task, output_column, score_column in (
            ("stance", "stance_label", "stance_score"),
            ("certainty", "certainty_label", "certainty_score"),
            ("time", "time_label", "time_score"),
        ):
            output = self._get_pipeline(task)(text, truncation=True)
            item = output[0] if isinstance(output, list) else output
            canonical = _canonical_label(task, item["label"])
            results[output_column] = canonical
            results[score_column] = float(item.get("score", 0.0))
        return results


def score_rbi_sentences(
    sentences: pd.DataFrame,
    method: str = "lexicon",
    model_config: dict[str, object] | None = None,
    *,
    transformer_adapter: RBITransformerAdapter | None = None,
    allow_model_download: bool = False,
) -> pd.DataFrame:
    """Score RBI sentences and fall back to the deterministic lexicon safely."""
    if method not in {"lexicon", "transformer"}:
        raise ValueError("method must be 'lexicon' or 'transformer'")
    if not isinstance(sentences, pd.DataFrame):
        raise TypeError("sentences must be a pandas DataFrame")
    if "sentence_text" not in sentences and "sentence" not in sentences:
        raise ValueError("sentences must contain sentence or sentence_text")

    scored = sentences.copy()
    if "sentence" not in scored:
        scored["sentence"] = scored["sentence_text"]
    if "sentence_text" not in scored:
        scored["sentence_text"] = scored["sentence"]
    lexicon_scores = [_lexicon_row(text) for text in scored["sentence"]]
    for column in (
        "stance_label",
        "stance_score",
        "certainty_label",
        "certainty_score",
        "time_label",
        "time_score",
        "hawkish_score",
        "dovish_score",
        "uncertainty_score",
        "forward_looking_score",
    ):
        scored[column] = [row[column] for row in lexicon_scores]
    scored["requested_method"] = method
    scored["scoring_method"] = "lexicon"
    scored["model_name"] = RBI_LEXICON_MODEL_NAME
    scored["model_version"] = RBI_LEXICON_MODEL_VERSION
    scored["fallback_used"] = False
    scored["fallback_reason"] = pd.NA

    if method == "transformer" and not scored.empty:
        config = dict(model_config or {})
        configured_adapter = config.pop("adapter", None)
        model_names = config.pop("model_names", RBI_HF_MODELS)
        local_files_only = bool(
            config.pop("local_files_only", not allow_model_download)
        )
        pipeline_factory = config.pop("pipeline_factory", None)
        if config:
            raise ValueError(
                f"unsupported model_config keys: {sorted(config)}"
            )
        adapter = (
            transformer_adapter
            or configured_adapter
            or RBITransformerAdapter(
                model_names=dict(model_names),
                local_files_only=local_files_only,
                pipeline_factory=pipeline_factory,
            )
        )
        for index, text in scored["sentence"].items():
            try:
                labels = adapter.score(str(text))
                for column, value in labels.items():
                    scored.at[index, column] = value
                scored.at[index, "scoring_method"] = "transformer"
                scored.at[index, "model_name"] = ";".join(
                    adapter.model_names.values()
                )
                scored.at[index, "model_version"] = "huggingface"
            except Exception as exc:
                scored.at[index, "fallback_used"] = True
                scored.at[index, "fallback_reason"] = str(exc)
    return scored
