"""Plain-language recommendation explanations."""

from __future__ import annotations


def build_recommendation_explanation(
    *,
    profile_name: str,
    main_strategy: str,
    overlay_strategy: str | None,
    scenarios: tuple[str, ...],
    confidence: str,
    sentiment_confirmation_status: str = "Insufficient Sentiment Data",
    macro_sentiment_confirmation: str = "Insufficient Macro Data",
    nlp_confirmation_status: str = "Insufficient NLP Data",
) -> str:
    overlay_text = (
        f"Use {overlay_strategy} as a risk-control overlay; it is not being presented "
        "as a replacement for the fixed growth portfolio."
        if overlay_strategy
        else "No adaptive overlay cleared the current evidence and safety gates."
    )
    sentiment_text = {
        "Confirmed Risk-Off": "Sentiment confirms the quantitative stress signal.",
        "Confirmed Risk-On": "Sentiment confirms the quantitative risk-on signal.",
        "Confirmed Neutral": "Sentiment confirms a neutral quantitative regime.",
        "Quant-Sentiment Disagreement": (
            "Sentiment disagrees with the quantitative regime, so recommendation "
            "confidence is not upgraded."
        ),
    }.get(
        sentiment_confirmation_status,
        "Sentiment coverage is insufficient, so the recommendation remains quantitative.",
    )
    macro_text = {
        "Confirmed Risk-Off": "RBI macro language also confirms quantitative stress.",
        "Confirmed Risk-On": "RBI macro language also confirms the quantitative risk-on state.",
        "Confirmed Neutral": "RBI macro language is consistent with a neutral quantitative state.",
        "Quant-Macro Disagreement": (
            "RBI macro language disagrees with the quantitative regime; this is "
            "reported as commentary and does not change the recommendation."
        ),
    }.get(
        macro_sentiment_confirmation,
        "RBI macro coverage is insufficient, so it does not affect the recommendation.",
    )
    nlp_text = {
        "Confirms Quantitative Stress": (
            "The composite ex-ante NLP monitor confirms quantitative stress."
        ),
        "Confirms Quantitative Risk-On": (
            "The composite ex-ante NLP monitor confirms the quantitative risk-on state."
        ),
        "Quant-NLP Disagreement": (
            "The composite NLP monitor disagrees with the quantitative regime; "
            "this remains commentary and does not change selection."
        ),
    }.get(
        nlp_confirmation_status,
        "Composite NLP coverage is insufficient, so it does not affect the recommendation.",
    )
    return (
        f"For a {profile_name} objective, {main_strategy} is the selected core portfolio. "
        f"{overlay_text} The recommendation is {confidence.lower()} confidence under "
        f"the current scenario assessment ({', '.join(scenarios)}). "
        f"{sentiment_text} {macro_text} {nlp_text}"
    )
