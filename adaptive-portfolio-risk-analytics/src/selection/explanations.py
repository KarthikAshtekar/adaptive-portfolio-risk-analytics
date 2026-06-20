"""Plain-language recommendation explanations."""

from __future__ import annotations


def build_recommendation_explanation(
    *,
    profile_name: str,
    main_strategy: str,
    overlay_strategy: str | None,
    scenarios: tuple[str, ...],
    confidence: str,
) -> str:
    overlay_text = (
        f"Use {overlay_strategy} as a risk-control overlay; it is not being presented "
        "as a replacement for the fixed growth portfolio."
        if overlay_strategy
        else "No adaptive overlay cleared the current evidence and safety gates."
    )
    return (
        f"For a {profile_name} objective, {main_strategy} is the selected core portfolio. "
        f"{overlay_text} The recommendation is {confidence.lower()} confidence under "
        f"the current scenario assessment ({', '.join(scenarios)})."
    )

