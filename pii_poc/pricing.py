"""Gemini pricing table — USD per 1 million tokens.

Rates below are for the Gemini Flash family we currently use. Update
them when Google changes prices.

Last updated: 2026-11-12
Source:      https://ai.google.dev/gemini-api/docs/pricing
"""

from __future__ import annotations


# Keys: model name as sent to the API.
# Values: {input, cached_input, output} per 1M tokens, in USD.
#
# Gemini 3.8 Flash — introductory rate effective through 2026-12-31.
# Standard rate kicks in on 2027-01-01 (roughly 2x: $1.50 / $7.50 / $0.14).
PRICES: dict[str, dict[str, float]] = {
    "gemini-3.8-flash": {
        "input":        0.75,
        "cached_input": 0.07,
        "output":       3.75,
    },
}

# Fallback when the caller passes a model we don't have in the table.
_DEFAULT_MODEL = "gemini-3.8-flash"


def cost_usd(
    model: str,
    prompt_tokens: int,
    cached_tokens: int,
    output_tokens: int,
) -> float:
    """Compute USD cost for one API call.

    `cached_tokens` is the subset of `prompt_tokens` that was served
    from Gemini's prompt cache (90% cheaper). `output_tokens` already
    includes any thinking/thoughts tokens, which Google bills at the
    output rate.

    Unknown models fall back to `_DEFAULT_MODEL`'s rates so the number
    is approximate rather than missing.
    """
    rates = PRICES.get(model) or PRICES[_DEFAULT_MODEL]
    uncached = max(0, prompt_tokens - cached_tokens)
    return (
        uncached * rates["input"]
        + cached_tokens * rates["cached_input"]
        + output_tokens * rates["output"]
    ) / 1_000_000.0
