"""Thin Gemini wrapper for the ICF-feedback call.

Reads `GEMINI_API_KEY` and `GEMINI_MODEL` from env (set in `.env`). The
app catches `LLMConfigError` and renders a clean message in the UI
rather than letting the API crash take the server down.

Returns `(text, usage)` where `usage` is a small dataclass with token
counts and a cost estimate in USD.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from google import genai


# The user-provided prompt, kept verbatim so it's easy to tweak later.
_PROMPT = (
    "You are reviewing a coaching session against ICF Core Competencies. "
    "Personal details have been replaced with placeholders such as "
    "[COACH_NAME], [CLIENT_NAME], [PERSON_1], [ORG_1]; treat each as a "
    "consistent real person or organisation. "
    "Refer to this session's coach as [COACH_NAME] and this session's "
    "client as [CLIENT_NAME] throughout your feedback — do not write "
    "'the coach' or 'the client' when you mean these specific people. "
    "When referencing anyone or anything else from the transcript, cite "
    "them by their exact placeholder (e.g. [PERSON_3], [ORG_1]), "
    "including the square brackets. "
    "Always keep placeholders exactly as written. "
    "Give 3 concise feedback points for [COACH_NAME], each naming the "
    "ICF competency and quoting one line of evidence from the transcript "
    "(copy the line verbatim, including any placeholders)."
)

# Default per-1M-token prices for Gemini 3.8 Flash (introductory rate
# effective until 2026-12-31 — per Google's pricing page). After that date
# the standard rate is 1.50 / 7.50. Override any of these via `.env`:
#   GEMINI_PRICE_INPUT_PER_1M, GEMINI_PRICE_OUTPUT_PER_1M,
#   GEMINI_PRICE_CACHED_INPUT_PER_1M
_DEFAULT_PRICE_INPUT = 0.75
_DEFAULT_PRICE_OUTPUT = 3.75
_DEFAULT_PRICE_CACHED_INPUT = 0.07


class LLMConfigError(Exception):
    """Raised when GEMINI_API_KEY is missing. Surface as a UI message."""


@dataclass
class Usage:
    """Token counts + computed USD cost for one Gemini call."""
    prompt_tokens: int       # tokens in the request
    cached_tokens: int       # subset of prompt tokens billed at the cache rate
    output_tokens: int       # tokens in the reply (incl. hidden thinking tokens)
    total_tokens: int        # prompt + output, as Gemini reports it
    cost_usd: float          # USD at the configured per-1M-token rates
    model: str               # the model name used


def _price(env_var: str, default: float) -> float:
    try:
        return float(os.environ.get(env_var, "") or default)
    except ValueError:
        return default


def _usage_from(response, model: str) -> Usage:
    meta = response.usage_metadata
    prompt = int(getattr(meta, "prompt_token_count", 0) or 0)
    candidates = int(getattr(meta, "candidates_token_count", 0) or 0)
    thoughts = int(getattr(meta, "thoughts_token_count", 0) or 0)
    cached = int(getattr(meta, "cached_content_token_count", 0) or 0)
    total = int(getattr(meta, "total_token_count", 0) or 0)

    output = candidates + thoughts  # thinking tokens are billed at the output rate

    price_in = _price("GEMINI_PRICE_INPUT_PER_1M", _DEFAULT_PRICE_INPUT)
    price_out = _price("GEMINI_PRICE_OUTPUT_PER_1M", _DEFAULT_PRICE_OUTPUT)
    price_cached = _price("GEMINI_PRICE_CACHED_INPUT_PER_1M", _DEFAULT_PRICE_CACHED_INPUT)

    uncached_input = max(0, prompt - cached)
    cost = (
        uncached_input * price_in
        + cached * price_cached
        + output * price_out
    ) / 1_000_000.0

    return Usage(
        prompt_tokens=prompt,
        cached_tokens=cached,
        output_tokens=output,
        total_tokens=total or (prompt + output),
        cost_usd=cost,
        model=model,
    )


def get_icf_feedback(redacted_text: str) -> tuple[str, Usage]:
    """Send `redacted_text` to Gemini and return (reply_text, usage).

    Raises `LLMConfigError` if the API key is missing. Other exceptions
    (network, quota, bad model name, etc.) propagate so the caller can
    render them.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise LLMConfigError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your Gemini key."
        )

    model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=f"{_PROMPT}\n\n--- Transcript ---\n{redacted_text}",
    )
    text = response.text or "(The model returned an empty response.)"
    return text, _usage_from(response, model)
