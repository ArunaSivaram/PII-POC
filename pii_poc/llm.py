"""Gemini wrapper for the ICF-feedback call.

Reads `GEMINI_API_KEY` and `GEMINI_MODEL` from env (set in `.env` or on
the deployment host). The app catches `LLMConfigError` and renders a
clean UI message rather than crashing.

Returns `(SessionReport, Usage)`:
  * `SessionReport` — our stable internal shape (see `pii_poc.report`).
    Populated from Gemini's JSON reply via `from_poc_feedback`. When
    parsing fails, structured fields are empty but `raw_text` still
    carries Gemini's exact output.
  * `Usage` — token counts + USD cost estimate for the call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from google import genai
from google.genai import types

from pii_poc.report import SessionReport, from_poc_feedback


# The prompt asks Gemini for a single JSON object matching the
# SessionReport / CompetencyResult / Evidence dataclasses. Response MIME
# type is also pinned to application/json in the request config.
_PROMPT = (
    "You are reviewing a coaching session against the eight ICF Core "
    "Competencies:\n"
    "  1. Demonstrates Ethical Practice\n"
    "  2. Embodies a Coaching Mindset\n"
    "  3. Establishes and Maintains Agreements\n"
    "  4. Cultivates Trust and Safety\n"
    "  5. Maintains Presence\n"
    "  6. Listens Actively\n"
    "  7. Evokes Awareness\n"
    "  8. Facilitates Client Growth\n\n"
    "Personal details in the transcript have been replaced with "
    "placeholders such as [COACH_NAME], [CLIENT_NAME], [PERSON_1], "
    "[ORG_1]. Treat each as a consistent real person or organisation. "
    "Refer to this session's coach as [COACH_NAME] and the client as "
    "[CLIENT_NAME]. When referencing anyone or anything else from the "
    "transcript, cite them by their exact placeholder (e.g. [PERSON_3], "
    "[ORG_1]). Always keep placeholders exactly as written, including "
    "the square brackets.\n\n"
    "Return a single JSON object with this exact shape and keys — no "
    "prose, no markdown fences, nothing before or after:\n\n"
    "{\n"
    '  "overall_summary": "2-3 sentence overview of the whole session",\n'
    '  "competencies": [\n'
    "    {\n"
    '      "competency": "<one of the 8 competency names above>",\n'
    '      "score_or_level": "<Not evident | Developing | ACC | PCC | MCC>",\n'
    '      "summary": "<one-line observation>",\n'
    '      "strengths": [\n'
    '        {"timestamp": "MM:SS",\n'
    '         "quote": "<transcript line verbatim, including any placeholders>",\n'
    '         "note": "<what this demonstrates>"}\n'
    "      ],\n"
    '      "development_areas": [\n'
    '        {"timestamp": "MM:SS", "quote": "...", "note": "..."}\n'
    "      ],\n"
    '      "pcc_markers_observed": ["<short phrase>", "..."],\n'
    '      "pcc_markers_missed":  ["<short phrase>", "..."]\n'
    "    }\n"
    "  ],\n"
    '  "next_steps": ["<action 1>", "<action 2>", "<action 3>"]\n'
    "}\n\n"
    "Include an entry for all 8 competencies, in the order listed. "
    "Provide 1-2 evidence items per competency combined across "
    "strengths and development_areas. Quote transcript lines verbatim "
    "(keep placeholders intact). Keep summaries and notes concise."
)

# Default per-1M-token prices for Gemini 3.8 Flash (introductory rate
# through 2026-12-31). After that, standard rates are 1.50 / 7.50.
# Override any of these via `.env`:
#   GEMINI_PRICE_INPUT_PER_1M, GEMINI_PRICE_OUTPUT_PER_1M,
#   GEMINI_PRICE_CACHED_INPUT_PER_1M
_DEFAULT_PRICE_INPUT = 0.75
_DEFAULT_PRICE_OUTPUT = 3.75
_DEFAULT_PRICE_CACHED_INPUT = 0.07


class LLMConfigError(Exception):
    """Raised when GEMINI_API_KEY is missing. Surface as a UI message."""


@dataclass
class Usage:
    """Token counts + computed USD cost for one Gemini call.

    `output_tokens` is the visible candidates tokens; `thinking_tokens`
    is the separate "thoughts" count (billed at the output rate). UI
    layers that want to show them combined should sum the two.
    """
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int
    thinking_tokens: int
    total_tokens: int
    cost_usd: float
    model: str


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

    billable_output = candidates + thoughts  # thinking tokens billed at output rate

    price_in = _price("GEMINI_PRICE_INPUT_PER_1M", _DEFAULT_PRICE_INPUT)
    price_out = _price("GEMINI_PRICE_OUTPUT_PER_1M", _DEFAULT_PRICE_OUTPUT)
    price_cached = _price(
        "GEMINI_PRICE_CACHED_INPUT_PER_1M", _DEFAULT_PRICE_CACHED_INPUT
    )

    uncached_input = max(0, prompt - cached)
    cost = (
        uncached_input * price_in
        + cached * price_cached
        + billable_output * price_out
    ) / 1_000_000.0

    return Usage(
        prompt_tokens=prompt,
        cached_tokens=cached,
        output_tokens=candidates,
        thinking_tokens=thoughts,
        total_tokens=total or (prompt + billable_output),
        cost_usd=cost,
        model=model,
    )


# Extra instruction appended to the prompt when the coach has
# explicitly overridden the severe-content block. Tells Gemini not to
# engage with what any [REVIEW_N] placeholder contains.
_SEVERE_OVERRIDE_INSTRUCTION = (
    "This transcript contains one or more [REVIEW_N] placeholders that "
    "mask content the coach has flagged as sensitive (self-harm, "
    "violence, or similar). Do NOT attempt to interpret, speculate "
    "about, or comment on what those placeholders contain. Limit your "
    "feedback strictly to the coach's craft and ICF competencies around "
    "how they navigated those moments."
)


def get_icf_feedback(
    redacted_text: str,
    *,
    allow_severe: bool = False,
) -> tuple[SessionReport, Usage]:
    """Send `redacted_text` to Gemini and return (SessionReport, Usage).

    `allow_severe` is set by the caller when the user has explicitly
    overridden the severe-content block — in that case we append a
    guardrail instruction to the prompt so Gemini stays on coaching
    craft and doesn't try to interpret the masked content.

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

    prompt = _PROMPT
    if allow_severe:
        prompt = prompt + "\n\n" + _SEVERE_OVERRIDE_INSTRUCTION

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=f"{prompt}\n\n--- Transcript ---\n{redacted_text}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    text = response.text or "{}"
    report = from_poc_feedback(text)
    return report, _usage_from(response, model)
