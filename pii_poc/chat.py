"""Multi-turn chat engine for one coaching session.

Shape of the world when this module is used:
  * Tab 1 has produced a `RedactionResult` (redacted transcript + mapping).
  * Tab 2 has produced a `SessionReport` from Gemini.
  * The coach now wants to talk about this session with an AI mentor or
    a peer coach. That's what `ChatSession` is for.

Privacy posture:
  * Everything that leaves this process has placeholders only.
  * Each coach message goes through a two-step redaction BEFORE anything
    reaches Gemini:
        1. Reverse-apply the existing session mapping (so known names
           and identifiers become their original placeholders).
        2. Run the redactor on whatever's left to catch new PII, adding
           new entries to the SAME mapping with continuing numbers.
  * The mapping is mutated in-place on the `ChatContext`, so once a new
    entity is detected it stays rehydrateable in subsequent replies.

Prompt construction:
  * `system_instruction` = `prompts/chat_shared_rules.md` + the chosen
    persona file. Re-read on every call so wording tweaks take effect
    without a restart.
  * The conversation starts with a FIXED first user turn (transcript +
    report) and a FIXED model acknowledgement, so the first two entries
    of `contents` are byte-identical across all calls in the session —
    enough to trigger Gemini's automatic prompt caching on the prefix.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types

from pii_poc.llm import LLMConfigError, Usage
from pii_poc.pricing import cost_usd
from pii_poc.redactor import _PREFIX_OF, redact
from pii_poc.rehydrate import rehydrate
from pii_poc.report import SessionReport, to_prompt_text


# ======================================================================
# Prompts + fixed strings
# ======================================================================

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

Persona = Literal["mentor", "peer"]

_PERSONA_FILES: dict[str, str] = {
    "mentor": "persona_mentor.md",
    "peer":   "persona_peer.md",
}

# Byte-identical ack sitting after the fixed context block, so the
# first two entries of `contents` qualify for prompt caching.
_FIXED_MODEL_ACK = (
    "Understood. I have the transcript and the report in front of me. "
    "What would you like to explore?"
)


def _read_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class ChatContext:
    """Session context — built once after tab 1 + tab 2 have run.

    `mapping` is MUTABLE: the chat engine extends it in place whenever
    a coach message mentions a person or entity we didn't catch during
    the original redaction pass.
    """
    redacted_transcript: str
    report: SessionReport
    mapping: dict[str, str]       # placeholder -> original (shared ref)
    coach_name: str
    client_name: str


@dataclass
class ChatTurn:
    """One Q&A exchange."""
    reply_shown: str      # rehydrated with real names — for the coach's view
    reply_raw: str        # exactly as Gemini returned it, placeholders intact
    message_sent: str     # the redacted user message that went to Gemini
    usage: Usage
    cost_usd: float


# ======================================================================
# Message redaction (two-step, mutates ctx.mapping)
# ======================================================================

def _is_person_placeholder(ph: str) -> bool:
    return ph in ("[COACH_NAME]", "[CLIENT_NAME]") or ph.startswith("[PERSON_")


def _reverse_apply_mapping(text: str, mapping: dict[str, str]) -> str:
    """Replace known `original` values in `text` with their placeholders.

    - Case-insensitive, word-bounded.
    - For PERSON-type placeholders, also matches the first-name token
      alone (so "Rohan" → `[PERSON_3]` even if mapping value is "Rohan Raman").
    - Longer phrases are tried first so "Mei Lin" is replaced before "Mei".
    - Same boundary guards as the redactor's name matcher — won't fire
      inside dotted identifiers like `priya.raman@example.com`.
    """
    pairs: list[tuple[str, str]] = []
    for ph, original in mapping.items():
        original = (original or "").strip()
        if not original:
            continue
        pairs.append((original, ph))
        if _is_person_placeholder(ph):
            tokens = original.split()
            if len(tokens) > 1:
                pairs.append((tokens[0], ph))

    pairs.sort(key=lambda p: -len(p[0]))

    out = text
    for original, ph in pairs:
        pattern = (
            r"(?<![\w@])(?<![a-zA-Z0-9]\.)"
            + re.escape(original)
            + r"(?![\w@])(?!\.[a-zA-Z0-9])"
        )
        out = re.sub(pattern, ph, out, flags=re.IGNORECASE)
    return out


# Matches our placeholder shape: [COACH_NAME], [PERSON_1], [ID_3], …
_PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+(?:_\d+)?\]")

# Pull the prefix letters out of a placeholder, e.g. "[PERSON_1]" -> "PERSON".
_PREFIX_OF_PLACEHOLDER = re.compile(r"\[([A-Z_]+?)(?:_\d+)?\]")


def _max_counters(mapping: dict[str, str]) -> dict[str, int]:
    """For each prefix in the mapping, return the highest N seen so new
    placeholders can continue the sequence."""
    out: dict[str, int] = {}
    for ph in mapping:
        m = re.fullmatch(r"\[([A-Z_]+?)_(\d+)\]", ph)
        if not m:
            continue
        prefix, n = m.group(1), int(m.group(2))
        out[prefix] = max(out.get(prefix, 0), n)
    return out


def _tag_new_pii(text: str, mapping: dict[str, str]) -> str:
    """Run the redactor on `text` and tag any new PII with continuing placeholders.

    Mutates `mapping`. Spans that fall inside existing placeholder tokens
    (GLiNER sometimes tags the contents of `[CLIENT_NAME]`) are ignored.
    """
    # No coach/client names needed — step 1 has already handled them.
    result = redact(text, coach_name="", client_name="", sensitive_mode="mask")

    placeholder_ranges = [
        (m.start(), m.end()) for m in _PLACEHOLDER_RE.finditer(text)
    ]

    def inside_placeholder(start: int, end: int) -> bool:
        return any(start < e and s < end for s, e in placeholder_ranges)

    new_spans = [s for s in result.spans if not inside_placeholder(s.start, s.end)]
    if not new_spans:
        return text

    # Canonical lookup so the same value gets one placeholder, not several.
    canonical: dict[tuple[str, str], str] = {}
    for ph, original in mapping.items():
        pm = _PREFIX_OF_PLACEHOLDER.fullmatch(ph)
        prefix = pm.group(1) if pm else "OTHER"
        canonical[(prefix, (original or "").strip().lower())] = ph

    counters = _max_counters(mapping)
    assignments: dict[int, str] = {}  # span.start -> placeholder

    for s in sorted(new_spans, key=lambda s: s.start):
        prefix = _PREFIX_OF.get(s.entity_type, "OTHER")
        key = (prefix, s.original.strip().lower())
        if key in canonical:
            assignments[s.start] = canonical[key]
            continue
        counters[prefix] = counters.get(prefix, 0) + 1
        ph = f"[{prefix}_{counters[prefix]}]"
        canonical[key] = ph
        mapping[ph] = s.original
        assignments[s.start] = ph

    # Substitute right-to-left so earlier offsets stay valid.
    out = text
    for s in sorted(new_spans, key=lambda s: -s.start):
        out = out[: s.start] + assignments[s.start] + out[s.end :]
    return out


def redact_user_message(message: str, ctx: ChatContext) -> str:
    """Return `message` with PII replaced by placeholders.

    Mutates `ctx.mapping` to add entries for any new entities caught by
    step 2 of the redaction.
    """
    stage1 = _reverse_apply_mapping(message, ctx.mapping)
    stage2 = _tag_new_pii(stage1, ctx.mapping)
    return stage2


# ======================================================================
# Prompt / contents assembly
# ======================================================================

def _build_system_instruction(persona: Persona) -> str:
    """Compose system instruction = shared rules + persona file.

    Both files are re-read on every call so wording tweaks don't need an
    app restart (each read is ~1 KB; negligible).
    """
    if persona not in _PERSONA_FILES:
        raise ValueError(f"Unknown persona: {persona!r}")
    shared = _read_prompt("chat_shared_rules.md")
    persona_text = _read_prompt(_PERSONA_FILES[persona])
    return shared + "\n\n" + persona_text


def _build_fixed_context_block(ctx: ChatContext) -> str:
    """The byte-identical first user turn: transcript + report.

    This block is identical across every call in the session so Gemini's
    automatic prompt caching catches it.
    """
    return (
        "SESSION TRANSCRIPT (redacted):\n"
        + ctx.redacted_transcript
        + "\n\nFEEDBACK REPORT:\n"
        + to_prompt_text(ctx.report)
    )


# ======================================================================
# The chat session
# ======================================================================

class ChatSession:
    """One ongoing chat about one coaching session.

    One instance per browser session. Tracks persona, message history,
    usage totals, and the per-session question cap.
    """

    def __init__(self, context: ChatContext, persona: Persona = "mentor"):
        if persona not in _PERSONA_FILES:
            raise ValueError(f"Unknown persona: {persona!r}")
        self.context = context
        self.persona: Persona = persona
        self.history: list[dict[str, str]] = []  # [{"role": ..., "content": ...}]
        try:
            self.questions_cap = max(1, int(os.environ.get("CHAT_MAX_QUESTIONS", "50")))
        except ValueError:
            self.questions_cap = 50
        self.questions_used = 0
        self.total_cost_usd = 0.0
        self.total_tokens = 0

    # ---- public state helpers ----
    @property
    def questions_left(self) -> int:
        return max(0, self.questions_cap - self.questions_used)

    def set_persona(self, persona: Persona) -> None:
        """Switch persona. Clears the chat history; usage totals persist.

        The cap counts per session regardless of persona, so a coach
        can't dodge the cap by hopping between personas.
        """
        if persona not in _PERSONA_FILES:
            raise ValueError(f"Unknown persona: {persona!r}")
        if persona == self.persona:
            return
        self.persona = persona
        self.history.clear()

    def reset(self) -> None:
        """Full reset — clears history AND usage counters."""
        self.history.clear()
        self.questions_used = 0
        self.total_cost_usd = 0.0
        self.total_tokens = 0

    # ---- core operation ----
    def ask(self, user_message: str) -> ChatTurn:
        """Send one user turn. Returns a ChatTurn describing what happened.

        On cap exhaustion the method returns a friendly message WITHOUT
        calling Gemini or mutating the mapping. Only successful calls
        count against the cap.
        """
        if self.questions_left == 0:
            friendly = (
                f"We've reached the chat limit for this session "
                f"({self.questions_cap} questions). Reset the chat to continue."
            )
            return ChatTurn(
                reply_shown=friendly,
                reply_raw=friendly,
                message_sent="",
                usage=self._zero_usage(),
                cost_usd=0.0,
            )

        # Nothing leaves this process until the two-step redaction finishes.
        redacted_msg = redact_user_message(user_message, self.context)

        # contents = [fixed block, fixed ack, history..., new msg]
        contents: list[dict] = [
            {"role": "user",  "parts": [{"text": _build_fixed_context_block(self.context)}]},
            {"role": "model", "parts": [{"text": _FIXED_MODEL_ACK}]},
        ]
        for m in self.history:
            api_role = "user" if m["role"] == "user" else "model"
            contents.append({"role": api_role, "parts": [{"text": m["content"]}]})
        contents.append({"role": "user", "parts": [{"text": redacted_msg}]})

        system_instruction = _build_system_instruction(self.persona)
        model = self._model()

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigError(
                "GEMINI_API_KEY is not set. Add it to .env or the host env."
            )

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
            ),
        )
        reply_raw = response.text or "(The model returned an empty response.)"

        usage = self._usage_from(response, model)

        # Rehydrate for display (placeholders -> real names). The reply_raw
        # keeps placeholders so the UI can show both views.
        reply_shown = rehydrate(reply_raw, self.context.mapping)

        # Only count & persist on successful completion.
        self.history.append({"role": "user",      "content": redacted_msg})
        self.history.append({"role": "assistant", "content": reply_raw})
        self.questions_used += 1
        self.total_cost_usd += usage.cost_usd
        self.total_tokens += usage.total_tokens

        return ChatTurn(
            reply_shown=reply_shown,
            reply_raw=reply_raw,
            message_sent=redacted_msg,
            usage=usage,
            cost_usd=usage.cost_usd,
        )

    # ---- plumbing ----
    def _model(self) -> str:
        return (
            os.environ.get("GEMINI_CHAT_MODEL", "").strip()
            or os.environ.get("GEMINI_MODEL", "").strip()
            or "gemini-3.8-flash"
        )

    def _zero_usage(self) -> Usage:
        return Usage(
            prompt_tokens=0, cached_tokens=0, output_tokens=0,
            thinking_tokens=0, total_tokens=0, cost_usd=0.0,
            model=self._model(),
        )

    def _usage_from(self, response, model: str) -> Usage:
        meta = response.usage_metadata
        prompt = int(getattr(meta, "prompt_token_count", 0) or 0)
        candidates = int(getattr(meta, "candidates_token_count", 0) or 0)
        thoughts = int(getattr(meta, "thoughts_token_count", 0) or 0)
        cached = int(getattr(meta, "cached_content_token_count", 0) or 0)
        total = int(getattr(meta, "total_token_count", 0) or 0)
        billable_output = candidates + thoughts  # thinking billed at output rate
        return Usage(
            prompt_tokens=prompt,
            cached_tokens=cached,
            output_tokens=candidates,
            thinking_tokens=thoughts,
            total_tokens=total or (prompt + billable_output),
            cost_usd=cost_usd(
                model=model,
                prompt_tokens=prompt,
                cached_tokens=cached,
                output_tokens=billable_output,
            ),
            model=model,
        )


# ======================================================================
# Starter questions
# ======================================================================

# Rough ordering from weakest to strongest on the ICF scale.
_LEVEL_ORDER = {
    "not evident": 0,
    "developing":  1,
    "acc":         2,
    "pcc":         3,
    "mcc":         4,
}

# The eight fixed-option starters from the spec. We pick the first three
# as "most relevant" by default — the PM can reorder the list to steer
# which ones surface.
_FIXED_STARTERS: list[str] = [
    "What's the one thing I should work on next?",
    "Where did I miss a chance to go deeper?",
    "Was I leading the client anywhere?",
    "Did the client reach their own insight, or did I hand it to them?",
    "How well did I set up the agreement and check in at the end?",
    "What could I have asked instead of my weakest question?",
    "Which PCC markers did I clearly show?",
    "What should I practise before my next session?",
]


def _level_rank(level: str) -> int:
    return _LEVEL_ORDER.get((level or "").strip().lower(), 99)


def _first_dev_timestamp(report: SessionReport) -> str:
    """Return the first non-empty timestamp from any development-area
    evidence, falling back to any strength evidence timestamp."""
    for c in report.competencies:
        for e in c.development_areas:
            if e.timestamp:
                return e.timestamp
    for c in report.competencies:
        for e in c.strengths:
            if e.timestamp:
                return e.timestamp
    return ""


def _dynamic_starters(report: SessionReport) -> list[str]:
    if not report or not report.competencies:
        return []

    out: list[str] = []

    lowest = min(report.competencies, key=lambda c: _level_rank(c.score_or_level))
    out.append(
        f"Why was my lowest-rated competency ({lowest.competency}) "
        f"rated {lowest.score_or_level}?"
    )

    strongest = max(report.competencies, key=lambda c: _level_rank(c.score_or_level))
    out.append(f"What did I do well in {strongest.competency}?")

    timestamp = _first_dev_timestamp(report)
    if timestamp:
        out.append(f"How did I do at {timestamp}?")

    return out


def suggest_starters(report: SessionReport) -> list[str]:
    """Return 6 starter questions — 3 dynamic from the report, 3 fixed.

    Starters are plain text; real names are allowed. They are intended
    to be fed into the UI and then through `ChatSession.ask()` like any
    other coach message, so the standard message-redaction step handles
    anything that might carry PII.
    """
    return _dynamic_starters(report) + list(_FIXED_STARTERS[:3])
