"""Transcript -> redacted text + placeholder map.

Public entry point: `redact(text, coach_name, client_name)` -> RedactionResult.

Guarantees:
- Same original value always maps to the same placeholder in one call.
- Coach / client names (if provided) always collapse to [COACH] / [CLIENT],
  including first-name-only or surname-only mentions, case-insensitive.
- City / street level locations are tagged; bare country names ('India',
  'Singapore', etc.) are deliberately left alone.
- Never prints the transcript or the mapping.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from pii_poc.recognizers import SUPPORTED_ENTITIES, build_analyzer
from pii_poc.severe import detect_severe_spans


# Country names kept unredacted even when LOCATION fires on them.
# Lowercased on comparison; trailing punctuation is stripped.
_COUNTRY_WHITELIST = {
    "india", "singapore",
    "usa", "u.s.", "us", "united states", "united states of america", "america",
    "uk", "u.k.", "united kingdom", "britain", "great britain", "england",
    "australia", "new zealand", "canada", "mexico",
    "china", "japan", "south korea", "korea", "taiwan", "hong kong",
    "indonesia", "malaysia", "thailand", "vietnam", "philippines",
    "germany", "france", "spain", "italy", "netherlands", "belgium",
    "sweden", "norway", "denmark", "finland", "switzerland", "ireland",
    "brazil", "argentina", "chile", "colombia",
    "south africa", "nigeria", "egypt", "kenya", "uae", "saudi arabia",
    "pakistan", "bangladesh", "sri lanka", "nepal",
}

# Canonical entity type -> placeholder prefix.
# - All ID-type entities (national IDs, bank account, credit card) share
#   the "ID" prefix so IDs get one tidy numbering sequence.
# - Family-member *names* share the "PERSON" prefix so readers see
#   [PERSON_1] rather than a separate [FAMILY_1] category.
_PREFIX_OF = {
    # Core PII
    "PERSON":        "PERSON",
    "ORGANIZATION":  "ORG",
    "LOCATION":      "LOCATION",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER":  "PHONE",
    "DATE_OF_BIRTH": "DOB",
    "FAMILY_MEMBER": "PERSON",  # merged with PERSON counter
    # Identifier-type PII
    "SG_NRIC":      "ID",
    "IN_AADHAAR":   "ID",
    "IN_PAN":       "ID",
    "BANK_ACCOUNT": "ID",
    "CREDIT_CARD":  "ID",
    # Sensitive topics (only replaced in sensitive_mode="mask")
    "HEALTH":         "HEALTH",
    "FINANCE_AMOUNT": "AMOUNT",
    "BELIEF":         "BELIEF",
    # Severe content — always masked regardless of sensitive_mode.
    # The UI uses the count to decide whether to block the Gemini call.
    "SEVERE_RISK":    "REVIEW",
}

# Which entity types are "sensitive content" (as opposed to identifying PII).
# These are kept in `spans` for the UI to highlight, but in sensitive_mode
# "flag" (default) they are NOT replaced in the redacted text. Note:
# SEVERE_RISK is deliberately NOT in this set — severe spans are always
# masked so the exact disclosure never reaches the LLM, even on override.
SENSITIVE_TOPIC_ENTITIES = {"HEALTH", "FINANCE_AMOUNT", "BELIEF"}


# Relationship / role words that GLiNER occasionally mislabels as PERSON
# (e.g. "mum", "my sister", "her son", "boss"). When a PERSON or
# FAMILY_MEMBER span is nothing more than one of these — optionally with a
# leading possessive pronoun or a trailing "'s" — we drop it so the word
# stays in the transcript. Real names ("Arjun", "Maya") are not in this
# set and continue to be masked.
_RELATIONSHIP_WORDS = {
    # Direct family
    "mum", "mom", "mother", "ma", "mama", "mummy", "mommy",
    "dad", "father", "pa", "papa", "daddy",
    "parent", "parents",
    "brother", "sister", "sibling", "bro", "sis",
    "son", "daughter", "child", "children", "kid", "kids",
    "husband", "wife", "spouse", "partner",
    # Extended family
    "uncle", "aunt", "aunty", "auntie",
    "cousin", "niece", "nephew",
    "grandmother", "grandfather", "grandma", "grandpa",
    "gran", "granddad", "nan", "nana",
    # Work / peer roles the user flagged as "stay as they are"
    "boss", "manager", "colleague", "coworker",
}

_POSSESSIVES = {"my", "your", "his", "her", "their", "our"}

# Transcript turn-marker tokens. GLiNER occasionally tags the literal words
# "COACH" and "CLIENT" from lines like `[12:34] COACH: ...` as PERSON — they
# are structure, not names, and masking them wrecks the transcript format.
_TURN_MARKER_TOKENS = {"coach", "client", "interviewer", "interviewee", "speaker"}


def _is_turn_marker_word(snippet: str) -> bool:
    """True if `snippet` is just a transcript turn-marker label."""
    cleaned = snippet.strip().lower().strip(".,!?;:'\"-")
    return cleaned in _TURN_MARKER_TOKENS


def _is_pure_relationship_word(snippet: str) -> bool:
    """True if `snippet` is just a relationship word.

    Accepts a leading possessive pronoun ("my sister") and a trailing
    possessive "'s" ("my brother's").
    """
    cleaned = snippet.strip().lower().strip(".,!?;:'\"-")
    if cleaned.endswith("'s"):
        cleaned = cleaned[:-2]
    tokens = cleaned.split()
    while tokens and tokens[0] in _POSSESSIVES:
        tokens.pop(0)
    return len(tokens) == 1 and tokens[0] in _RELATIONSHIP_WORDS


@dataclass
class DetectedSpan:
    """One PII detection on the original text."""
    start: int
    end: int
    entity_type: str  # PERSON / ORGANIZATION / ... / COACH / CLIENT
    original: str


@dataclass
class RedactionResult:
    """Everything the UI and the rehydrator need from a redaction pass."""
    redacted_text: str
    mapping: dict[str, str]        # placeholder -> original (first occurrence)
    counts: dict[str, int]         # entity_type -> number of spans
    seconds: float
    spans: list[DetectedSpan] = field(default_factory=list)


# Module-level analyzer cache. Built lazily on first call so importing this
# module (e.g. in tests, in the UI startup banner) is cheap. Rebuilding it
# is slow because GLiNER + spaCy load heavy weights.
_ANALYZER = None


def _get_analyzer():
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = build_analyzer()
    return _ANALYZER


def warmup() -> None:
    """Force the analyzer to load now.

    Call this at app startup (e.g. in `app.py`) so the first user click
    doesn't pay the ~10s spaCy + GLiNER load cost. On Hugging Face
    Spaces this means the "Building" phase covers the model download
    instead of the user's first interaction.
    """
    _get_analyzer()


def _name_spans(text: str, name: str, label: str) -> list[DetectedSpan]:
    """Find every mention of `name`, case-insensitive.

    Matches the full name AND each individual token (so 'Priya Patel'
    matches 'Priya' and 'Patel' on their own too). Short tokens (< 2
    chars) are skipped so initials don't match everywhere.
    """
    name = name.strip()
    if not name:
        return []
    phrases = [name]
    tokens = [t for t in name.split() if len(t) >= 2]
    if len(tokens) > 1:
        phrases.extend(tokens)

    # Boundary rules:
    #   - Not preceded/followed by a word char or '@' (ordinary word boundary
    #     plus email-address protection).
    #   - Not preceded by "<letter/digit>." and not followed by ".<letter/digit>"
    #     — this stops us tagging `priya` or `raman` inside dotted identifiers
    #     such as `priya.raman@example.com`, which would otherwise leak the
    #     domain by displacing the EMAIL_ADDRESS span.
    found: list[DetectedSpan] = []
    for phrase in phrases:
        pattern = (
            r"(?<![\w@])(?<![a-zA-Z0-9]\.)"
            + re.escape(phrase)
            + r"(?![\w@])(?!\.[a-zA-Z0-9])"
        )
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            found.append(
                DetectedSpan(
                    start=m.start(),
                    end=m.end(),
                    entity_type=label,
                    original=m.group(0),
                )
            )
    return found


def _is_country(span: DetectedSpan, text: str) -> bool:
    """True if a LOCATION span is just a country name (keep it unredacted)."""
    if span.entity_type != "LOCATION":
        return False
    snippet = text[span.start : span.end].strip().strip(".,;:!?")
    return snippet.lower() in _COUNTRY_WHITELIST


def _is_bare_relationship(span: DetectedSpan, text: str) -> bool:
    """True if a PERSON/FAMILY_MEMBER span is only a relationship word."""
    if span.entity_type not in ("PERSON", "FAMILY_MEMBER"):
        return False
    return _is_pure_relationship_word(text[span.start : span.end])


def _is_turn_marker(span: DetectedSpan, text: str) -> bool:
    """True if a PERSON span is a transcript turn-marker label (COACH/CLIENT)."""
    if span.entity_type != "PERSON":
        return False
    return _is_turn_marker_word(text[span.start : span.end])


def _resolve_overlaps(spans: list[DetectedSpan]) -> list[DetectedSpan]:
    """Deduplicate overlapping spans.

    Strategy:
      - COACH / CLIENT spans are kept first (they always win over analyzer
        detections — they are the ground-truth labels we were given).
      - Among remaining analyzer spans, the first one we see for an offset
        range wins. Presidio has already resolved intra-recognizer
        duplicates for us; this just protects against coach/client-vs-model
        collisions and model-vs-pattern collisions.
    """
    # Sort so longer spans come first within the same start offset -
    # that way a full-name match beats a single-token match.
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    kept: list[DetectedSpan] = []

    # Pass 1: COACH/CLIENT get right of way.
    for span in ordered:
        if span.entity_type not in ("COACH", "CLIENT"):
            continue
        if not any(span.start < e.end and e.start < span.end for e in kept):
            kept.append(span)

    # Pass 2: everything else, as long as it doesn't overlap a kept span.
    for span in ordered:
        if span.entity_type in ("COACH", "CLIENT"):
            continue
        if not any(span.start < e.end and e.start < span.end for e in kept):
            kept.append(span)

    return sorted(kept, key=lambda s: s.start)


def redact(
    text: str,
    coach_name: str = "",
    client_name: str = "",
    sensitive_mode: str = "flag",
) -> RedactionResult:
    """Redact PII in `text` and return a RedactionResult.

    Parameters
    ----------
    sensitive_mode : "flag" | "mask"
        How to treat sensitive-topic spans (HEALTH, FINANCE_AMOUNT, BELIEF).
        - "flag" (default): leave the original text intact so the LLM can
          analyse the coaching content, but still return the spans in
          `spans` so the UI can highlight them in a reviewer colour.
        - "mask": also redact these with neutral tags ([HEALTH_1],
          [AMOUNT_1], [BELIEF_1]).

    `text` is never printed or logged. Only entity *counts* are safe to log.
    """
    if sensitive_mode not in ("flag", "mask"):
        raise ValueError(
            f"sensitive_mode must be 'flag' or 'mask', got {sensitive_mode!r}"
        )

    t0 = time.perf_counter()
    analyzer = _get_analyzer()

    # 1) Run Presidio. Char offsets in results are against the full `text`,
    #    thanks to GLiNER's token-based chunker handling the position math.
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=SUPPORTED_ENTITIES,
    )
    spans: list[DetectedSpan] = [
        DetectedSpan(
            start=r.start,
            end=r.end,
            entity_type=r.entity_type,
            original=text[r.start : r.end],
        )
        for r in results
    ]

    # 1b) Add SEVERE_RISK spans from the keyword/phrase matcher. These
    #     run on the original text so they aren't affected by PII masking
    #     and are always masked themselves (see SENSITIVE_TOPIC_ENTITIES
    #     comment at the top of this file).
    for s in detect_severe_spans(text):
        spans.append(
            DetectedSpan(
                start=s.start,
                end=s.end,
                entity_type="SEVERE_RISK",
                original=s.phrase,
            )
        )

    # 2) Drop country-level LOCATIONs.
    spans = [s for s in spans if not _is_country(s, text)]

    # 2b) Drop PERSON / FAMILY_MEMBER spans that are only a relationship
    #     word (GLiNER sometimes tags "mum", "my sister", "boss" at
    #     moderate confidence). Real names survive this filter.
    spans = [s for s in spans if not _is_bare_relationship(s, text)]

    # 2c) Drop PERSON spans that are only a transcript turn-marker label
    #     ("COACH" / "CLIENT" from lines like `[12:34] COACH: ...`).
    spans = [s for s in spans if not _is_turn_marker(s, text)]

    # 3) Layer in coach/client ground-truth spans.
    spans.extend(_name_spans(text, coach_name, "COACH"))
    spans.extend(_name_spans(text, client_name, "CLIENT"))

    # 4) Resolve overlaps.
    spans = _resolve_overlaps(spans)

    # 5) Decide which spans get rewritten vs left in place.
    #    Sensitive-topic spans stay in `spans` either way so the UI can
    #    highlight them; we only skip them during substitution when flagging.
    if sensitive_mode == "flag":
        to_mask = [s for s in spans if s.entity_type not in SENSITIVE_TOPIC_ENTITIES]
    else:
        to_mask = list(spans)
    to_mask_ids = {id(s) for s in to_mask}

    # 6) Assign placeholders. Numbering is by first appearance in the text.
    mapping: dict[str, str] = {}
    counts: dict[str, int] = defaultdict(int)
    counters: dict[str, int] = defaultdict(int)
    seen: dict[tuple[str, str], str] = {}  # (prefix, normalized_original) -> placeholder

    def placeholder_for(span: DetectedSpan) -> str:
        if span.entity_type == "COACH":
            return "[COACH_NAME]"
        if span.entity_type == "CLIENT":
            return "[CLIENT_NAME]"
        prefix = _PREFIX_OF.get(span.entity_type, "OTHER")
        key = (prefix, span.original.strip().lower())
        if key in seen:
            return seen[key]
        counters[prefix] += 1
        tag = f"[{prefix}_{counters[prefix]}]"
        seen[key] = tag
        mapping[tag] = span.original
        return tag

    # Walk all spans in text order so counts reflect everything the UI
    # might show; populate placeholders only for spans that will be masked.
    for span in sorted(spans, key=lambda s: s.start):
        counts[span.entity_type] += 1
        if id(span) in to_mask_ids:
            placeholder_for(span)

    # Record coach/client in mapping too (for the rehydrator). Only if we
    # actually masked something that way.
    if any(s.entity_type == "COACH" for s in spans):
        mapping["[COACH_NAME]"] = coach_name
    if any(s.entity_type == "CLIENT" for s in spans):
        mapping["[CLIENT_NAME]"] = client_name

    # 7) Apply replacements right-to-left for the masked spans only.
    out = text
    for span in sorted(to_mask, key=lambda s: -s.start):
        tag = (
            "[COACH_NAME]" if span.entity_type == "COACH"
            else "[CLIENT_NAME]" if span.entity_type == "CLIENT"
            else seen[(_PREFIX_OF.get(span.entity_type, "OTHER"),
                       span.original.strip().lower())]
        )
        out = out[: span.start] + tag + out[span.end :]

    return RedactionResult(
        redacted_text=out,
        mapping=mapping,
        counts=dict(counts),
        seconds=time.perf_counter() - t0,
        spans=sorted(spans, key=lambda s: s.start),
    )
