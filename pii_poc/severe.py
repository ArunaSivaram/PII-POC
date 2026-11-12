"""Severe-content detection — keyword/phrase matching for topics that
need specialist attention (self-harm, interpersonal violence, child
harm, substance crisis).

This is NOT a clinical triage. It's a routing signal: "this transcript
references topics that warrant a human reviewer before any AI runs over
them." Over-detection is acceptable — the UI offers a block-with-override
pattern. Under-detection is not — anything it catches is handled with
care.

Phrases live in `prompts/severe_phrases.txt` so the PM can tune them
without touching code. Format: `## category` headers; one phrase per
line. Matching is case-insensitive and word-bounded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_PHRASES_FILE = Path(__file__).resolve().parent.parent / "prompts" / "severe_phrases.txt"


@dataclass
class SevereSpan:
    """One match for a severe-content phrase."""
    start: int
    end: int
    phrase: str      # the exact substring from the input (preserves case)
    category: str    # e.g. "self-harm", "interpersonal-violence"


def _load_phrases() -> list[tuple[str, str]]:
    """Read `prompts/severe_phrases.txt` and return (phrase, category) pairs.

    Longest phrases come first so they're tried before their shorter
    substrings (avoids "hit me" stealing a match from "hits me" etc.).
    """
    if not _PHRASES_FILE.exists():
        return []

    pairs: list[tuple[str, str]] = []
    current_category: str | None = None

    for raw in _PHRASES_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("##"):
            current_category = line.lstrip("#").strip().lower()
            continue
        if line.startswith("#"):
            continue  # ordinary comment line
        if current_category:
            pairs.append((line, current_category))

    pairs.sort(key=lambda p: -len(p[0]))
    return pairs


# Cache phrases at import time; the file is tiny and this keeps
# per-call cost at zero. If you tune the phrases file in the UI,
# restart the app to pick them up.
_PHRASE_PAIRS: list[tuple[str, str]] = _load_phrases()


def detect_severe_spans(text: str) -> list[SevereSpan]:
    """Find every severe-content phrase match in `text`.

    - Case-insensitive, word-bounded on both sides.
    - Longer phrases win over shorter overlapping ones (so "beaten up"
      beats "beaten" at the same offset).
    """
    if not text:
        return []

    occupied: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(start < e and s < end for s, e in occupied)

    hits: list[SevereSpan] = []
    for phrase, category in _PHRASE_PAIRS:
        # Word boundaries on both sides so "rape" doesn't match in "grape".
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            if _overlaps(m.start(), m.end()):
                continue
            occupied.append((m.start(), m.end()))
            hits.append(
                SevereSpan(
                    start=m.start(),
                    end=m.end(),
                    phrase=m.group(0),
                    category=category,
                )
            )

    return sorted(hits, key=lambda s: s.start)


def has_severe(text: str) -> bool:
    """Quick True/False check — faster than returning spans if that's all you need."""
    if not text or not _PHRASE_PAIRS:
        return False
    for phrase, _ in _PHRASE_PAIRS:
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False
