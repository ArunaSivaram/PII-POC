"""Swap placeholders back to their originals, tolerating LLM reformatting.

Placeholder forms accepted:

    [PERSON_1]       [Person_1]       [PERSON 1]       (bracketed)
    PERSON_1         Person_1                           (bare, word-bounded)

Plain words such as "coach", "client", "COACH:" or "the client's goal" are
**never** rehydrated — the bare form requires an underscore-delimited
suffix, so an ordinary word can't accidentally match.

Placeholders not in `mapping` are left untouched.
"""

from __future__ import annotations

import re


def _variants_regex(placeholder: str) -> str:
    """Build a case-insensitive regex that matches only placeholder-shaped tokens.

    A placeholder like `[COACH_NAME]` is matched in two forms:

      1. Bracketed — `[COACH_NAME]`, `[Coach_Name]`, `[COACH NAME]`, etc.
         Case flexible (via re.IGNORECASE at the call site); the separator
         inside the brackets may be an underscore or whitespace.
      2. Bare, word-bounded — `COACH_NAME` or `Coach_Name`.
         Underscore separator **only** (no space form for the bare case)
         so plain prose like "coach name" never matches.

    The bare form requires `<prefix>_<suffix>` structure, which rules out
    plain words like "coach" or speaker labels like "COACH:".
    """
    core = placeholder.strip("[]")
    parts = core.rsplit("_", 1)
    if len(parts) != 2:
        # Defensive fallback: a placeholder without an underscore suffix is
        # only rehydrated when brackets are present. (In practice our
        # placeholders always have an underscore suffix post-rename, so this
        # branch is unreachable with well-formed mappings.)
        return r"\[" + re.escape(core) + r"\]"

    prefix, suffix = parts
    p = re.escape(prefix)
    s = re.escape(suffix)
    bracketed = r"\[" + p + r"[_\s]" + s + r"\]"
    bare      = r"\b" + p + r"_" + s + r"\b"
    return r"(?:" + bracketed + r"|" + bare + r")"


def rehydrate(text: str, mapping: dict[str, str]) -> str:
    """Replace placeholders in `text` with their originals from `mapping`.

    Only placeholder-shaped tokens are replaced (see module docstring for
    the exact forms). Unknown placeholders are left untouched. Replacement
    goes through a function so that `\\1`-style back-references in the
    original value are treated as literal text.
    """
    if not mapping:
        return text

    # Sort placeholders by length descending so [PERSON_10] is matched
    # before [PERSON_1] when both exist. The `\b` boundary already blocks
    # [PERSON_1] from eating [PERSON_10], but ordering gives belt-and-braces
    # safety.
    items = sorted(mapping.items(), key=lambda kv: -len(kv[0]))

    for placeholder, original in items:
        pattern = _variants_regex(placeholder)
        text = re.sub(
            pattern,
            lambda _m, o=original: o,
            text,
            flags=re.IGNORECASE,
        )
    return text
