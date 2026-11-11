"""Custom Presidio recognizers + AnalyzerEngine builder for the PoC.

Entity types produced here (canonical internal names):
    PERSON, ORGANIZATION, LOCATION, EMAIL, PHONE_NUMBER,
    DATE_OF_BIRTH, HEALTH_CONDITION,
    SG_NRIC, IN_AADHAAR, IN_PAN

The `redactor` module maps these onto the user-facing placeholder prefixes
(PERSON, ORG, LOCATION, EMAIL, PHONE, DOB, HEALTH, ID).
"""

from __future__ import annotations

import re
from typing import Optional

from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
)
from presidio_analyzer.chunkers import TokenizerBasedTextChunker
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import GLiNERRecognizer


class ScopedGLiNERRecognizer(GLiNERRecognizer):
    """GLiNERRecognizer that refuses to invent labels on the fly.

    The AnalyzerEngine passes every user-requested entity to every recognizer.
    The default GLiNER recognizer treats any unmapped entity name as a new
    zero-shot label prompt — so asking it for e.g. "SG_NRIC" makes it
    hallucinate matches. We restrict `entities` to this recognizer's own
    `supported_entities` before delegating.
    """

    def analyze(self, text, entities, nlp_artifacts=None):
        scoped = [e for e in entities if e in self.supported_entities]
        if not scoped:
            return []
        return super().analyze(
            text=text, entities=scoped, nlp_artifacts=nlp_artifacts
        )


# ======================================================================
# Singapore NRIC / FIN checksum
# ======================================================================
# Public algorithm. Weights apply to the 7 middle digits. An offset is
# added by prefix, the result mod 11 is looked up in a letter table that
# depends on the prefix family (S/T, F/G, or M).

_NRIC_WEIGHTS = (2, 7, 6, 5, 4, 3, 2)
_NRIC_OFFSET = {"S": 0, "T": 4, "F": 0, "G": 4, "M": 3}
_NRIC_LOOKUP_ST = "JZIHGFEDCBA"
_NRIC_LOOKUP_FG = "XWUTRQPNMLK"
_NRIC_LOOKUP_M = "KLJNPQRTUWX"  # reversed F/G letters; M-series (2022+)


def _nric_checksum_valid(nric: str) -> bool:
    """True iff the final letter of `nric` matches the computed checksum."""
    if not re.fullmatch(r"[STFGM]\d{7}[A-Z]", nric):
        return False
    prefix = nric[0]
    digits = [int(c) for c in nric[1:8]]
    given = nric[8]
    weighted = sum(d * w for d, w in zip(digits, _NRIC_WEIGHTS))
    remainder = (weighted + _NRIC_OFFSET[prefix]) % 11
    if prefix in ("S", "T"):
        expected = _NRIC_LOOKUP_ST[remainder]
    elif prefix in ("F", "G"):
        expected = _NRIC_LOOKUP_FG[remainder]
    else:  # M
        expected = _NRIC_LOOKUP_M[remainder]
    return given == expected


class SingaporeNricRecognizer(PatternRecognizer):
    """SG NRIC / FIN: letter + 7 digits + letter, final letter is a checksum."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="SG_NRIC",
            patterns=[Pattern("NRIC/FIN", r"\b[STFGM]\d{7}[A-Z]\b", 0.4)],
            context=["NRIC", "FIN", "identification", "IC", "identity card"],
        )

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        return _nric_checksum_valid(pattern_text)


# ======================================================================
# India Aadhaar (Verhoeff) and PAN
# ======================================================================
# Verhoeff is a decimal checksum algorithm. Last digit of a valid number
# makes the running state return to 0.

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def _verhoeff_valid(number: str) -> bool:
    """Validate the Verhoeff checksum on an all-digit string."""
    if not number.isdigit():
        return False
    c = 0
    for i, digit in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(digit)]]
    return c == 0


class IndiaAadhaarRecognizer(PatternRecognizer):
    """Aadhaar: 12 digits, optionally 4-4-4 with spaces. Verhoeff-validated."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="IN_AADHAAR",
            patterns=[
                # 4-4-4 spaced form gets a higher score than bare 12 digits.
                Pattern("Aadhaar 4-4-4", r"\b\d{4}\s\d{4}\s\d{4}\b", 0.4),
                Pattern("Aadhaar 12",    r"\b\d{12}\b",             0.3),
            ],
            context=["aadhaar", "aadhar", "UIDAI"],
        )

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        digits = re.sub(r"\s", "", pattern_text)
        return _verhoeff_valid(digits)


class IndiaPanRecognizer(PatternRecognizer):
    """PAN: 5 letters, 4 digits, 1 letter (format-only; PAN has no simple checksum)."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="IN_PAN",
            patterns=[Pattern("PAN", r"\b[A-Z]{5}\d{4}[A-Z]\b", 0.6)],
            context=["PAN", "permanent account number"],
        )


# ======================================================================
# Phone numbers: +65 and +91
# ======================================================================
# Both allow spaces, dashes, or dots between the digit groups, or no separator.

class SingaporePhoneRecognizer(PatternRecognizer):
    """+65 followed by 8 digits (4-4 grouped), flexible separator."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    "+65 phone",
                    r"\+65[\s\-\.]?\d{4}[\s\-\.]?\d{4}",
                    0.75,
                ),
            ],
            context=["phone", "call", "mobile", "contact"],
        )


class IndiaPhoneRecognizer(PatternRecognizer):
    """+91 followed by 10 digits (5-5 grouped), flexible separator."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    "+91 phone",
                    r"\+91[\s\-\.]?\d{5}[\s\-\.]?\d{5}",
                    0.75,
                ),
            ],
            context=["phone", "call", "mobile", "contact"],
        )


# ======================================================================
# AnalyzerEngine builder
# ======================================================================
# GLiNER free-text label -> our canonical Presidio entity type.
# Several GLiNER label spellings map to the same type so the model has
# multiple phrasings to latch onto.
GLINER_ENTITY_MAPPING = {
    # ---- Core PII (always masked) ----
    "person":        "PERSON",
    "organization":  "ORGANIZATION",
    "company":       "ORGANIZATION",
    "employer":      "ORGANIZATION",
    "location":      "LOCATION",
    "city":          "LOCATION",
    "address":       "LOCATION",
    # Presidio's built-in EmailRecognizer emits "EMAIL_ADDRESS"; use the
    # same canonical name so both the regex-based and GLiNER detectors
    # contribute to one bucket.
    "email":         "EMAIL_ADDRESS",
    "email address": "EMAIL_ADDRESS",
    "phone number":  "PHONE_NUMBER",
    "date of birth": "DATE_OF_BIRTH",

    # Family member *name* — always masked, numbered under PERSON so
    # [PERSON_1]/[PERSON_2] remain readable. Relationship words themselves
    # (brother / wife / boss) are not labels and stay untouched.
    "family member name":  "FAMILY_MEMBER",

    # Bank account number is identifying PII, not sensitive "content".
    # Always masked under the ID counter.
    "bank account number": "BANK_ACCOUNT",

    # ---- Sensitive topics (flagged by default; masked in sensitive_mode="mask") ----
    # HEALTH category
    "medical condition":       "HEALTH",
    "mental health condition": "HEALTH",
    "medication":              "HEALTH",
    "treatment or therapy":    "HEALTH",

    # FINANCE category (amounts and hardship together)
    "salary or income amount": "FINANCE_AMOUNT",
    "debt or loan amount":     "FINANCE_AMOUNT",
    "financial hardship":      "FINANCE_AMOUNT",

    # BELIEF category (special categories under most privacy regimes)
    "sexual orientation": "BELIEF",
    "religious belief":   "BELIEF",
    "political view":     "BELIEF",
}

# Entities the AnalyzerEngine is asked to return. Presidio ignores
# recognizers that don't support any of these, so built-in recognizers
# for SSN etc. won't fire even though they're loaded. CREDIT_CARD IS
# listed here so Presidio's built-in CreditCardRecognizer (Luhn) runs.
SUPPORTED_ENTITIES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "DATE_OF_BIRTH",
    "FAMILY_MEMBER",
    "BANK_ACCOUNT",
    "CREDIT_CARD",
    "SG_NRIC",
    "IN_AADHAAR",
    "IN_PAN",
    "HEALTH",
    "FINANCE_AMOUNT",
    "BELIEF",
]


def build_analyzer(threshold: float = 0.5) -> AnalyzerEngine:
    """Build the AnalyzerEngine used by this PoC.

    - NLP engine: spaCy with `en_core_web_sm` only.
    - Removes the default `SpacyRecognizer` so spaCy NER doesn't produce
      competing PERSON/LOCATION/ORG results alongside GLiNER.
    - Adds the `nvidia/gliner-PII` GLiNER model with a token-based chunker
      (384 tokens, 48-token overlap). The chunker is deferred and resolved
      to GLiNER's own tokenizer when the model loads, so char offsets in
      results map back to the original text automatically.
    - Adds our custom SG/IN pattern recognizers.
    """
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    ).create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en"])
    # spaCy NER would duplicate GLiNER on PERSON / ORG / LOCATION with weaker results.
    registry.remove_recognizer("SpacyRecognizer")

    registry.add_recognizer(SingaporeNricRecognizer())
    registry.add_recognizer(IndiaAadhaarRecognizer())
    registry.add_recognizer(IndiaPanRecognizer())
    registry.add_recognizer(SingaporePhoneRecognizer())
    registry.add_recognizer(IndiaPhoneRecognizer())

    registry.add_recognizer(
        ScopedGLiNERRecognizer(
            model_name="nvidia/gliner-PII",
            entity_mapping=GLINER_ENTITY_MAPPING,
            threshold=threshold,
            text_chunker=TokenizerBasedTextChunker(
                max_tokens=384, overlap_tokens=48
            ),
        )
    )

    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["en"],
    )
