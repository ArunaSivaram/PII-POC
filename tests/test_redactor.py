"""Behavioural tests for the redactor + custom recognizers.

These tests load the analyzer once (module-level cache) so pytest's first
test takes a few seconds; the rest are fast.
"""

from __future__ import annotations

import pytest

from pii_poc.recognizers import _nric_checksum_valid, _verhoeff_valid
from pii_poc.redactor import redact


# ----------------------------------------------------------------------
# Pure checksum helpers (no model load involved)
# ----------------------------------------------------------------------

def test_nric_checksum_valid_and_invalid():
    # S1234567D is a well-known valid test NRIC; A is the wrong checksum.
    assert _nric_checksum_valid("S1234567D") is True
    assert _nric_checksum_valid("S1234567A") is False
    # Non-conforming strings must return False, not raise.
    assert _nric_checksum_valid("X1234567D") is False
    assert _nric_checksum_valid("S123456D") is False


def test_verhoeff_valid_and_invalid():
    # Known valid demo Aadhaar (Verhoeff checksum passes).
    assert _verhoeff_valid("234123412346") is True
    # Any digit flipped should fail.
    assert _verhoeff_valid("234123412345") is False
    # Non-digit input must return False.
    assert _verhoeff_valid("23412341234A") is False


# ----------------------------------------------------------------------
# End-to-end redactor behaviour
# ----------------------------------------------------------------------

def test_consistent_placeholders_for_same_person():
    text = "Priya called Priya back after the meeting with Priya's team."
    result = redact(text)
    # All three mentions should collapse to [PERSON_1] (one placeholder).
    assert result.redacted_text.count("[PERSON_1]") == 3
    # No second person placeholder.
    assert "[PERSON_2]" not in result.redacted_text
    # Original name must be gone.
    assert "Priya" not in result.redacted_text


def test_client_matches_first_name_only():
    text = "I met with Priya today. Priya mentioned Rohan would join tomorrow."
    result = redact(text, client_name="Priya Kumar")
    # 'Priya' alone (no surname in the text) should still become [CLIENT_NAME].
    assert result.redacted_text.count("[CLIENT_NAME]") == 2
    assert "Priya" not in result.redacted_text


def test_coach_and_client_in_mapping():
    text = "Jamie spoke with Priya about next steps."
    result = redact(text, coach_name="Jamie", client_name="Priya")
    assert "Jamie" not in result.redacted_text
    assert "Priya" not in result.redacted_text
    assert "[COACH_NAME]" in result.redacted_text
    assert "[CLIENT_NAME]" in result.redacted_text
    assert result.mapping["[COACH_NAME]"] == "Jamie"
    assert result.mapping["[CLIENT_NAME]"] == "Priya"


def test_valid_nric_is_redacted():
    text = "For records, my NRIC is S1234567D."
    result = redact(text)
    assert "S1234567D" not in result.redacted_text
    # It should have been tagged as an ID.
    assert "[ID_1]" in result.redacted_text
    assert result.counts.get("SG_NRIC", 0) == 1


def test_invalid_nric_is_not_redacted_as_id():
    # S1234567A has the wrong checksum letter; our NRIC recognizer should pass.
    text = "The reference code was S1234567A, not a real NRIC."
    result = redact(text)
    # The specific string survives — at least, the SG_NRIC recognizer didn't fire.
    assert result.counts.get("SG_NRIC", 0) == 0


def test_valid_aadhaar_is_redacted():
    # 234123412346 passes Verhoeff.
    text = "Please note the Aadhaar number 2341 2341 2346 on file."
    result = redact(text)
    assert "2341 2341 2346" not in result.redacted_text
    assert "[ID_1]" in result.redacted_text
    assert result.counts.get("IN_AADHAAR", 0) == 1


def test_singapore_phone_redacted():
    text = "Call Priya on +65 9123 4567 after lunch."
    result = redact(text)
    assert "+65 9123 4567" not in result.redacted_text
    assert "[PHONE_1]" in result.redacted_text


def test_india_phone_redacted():
    text = "Aarav's number is +91 98765 43210 if you need him."
    result = redact(text)
    assert "+91 98765 43210" not in result.redacted_text
    assert "[PHONE_1]" in result.redacted_text


def test_country_names_stay_unredacted():
    text = (
        "She moved from India to Singapore last year "
        "and is now looking for a role based in Mumbai."
    )
    result = redact(text)
    # Country names survive even though a LOCATION-type model might tag them.
    assert "India" in result.redacted_text
    assert "Singapore" in result.redacted_text
    # City-level LOCATION should still be redacted.
    assert "Mumbai" not in result.redacted_text


# ----------------------------------------------------------------------
# Sensitive topics — GLiNER label probes (one sentence per label)
# ----------------------------------------------------------------------
# Each row tells us: "given this sentence, this GLiNER label should map
# to this canonical entity type in the result". If a label fails to fire
# we'll see exactly which one is weak and can pick a different phrasing.

_LABEL_PROBES = [
    # label               expected_type        sentence
    ("medical condition",       "HEALTH",
        "She has been diagnosed with hypertension since January."),
    ("mental health condition", "HEALTH",
        "He has been struggling with severe anxiety and depression this quarter."),
    ("medication",              "HEALTH",
        "Her GP prescribed sertraline and propranolol last week."),
    ("treatment or therapy",    "HEALTH",
        "She has been in cognitive behavioural therapy for six months."),
    ("salary or income amount", "FINANCE_AMOUNT",
        "My base salary is 85,000 pounds before bonus."),
    ("debt or loan amount",     "FINANCE_AMOUNT",
        "I still owe about 45,000 dollars on my student loan."),
    ("bank account number",     "BANK_ACCOUNT",
        "Please transfer it to account 1234567890 at OCBC."),
    ("financial hardship",      "FINANCE_AMOUNT",
        "After the layoff I have been struggling to pay rent and bills."),
    ("family member name",      "FAMILY_MEMBER",
        "My sister Maya called me yesterday afternoon."),
    ("sexual orientation",      "BELIEF",
        "I came out as bisexual to my family last year."),
    ("religious belief",        "BELIEF",
        "Growing up Hindu shaped a lot of my values."),
    ("political view",          "BELIEF",
        "I lean quite progressive on economic policy."),
]


@pytest.mark.parametrize(
    "label,expected_type,sentence",
    _LABEL_PROBES,
    ids=[row[0] for row in _LABEL_PROBES],
)
def test_sensitive_label_fires(label, expected_type, sentence):
    """Each configured GLiNER label should fire on its probe sentence."""
    # Use mask mode so the pipeline exercises the full masking code path too.
    result = redact(sentence, sensitive_mode="mask")
    found_types = {s.entity_type for s in result.spans}
    assert expected_type in found_types, (
        f"GLiNER label {label!r} did not fire on probe sentence "
        f"(expected type {expected_type}, got types {sorted(found_types)})"
    )


# ----------------------------------------------------------------------
# flag vs mask mode
# ----------------------------------------------------------------------

def test_flag_mode_leaves_sensitive_text_intact():
    text = "I have been struggling with severe anxiety and depression for months."
    result = redact(text, sensitive_mode="flag")
    # Original sensitive wording is still there.
    assert "anxiety" in result.redacted_text or "depression" in result.redacted_text
    # But HEALTH spans ARE reported so the UI can highlight them.
    assert any(s.entity_type == "HEALTH" for s in result.spans)
    # No HEALTH placeholder was inserted.
    assert "[HEALTH_" not in result.redacted_text


def test_mask_mode_replaces_sensitive_text_with_neutral_tag():
    text = "I have been struggling with severe anxiety and depression for months."
    result = redact(text, sensitive_mode="mask")
    # Sensitive wording is gone.
    assert "anxiety" not in result.redacted_text
    assert "depression" not in result.redacted_text
    assert "[HEALTH_1]" in result.redacted_text


def test_mask_mode_uses_amount_tag_for_finance():
    text = "My salary is 85,000 pounds before bonus this year."
    result = redact(text, sensitive_mode="mask")
    assert "[AMOUNT_1]" in result.redacted_text


def test_mask_mode_uses_belief_tag():
    text = "Growing up Hindu shaped a lot of my values."
    result = redact(text, sensitive_mode="mask")
    assert "[BELIEF_1]" in result.redacted_text


# ----------------------------------------------------------------------
# Family-member and card/bank: always masked
# ----------------------------------------------------------------------

def test_family_member_name_always_masked_even_in_flag_mode():
    """Family member NAMES are PII and should always be masked, even when
    sensitive topics are only flagged. The relationship WORD stays."""
    text = "My brother Arjun lives in Delhi with his wife."
    result = redact(text, sensitive_mode="flag")
    # The name must be gone. The relationship word (brother / wife) should
    # survive because it carries no identifying information.
    assert "Arjun" not in result.redacted_text
    assert "brother" in result.redacted_text


def test_credit_card_always_masked():
    # 4111 1111 1111 1111 is a Visa test card; Luhn-valid.
    text = "My card number was 4111 1111 1111 1111 but I cancelled it."
    result = redact(text)
    assert "4111 1111 1111 1111" not in result.redacted_text
    # It should have been tagged as an ID-class entity.
    assert "[ID_1]" in result.redacted_text


def test_bank_account_always_masked():
    text = "Transfer it to account 1234567890 at OCBC."
    result = redact(text, sensitive_mode="flag")
    # Even in flag mode (which only affects sensitive TOPICS), a bank account
    # number is identifying PII and must be masked.
    assert "1234567890" not in result.redacted_text


def test_relationship_words_stay_unmasked():
    """Relationship / role words on their own should survive, even though
    GLiNER sometimes tags them as PERSON at moderate confidence."""
    text = (
        "My mum called about my brother's wedding. "
        "My boss approved the leave and my manager helped with cover."
    )
    result = redact(text)
    # Each relationship word must still be in the output.
    for word in ("mum", "brother", "boss", "manager"):
        assert word in result.redacted_text, (
            f"{word!r} was redacted but should have stayed"
        )


def test_family_name_still_masked_but_relationship_word_stays():
    """The filter must NOT accidentally un-mask real family names."""
    text = "My sister Maya called yesterday to say hi."
    result = redact(text)
    # "sister" stays (relationship word), "Maya" goes (real name).
    assert "sister" in result.redacted_text
    assert "Maya" not in result.redacted_text
