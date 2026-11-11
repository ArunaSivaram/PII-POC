"""Tests for the rehydrator — placeholder variations the LLM might emit.

Also verifies the critical anti-regression rule: ordinary words such as
"coach" or "client" must never be rehydrated, even case-insensitively.
"""

from __future__ import annotations

from pii_poc.rehydrate import rehydrate


# ----------------------------------------------------------------------
# Happy-path placeholder forms
# ----------------------------------------------------------------------

def test_exact_placeholder():
    assert (
        rehydrate("Hello [PERSON_1], welcome.", {"[PERSON_1]": "Priya"})
        == "Hello Priya, welcome."
    )


def test_case_insensitive():
    out = rehydrate(
        "Hello [Person_1] and [person_1].",
        {"[PERSON_1]": "Priya"},
    )
    assert out == "Hello Priya and Priya."


def test_space_instead_of_underscore_inside_brackets():
    assert (
        rehydrate("Hello [PERSON 1], welcome.", {"[PERSON_1]": "Priya"})
        == "Hello Priya, welcome."
    )


def test_bare_token_with_underscore():
    assert (
        rehydrate("Hello PERSON_1, welcome.", {"[PERSON_1]": "Priya"})
        == "Hello Priya, welcome."
    )


def test_bare_token_case_insensitive():
    assert (
        rehydrate("Hello Person_1, welcome.", {"[PERSON_1]": "Priya"})
        == "Hello Priya, welcome."
    )


def test_bare_coach_name_token_rehydrated():
    assert (
        rehydrate("COACH_NAME did a great job.", {"[COACH_NAME]": "Anita"})
        == "Anita did a great job."
    )


def test_bracketed_token_inside_parens_keeps_parens():
    # The placeholder matches; the surrounding punctuation stays.
    assert (
        rehydrate("Hello (PERSON_1), welcome.", {"[PERSON_1]": "Priya"})
        == "Hello (Priya), welcome."
    )


def test_coach_and_client():
    out = rehydrate(
        "[COACH_NAME] asked [CLIENT_NAME] about the week.",
        {"[COACH_NAME]": "Anita", "[CLIENT_NAME]": "Priya"},
    )
    assert out == "Anita asked Priya about the week."


# ----------------------------------------------------------------------
# Anti-regression — ordinary words must never be rehydrated
# ----------------------------------------------------------------------

def test_plain_word_coach_not_rehydrated():
    """Ordinary word 'coach' must never be replaced. This is the real bug."""
    text = "The coach asked a powerful question."
    out = rehydrate(text, {"[COACH_NAME]": "Ramya Nair"})
    assert out == text


def test_bracketed_coach_name_rehydrated():
    out = rehydrate(
        "[COACH_NAME] asked a powerful question.",
        {"[COACH_NAME]": "Ramya Nair"},
    )
    assert out == "Ramya Nair asked a powerful question."


def test_bracketed_coach_name_case_insensitive_rehydrated():
    out = rehydrate(
        "[Coach_Name] asked a powerful question.",
        {"[COACH_NAME]": "Ramya Nair"},
    )
    assert out == "Ramya Nair asked a powerful question."


def test_speaker_label_not_rehydrated():
    """Transcript speaker labels like `COACH:` / `CLIENT:` must stay intact."""
    text = "COACH: hello. CLIENT: hi there."
    out = rehydrate(
        text,
        {"[COACH_NAME]": "Anita", "[CLIENT_NAME]": "Priya"},
    )
    assert out == text


def test_possessive_client_word_not_rehydrated():
    text = "the client's goal is clearer now."
    out = rehydrate(text, {"[CLIENT_NAME]": "Priya"})
    assert out == text


def test_plain_role_words_mid_sentence_untouched():
    text = "In this coach-client dyad, trust was growing."
    out = rehydrate(
        text,
        {"[COACH_NAME]": "Anita", "[CLIENT_NAME]": "Priya"},
    )
    assert out == text


# ----------------------------------------------------------------------
# Safety / edge cases
# ----------------------------------------------------------------------

def test_unknown_placeholder_is_left_alone():
    out = rehydrate(
        "[PERSON_1] met [PERSON_99] today.",
        {"[PERSON_1]": "Priya"},
    )
    assert "Priya" in out
    assert "[PERSON_99]" in out


def test_longer_number_not_eaten_by_shorter():
    """[PERSON_10] must be replaced as a whole, not as [PERSON_1] + '0'."""
    out = rehydrate(
        "[PERSON_1] and [PERSON_10] talked.",
        {"[PERSON_1]": "Priya", "[PERSON_10]": "Jamie"},
    )
    assert "Priya" in out and "Jamie" in out
    assert "[PERSON_1]" not in out and "[PERSON_10]" not in out
    assert "Priya0" not in out


def test_empty_mapping_returns_text_unchanged():
    assert (
        rehydrate("[PERSON_1] said hi.", {})
        == "[PERSON_1] said hi."
    )


def test_does_not_touch_ordinary_words_around_placeholder():
    assert (
        rehydrate("The person met [PERSON_1] at 3pm.", {"[PERSON_1]": "Priya"})
        == "The person met Priya at 3pm."
    )


def test_original_may_contain_special_chars():
    """`$` and `\\1` in the replacement must be treated as literals."""
    out = rehydrate("Amount was [AMOUNT_1].", {"[AMOUNT_1]": "$1,200"})
    assert out == "Amount was $1,200."


def test_mixed_variations_in_one_text():
    text = "[COACH_NAME] met Person_1 and [PERSON 2] at [ORG_1]."
    mapping = {
        "[COACH_NAME]": "Anita",
        "[PERSON_1]":  "Priya",
        "[PERSON_2]":  "Rohan",
        "[ORG_1]":     "Coastline Bank",
    }
    out = rehydrate(text, mapping)
    assert out == "Anita met Priya and Rohan at Coastline Bank."
