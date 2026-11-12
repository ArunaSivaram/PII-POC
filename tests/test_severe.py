"""Tests for severe-content phrase detection."""

from __future__ import annotations

from pii_poc.severe import detect_severe_spans, has_severe


def test_detects_hit_me_in_sentence():
    spans = detect_severe_spans("He hit me across the face last week.")
    assert len(spans) == 1
    assert spans[0].phrase.lower() == "hit me"
    assert spans[0].category == "interpersonal-violence"


def test_case_insensitive():
    spans = detect_severe_spans("Suicide and SUICIDE and Suicidal.")
    assert len(spans) == 3
    assert {s.phrase.lower() for s in spans} == {"suicide", "suicidal"}


def test_word_bounded_no_false_positive_in_grape():
    """'rape' must not match inside 'grape', 'landscape', etc."""
    spans = detect_severe_spans("She loved the grape harvest on that landscape.")
    assert spans == []


def test_longest_phrase_wins():
    """`beaten up` should outrank `beaten` at the same offset."""
    spans = detect_severe_spans("He had beaten up his younger brother.")
    assert len(spans) == 1
    assert spans[0].phrase.lower() == "beaten up"


def test_empty_text_returns_empty_list():
    assert detect_severe_spans("") == []


def test_no_match_returns_empty_list():
    assert detect_severe_spans("We talked about budgets and timelines.") == []


def test_has_severe_quick_check():
    assert has_severe("He hit me yesterday.") is True
    assert has_severe("We discussed goals.") is False


def test_multiple_different_categories_in_one_text():
    """A transcript with both self-harm and violence phrases catches both."""
    spans = detect_severe_spans(
        "She mentioned suicide once, and said he hit me earlier that week."
    )
    cats = {s.category for s in spans}
    assert "self-harm" in cats
    assert "interpersonal-violence" in cats
