"""Test the sample loader on the three fictional transcripts."""

from __future__ import annotations

from pathlib import Path

from pii_poc.samples import load_sample

_SAMPLES = Path(__file__).parent.parent / "samples"


def _turn_count(text: str) -> int:
    """Count COACH + CLIENT turn markers."""
    return text.count("COACH:") + text.count("CLIENT:")


def test_sample_01_header_and_body():
    text, coach, client = load_sample(_SAMPLES / "01_career_transition.txt")
    assert coach == "Anita Menon"
    assert client == "Priya Raman"
    assert "# coach:" not in text            # header was stripped
    assert text.startswith("[00:00] COACH:")  # body starts cleanly
    assert 60 <= _turn_count(text) <= 100  # brief said "~60-80"; soft cap


def test_sample_02_header_and_body():
    text, coach, client = load_sample(_SAMPLES / "02_burnout.txt")
    assert coach == "Suresh Nair"
    assert client == "Arjun Krishnan"
    assert "# coach:" not in text
    assert text.startswith("[00:00] COACH:")
    assert 60 <= _turn_count(text) <= 100  # brief said "~60-80"; soft cap


def test_sample_03_header_and_body():
    text, coach, client = load_sample(_SAMPLES / "03_tricky_names.txt")
    assert coach == "Marcus Hale"
    assert client == "Alex Carter"
    assert "# coach:" not in text
    assert text.startswith("[00:00] COACH:")
    assert 60 <= _turn_count(text) <= 100  # brief said "~60-80"; soft cap


def test_sample_03_coach_name_appears_exactly_once():
    """Per the sample's brief, 'Marcus' should appear in the body exactly once."""
    text, _, _ = load_sample(_SAMPLES / "03_tricky_names.txt")
    assert text.count("Marcus") == 1


def test_sample_04_header_and_body():
    text, coach, client = load_sample(_SAMPLES / "04_disclosure.txt")
    assert coach == "Dana Lewis"
    assert client == "Maria Santos"
    assert "# coach:" not in text
    assert text.startswith("[00:00] COACH:")


def test_sample_04_contains_severe_phrase():
    """The sample is designed to exercise the SEVERE_RISK path."""
    text, _, _ = load_sample(_SAMPLES / "04_disclosure.txt")
    # The exact phrase used by the detector.
    assert "hit me" in text.lower()
