"""Tests for the internal SessionReport format and its parsers."""

from __future__ import annotations

import json

import pytest

from pii_poc.report import (
    CompetencyResult,
    Evidence,
    SessionReport,
    from_coachbetter_report,
    from_poc_feedback,
    to_prompt_text,
)


# ----------------------------------------------------------------------
# to_prompt_text
# ----------------------------------------------------------------------

def test_to_prompt_text_renders_full_report():
    report = SessionReport(
        overall_summary="[COACH_NAME] held a steady presence.",
        competencies=[
            CompetencyResult(
                competency="Listens Actively",
                score_or_level="PCC",
                summary="Deep listening throughout.",
                strengths=[
                    Evidence(
                        timestamp="02:41",
                        quote="What else is there?",
                        note="open stance",
                    ),
                ],
                development_areas=[
                    Evidence(
                        timestamp="18:10",
                        quote="So basically you need X.",
                        note="overt framing",
                    ),
                ],
                pcc_markers_observed=["reflects back"],
                pcc_markers_missed=["invites silence"],
            ),
        ],
        next_steps=["Practice silence", "Reflect feelings"],
        raw_text="{...original response...}",
    )
    out = to_prompt_text(report)

    # Structural header.
    assert "Session feedback report" in out
    # Content pieces all land in the output.
    assert "steady presence" in out
    assert "Listens Actively" in out
    assert "PCC" in out
    assert "02:41" in out
    assert "What else is there?" in out
    assert "reflects back" in out
    assert "invites silence" in out
    assert "Practice silence" in out
    # raw_text should NOT leak into the LLM prompt rendering.
    assert "original response" not in out


def test_to_prompt_text_handles_minimal_report():
    report = SessionReport(
        overall_summary="",
        competencies=[],
        next_steps=[],
        raw_text="",
    )
    # Must not crash on an empty report and should still identify itself.
    out = to_prompt_text(report)
    assert "Session feedback report" in out


def test_to_prompt_text_stays_under_token_budget():
    """A full 8-competency report should fit comfortably under ~4,000
    tokens. Using the ~4 chars/token rule of thumb => 16,000 chars.
    """
    comps = [
        CompetencyResult(
            competency=f"Competency Number {i}",
            score_or_level="PCC",
            summary="One-line observation that isn't too long.",
            strengths=[
                Evidence("12:34", "a short transcript quote", "brief note"),
                Evidence("13:00", "another short quote",      "brief note"),
            ],
            development_areas=[
                Evidence("12:34", "a quote for development", "brief note"),
            ],
            pcc_markers_observed=["m1", "m2", "m3"],
            pcc_markers_missed=["m4"],
        )
        for i in range(8)
    ]
    report = SessionReport(
        overall_summary=(
            "A couple-sentence overall summary that gives context before "
            "we dive into the competency-by-competency breakdown."
        ),
        competencies=comps,
        next_steps=[
            "First concrete step with a bit of detail.",
            "Second concrete step with a bit of detail.",
            "Third concrete step with a bit of detail.",
        ],
        raw_text="",
    )
    out = to_prompt_text(report)
    assert len(out) < 16_000, f"rendering too long: {len(out)} chars"


# ----------------------------------------------------------------------
# from_poc_feedback — JSON parsing
# ----------------------------------------------------------------------

def test_from_poc_feedback_parses_plain_json():
    payload = {
        "overall_summary": "A solid session.",
        "competencies": [
            {
                "competency": "Maintains Presence",
                "score_or_level": "PCC",
                "summary": "Grounded throughout.",
                "strengths": [
                    {"timestamp": "05:12", "quote": "Take your time.", "note": "spacious"},
                ],
                "development_areas": [],
                "pcc_markers_observed": ["observes shifts"],
                "pcc_markers_missed": [],
            }
        ],
        "next_steps": ["s1", "s2", "s3"],
    }
    text = json.dumps(payload)
    report = from_poc_feedback(text)

    assert report.overall_summary == "A solid session."
    assert len(report.competencies) == 1
    c = report.competencies[0]
    assert c.competency == "Maintains Presence"
    assert c.score_or_level == "PCC"
    assert c.strengths[0].timestamp == "05:12"
    assert c.strengths[0].quote == "Take your time."
    assert c.pcc_markers_observed == ["observes shifts"]
    assert report.next_steps == ["s1", "s2", "s3"]
    # raw_text always preserved.
    assert report.raw_text == text


def test_from_poc_feedback_tolerates_json_markdown_fences():
    payload = {
        "overall_summary": "fenced",
        "competencies": [],
        "next_steps": [],
    }
    text = "```json\n" + json.dumps(payload) + "\n```"
    report = from_poc_feedback(text)
    assert report.overall_summary == "fenced"
    assert report.raw_text == text


def test_from_poc_feedback_tolerates_plain_fences():
    payload = {"overall_summary": "no-lang-fence", "competencies": [], "next_steps": []}
    text = "```\n" + json.dumps(payload) + "\n```"
    report = from_poc_feedback(text)
    assert report.overall_summary == "no-lang-fence"


def test_from_poc_feedback_tolerates_surrounding_prose():
    payload = {"overall_summary": "surrounded", "competencies": [], "next_steps": []}
    text = (
        "Sure! Here's the feedback:\n\n"
        + json.dumps(payload)
        + "\n\nLet me know if you need more."
    )
    report = from_poc_feedback(text)
    assert report.overall_summary == "surrounded"


def test_from_poc_feedback_falls_back_to_raw_text_on_prose():
    text = "This is not JSON. Just regular Gemini prose output."
    report = from_poc_feedback(text)
    assert report.overall_summary == ""
    assert report.competencies == []
    assert report.next_steps == []
    assert report.raw_text == text


def test_from_poc_feedback_handles_missing_top_level_fields():
    text = json.dumps({"overall_summary": "partial"})
    report = from_poc_feedback(text)
    assert report.overall_summary == "partial"
    assert report.competencies == []
    assert report.next_steps == []
    assert report.raw_text == text


def test_from_poc_feedback_handles_malformed_evidence():
    """Gemini sometimes puts a bare string where an evidence dict belongs —
    we should still accept the item and keep the information in `quote`."""
    text = json.dumps({
        "overall_summary": "x",
        "competencies": [{
            "competency": "Listens Actively",
            "score_or_level": "Developing",
            "summary": "s",
            "strengths": ["just a string"],
            "development_areas": [],
            "pcc_markers_observed": [],
            "pcc_markers_missed": [],
        }],
        "next_steps": [],
    })
    report = from_poc_feedback(text)
    assert report.competencies[0].strengths[0].quote == "just a string"
    assert report.competencies[0].strengths[0].timestamp == ""


def test_from_poc_feedback_ignores_non_dict_competencies():
    text = json.dumps({
        "overall_summary": "x",
        "competencies": ["string instead of object", None, 42],
        "next_steps": [],
    })
    report = from_poc_feedback(text)
    # None of the bogus items should have made it through.
    assert report.competencies == []


# ----------------------------------------------------------------------
# External adapter stub
# ----------------------------------------------------------------------

def test_from_coachbetter_report_not_implemented():
    with pytest.raises(NotImplementedError, match="Waiting for CoachBetter report sample"):
        from_coachbetter_report({})
