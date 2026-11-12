"""Tests for the chat engine.

The Gemini call itself is mocked (no API key required and no network
traffic). The redactor is NOT mocked because we want to verify that
step 2 of the message redaction actually catches a new name.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pii_poc.chat import (
    ChatContext,
    ChatSession,
    redact_user_message,
    suggest_starters,
)
from pii_poc.report import (
    CompetencyResult,
    Evidence,
    SessionReport,
)


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------

def _report() -> SessionReport:
    return SessionReport(
        overall_summary="Steady session.",
        competencies=[
            CompetencyResult(
                competency="Listens Actively",
                score_or_level="PCC",
                summary="Good listening.",
                strengths=[Evidence("02:41", "What else is there?", "open stance")],
                development_areas=[Evidence("18:10", "You should...", "leading")],
                pcc_markers_observed=["reflects back"],
                pcc_markers_missed=[],
            ),
            CompetencyResult(
                competency="Maintains Presence",
                score_or_level="Developing",
                summary="Sometimes rushed.",
                strengths=[],
                development_areas=[Evidence("08:15", "Let me ask...", "impatient")],
                pcc_markers_observed=[],
                pcc_markers_missed=["observes shifts"],
            ),
        ],
        next_steps=["Practice silence"],
        raw_text="{}",
    )


def _context(**overrides) -> ChatContext:
    defaults = dict(
        redacted_transcript="[00:01] COACH: Hi [CLIENT_NAME].\n[00:05] CLIENT: Hi.",
        report=_report(),
        mapping={
            "[COACH_NAME]":  "Anita Menon",
            "[CLIENT_NAME]": "Priya Raman",
            "[PERSON_1]":    "Kenneth",
        },
        coach_name="Anita Menon",
        client_name="Priya Raman",
    )
    defaults.update(overrides)
    return ChatContext(**defaults)


def _mock_response(
    reply_text: str,
    *,
    prompt_tokens: int = 200,
    output_tokens: int = 60,
    cached_tokens: int = 0,
    thoughts_tokens: int = 0,
    total_tokens: int | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.text = reply_text
    resp.usage_metadata = MagicMock(
        prompt_token_count=prompt_tokens,
        candidates_token_count=output_tokens,
        thoughts_token_count=thoughts_tokens,
        cached_content_token_count=cached_tokens,
        total_token_count=total_tokens or (prompt_tokens + output_tokens),
    )
    return resp


@pytest.fixture
def patched_gemini(monkeypatch):
    """Mock google-genai so no network calls happen."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with patch("pii_poc.chat.genai.Client") as client_cls:
        client = MagicMock()
        client_cls.return_value = client
        yield client


# ----------------------------------------------------------------------
# Message redaction (step 1: known mapping values)
# ----------------------------------------------------------------------

def test_client_first_name_replaced_before_send(patched_gemini):
    patched_gemini.models.generate_content.return_value = _mock_response("ok")
    ctx = _context()
    session = ChatSession(ctx)

    turn = session.ask("What did Priya say about her manager?")

    assert "Priya" not in turn.message_sent
    assert "[CLIENT_NAME]" in turn.message_sent
    assert patched_gemini.models.generate_content.call_count == 1


def test_redact_user_message_also_catches_coach_name(patched_gemini):
    """Smoke-test the step-1 reverse mapping for coach's own name."""
    ctx = _context()
    out = redact_user_message("I'm Anita and I think we did fine.", ctx)
    assert "Anita" not in out
    assert "[COACH_NAME]" in out


# ----------------------------------------------------------------------
# Message redaction (step 2: new PII gets a new placeholder)
# ----------------------------------------------------------------------

def test_new_person_gets_new_placeholder_and_rehydrates(patched_gemini):
    # Gemini "returns" a reply that uses the new placeholder we expect.
    patched_gemini.models.generate_content.return_value = _mock_response(
        "I noticed [PERSON_2] came up only once."
    )
    ctx = _context()
    session = ChatSession(ctx)

    turn = session.ask("What about Marcus's feedback this session?")

    # Marcus should have been added to the mapping as [PERSON_2]
    # (continuing from [PERSON_1]: Kenneth).
    assert "[PERSON_2]" in ctx.mapping
    assert ctx.mapping["[PERSON_2]"] == "Marcus"
    # The sent message carries the placeholder, not the real name.
    assert "Marcus" not in turn.message_sent
    assert "[PERSON_2]" in turn.message_sent
    # The reply is rehydrated for the coach's view.
    assert "[PERSON_2]" not in turn.reply_shown
    assert "Marcus" in turn.reply_shown


# ----------------------------------------------------------------------
# Byte-identical fixed context block
# ----------------------------------------------------------------------

def test_fixed_context_block_byte_identical_across_turns(patched_gemini):
    patched_gemini.models.generate_content.side_effect = [
        _mock_response("reply one"),
        _mock_response("reply two"),
    ]
    ctx = _context()
    session = ChatSession(ctx)

    session.ask("first question")
    session.ask("second question")

    calls = patched_gemini.models.generate_content.call_args_list
    c1 = calls[0].kwargs["contents"]
    c2 = calls[1].kwargs["contents"]

    # The first two entries (fixed transcript+report, fixed ack) must
    # match byte-for-byte so Gemini's prompt cache applies.
    assert c1[0] == c2[0]
    assert c1[1] == c2[1]


def test_contents_shape_matches_spec(patched_gemini):
    """[fixed-user][fixed-model-ack][history…][new-user]."""
    patched_gemini.models.generate_content.return_value = _mock_response("ok")
    ctx = _context()
    session = ChatSession(ctx)

    session.ask("hello")

    contents = patched_gemini.models.generate_content.call_args.kwargs["contents"]
    assert contents[0]["role"] == "user"
    assert "SESSION TRANSCRIPT (redacted):" in contents[0]["parts"][0]["text"]
    assert "FEEDBACK REPORT:" in contents[0]["parts"][0]["text"]
    assert contents[1]["role"] == "model"           # fixed ack
    assert contents[-1]["role"] == "user"           # the new message
    # No history yet, so exactly three entries on the first turn.
    assert len(contents) == 3


# ----------------------------------------------------------------------
# The question cap
# ----------------------------------------------------------------------

def test_cap_stops_at_limit(patched_gemini, monkeypatch):
    monkeypatch.setenv("CHAT_MAX_QUESTIONS", "2")
    patched_gemini.models.generate_content.return_value = _mock_response("ok")
    ctx = _context()
    session = ChatSession(ctx)

    session.ask("q1")
    session.ask("q2")
    blocked = session.ask("q3")  # must NOT reach Gemini

    assert patched_gemini.models.generate_content.call_count == 2
    assert session.questions_used == 2
    assert session.questions_left == 0
    # The blocked turn signals that nothing was sent.
    assert blocked.message_sent == ""
    assert "limit" in blocked.reply_shown.lower()


# ----------------------------------------------------------------------
# set_persona + reset
# ----------------------------------------------------------------------

def test_set_persona_clears_history_but_keeps_usage(patched_gemini):
    patched_gemini.models.generate_content.return_value = _mock_response("ok")
    ctx = _context()
    session = ChatSession(ctx, persona="mentor")
    session.ask("first")
    assert len(session.history) == 2
    assert session.questions_used == 1

    session.set_persona("peer")
    assert session.persona == "peer"
    assert session.history == []
    # Usage persists — the cap is a per-session resource.
    assert session.questions_used == 1


def test_reset_clears_everything(patched_gemini):
    patched_gemini.models.generate_content.return_value = _mock_response("ok")
    ctx = _context()
    session = ChatSession(ctx)
    session.ask("q1")

    session.reset()
    assert session.history == []
    assert session.questions_used == 0
    assert session.total_cost_usd == 0.0
    assert session.total_tokens == 0


# ----------------------------------------------------------------------
# Starter questions
# ----------------------------------------------------------------------

def test_suggest_starters_returns_six_mixed():
    starters = suggest_starters(_report())
    assert len(starters) == 6
    # Dynamic #1 names the lowest-rated competency (Maintains Presence).
    assert "Maintains Presence" in starters[0]
    # Dynamic #2 names the strongest competency (Listens Actively).
    assert "Listens Actively" in starters[1]
    # Dynamic #3 cites the FIRST dev-area timestamp encountered when
    # walking competencies in order — "18:10" from Listens Actively,
    # which precedes Maintains Presence in the report.
    assert "18:10" in starters[2]
    # Remaining three are from the fixed list.
    assert starters[3].startswith("What's the one thing")
