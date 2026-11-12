"""Privacy Shield for Coaching Transcripts (proof of concept).

Three-tab Gradio UI for the PII redaction demo.
  * Redact        — load/paste a transcript, highlight detected PII by
                    type, produce the redacted version + mapping table.
  * Send to AI    — show the redacted text, call Gemini for ICF
                    feedback, display tokens/cost, render raw vs
                    rehydrated panels.
  * Talk it through — chat with an AI mentor or peer about the session,
                    using the redacted transcript + the SessionReport
                    from the previous tabs.

Guarantees:
  * `.env` loads before gradio/gliner are imported so their telemetry-
    off flags take effect.
  * The transcript is only ever displayed inside the UI widgets; nothing
    is printed to stdout or written to a log.
"""

from __future__ import annotations

# Load .env first — gradio and gliner read env vars at import time.
from dotenv import load_dotenv

load_dotenv()

import os
import re
from pathlib import Path
from typing import Optional

import gradio as gr

from pii_poc.chat import ChatContext, ChatSession, suggest_starters
from pii_poc.llm import LLMConfigError, get_icf_feedback
from pii_poc.redactor import (
    RedactionResult,
    SENSITIVE_TOPIC_ENTITIES,
    redact,
    warmup,
)
from pii_poc.rehydrate import rehydrate
from pii_poc.report import SessionReport, to_prompt_text
from pii_poc.samples import load_sample

# Load the analyzer at import time so Hugging Face Spaces (or any host)
# pays the ~10s spaCy + GLiNER load during startup, not on the first
# user click. Safe to call at module scope — `warmup()` is idempotent
# (second call hits the module-level cache).
warmup()


_SAMPLES_DIR = Path(__file__).parent / "samples"


# ----------------------------------------------------------------------
# Two-tone palette:
#   - Blue  — all PII entities (the stuff that gets masked to placeholders)
#   - Amber — sensitive-topic flags (the stuff a coach should review)
# One colour per role keeps the transcript calm and legible. Hover still
# reveals the specific entity label via the HighlightedText tooltip.
# ----------------------------------------------------------------------
_PII_BLUE = "#DBEAFE"         # blue-100  — soft, neutral
_SENSITIVE_AMBER = "#FEF3C7"  # amber-100 — subtle warning
_SEVERE_RED = "#FECACA"       # red-200   — needs reviewer attention

COLOR_MAP = {
    # Masked PII
    "COACH":          _PII_BLUE,
    "CLIENT":         _PII_BLUE,
    "PERSON":         _PII_BLUE,
    "FAMILY_MEMBER":  _PII_BLUE,
    "ORGANIZATION":   _PII_BLUE,
    "LOCATION":       _PII_BLUE,
    "EMAIL_ADDRESS":  _PII_BLUE,
    "PHONE_NUMBER":   _PII_BLUE,
    "DATE_OF_BIRTH":  _PII_BLUE,
    "SG_NRIC":        _PII_BLUE,
    "IN_AADHAAR":     _PII_BLUE,
    "IN_PAN":         _PII_BLUE,
    "BANK_ACCOUNT":   _PII_BLUE,
    "CREDIT_CARD":    _PII_BLUE,
    # Sensitive topics (flagged or masked depending on mode)
    "HEALTH":         _SENSITIVE_AMBER,
    "FINANCE_AMOUNT": _SENSITIVE_AMBER,
    "BELIEF":         _SENSITIVE_AMBER,
    # Severe content — always masked; red to flag "reviewer attention".
    "SEVERE_RISK":    _SEVERE_RED,
}


# Placeholder-prefix → entity-type label (for colouring the Redacted
# pane). We derive the prefix from the placeholder text, then look up
# any representative entity type that shares the right colour.
_PLACEHOLDER_PREFIX_TO_LABEL = {
    "COACH_NAME":  "COACH",
    "CLIENT_NAME": "CLIENT",
    "PERSON":      "PERSON",
    "ORG":         "ORGANIZATION",
    "LOCATION":    "LOCATION",
    "EMAIL":       "EMAIL_ADDRESS",
    "PHONE":       "PHONE_NUMBER",
    "DOB":         "DATE_OF_BIRTH",
    "ID":          "SG_NRIC",      # all ID types share the blue
    "HEALTH":      "HEALTH",
    "AMOUNT":      "FINANCE_AMOUNT",
    "BELIEF":      "BELIEF",
    "REVIEW":      "SEVERE_RISK",
}

_PLACEHOLDER_SPAN_RE = re.compile(r"\[([A-Z]+(?:_[A-Z]+)*?)(?:_\d+)?\]")


def _redacted_highlight_segments(redacted_text: str):
    """Build (chunk, label) tuples for the Redacted pane so each
    placeholder gets its PII/sensitive/severe colour."""
    if not redacted_text:
        return [("", None)]
    segments: list[tuple[str, Optional[str]]] = []
    cursor = 0
    for m in _PLACEHOLDER_SPAN_RE.finditer(redacted_text):
        if m.start() > cursor:
            segments.append((redacted_text[cursor : m.start()], None))
        prefix = m.group(1)
        label = _PLACEHOLDER_PREFIX_TO_LABEL.get(prefix)
        segments.append((redacted_text[m.start() : m.end()], label))
        cursor = m.end()
    if cursor < len(redacted_text):
        segments.append((redacted_text[cursor:], None))
    return segments


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _sample_choices() -> list[tuple[str, str]]:
    """List (display_name, path) tuples for every *.txt in samples/."""
    files = sorted(_SAMPLES_DIR.glob("*.txt"))
    return [(p.stem.replace("_", " ").title(), str(p)) for p in files]


def _on_sample_select(path: Optional[str]) -> tuple[str, str, str]:
    """Dropdown change handler — fills transcript, coach, and client fields."""
    if not path:
        return "", "", ""
    text, coach, client = load_sample(path)
    return text, coach, client


def _segments_for_highlight(text: str, spans) -> list[tuple[str, Optional[str]]]:
    """Convert spans into the (chunk, label) list that HighlightedText wants."""
    if not spans:
        return [(text, None)]
    out: list[tuple[str, Optional[str]]] = []
    cursor = 0
    for s in sorted(spans, key=lambda x: x.start):
        if s.start > cursor:
            out.append((text[cursor : s.start], None))
        out.append((text[s.start : s.end], s.entity_type))
        cursor = s.end
    if cursor < len(text):
        out.append((text[cursor:], None))
    return out


def _placeholder_type(placeholder: str, original: str, spans) -> str:
    """Infer the entity type behind a placeholder by looking up a span."""
    if placeholder == "[COACH_NAME]":
        return "COACH"
    if placeholder == "[CLIENT_NAME]":
        return "CLIENT"
    norm = original.strip().lower()
    for s in spans:
        if s.entity_type in ("COACH", "CLIENT"):
            continue
        if s.original.strip().lower() == norm:
            return s.entity_type
    return "?"


# Groupings for the summary line. (singular, plural, [entity types]).
_PII_GROUPS = [
    ("name",          "names",          ["PERSON", "FAMILY_MEMBER", "COACH", "CLIENT"]),
    ("organisation",  "organisations",  ["ORGANIZATION"]),
    ("location",      "locations",      ["LOCATION"]),
    ("email",         "emails",         ["EMAIL_ADDRESS"]),
    ("phone",         "phones",         ["PHONE_NUMBER"]),
    ("ID",            "IDs",            ["SG_NRIC", "IN_AADHAAR", "IN_PAN",
                                         "BANK_ACCOUNT", "CREDIT_CARD"]),
    ("date of birth", "dates of birth", ["DATE_OF_BIRTH"]),
]

_SENSITIVE_GROUPS = [
    ("health flag",  "health flags",  ["HEALTH"]),
    ("finance flag", "finance flags", ["FINANCE_AMOUNT"]),
    ("belief flag",  "belief flags",  ["BELIEF"]),
    ("severe span",  "severe spans",  ["SEVERE_RISK"]),
]


def _summarise(counts: dict[str, int], seconds: float, mode: str) -> str:
    """Build the one-line summary shown under the transcript."""
    def gather(groups):
        parts = []
        total = 0
        for singular, plural, keys in groups:
            n = sum(counts.get(k, 0) for k in keys)
            total += n
            if n > 0:
                parts.append(f"{n} {singular if n == 1 else plural}")
        return total, parts

    pii_total, pii_parts = gather(_PII_GROUPS)
    sens_total, sens_parts = gather(_SENSITIVE_GROUPS)

    lines: list[str] = []
    if pii_total:
        lines.append(
            f"Masked **{pii_total}** personal details "
            f"({', '.join(pii_parts)}) in **{seconds:.1f}** seconds."
        )
    if sens_total:
        verb = "Flagged" if mode == "flag" else "Masked"
        lines.append(
            f"{verb} **{sens_total}** sensitive-topic spans "
            f"({', '.join(sens_parts)})."
        )
    if not lines:
        return f"_No personal details detected (processed in {seconds:.1f}s)._"
    return "  \n".join(lines)


# ----------------------------------------------------------------------
# Redact callback
# ----------------------------------------------------------------------
def _on_redact(
    transcript: str,
    coach: str,
    client: str,
    mode_label: str,
    _state,
):
    if not transcript.strip():
        return (
            [("", None)],
            [("", None)],                          # Tab 1 redacted pane (HighlightedText)
            [],
            "_Paste or load a transcript first._",
            gr.update(value="", visible=False),    # Tab 1 severe banner
            None,
            "",                                    # Tab 2 preview
            gr.update(value="", visible=False),    # Tab 2 severe banner
            gr.update(visible=False, value=False), # Tab 2 override checkbox
            "",                                    # Tab 2 usage
            "",                                    # Tab 2 raw output
            "",                                    # Tab 2 rehydrated output
            None,                                  # Tab 2 report state
            gr.update(value="", visible=False),    # Tab 3 severe banner
        )

    mode = mode_label.lower()
    result: RedactionResult = redact(
        transcript,
        coach_name=coach,
        client_name=client,
        sensitive_mode=mode,
    )

    highlighted = _segments_for_highlight(transcript, result.spans)

    mapping_rows = [
        [
            placeholder,
            original,
            _placeholder_type(placeholder, original, result.spans),
        ]
        for placeholder, original in result.mapping.items()
    ]

    summary = _summarise(result.counts, result.seconds, mode)

    redacted_segments = _redacted_highlight_segments(result.redacted_text)

    severe_count = result.counts.get("SEVERE_RISK", 0)
    if severe_count:
        severe_update = gr.update(
            value=(
                "<div style='background:#FEE2E2;border:1px solid #FCA5A5;"
                "border-radius:8px;padding:0.75em 1em;'>"
                "<strong>⚠ Severe content detected</strong> &nbsp;·&nbsp; "
                f"{severe_count} phrase"
                f"{'s' if severe_count != 1 else ''} masked as "
                "<code>[REVIEW_N]</code>. "
                "The AI feedback on tab 2 is blocked by default — review the "
                "ICF Code of Ethics on referral before proceeding."
                "</div>"
            ),
            visible=True,
        )
    else:
        severe_update = gr.update(value="", visible=False)

    # Tab 2 severe-content UI: show banner + checkbox only when there
    # are SEVERE_RISK spans. Checkbox always resets to unchecked so the
    # coach confirms per redaction.
    if severe_count:
        tab2_banner = gr.update(
            value=(
                "<div style='background:#FEE2E2;border:1px solid #FCA5A5;"
                "border-radius:8px;padding:0.75em 1em;'>"
                "<strong>Severe content detected in this transcript.</strong> "
                "The <em>Get feedback from Gemini</em> button will refuse to "
                "send unless you confirm you've consulted a supervisor. "
                "Masked content is never sent to Gemini either way."
                "</div>"
            ),
            visible=True,
        )
        tab2_checkbox = gr.update(visible=True, value=False)
        tab3_banner = gr.update(
            value=(
                "<div style='background:#FEE2E2;border:1px solid #FCA5A5;"
                "border-radius:8px;padding:0.75em 1em;'>"
                "<strong>This session contains severe-content flags.</strong> "
                "The chat will discuss the coach's craft — it will not analyse "
                "what was disclosed. Consider talking to a supervisor or "
                "mentor coach."
                "</div>"
            ),
            visible=True,
        )
    else:
        tab2_banner = gr.update(value="", visible=False)
        tab2_checkbox = gr.update(visible=False, value=False)
        tab3_banner = gr.update(value="", visible=False)

    # Final values populate Tab 2's preview and clear any stale Gemini
    # output from a previous run, so Tab 2 always reflects the latest
    # redaction state.
    return (
        highlighted,
        redacted_segments,
        mapping_rows,
        summary,
        severe_update,
        result,
        result.redacted_text,  # Tab 2 preview (plain text — unchanged)
        tab2_banner,
        tab2_checkbox,
        "",                    # Tab 2 usage (cleared)
        "",                    # Tab 2 raw output (cleared)
        "",                    # Tab 2 rehydrated output (cleared)
        None,                  # Tab 2 report state (cleared)
        tab3_banner,
    )


def _format_usage(u) -> str:
    """One-line Markdown summary of tokens and estimated cost."""
    return (
        f"**{u.prompt_tokens:,}** tokens in &nbsp;→&nbsp; "
        f"**{u.output_tokens:,}** tokens out &nbsp;•&nbsp; "
        f"cost ≈ **${u.cost_usd:.4f}** USD &nbsp;•&nbsp; "
        f"model `{u.model}`"
    )


def _render_report(report: SessionReport) -> str:
    """Pick the best display text for a SessionReport.

    If the structured fields came back, render via `to_prompt_text` so
    the user sees the organised view. Otherwise fall back to the raw
    Gemini reply (so a parse failure stays visible / debuggable instead
    of showing an empty panel).
    """
    if report.competencies or report.overall_summary or report.next_steps:
        return to_prompt_text(report)
    return report.raw_text or "_(Gemini returned an empty response.)_"


def _on_llm_click(state, severe_override: bool):
    """Tab 2 button: send redacted text to Gemini, show usage + raw + rehydrated.

    If the current RedactionResult contains SEVERE_RISK spans and the
    coach has NOT ticked `severe_override`, we refuse to send and
    render a block message instead. On override, `allow_severe=True`
    is passed down so Gemini is told not to interpret the masked
    [REVIEW_N] content.
    """
    if state is None or not getattr(state, "redacted_text", "").strip():
        msg = (
            "**Run tab 1 first.** Redact a transcript on the Redact tab, "
            "then come back here."
        )
        return "", msg, msg, None

    severe_count = state.counts.get("SEVERE_RISK", 0) if state.counts else 0
    if severe_count and not severe_override:
        block_msg = (
            "**AI feedback blocked.**\n\n"
            f"This transcript contains **{severe_count}** span"
            f"{'s' if severe_count != 1 else ''} the detector flagged as "
            "severe (self-harm, violence, or similar). The ICF Code of "
            "Ethics recommends talking to a supervisor or mentor coach "
            "and/or considering a specialist referral before running AI "
            "analysis over these moments.\n\n"
            "If you've done that and still want craft-level feedback from "
            "the AI, tick the checkbox above and press the button again."
        )
        return "", block_msg, block_msg, None

    try:
        report, usage = get_icf_feedback(
            state.redacted_text,
            allow_severe=bool(severe_count),
        )
    except LLMConfigError as e:
        msg = (
            "**Can't call Gemini yet.**\n\n"
            f"{e}\n\n"
            "Then restart the app."
        )
        return "", msg, msg, None
    except Exception as e:  # network / quota / bad model name
        msg = f"**Gemini request failed:** `{type(e).__name__}` — {e}"
        return "", msg, msg, None

    display = _render_report(report)
    rehydrated = rehydrate(display, state.mapping)
    return _format_usage(usage), display, rehydrated, report


# ======================================================================
# Tab 3 — "Talk it through"
# ======================================================================

_PERSONA_DESCRIPTIONS: dict[str, str] = {
    "Mentor coach": (
        "_Direct, evidence-based feedback against the ICF competencies, "
        "with lines you could try._"
    ),
    "Peer coach": (
        "_A fellow coach reflecting with you: curious, collegial, "
        "fewer verdicts._"
    ),
}


def _persona_key(label: str) -> str:
    """Map the radio label to the ChatSession persona key."""
    return "peer" if label == "Peer coach" else "mentor"


def _persona_desc(label: str) -> str:
    return _PERSONA_DESCRIPTIONS.get(label, "")


def _default_cap() -> int:
    try:
        return max(1, int(os.environ.get("CHAT_MAX_QUESTIONS", "50")))
    except ValueError:
        return 50


def _counter_text(session) -> str:
    """Return the 'Questions left: X / N' line."""
    if session is None:
        cap = _default_cap()
        return f"Questions left: **{cap}** / {cap}"
    return f"Questions left: **{session.questions_left}** / {session.questions_cap}"


def _format_chat_usage(turn, session) -> str:
    """Markdown block for the Usage accordion on tab 3."""
    u = turn.usage
    return (
        f"**Last message**  \n"
        f"Input: {u.prompt_tokens:,} ({u.cached_tokens:,} cached)  \n"
        f"Output: {u.output_tokens:,} ({u.thinking_tokens:,} thinking)  \n"
        f"Cost: ${u.cost_usd:.4f}\n\n"
        f"**This session**  \n"
        f"Total cost: ${session.total_cost_usd:.4f} "
        f"across {session.questions_used} question"
        f"{'s' if session.questions_used != 1 else ''}."
    )


_COACH_TURN_RE = re.compile(
    r"\[(\d\d:\d\d)\]\s*COACH:\s*(.+?)(?=\n\[\d\d:\d\d\]|\Z)",
    re.DOTALL,
)


def _coach_turns_from_transcript(
    text: str, max_chars: int = 80
) -> tuple[list[str], dict[str, tuple[str, str]]]:
    """Parse the ORIGINAL transcript into dropdown choices + a lookup.

    Returns `(choices, lookup)` where each choice is a display string
    like `[02:41] What does that number mean to you?` and the lookup
    maps the display back to `(timestamp, full_quote)` so the moment-
    select handler can build the fill-in template.
    """
    choices: list[str] = []
    lookup: dict[str, tuple[str, str]] = {}
    for m in _COACH_TURN_RE.finditer(text or ""):
        ts = m.group(1)
        quote = m.group(2).strip().replace("\n", " ")
        truncated = quote[:max_chars] + ("…" if len(quote) > max_chars else "")
        display = f"[{ts}] {truncated}"
        # Guard against duplicate display strings (rare) so dict keys stay unique.
        if display in lookup:
            display = f"{display} ({len(lookup)})"
        choices.append(display)
        lookup[display] = (ts, quote)
    return choices, lookup


def _readiness_update(red_state, rep_state):
    """Show or hide the 'run tabs 1 and 2 first' banner based on state."""
    if red_state is None or rep_state is None:
        return gr.update(
            value=(
                "_Redact a transcript and generate feedback first "
                "(tabs 1 and 2)._"
            ),
            visible=True,
        )
    return gr.update(visible=False)


# ---- send pipeline (shared by Send button, Enter, and starter clicks) ----

_NOT_READY = (
    "**Please run tab 1 (Redact) and tab 2 (Send to AI) first.** "
    "I need a redacted transcript and the ICF feedback report to talk about."
)


def _on_send_core(
    msg: str,
    chat_history,
    chat_sess,
    redaction,
    report,
    coach_name: str,
    client_name: str,
    persona_label: str,
):
    """Shared send logic — returns 7 values (no msg_input in this list).

    Order:
      chat_history, chat_sess, counter_text,
      ai_sent_md, ai_raw_md, usage_md, persona_notice_update
    """
    chat_history = list(chat_history or [])

    if not msg.strip():
        return (
            chat_history, chat_sess,
            _counter_text(chat_sess),
            gr.update(), gr.update(), gr.update(),
            gr.update(visible=False),
        )

    if redaction is None or report is None:
        chat_history.append({"role": "user", "content": msg})
        chat_history.append({"role": "assistant", "content": _NOT_READY})
        return (
            chat_history, chat_sess,
            _counter_text(chat_sess),
            gr.update(), gr.update(), gr.update(),
            gr.update(visible=False),
        )

    persona_key = _persona_key(persona_label)
    if chat_sess is None:
        ctx = ChatContext(
            redacted_transcript=redaction.redacted_text,
            report=report,
            mapping=redaction.mapping,
            coach_name=coach_name or "",
            client_name=client_name or "",
        )
        chat_sess = ChatSession(ctx, persona=persona_key)
    elif chat_sess.persona != persona_key:
        chat_sess.set_persona(persona_key)

    try:
        turn = chat_sess.ask(msg)
    except LLMConfigError as e:
        chat_history.append({"role": "user", "content": msg})
        chat_history.append({
            "role": "assistant",
            "content": f"**Can't call Gemini.**\n\n{e}",
        })
        return (
            chat_history, chat_sess,
            _counter_text(chat_sess),
            gr.update(), gr.update(), gr.update(),
            gr.update(visible=False),
        )
    except Exception as e:
        chat_history.append({"role": "user", "content": msg})
        chat_history.append({
            "role": "assistant",
            "content": f"**Gemini request failed:** `{type(e).__name__}` — {e}",
        })
        return (
            chat_history, chat_sess,
            _counter_text(chat_sess),
            gr.update(), gr.update(), gr.update(),
            gr.update(visible=False),
        )

    chat_history.append({"role": "user", "content": msg})
    chat_history.append({"role": "assistant", "content": turn.reply_shown})

    # When the cap is hit, ChatSession.ask returns an empty message_sent
    # and a friendly reply; leave the right-column accordions as-is.
    if turn.message_sent:
        ai_sent = f"```\n{turn.message_sent}\n```"
        ai_raw = turn.reply_raw
        usage_md = _format_chat_usage(turn, chat_sess)
    else:
        ai_sent = gr.update()
        ai_raw = gr.update()
        usage_md = gr.update()

    return (
        chat_history, chat_sess,
        _counter_text(chat_sess),
        ai_sent, ai_raw, usage_md,
        gr.update(visible=False),
    )


def _on_send_from_input(msg, *args):
    """Send button / Enter: clear the input textbox after sending."""
    hist, sess, counter, ai_sent, ai_raw, usage, notice = _on_send_core(msg, *args)
    return hist, sess, "", counter, ai_sent, ai_raw, usage, notice


def _on_send_from_starter(msg, *args):
    """Starter click: don't touch the input textbox (preserve any draft)."""
    hist, sess, counter, ai_sent, ai_raw, usage, notice = _on_send_core(msg, *args)
    return hist, sess, gr.update(), counter, ai_sent, ai_raw, usage, notice


def _make_starter_handler(idx: int):
    """Produce a per-button click handler that reads its starter text from state."""
    def handler(starters, *args):
        msg = starters[idx] if 0 <= idx < len(starters) else ""
        return _on_send_from_starter(msg, *args)
    return handler


def _on_moment_select(display, lookup):
    """Fill the msg_input with the 'How did I do at …' template."""
    if not display or not lookup or display not in lookup:
        return gr.update()
    ts, quote = lookup[display]
    return f'How did I do at [{ts}] when I said "{quote}"?'


def _on_persona_change(new_label, chat_sess):
    """Persona switch — if a session exists, start fresh conversation."""
    desc = _persona_desc(new_label)
    if chat_sess is None:
        # Nothing to reset yet — just update the description.
        return (
            desc,
            gr.update(visible=False),
            gr.update(),  # chat_display unchanged
            None,
            gr.update(),  # counter unchanged
        )
    chat_sess.set_persona(_persona_key(new_label))
    notice = f"_Switched to {new_label} — fresh conversation._"
    return (
        desc,
        gr.update(value=notice, visible=True),
        [],  # clear chat_display
        chat_sess,
        _counter_text(chat_sess),
    )


def _on_chat_reset():
    """Full reset of the chat tab's visible state."""
    return (
        [],                                         # chat_display
        None,                                       # chat_sess
        "",                                         # msg_input
        _counter_text(None),                        # counter
        "_No message sent yet._",                   # ai_sent
        "_No reply received yet._",                 # ai_raw
        "_No usage yet._",                          # usage
        gr.update(visible=False),                   # persona_notice
    )


def _populate_moments(transcript):
    """Build the moment-dropdown choices from the current transcript."""
    choices, lookup = _coach_turns_from_transcript(transcript or "")
    return (
        gr.update(choices=choices, value=None),
        lookup,
    )


def _populate_starters(report):
    """Build the 6 starter-button updates from the current SessionReport."""
    if report is None:
        empties = [gr.update(value="", visible=False) for _ in range(6)]
        return (*empties, [""] * 6)
    starters = suggest_starters(report)
    # Pad to exactly 6 if the dynamic helper returned fewer (e.g. no report).
    while len(starters) < 6:
        starters.append("")
    starters = starters[:6]
    updates = [
        gr.update(value=s, visible=bool(s))
        for s in starters
    ]
    return (*updates, starters)


# ----------------------------------------------------------------------
# Blocks layout
# ----------------------------------------------------------------------
SOFT_THEME = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Privacy Shield for Coaching Transcripts") as app:

        # Cross-tab state.
        # `state`              — RedactionResult from Tab 1
        # `report_state`       — SessionReport from Tab 2
        # `chat_state`         — the live ChatSession (Tab 3)
        # `moment_lookup`      — {display -> (timestamp, quote)} for the moment dropdown
        # `starters_state`     — the 6 starter strings suggest_starters() produced
        state = gr.State(None)
        report_state = gr.State(None)
        chat_state = gr.State(None)
        moment_lookup = gr.State({})
        starters_state = gr.State([""] * 6)

        gr.Markdown("# Privacy Shield for Coaching Transcripts (proof of concept)")

        with gr.Tabs():
            # ---------- Redact tab ----------
            with gr.Tab("Redact"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sample = gr.Dropdown(
                            label="Load a sample",
                            choices=_sample_choices(),
                            value=None,
                            info="Fills the transcript and the coach / client fields.",
                        )
                        coach = gr.Textbox(
                            label="Coach name",
                            placeholder="e.g. Anita Menon",
                        )
                        client = gr.Textbox(
                            label="Client name",
                            placeholder="e.g. Priya Raman",
                        )
                        mode = gr.Radio(
                            label="Sensitive topics",
                            choices=["Flag", "Mask"],
                            value="Flag",
                            info="Flag = highlight for review. Mask = replace with neutral tags.",
                        )
                        redact_btn = gr.Button(
                            "Redact",
                            variant="primary",
                            size="lg",
                        )
                    with gr.Column(scale=2):
                        transcript = gr.Textbox(
                            label="Transcript",
                            lines=18,
                            max_lines=40,
                            placeholder=(
                                "Paste a transcript here, or pick a sample from "
                                "the dropdown on the left..."
                            ),
                        )

                summary = gr.Markdown("")

                gr.Markdown(
                    "<p style='color:#64748B; font-size:0.85em; margin-top:0.5em;'>"
                    "Blue = masked personal detail &nbsp;·&nbsp; "
                    "Amber = sensitive topic flagged for review &nbsp;·&nbsp; "
                    "Red = severe content, always masked &amp; blocked from AI"
                    "</p>"
                )
                severe_banner_md = gr.Markdown("", visible=False)
                with gr.Row():
                    with gr.Column():
                        highlighted = gr.HighlightedText(
                            label="Original (highlighted)",
                            color_map=COLOR_MAP,
                            combine_adjacent=False,
                            show_legend=False,
                        )
                    with gr.Column():
                        redacted_out = gr.HighlightedText(
                            label="Redacted",
                            color_map=COLOR_MAP,
                            combine_adjacent=False,
                            show_legend=False,
                        )

                mapping_table = gr.Dataframe(
                    label="Mapping",
                    headers=["Placeholder", "Original", "Type"],
                    interactive=False,
                    wrap=True,
                )

            # ---------- Send to AI tab ----------
            with gr.Tab("Send to AI"):
                gr.Markdown(
                    "### This is exactly what the AI receives.\n"
                    "_Personal details have been replaced with placeholders. "
                    "No real names or identifiers leave this machine._"
                )
                ai_input = gr.Textbox(
                    label="Redacted transcript (what Gemini sees)",
                    lines=18,
                    max_lines=40,
                    interactive=False,
                    placeholder="Run the Redact tab first — the redacted text will appear here.",
                )
                severe_tab2_banner_md = gr.Markdown("", visible=False)
                severe_override_cb = gr.Checkbox(
                    label=(
                        "I've consulted a supervisor or mentor coach "
                        "and understand the limits of AI feedback here. "
                        "Enable the Gemini call."
                    ),
                    value=False,
                    visible=False,
                )
                llm_btn = gr.Button(
                    "Get feedback from Gemini",
                    variant="primary",
                    size="lg",
                )
                usage_md = gr.Markdown("")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**What Gemini returned** (placeholders intact)")
                        raw_out = gr.Markdown("")
                    with gr.Column():
                        gr.Markdown("**What the coach sees** (rehydrated to real names)")
                        rehydrated_out = gr.Markdown("")

            # ---------- Talk it through tab ----------
            with gr.Tab("Talk it through"):
                readiness_md = gr.Markdown(
                    "_Redact a transcript and generate feedback first "
                    "(tabs 1 and 2)._",
                    visible=True,
                )
                severe_tab3_banner_md = gr.Markdown("", visible=False)

                with gr.Row():
                    persona = gr.Radio(
                        choices=["Mentor coach", "Peer coach"],
                        value="Mentor coach",
                        label="Persona",
                    )
                persona_desc_md = gr.Markdown(
                    _PERSONA_DESCRIPTIONS["Mentor coach"]
                )
                persona_notice_md = gr.Markdown("", visible=False)

                with gr.Row():
                    # ---- LEFT: conversation + input + starters + moment ----
                    with gr.Column(scale=3):
                        chat_display = gr.Chatbot(
                            height=420,
                            show_label=False,
                            value=[],
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                show_label=False,
                                placeholder=(
                                    "Ask about a moment, a competency, "
                                    "or what to try next..."
                                ),
                                scale=5,
                                lines=2,
                                max_lines=6,
                            )
                            send_btn = gr.Button(
                                "Send", variant="primary", scale=1,
                            )
                        counter_md = gr.Markdown(_counter_text(None))

                        gr.Markdown("**Starter questions**")
                        with gr.Row():
                            s_btn1 = gr.Button("", visible=False, size="sm")
                            s_btn2 = gr.Button("", visible=False, size="sm")
                            s_btn3 = gr.Button("", visible=False, size="sm")
                        with gr.Row():
                            s_btn4 = gr.Button("", visible=False, size="sm")
                            s_btn5 = gr.Button("", visible=False, size="sm")
                            s_btn6 = gr.Button("", visible=False, size="sm")
                        starter_btns = [
                            s_btn1, s_btn2, s_btn3, s_btn4, s_btn5, s_btn6,
                        ]

                        gr.Markdown("**Ask about a moment**")
                        moment_dd = gr.Dropdown(
                            choices=[],
                            label="Coach turns from the transcript",
                            interactive=True,
                            info=(
                                "Pick one and it fills the box with a "
                                '"How did I do at…" question you can edit.'
                            ),
                        )

                    # ---- RIGHT: inspector accordions ----
                    with gr.Column(scale=1):
                        with gr.Accordion("What the AI saw", open=False):
                            ai_sent_md = gr.Markdown("_No message sent yet._")
                            gr.Markdown("---")
                            ai_raw_md = gr.Markdown("_No reply received yet._")
                        with gr.Accordion("Usage", open=False):
                            usage_detail_md = gr.Markdown("_No usage yet._")

                reset_chat_btn = gr.Button(
                    "Reset conversation", variant="secondary",
                )

        gr.Markdown(
            "<p style='color:#94A3B8; font-size:0.85em; margin-top:2em; "
            "text-align:center;'>"
            "Demo only — use fictional transcripts."
            "</p>"
        )

        # Wire up callbacks.
        sample.change(
            _on_sample_select,
            inputs=[sample],
            outputs=[transcript, coach, client],
        )
        # Common "reset Tab 3 outputs" chain, used whenever Tab 1 or Tab 2
        # re-runs (new context => stale chat).
        reset_tab3_outputs = [
            chat_display, chat_state, msg_input, counter_md,
            ai_sent_md, ai_raw_md, usage_detail_md, persona_notice_md,
        ]

        redact_event = redact_btn.click(
            _on_redact,
            inputs=[transcript, coach, client, mode, state],
            outputs=[
                highlighted,
                redacted_out,
                mapping_table,
                summary,
                severe_banner_md,
                state,
                ai_input,
                severe_tab2_banner_md,
                severe_override_cb,
                usage_md,
                raw_out,
                rehydrated_out,
                report_state,
                severe_tab3_banner_md,
            ],
        )
        redact_event.then(
            _populate_moments,
            inputs=[transcript],
            outputs=[moment_dd, moment_lookup],
        ).then(
            _populate_starters,
            inputs=[report_state],
            outputs=[*starter_btns, starters_state],
        ).then(
            _readiness_update,
            inputs=[state, report_state],
            outputs=[readiness_md],
        ).then(
            _on_chat_reset,
            outputs=reset_tab3_outputs,
        )

        llm_event = llm_btn.click(
            _on_llm_click,
            inputs=[state, severe_override_cb],
            outputs=[usage_md, raw_out, rehydrated_out, report_state],
        )
        llm_event.then(
            _populate_starters,
            inputs=[report_state],
            outputs=[*starter_btns, starters_state],
        ).then(
            _readiness_update,
            inputs=[state, report_state],
            outputs=[readiness_md],
        ).then(
            _on_chat_reset,
            outputs=reset_tab3_outputs,
        )

        # ---------- Tab 3 events ----------
        send_outputs = [
            chat_display, chat_state, msg_input, counter_md,
            ai_sent_md, ai_raw_md, usage_detail_md, persona_notice_md,
        ]
        send_inputs = [
            msg_input, chat_display, chat_state,
            state, report_state, coach, client, persona,
        ]

        send_btn.click(_on_send_from_input, inputs=send_inputs, outputs=send_outputs)
        msg_input.submit(_on_send_from_input, inputs=send_inputs, outputs=send_outputs)

        starter_inputs = [
            starters_state, chat_display, chat_state,
            state, report_state, coach, client, persona,
        ]
        for i, btn in enumerate(starter_btns):
            btn.click(
                _make_starter_handler(i),
                inputs=starter_inputs,
                outputs=send_outputs,
            )

        moment_dd.change(
            _on_moment_select,
            inputs=[moment_dd, moment_lookup],
            outputs=[msg_input],
        )

        persona.change(
            _on_persona_change,
            inputs=[persona, chat_state],
            outputs=[
                persona_desc_md, persona_notice_md,
                chat_display, chat_state, counter_md,
            ],
        )

        reset_chat_btn.click(
            _on_chat_reset,
            outputs=reset_tab3_outputs,
        )

    return app


demo = build_app()

if __name__ == "__main__":
    # Plain launch — Hugging Face Spaces supplies the server host and port.
    # Theme stays as a styling kwarg (not a hosting kwarg).
    demo.launch(theme=SOFT_THEME)
