"""CoachBetter — Privacy Shield (proof of concept).

Minimal Gradio UI for the PII redaction demo. Two tabs:
  * Redact     — load/paste a transcript, highlight detected PII by type,
                 show a redacted version and a mapping table.
  * Rehydrate  — (placeholder for the next step).

Guarantees:
  * `.env` loads before gradio/gliner are imported so their telemetry-off
    flags take effect.
  * The transcript is only ever displayed inside the UI widgets; nothing
    is printed to stdout or written to a log.
"""

from __future__ import annotations

# Load .env first — gradio and gliner read env vars at import time.
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from typing import Optional

import gradio as gr

from pii_poc.llm import LLMConfigError, get_icf_feedback
from pii_poc.redactor import (
    RedactionResult,
    SENSITIVE_TOPIC_ENTITIES,
    redact,
    warmup,
)
from pii_poc.rehydrate import rehydrate
from pii_poc.samples import load_sample

# Load the analyzer at import time so Hugging Face Spaces (or any host)
# pays the ~10s spaCy + GLiNER load during startup, not on the first
# user click. Safe to call at module scope — `warmup()` is idempotent
# (second call hits the module-level cache).
warmup()


_SAMPLES_DIR = Path(__file__).parent / "samples"


# ----------------------------------------------------------------------
# Visual palette — soft pastels grouped by meaning.
# Brighter blue for the primary participants (coach/client), paler blues
# for other people, warmer colours for identifiers, pink/amber/grey for
# sensitive-topic flags so reviewers can scan them at a glance.
# ----------------------------------------------------------------------
COLOR_MAP = {
    "COACH":          "#60A5FA",  # blue-400
    "CLIENT":         "#60A5FA",
    "PERSON":         "#BFDBFE",  # blue-200
    "FAMILY_MEMBER":  "#BFDBFE",
    "ORGANIZATION":   "#DDD6FE",  # violet-200
    "LOCATION":       "#BBF7D0",  # green-200
    "EMAIL_ADDRESS":  "#A5F3FC",  # cyan-200
    "PHONE_NUMBER":   "#FED7AA",  # orange-200
    "DATE_OF_BIRTH":  "#FECACA",  # red-200
    "SG_NRIC":        "#FCA5A5",  # red-300
    "IN_AADHAAR":     "#FCA5A5",
    "IN_PAN":         "#FCA5A5",
    "BANK_ACCOUNT":   "#FCA5A5",
    "CREDIT_CARD":    "#FCA5A5",
    "HEALTH":         "#FBCFE8",  # pink-200  (sensitive topic)
    "FINANCE_AMOUNT": "#FEF08A",  # yellow-200 (sensitive topic)
    "BELIEF":         "#E5E7EB",  # gray-200   (sensitive topic)
}


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
            "",
            [],
            "_Paste or load a transcript first._",
            None,
            "",  # Tab 2 preview
            "",  # Tab 2 usage
            "",  # Tab 2 raw output
            "",  # Tab 2 rehydrated output
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

    # Final values populate Tab 2's preview and clear any stale Gemini
    # output from a previous run, so Tab 2 always reflects the latest
    # redaction state.
    return (
        highlighted,
        result.redacted_text,
        mapping_rows,
        summary,
        result,
        result.redacted_text,  # Tab 2 preview
        "",                    # Tab 2 usage (cleared)
        "",                    # Tab 2 raw output (cleared)
        "",                    # Tab 2 rehydrated output (cleared)
    )


def _format_usage(u) -> str:
    """One-line Markdown summary of tokens and estimated cost."""
    return (
        f"**{u.prompt_tokens:,}** tokens in &nbsp;→&nbsp; "
        f"**{u.output_tokens:,}** tokens out &nbsp;•&nbsp; "
        f"cost ≈ **${u.cost_usd:.4f}** USD &nbsp;•&nbsp; "
        f"model `{u.model}`"
    )


def _on_llm_click(state):
    """Tab 2 button: send redacted text to Gemini, show usage + raw + rehydrated."""
    if state is None or not getattr(state, "redacted_text", "").strip():
        msg = "**Run tab 1 first.** Redact a transcript on the Redact tab, then come back here."
        return "", msg, msg

    try:
        raw, usage = get_icf_feedback(state.redacted_text)
    except LLMConfigError as e:
        msg = (
            "**Can't call Gemini yet.**\n\n"
            f"{e}\n\n"
            "Then restart the app."
        )
        return "", msg, msg
    except Exception as e:  # network / quota / bad model name
        msg = f"**Gemini request failed:** `{type(e).__name__}` — {e}"
        return "", msg, msg

    rehydrated = rehydrate(raw, state.mapping)
    return _format_usage(usage), raw, rehydrated


# ----------------------------------------------------------------------
# Blocks layout
# ----------------------------------------------------------------------
SOFT_THEME = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="CoachBetter — Privacy Shield") as app:

        # Cross-tab state. Populated on every Redact click.
        state = gr.State(None)

        gr.Markdown("# CoachBetter — Privacy Shield (proof of concept)")

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

                with gr.Row():
                    with gr.Column():
                        highlighted = gr.HighlightedText(
                            label="Original (highlighted)",
                            color_map=COLOR_MAP,
                            combine_adjacent=False,
                            show_legend=True,
                        )
                    with gr.Column():
                        redacted_out = gr.Textbox(
                            label="Redacted",
                            lines=18,
                            max_lines=40,
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
        redact_btn.click(
            _on_redact,
            inputs=[transcript, coach, client, mode, state],
            outputs=[
                highlighted,
                redacted_out,
                mapping_table,
                summary,
                state,
                ai_input,
                usage_md,
                raw_out,
                rehydrated_out,
            ],
        )
        llm_btn.click(
            _on_llm_click,
            inputs=[state],
            outputs=[usage_md, raw_out, rehydrated_out],
        )

    return app


demo = build_app()

if __name__ == "__main__":
    # Plain launch — Hugging Face Spaces supplies the server host and port.
    # Theme stays as a styling kwarg (not a hosting kwarg).
    demo.launch(theme=SOFT_THEME)
