"""Internal format for coaching session feedback.

A stable intermediate shape keeps the chat feature (and anything else
built on top of feedback) from depending on one particular report
layout. The PoC's Gemini output, future richer prompts, and the
external CoachBetter report format will all funnel into a
`SessionReport` before they reach a downstream consumer.

**All data here is in REDACTED form with placeholders.** The function
contracts assume the transcript has been through `pii_poc.redactor.redact()`,
so real names and identifiers never land in a `SessionReport`. Rehydration
happens at render time, not inside this module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# ======================================================================
# Dataclasses
# ======================================================================

@dataclass
class Evidence:
    """A single transcript citation backing a feedback point."""
    timestamp: str   # "MM:SS"
    quote: str       # verbatim line from the redacted transcript
    note: str        # what this evidence demonstrates


@dataclass
class CompetencyResult:
    """Feedback on one of the ICF Core Competencies."""
    competency: str
    # Level/score is deliberately a string so different scales fit
    # (e.g. "PCC", "4/5", "Developing").
    score_or_level: str
    summary: str
    strengths: list[Evidence] = field(default_factory=list)
    development_areas: list[Evidence] = field(default_factory=list)
    pcc_markers_observed: list[str] = field(default_factory=list)
    pcc_markers_missed: list[str] = field(default_factory=list)


@dataclass
class SessionReport:
    """Whole session feedback in the stable internal format.

    `raw_text` always holds the original report (JSON string, free-form
    prose, whatever the source was) so no information is lost even when
    structured parsing fails.
    """
    overall_summary: str
    competencies: list[CompetencyResult]
    next_steps: list[str]
    raw_text: str


# ======================================================================
# Serialise for LLM context
# ======================================================================

def to_prompt_text(report: SessionReport) -> str:
    """Compact, readable plain-text rendering for sending to the LLM.

    Target under ~4,000 tokens (~16k chars) for a full 8-competency
    report. Stable layout so the chat LLM doesn't need to re-infer
    structure each turn. `raw_text` is intentionally **not** included
    — the caller has it separately if needed.
    """
    lines: list[str] = ["Session feedback report", ""]

    if report.overall_summary:
        lines.append(f"Overall summary: {report.overall_summary}")
        lines.append("")

    if report.competencies:
        lines.append("Competencies:")
        for i, c in enumerate(report.competencies, 1):
            lines.append("")
            lines.append(f"[{i}] {c.competency} — level: {c.score_or_level}")
            if c.summary:
                lines.append(f"    Summary: {c.summary}")
            if c.strengths:
                lines.append("    Strengths:")
                for e in c.strengths:
                    lines.append(
                        f"      - [{e.timestamp}] \"{e.quote}\" — {e.note}"
                    )
            if c.development_areas:
                lines.append("    Development areas:")
                for e in c.development_areas:
                    lines.append(
                        f"      - [{e.timestamp}] \"{e.quote}\" — {e.note}"
                    )
            if c.pcc_markers_observed:
                lines.append(
                    f"    PCC markers observed: {', '.join(c.pcc_markers_observed)}"
                )
            if c.pcc_markers_missed:
                lines.append(
                    f"    PCC markers missed: {', '.join(c.pcc_markers_missed)}"
                )

    if report.next_steps:
        lines.append("")
        lines.append("Next steps:")
        for i, step in enumerate(report.next_steps, 1):
            lines.append(f"  {i}. {step}")

    return "\n".join(lines)


# ======================================================================
# Parsers
# ======================================================================

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*\n", flags=re.IGNORECASE)


def _extract_json_blob(text: str) -> dict[str, Any]:
    """Find a JSON object inside `text`, tolerating markdown fences.

    Strategy:
      1. Strip opening/closing ```...``` fences if present.
      2. Try `json.loads` on the stripped body.
      3. Fall back to the outermost `{...}` substring and parse that.

    Raises `ValueError` when no JSON object can be recovered.
    """
    s = text.strip()
    fence = _FENCE_OPEN.match(s)
    if fence:
        s = s[fence.end() :]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()

    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: find outermost {...} and try again.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(s[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    raise ValueError("no JSON object found")


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def _as_str_list(x: Any) -> list[str]:
    if not isinstance(x, list):
        return []
    return [str(item) for item in x if item]


def _evidence_from(d: Any) -> Evidence:
    """Coerce an arbitrary value into an Evidence.

    Gemini sometimes puts a bare string where a dict was expected; keep
    the information in `quote` rather than dropping the whole item.
    """
    if not isinstance(d, dict):
        return Evidence(timestamp="", quote=_as_str(d), note="")
    return Evidence(
        timestamp=_as_str(d.get("timestamp")),
        quote=_as_str(d.get("quote")),
        note=_as_str(d.get("note")),
    )


def from_poc_feedback(text: str) -> SessionReport:
    """Build a SessionReport from Gemini's PoC feedback reply.

    Best-effort — if the reply doesn't contain valid JSON matching the
    expected shape, the structured fields come back empty but `raw_text`
    still holds the original reply so nothing is thrown away.
    """
    try:
        data = _extract_json_blob(text)
    except ValueError:
        return SessionReport(
            overall_summary="",
            competencies=[],
            next_steps=[],
            raw_text=text,
        )

    competencies: list[CompetencyResult] = []
    for c in (data.get("competencies") or []):
        if not isinstance(c, dict):
            continue
        competencies.append(
            CompetencyResult(
                competency=_as_str(c.get("competency")),
                score_or_level=_as_str(c.get("score_or_level")),
                summary=_as_str(c.get("summary")),
                strengths=[_evidence_from(e) for e in (c.get("strengths") or [])],
                development_areas=[
                    _evidence_from(e) for e in (c.get("development_areas") or [])
                ],
                pcc_markers_observed=_as_str_list(c.get("pcc_markers_observed")),
                pcc_markers_missed=_as_str_list(c.get("pcc_markers_missed")),
            )
        )

    return SessionReport(
        overall_summary=_as_str(data.get("overall_summary")),
        competencies=competencies,
        next_steps=_as_str_list(data.get("next_steps")),
        raw_text=text,
    )


def from_coachbetter_report(data: dict) -> SessionReport:
    """Adapt the external CoachBetter report format into a SessionReport.

    Not yet implemented — we'll fill this in once we have a sample of the
    real report shape to map against.
    """
    raise NotImplementedError("Waiting for CoachBetter report sample")
