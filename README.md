---
title: CoachBetter Privacy Shield PoC
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
python_version: "3.11"
pinned: false
---

# CoachBetter — Privacy Shield (proof of concept)

A small demo that redacts personal details in coaching session transcripts
with consistent placeholders **before** an LLM sees the text, then swaps
them back for the coach after the LLM responds.

- Tab 1 **Redact** — paste a transcript (or load a fictional sample),
  highlight detected details by type, produce the redacted version and a
  placeholder mapping.
- Tab 2 **Send to AI** — show exactly what Gemini receives, call Gemini
  for ICF-competency feedback, display token count + estimated cost, and
  render the response in two panels (placeholders intact / rehydrated to
  real names).
- Tab 3 **Talk it through** — chat with an AI mentor coach or peer coach
  about the session. Each coach message is redacted before it reaches
  Gemini; replies are rehydrated back to real names for the coach's
  view.

## Secrets required on the Space

Set these in **Settings → Variables and secrets**:

| Name                 | Required | Notes                                                         |
|----------------------|----------|---------------------------------------------------------------|
| `GEMINI_API_KEY`     | Yes      | Needed for Tab 2 ("Get feedback") and Tab 3 (chat).           |
| `GEMINI_MODEL`       | No       | Defaults to `gemini-3.8-flash`.                               |
| `GEMINI_CHAT_MODEL`  | No       | Overrides the model used by Tab 3 only. Falls back to above.  |
| `CHAT_MAX_QUESTIONS` | No       | Per-session chat cap on Tab 3 (default `50`).                 |

All are read from the process environment, so no `.env` file is needed
on the Space.

## Demo-only

Use **fictional** transcripts only. The three bundled samples under
`samples/` are fictional and safe to share.
