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

## Secrets required on the Space

Set these in **Settings → Variables and secrets**:

| Name              | Required | Notes                                               |
|-------------------|----------|-----------------------------------------------------|
| `GEMINI_API_KEY`  | Yes      | Needed for the Tab 2 "Get feedback" button.         |
| `GEMINI_MODEL`    | No       | Defaults to `gemini-3.8-flash`.                     |

Both are read from the process environment, so no `.env` file is needed
on the Space.

## Demo-only

Use **fictional** transcripts only. The three bundled samples under
`samples/` are fictional and safe to share.
