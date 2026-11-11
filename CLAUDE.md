# PII-POC — Project rules for Claude Code

## Purpose
A small, standalone proof-of-concept that shows **personal details in a coaching
session transcript can be replaced with consistent placeholders *before* an LLM
sees the text, and swapped back afterwards for the coach.** The audience is a
product manager demoing the pattern, not a production system.

## Non-negotiables

### Privacy
- **Only use fictional transcripts.** Never put real names, phone numbers,
  emails, or any real client data in `samples/`, tests, screenshots, commit
  messages, or anywhere else in this repo.
- **Do not log or print transcript text to the terminal** beyond what the Gradio
  UI already displays to the user. No `print(transcript)`, no debug dumps of
  raw or redacted transcript content to stdout/stderr/log files.
- Entity *types* and *counts* are fine to log (e.g. `found 3 PERSON, 1 PHONE`).
  Entity *values* and surrounding transcript text are not.

### Toolchain
- **Python 3.11.**
- **`uv` only** for dependency and environment management — always `uv add`,
  `uv run`, `uv sync`. Never invoke `pip` directly.
- Secrets live in `.env` (gitignored). A committed `.env.example` documents the
  required variables.
- `.env` must be loaded (via `python-dotenv`) **before** `gradio` or `gliner`
  are imported — several of their env flags (analytics, telemetry) are read at
  import time.

### Code style
- Keep files **small and readable**. This is a demo, not production code.
- Comment the *why*, not the *what*. A one-line docstring at the top of each
  module explaining its role is helpful.
- No premature abstractions, no factories, no plugin layers. Straight-line code
  is fine.

## Layout
```
app.py              # Gradio UI entry point
pii_poc/
  __init__.py
  recognizers.py    # Presidio + GLiNER recognizer setup
  redactor.py       # transcript -> redacted text + placeholder map
  rehydrate.py      # LLM output + placeholder map -> rehydrated text
  llm.py            # Gemini call wrapper
samples/            # fictional transcripts
scripts/            # one-off scripts (smoke tests etc.)
tests/              # pytest tests
.env.example        # committed; real .env is gitignored
```

## Models
- **PII recognition:** GLiNER with `nvidia/gliner-PII` from Hugging Face,
  wired in as a Presidio custom recognizer.
- **spaCy:** `en_core_web_sm` only. Do **not** add `en_core_web_lg` — the small
  model is enough for this demo and keeps the repo lean.
- **LLM:** Google Gemini via `google-genai`. Default fast model is set in
  `.env.example` (`GEMINI_MODEL`); override via `.env`.
