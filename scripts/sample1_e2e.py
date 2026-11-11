"""End-to-end smoke test on sample 1: load -> redact -> (fake LLM reply) -> rehydrate.

Reports counts, durations, and rehydration correctness WITHOUT printing
any transcript text — per the project rule, only entity types/counts are
safe to log.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root on sys.path so `pii_poc` is importable when this script
# is invoked directly (uv run python scripts/sample1_e2e.py).
sys.path.insert(0, str(Path(__file__).parent.parent))

# .env must load before gliner / gradio imports.
from dotenv import load_dotenv

load_dotenv()

from pii_poc.llm import LLMConfigError, get_icf_feedback
from pii_poc.redactor import redact
from pii_poc.rehydrate import rehydrate
from pii_poc.samples import load_sample


def main() -> None:
    sample_path = Path(__file__).parent.parent / "samples" / "01_career_transition.txt"
    transcript, coach, client = load_sample(sample_path)
    print(f"Loaded sample: coach={coach!r}  client={client!r}")
    print(f"Transcript size: {len(transcript)} chars")

    print("Running redactor (sensitive_mode=flag)...")
    result = redact(transcript, coach_name=coach, client_name=client, sensitive_mode="flag")

    # Counts only — no transcript content.
    print(f"Redact took {result.seconds:.2f}s")
    print(f"Mapping entries: {len(result.mapping)}")
    print(f"Counts by type: {result.counts}")
    print(f"Spans detected: {len(result.spans)}")
    print(f"Email count (should be > 0 after fix): {result.counts.get('EMAIL_ADDRESS', 0)}")

    # ---- Live Gemini call ----
    print()
    print("Calling Gemini for ICF feedback...")
    try:
        raw, usage = get_icf_feedback(result.redacted_text)
    except LLMConfigError as e:
        print(f"[skip] {e}")
        return

    print(f"Gemini model:   {usage.model}")
    print(f"Prompt tokens:  {usage.prompt_tokens:,}")
    print(f"Output tokens:  {usage.output_tokens:,}")
    print(f"Total tokens:   {usage.total_tokens:,}")
    print(f"Cost (USD):     ${usage.cost_usd:.4f}")
    print(f"Reply length:   {len(raw)} chars")

    # Verify the reply contains at least one placeholder (Gemini should
    # follow the prompt and quote them).
    any_placeholder = any(p in raw for p in result.mapping)
    print(f"Reply mentions at least one placeholder: {any_placeholder}")

    # Rehydrate and spot-check that the coach/client name comes back in.
    rehydrated = rehydrate(raw, result.mapping)
    print(f"Rehydrated reply contains coach name '{coach}': {coach in rehydrated}")

    # Make sure no placeholder survived rehydration (the mapping covered it).
    still_present = [p for p in result.mapping if p in rehydrated]
    print(f"Placeholders left after rehydrate: {still_present or 'none'}")


if __name__ == "__main__":
    main()
