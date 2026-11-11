"""Smoke test: load nvidia/gliner-PII and extract entities from one fictional sentence.

First run downloads ~1–2 GB of model weights into the Hugging Face cache; later
runs are near-instant. Run with: `uv run python scripts/smoke_test.py`.
"""

from __future__ import annotations

# Load env flags (HF telemetry off, etc.) before importing gliner.
from dotenv import load_dotenv

load_dotenv()

from gliner import GLiNER

# Fictional test sentence — the one from the project brief.
SENTENCE = (
    "Hi Priya, I spoke to my manager Ravi at DBS yesterday, "
    "you can reach me on +65 9123 4567."
)

# Labels we want GLiNER to look for. GLiNER is a zero-shot model, so we tell
# it the entity types at inference time rather than it having a fixed schema.
LABELS = ["person", "organization", "phone number"]


def main() -> None:
    print("Loading model nvidia/gliner-PII (first run downloads ~1–2 GB)...")
    model = GLiNER.from_pretrained("nvidia/gliner-PII")

    print(f"Running on sentence: {SENTENCE!r}")
    entities = model.predict_entities(SENTENCE, LABELS)

    if not entities:
        print("No entities found.")
        return

    print(f"Found {len(entities)} entities:")
    for e in entities:
        print(f"  - {e['label']:<14} {e['text']!r}  (score={e['score']:.2f})")


if __name__ == "__main__":
    main()
