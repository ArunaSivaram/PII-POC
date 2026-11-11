"""Side-by-side phone-number detection: GLiNER vs Presidio's PhoneRecognizer.

Runs two test sets through both detectors:
  1. Clean    — well-formatted intl numbers in natural phone-number context.
  2. Hard     — context-free digits, false-positive bait, obfuscation,
                multiple numbers, non-phone digit strings.

Expected PASS result per row:
  - "yes"  = row contains a phone number; the tool should return it.
  - "no"   = row does NOT contain a phone number; the tool should return nothing.

Run with: `uv run python scripts/phone_bench.py`.
"""

from __future__ import annotations

# Load env flags before importing gliner / presidio.
from dotenv import load_dotenv

load_dotenv()

from gliner import GLiNER
from presidio_analyzer.predefined_recognizers import PhoneRecognizer

# Clean set: (country, format, sentence).
CLEAN = [
    ("SG", "intl +65",    "Call Priya on +65 9123 4567 after lunch."),
    ("GB", "intl +44",    "Reach James at +44 20 7946 0958 tomorrow."),
    ("GB", "local 07",    "Jen's mobile is 07700 900123 if you need it."),
    ("US", "(area) ###",  "You can dial (415) 555-0132 for the front desk."),
    ("US", "intl +1",     "Rob answers +1 212-555-0199 most evenings."),
    ("IN", "intl +91",    "Try Aarav on +91 98765 43210 after 7pm."),
    ("DE", "intl +49",    "Anna picks up on +49 30 12345678 usually."),
    ("FR", "intl +33",    "The Paris office is +33 1 42 86 82 00."),
    ("AU", "intl +61",    "Sydney contact: +61 2 9876 5432, ask for Ellie."),
    ("JP", "intl +81",    "Yuki's number is +81 3-1234-5678."),
    ("CN", "intl +86",    "Mei is on +86 138 0013 8000 if needed."),
    ("BR", "intl +55",    "Try Lucas at +55 11 98765-4321."),
]

# Hard set: each row says what we want the tool to do.
# (tag, expect — "yes" or "no", sentence, note).
HARD = [
    ("bare",       "yes", "+65 9123 4567",
        "context-free digits — no verb/intro, should still be caught"),
    ("spelled",    "no",  "Call me on nine one two three four five six seven.",
        "digits spelled out as words — neither tool handles this"),
    ("creditcard", "no",  "Charge went through on 4111 1111 1111 1111 last night.",
        "16-digit card (Visa test) — looks phone-ish, isn't a phone"),
    ("tracking",   "no",  "Your tracking ID is 1Z999AA10123456784, delivery Monday.",
        "alphanumeric courier ID — should NOT be tagged as a phone"),
    ("two_nums",   "yes", "Try +44 20 7946 0958 or if no answer +44 20 7946 0959.",
        "two phone numbers in one sentence — does the tool catch both?"),
    ("dotted",     "yes", "Priya is on +91.98765.43210 these days.",
        "dots as separators instead of spaces"),
    ("date",       "no",  "The audit meeting is on 2024-12-31 at 10 AM.",
        "date in ISO format — digit groups but not a phone"),
    ("short_code", "no",  "Dial 999 if it's urgent.",
        "emergency short code — not really a 'personal phone' PII"),
]

REGIONS = ("US", "GB", "SG", "IN", "DE", "FR", "AU", "JP", "CN", "BR")


def gliner_hits(model: GLiNER, text: str) -> list[str]:
    """Return every 'phone number' span GLiNER finds in `text`."""
    return [e["text"] for e in model.predict_entities(text, ["phone number"])]


def presidio_hits(rec: PhoneRecognizer, text: str) -> list[str]:
    """Return every phone span Presidio finds in `text`."""
    results = rec.analyze(text=text, entities=["PHONE_NUMBER"], nlp_artifacts=None)
    return [text[r.start : r.end] for r in results]


def fmt_hits(hits: list[str]) -> str:
    """Format a hit list for the table: empty -> '—', else comma-joined."""
    return ", ".join(hits) if hits else "—"


def score(hits: list[str], expect: str) -> str:
    """Compare what the tool returned vs what we expected ('yes' / 'no')."""
    if expect == "yes":
        return "pass" if hits else "FAIL (missed)"
    return "pass" if not hits else f"FAIL (false +)"


def run_clean(model: GLiNER, rec: PhoneRecognizer) -> None:
    print("\n=== Clean set: well-formatted intl numbers in natural context ===\n")
    header = f"{'Country':<8}{'Format':<14}{'GLiNER':<22}{'Presidio':<22}"
    print(header)
    print("-" * len(header))
    g = p = 0
    for country, fmt, sentence in CLEAN:
        gh = gliner_hits(model, sentence)
        ph = presidio_hits(rec, sentence)
        g += bool(gh); p += bool(ph)
        print(f"{country:<8}{fmt:<14}{fmt_hits(gh):<22}{fmt_hits(ph):<22}")
    total = len(CLEAN)
    print(f"\nGLiNER: {g}/{total} hits    Presidio: {p}/{total} hits")


def run_hard(model: GLiNER, rec: PhoneRecognizer) -> None:
    print("\n=== Hard set: adversarial cases (see 'Expect' column) ===\n")
    header = (f"{'Tag':<12}{'Expect':<8}{'GLiNER hits':<26}{'GLiNER':<16}"
              f"{'Presidio hits':<26}{'Presidio':<16}")
    print(header)
    print("-" * len(header))
    g_pass = p_pass = 0
    for tag, expect, sentence, _note in HARD:
        gh = gliner_hits(model, sentence)
        ph = presidio_hits(rec, sentence)
        gs = score(gh, expect)
        ps = score(ph, expect)
        g_pass += gs == "pass"
        p_pass += ps == "pass"
        print(f"{tag:<12}{expect:<8}{fmt_hits(gh)[:24]:<26}{gs:<16}"
              f"{fmt_hits(ph)[:24]:<26}{ps:<16}")
    total = len(HARD)
    print(f"\nGLiNER: {g_pass}/{total} correct    Presidio: {p_pass}/{total} correct")
    print("\nNotes on each case:")
    for tag, expect, _sentence, note in HARD:
        print(f"  [{tag}] ({expect}) {note}")


def main() -> None:
    print("Loading nvidia/gliner-PII (cached from smoke test)...")
    model = GLiNER.from_pretrained("nvidia/gliner-PII")
    rec = PhoneRecognizer(supported_regions=REGIONS)

    run_clean(model, rec)
    run_hard(model, rec)


if __name__ == "__main__":
    main()
