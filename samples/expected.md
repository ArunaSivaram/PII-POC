# Expected PII — ground truth per sample

For each transcript below, "**Should mask**" lists every personal detail the
redactor is expected to catch (in **mask** mode any sensitive-topic row below
should also be replaced; in the default **flag** mode those stay in text but
should still appear in `result.spans`).

"**Should NOT mask**" lists the adversarial cases the redactor must leave
untouched — mostly country names, relationship / role words, and common-word
homonyms of names.

---

## samples/01_career_transition.txt

### Should mask

Coach / client (always collapsed to `[COACH]` / `[CLIENT]`):
- `Anita Menon` — coach (header + "Thanks, Anita.")
- `Priya Raman` — client
- Every stand-alone `Priya` → `[CLIENT]`

Other people (`PERSON`):
- `Kenneth` — her manager
- `Rohan` — her husband
- `Mei Lin` — colleague

Organisations (`ORGANIZATION`):
- `Coastline Bank`
- Every stand-alone `Coastline` (short reference)
- `Tanglin Trust` — daughter's school

Locations (`LOCATION`):
- `Melbourne`

Emails (`EMAIL`):
- `priya.raman@coastline-bank.example.com`
- `priya.raman@fastmail.example.com`

Phones (`PHONE_NUMBER`):
- `+65 9123 4567`

### Should NOT mask

- Relationship / role words: `husband`, `daughter`, `father` ("my father's voice"), `teacher`, `manager` (when used generically, not as a name), `colleague`, `founder`
- Date-ish phrases that are not DOB: `Thursday`, `last Tuesday`, `six months`, `two months ago`

### Sensitive topics

- None in this transcript.

---

## samples/02_burnout.txt

### Should mask

Coach / client:
- `Suresh Nair` — coach ("Thanks, Suresh.")
- `Arjun Krishnan` — client
- Every stand-alone `Arjun` → `[CLIENT]`

People (`PERSON`):
- `Rahul Mehta` and every stand-alone `Rahul` — CEO
- `Maya` — COO
- `Priya` — wife (note: different person from the client in sample 01)

Organisations (`ORGANIZATION`):
- `Spectra Technologies`
- Every stand-alone `Spectra`

Locations (`LOCATION`):
- `Palo Alto`
- `Bangalore` (two occurrences)

Identifiers (`[ID_1]`):
- Aadhaar `2341 2341 2346` (Verhoeff-valid)

### Should NOT mask

- Countries: `Singapore`, `India` (country whitelist)
- Relationship / role words: `wife`, `parents`, `doctor`, `CEO`, `COO`, `HR`, `team`, `landlord`, `coach`
- `Amma` (Indian word for mother) — treat as a relationship word, not a name
- Months / weekdays: `July`, `November`, `December`, `March`, `Sunday`, `Tuesday`

### Sensitive topics

Flag in `flag` mode, mask as `[HEALTH_n]` in `mask` mode:
- `generalised anxiety` — the diagnosis
- `burnout` — named condition
- `medication` — the medication the doctor prescribed (two references)

No FINANCE amounts. No BELIEF.

---

## samples/03_tricky_names.txt

### Should mask

Coach / client:
- `Marcus Hale` — coach (appears as `Marcus` exactly once: "Thanks, Marcus.")
- `Alex Carter` — client
- Every stand-alone `Alex` → `[CLIENT]`

People — **only when used as a name** (capitalised, not at the start of a
sentence as an ordinary word):
- `Grace Okonkwo` and every stand-alone `Grace` used as a name (~9 occurrences)
- `Will Chen` and stand-alone `Will` used as a name (~5 occurrences)
- `Hope Alvaro` and stand-alone `Hope` used as a name (~5 occurrences)
- Stand-alone `Rose` used as a name (~3 occurrences)

Organisations (`ORGANIZATION`):
- `Horizon Partners`

### Should NOT mask (adversarial homonyms + role words)

- `with grace` — manner adverbial, not the name
- `I will`, `Will will take` (the second `will` is a modal verb), `will take`, `will do` — modal `will`, not the name
- `I hope`, `hope she'll`, `I hope I rose` — `hope` as a verb
- `I rose`, `rose to it` — `rose` as a verb
- `my boss` — role word, no name attached (four occurrences)
- Other role words: `team`, `colleagues`, `senior engineers`, `founder`

### Sensitive topics

- None in this transcript.

---

## Scoring notes

- Case-insensitive match is expected for the `[COACH]` / `[CLIENT]` pseudo-ids
  (the coach/client matcher is case-insensitive by design); it is **not**
  expected for other `PERSON` tags (GLiNER is case-sensitive, which is the
  whole reason sample 03 works).
- The `Priya` in sample 02 is a different real person from the `Priya` in
  sample 01. Each sample is scored independently.
- A span is a "hit" if it's captured at all; a "false positive" is a span
  masked that shouldn't have been.
