"""Read a sample transcript file and extract its coach/client header.

Sample files begin with a comment line of the form:

    # coach: Anita Menon | client: Priya Raman

`load_sample(path)` returns `(transcript_text, coach_name, client_name)`.
The header is stripped from the returned transcript. If the header is
missing, both names come back as empty strings and the full file content
is returned.
"""

from __future__ import annotations

import re
from pathlib import Path

_HEADER = re.compile(
    r"^#\s*coach\s*:\s*(.+?)\s*\|\s*client\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)


def load_sample(path: str | Path) -> tuple[str, str, str]:
    """Return `(transcript, coach_name, client_name)` from a sample file."""
    raw = Path(path).read_text(encoding="utf-8")
    lines = raw.splitlines()
    coach = client = ""

    # Look at the first non-blank line for the header.
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        m = _HEADER.match(line)
        if m:
            coach = m.group(1).strip()
            client = m.group(2).strip()
            lines = lines[i + 1 :]
        break

    # Trim leading blank lines from the body.
    while lines and not lines[0].strip():
        lines.pop(0)

    return "\n".join(lines), coach, client
