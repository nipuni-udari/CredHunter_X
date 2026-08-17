from __future__ import annotations

import math
from collections import Counter


def shannon_entropy(value: str) -> float:
    """Shannon entropy in bits per character. 0.0 for an empty string."""
    if not value:
        return 0.0
    length = len(value)
    counts = Counter(value)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def classify_charset(value: str) -> str:
    """Compact descriptor of which character classes appear in value, e.g.
    "A-Za-z0-9" or "A-Z0-9symbols". Empty string for an empty value."""
    parts = []
    if any(c.isupper() for c in value):
        parts.append("A-Z")
    if any(c.islower() for c in value):
        parts.append("a-z")
    if any(c.isdigit() for c in value):
        parts.append("0-9")
    if any(not c.isalnum() for c in value):
        parts.append("symbols")
    return "".join(parts)
