from __future__ import annotations

import re
from dataclasses import dataclass, field

from credhunter_x.models.candidate import Candidate

_FRAGMENT_LEN = 16  # 8 was short enough to coincidentally collide with common
# short boilerplate (code formatting, sequential dummy digits in test
# fixtures) between two otherwise-unrelated secrets. 16 consecutive matching
# characters is not something short structural/dummy text produces by
# chance -- only an actual shared secret does. Values shorter than
# FRAGMENT_LEN are still registered and matched whole, so short secrets
# remain fully covered.
_PEM_MARKER_RE = re.compile(r"^-----(BEGIN|END) [^-]+-----$")


def _strip_pem_boilerplate(value: str) -> str:
    """PEM header/footer marker lines (e.g. "-----BEGIN RSA PRIVATE KEY-----")
    are identical, public format text shared by every key of that type — not
    secret material. Left in fragments(), one key's boilerplate can collide
    with a differently-formatted key's boilerplate (e.g. "-----END PRIVATE
    KEY-----" vs "-----END RSA PRIVATE KEY-----" don't align at every 8-char
    offset) and trip the guard on content that was never actually sensitive.
    Stripped only for fragment generation — value_for() still returns the
    full original value, markers included, since raw_permit / masking
    replacement elsewhere need the exact matched text."""
    lines = value.split("\n")
    kept = [line for line in lines if not _PEM_MARKER_RE.match(line.strip())]
    return "\n".join(kept)


@dataclass
class SecretRegistry:
    """Holds every known real secret value across a batch (not just the
    current candidate) — the leak guard checks outgoing payloads against
    this, so it must be populated with everything a run could touch,
    including secrets in other files an agentic tool call might pull in.

    Keyed by candidate id (not just a flat set of values) so the guard's
    raw-mode permit mechanism can look up "which value does this specific
    candidate own" rather than permitting by value directly.
    """

    _values_by_candidate: dict[str, str] = field(default_factory=dict)

    def register(self, candidate_id: str, value: str) -> None:
        if value:
            self._values_by_candidate[candidate_id] = value

    def register_candidates(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            self.register(candidate.id, candidate.matched_value)

    def value_for(self, candidate_id: str) -> str | None:
        return self._values_by_candidate.get(candidate_id)

    def fragments(self) -> set[str]:
        """All contiguous FRAGMENT_LEN-character substrings of every
        registered value's non-boilerplate content. Values shorter than
        FRAGMENT_LEN are registered whole — they have no such substring, but
        are still worth catching verbatim."""
        result: set[str] = set()
        for value in self._values_by_candidate.values():
            stripped = _strip_pem_boilerplate(value)
            if not stripped:
                continue
            if len(stripped) < _FRAGMENT_LEN:
                result.add(stripped)
                continue
            for i in range(len(stripped) - _FRAGMENT_LEN + 1):
                result.add(stripped[i : i + _FRAGMENT_LEN])
        return result

    def __len__(self) -> int:
        return len(self._values_by_candidate)

    def __bool__(self) -> bool:
        return bool(self._values_by_candidate)
