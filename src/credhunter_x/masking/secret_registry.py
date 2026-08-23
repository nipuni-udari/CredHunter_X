from __future__ import annotations

import re
from dataclasses import dataclass, field

from credhunter_x.models.candidate import Candidate

# 8 was short enough to collide with unrelated boilerplate by chance; 16
# consecutive matching characters isn't. Shorter values are still matched
# whole.
_FRAGMENT_LEN = 16
_PEM_MARKER_RE = re.compile(r"^-----(BEGIN|END) [^-]+-----$")


def _strip_pem_boilerplate(value: str) -> str:
    """PEM header/footer lines are public format text, not secret
    material -- left in fragments(), they'd trip the guard on content
    that was never sensitive. Stripped only here; value_for() still
    returns the full original value."""
    lines = value.split("\n")
    kept = [line for line in lines if not _PEM_MARKER_RE.match(line.strip())]
    return "\n".join(kept)


@dataclass
class SecretRegistry:
    """Holds every known real secret value across a batch, not just the
    current candidate -- the guard checks payloads against all of it.
    Keyed by candidate id so raw-mode permit can look up which value a
    specific candidate owns."""

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
        registered value's non-boilerplate content. Shorter values are
        registered whole."""
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
