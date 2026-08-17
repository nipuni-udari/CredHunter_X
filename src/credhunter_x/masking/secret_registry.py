from __future__ import annotations

from dataclasses import dataclass, field

from credhunter_x.models.candidate import Candidate

_FRAGMENT_LEN = 8


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
        registered value. Values shorter than FRAGMENT_LEN are registered
        whole — they have no such substring, but are still worth catching
        verbatim."""
        result: set[str] = set()
        for value in self._values_by_candidate.values():
            if len(value) < _FRAGMENT_LEN:
                result.add(value)
                continue
            for i in range(len(value) - _FRAGMENT_LEN + 1):
                result.add(value[i : i + _FRAGMENT_LEN])
        return result

    def __len__(self) -> int:
        return len(self._values_by_candidate)

    def __bool__(self) -> bool:
        return bool(self._values_by_candidate)
