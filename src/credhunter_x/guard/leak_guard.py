from __future__ import annotations

import logging
import re

from credhunter_x.guard.errors import LeakError
from credhunter_x.masking.secret_registry import SecretRegistry

logger = logging.getLogger(__name__)

# CERTIFICATE and PUBLIC KEY blocks are public, and overlap with a private
# key's modulus is expected. PRIVATE KEY blocks are never excused.
_SAFE_PEM_BLOCK_RE = re.compile(
    r"-----BEGIN ([A-Z ]*?(?:CERTIFICATE|PUBLIC KEY))-----.*?-----END \1-----",
    re.DOTALL,
)


def _safe_pem_spans(payload: str) -> list[tuple[int, int]]:
    return [m.span() for m in _SAFE_PEM_BLOCK_RE.finditer(payload)]


def _fully_within_any_span(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(span_start <= start and end <= span_end for span_start, span_end in spans)


class LeakGuard:
    """Fail-closed check on every outgoing LLM payload. Used inside
    GuardedLLMClient, so no classifier call can skip it."""

    def __init__(self, registry: SecretRegistry) -> None:
        self._registry = registry

    def check(
        self,
        payload: str,
        *,
        candidate_id: str = "",
        rule_id: str = "",
        raw_permit_candidate_id: str | None = None,
    ) -> None:
        """Raises LeakError if payload contains part of any registered secret.
        raw_permit_candidate_id excuses only that candidate's own value, and a
        fragment inside a CERTIFICATE or PUBLIC KEY block is also excused."""
        if not self._registry:
            raise LeakError("LeakGuard registry is empty — refusing to permit any outbound call")

        # Every string that belongs to this candidate, including components such
        # as a decoded url password, which may not be a substring of the full value.
        permitted_values: set[str] = set()
        if raw_permit_candidate_id is not None:
            permitted_values = self._registry.values_for(raw_permit_candidate_id)

        safe_spans = _safe_pem_spans(payload)

        for fragment in self._registry.fragments():
            if fragment not in payload:
                continue
            if any(fragment in permitted for permitted in permitted_values):
                continue
            if self._all_occurrences_are_safe(fragment, payload, safe_spans):
                continue

            logger.error(
                "Potential secret leak blocked (candidate=%s, rule=%s)", candidate_id, rule_id
            )
            raise LeakError(
                f"Outgoing payload contains a fragment of a known secret (candidate={candidate_id})"
            )

    @staticmethod
    def _all_occurrences_are_safe(
        fragment: str, payload: str, safe_spans: list[tuple[int, int]]
    ) -> bool:
        if not safe_spans:
            return False
        index = payload.find(fragment)
        while index != -1:
            if not _fully_within_any_span(index, index + len(fragment), safe_spans):
                return False
            index = payload.find(fragment, index + 1)
        return True
