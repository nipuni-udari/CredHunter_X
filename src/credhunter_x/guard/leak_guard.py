from __future__ import annotations

import logging
import re

from credhunter_x.guard.errors import LeakError
from credhunter_x.masking.secret_registry import SecretRegistry

logger = logging.getLogger(__name__)

# CERTIFICATE/PUBLIC KEY blocks aren't secret -- byte overlap with a
# private key's modulus is expected. Never matches PRIVATE KEY headers: a
# second private key sharing bytes with the one under review is a real leak.
_SAFE_PEM_BLOCK_RE = re.compile(
    r"-----BEGIN ([A-Z ]*?(?:CERTIFICATE|PUBLIC KEY))-----.*?-----END \1-----",
    re.DOTALL,
)


def _safe_pem_spans(payload: str) -> list[tuple[int, int]]:
    return [m.span() for m in _SAFE_PEM_BLOCK_RE.finditer(payload)]


def _fully_within_any_span(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(span_start <= start and end <= span_end for span_start, span_end in spans)


class LeakGuard:
    """Fail-closed check run on every outgoing LLM payload. Composed into
    GuardedLLMClient, which intercepts every call a classifier makes --
    production code should never construct an LLM client without it."""

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
        """Raises LeakError if `payload` contains a fragment of any
        registered secret. `raw_permit_candidate_id`, if given, excuses
        only that candidate's own value -- every other candidate's secret
        still trips it. A fragment fully inside a CERTIFICATE/PUBLIC KEY
        PEM block is also excused (see _SAFE_PEM_BLOCK_RE); any occurrence
        outside one still trips the guard."""
        if not self._registry:
            raise LeakError("LeakGuard registry is empty — refusing to permit any outbound call")

        # Every string belonging to this candidate, not just its whole
        # matched value -- a credential component extracted from it (e.g. a
        # decoded url password) is equally its own secret, and is not
        # necessarily a substring of the full value.
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
