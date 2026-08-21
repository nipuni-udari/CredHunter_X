from __future__ import annotations

import logging
import re

from credhunter_x.guard.errors import LeakError
from credhunter_x.masking.secret_registry import SecretRegistry

logger = logging.getLogger(__name__)

# A PEM block whose own header names it a public key or certificate is not
# secret material by definition -- public keys and certificates are meant to
# be shared. Byte overlap with a private key's modulus (e.g. a certificate
# embedding the public half of a key pair) is expected, harmless
# cryptography, not a leak. The backreference ties BEGIN and END to the
# exact same header text, so a "PUBLIC KEY" block can't be closed by an
# unrelated "PRIVATE KEY" END marker. Deliberately does NOT match a block
# whose own header says PRIVATE KEY, at any prefix -- a second, genuinely
# different private key sharing fragments with the one under review is
# exactly the leak this guard exists to catch, not a false positive to wave
# through. This trusts the header text at face value rather than
# cryptographically verifying the key relationship -- a reasonable
# tradeoff scanning a known research dataset, not a defence against someone
# deliberately crafting a fake header to smuggle a real secret past this
# check.
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
    GuardedLLMClient (Milestone 5), which intercepts every call a classifier
    makes — production wiring should never construct an LLM client without
    it (see the plan: "no code path that can obtain an unguarded client")."""

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
        fragments that belong to that one candidate's own value (the RAW
        treatment's intended send) — every other candidate's secret is
        still caught, including ones that share the same context window.

        A fragment match that falls entirely inside a PEM block explicitly
        labeled CERTIFICATE or PUBLIC KEY is also excused — see
        _SAFE_PEM_BLOCK_RE. Any occurrence of the fragment outside such a
        block still trips the guard, even if another occurrence is safe."""
        if not self._registry:
            raise LeakError("LeakGuard registry is empty — refusing to permit any outbound call")

        permitted_value = ""
        if raw_permit_candidate_id is not None:
            permitted_value = self._registry.value_for(raw_permit_candidate_id) or ""

        safe_spans = _safe_pem_spans(payload)

        for fragment in self._registry.fragments():
            if fragment not in payload or fragment in permitted_value:
                continue
            if self._all_occurrences_are_safe(fragment, payload, safe_spans):
                continue

            logger.error(
                "Potential secret leak blocked (candidate=%s, rule=%s)", candidate_id, rule_id
            )
            raise LeakError(
                "Outgoing payload contains a fragment of a known secret "
                f"(candidate={candidate_id})"
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
