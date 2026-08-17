from __future__ import annotations

import logging

from credhunter_x.guard.errors import LeakError
from credhunter_x.masking.secret_registry import SecretRegistry

logger = logging.getLogger(__name__)


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
        still caught, including ones that share the same context window."""
        if not self._registry:
            raise LeakError("LeakGuard registry is empty — refusing to permit any outbound call")

        permitted_value = ""
        if raw_permit_candidate_id is not None:
            permitted_value = self._registry.value_for(raw_permit_candidate_id) or ""

        for fragment in self._registry.fragments():
            if fragment in payload and fragment not in permitted_value:
                logger.error(
                    "Potential secret leak blocked (candidate=%s, rule=%s)", candidate_id, rule_id
                )
                raise LeakError(
                    "Outgoing payload contains a fragment of a known secret "
                    f"(candidate={candidate_id})"
                )
