from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from credhunter_x.guard.leak_guard import LeakGuard
from credhunter_x.llm.client import LLMClient, LLMResponse, LLMToolResponse
from credhunter_x.llm.schema import ClassificationSchema


class GuardedLLMClient:
    """Wraps any LLMClient and makes the leak-guard check unavoidable for
    every outbound call. Production wiring should never construct an LLM
    client without this — the only place an unwrapped client should exist
    is a test exercising the concrete client in isolation (see
    test_litellm_client_mocked.py)."""

    def __init__(self, client: LLMClient, guard: LeakGuard) -> None:
        self._client = client
        self._guard = guard

    def generate(
        self,
        prompt: str,
        *,
        candidate_id: str = "",
        rule_id: str = "",
        raw_permit_candidate_id: str | None = None,
        response_schema: type[BaseModel] = ClassificationSchema,
    ) -> LLMResponse:
        self._guard.check(
            prompt,
            candidate_id=candidate_id,
            rule_id=rule_id,
            raw_permit_candidate_id=raw_permit_candidate_id,
        )
        return self._client.generate(prompt, response_schema=response_schema)

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        candidate_id: str = "",
        rule_id: str = "",
        raw_permit_candidate_id: str | None = None,
    ) -> LLMToolResponse:
        # Checks the FULL conversation history every call, not just the
        # newest message — an unmasked tool result appended on an earlier
        # turn must still be caught here on the next one.
        self._guard.check(
            json.dumps(messages, default=str),
            candidate_id=candidate_id,
            rule_id=rule_id,
            raw_permit_candidate_id=raw_permit_candidate_id,
        )
        return self._client.generate_with_tools(messages, tools)
